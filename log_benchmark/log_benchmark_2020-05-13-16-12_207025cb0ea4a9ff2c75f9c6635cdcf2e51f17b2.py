
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f5a37e62fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 16:12:55.363130
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 16:12:55.367311
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 16:12:55.370761
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 16:12:55.374455
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f5a43c2c438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356168.2188
Epoch 2/10

1/1 [==============================] - 0s 113ms/step - loss: 321121.0625
Epoch 3/10

1/1 [==============================] - 0s 100ms/step - loss: 254022.1719
Epoch 4/10

1/1 [==============================] - 0s 109ms/step - loss: 182345.4844
Epoch 5/10

1/1 [==============================] - 0s 102ms/step - loss: 117165.7109
Epoch 6/10

1/1 [==============================] - 0s 112ms/step - loss: 70606.3281
Epoch 7/10

1/1 [==============================] - 0s 104ms/step - loss: 42416.4180
Epoch 8/10

1/1 [==============================] - 0s 98ms/step - loss: 26673.4902
Epoch 9/10

1/1 [==============================] - 0s 110ms/step - loss: 17816.4551
Epoch 10/10

1/1 [==============================] - 0s 104ms/step - loss: 12591.5371

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.09069262  4.082408    6.8414116   6.6094365   5.3371634   5.653812
   5.4054513   5.4794264   5.3684506   6.106044    5.698079    6.144597
   5.4846563   5.414894    6.037649    5.972601    4.644856    5.3664017
   4.6043363   6.1247153   5.670611    5.6850014   6.0467234   5.454176
   6.1765265   6.763973    4.487205    5.4692955   6.344596    5.821742
   3.9339888   6.408799    6.203635    5.026435    5.194275    6.3569937
   5.213689    4.517432    6.0545855   6.2513433   6.689634    5.219574
   4.631667    4.083979    4.870955    4.7915077   4.4906864   5.5212116
   4.071425    6.9235287   5.692241    5.698596    6.1955028   5.1519947
   5.6028895   4.85504     6.7696133   6.4534006   5.2719965   5.983312
  -1.0133622   0.61489725  0.21668142  0.5788822  -0.6646657   0.1769167
   0.7734644  -0.19966935  1.4585819   1.0264335   0.18286543 -0.10645013
  -1.5067143  -0.2360542   0.36590755  1.5319538  -0.6322414   1.23148
  -0.8881282   1.733791   -1.9350169   0.30363625 -1.370994    1.6091121
   0.8995079   0.12172131 -0.49898994 -0.8956439  -0.3783378   1.455313
  -0.41192073  0.47519082  0.09491287  1.8908539   0.19167684  1.3079753
   0.6523666   1.226304    0.69450825  0.58517766  1.0424626   1.3615594
  -0.5264294   2.004817   -0.23492385 -0.16668706 -0.02132212 -0.02869244
   0.9762631  -0.13415267 -0.11983193  0.8007238  -0.71548367  1.5367666
   0.27122396 -0.4352188   1.0795715  -0.35711724  0.5549883   0.3969891
  -1.2405789  -0.5271348   1.1034205   0.26270074  1.5186777  -0.765848
  -1.366566   -0.32398754 -0.94816214  1.2717588   1.0186713   0.12494029
   0.2828555  -0.45630258 -0.27299362 -0.01653634  2.2449968  -0.16864441
   0.38745624 -0.509365   -0.44832778  1.4040564  -0.29988438  0.61102194
   0.7847273   0.32389003  0.7547448  -0.30149144 -0.1289003   0.34531456
  -0.18578945 -1.290177   -0.17425741 -1.2427425   0.552627    1.6153498
   1.3581808  -0.37598163  1.2963867   0.9260593   0.44921243  0.4494053
   1.05256     1.1885815   0.40962255  0.8884309  -0.33741373 -0.11696
   0.33427703 -0.05769826  0.27298886  1.6347611   0.41889757  1.6994911
  -0.27062112 -0.36300254  0.0188915  -0.56136525  0.10557802  0.58092946
   0.07636189  6.529544    5.17502     6.4735904   6.4843173   6.38947
   6.01968     5.8922877   6.0026736   5.5772953   5.9023027   5.384307
   6.323185    5.138002    5.4131765   7.415593    7.3342314   7.140688
   6.736669    6.317201    5.745243    5.952282    4.841633    7.0685034
   6.155535    6.970076    6.443313    5.7949014   6.4659534   6.2978125
   5.9074984   5.0347943   4.721315    6.038958    5.131574    7.120668
   6.6961756   6.8276796   6.2903185   7.864148    7.082983    5.3650246
   6.8756766   7.255253    5.7066503   7.2877192   5.8202925   5.140016
   5.905588    6.332247    5.381197    7.259975    7.552845    5.96568
   6.626489    6.375041    6.7996764   5.0443306   6.0602074   5.839611
   0.24004132  1.9293153   1.8102689   0.8336512   0.43541938  1.0861044
   1.4089489   1.062232    1.1972892   1.2213614   2.202364    0.59357077
   1.4441948   0.4342156   0.5663088   3.2004137   0.7563105   0.9198207
   0.37520498  0.5638815   2.9221897   1.0738844   0.37592995  2.3398554
   0.67706156  0.24943727  0.9902245   0.7707521   1.198512    0.7912299
   0.7691663   1.5918667   2.8995414   0.3307569   2.2619896   0.7962408
   1.3070729   0.27168572  2.0016675   0.76376045  0.98402876  1.5920725
   2.876279    0.6763139   3.0249553   2.2651224   1.4412775   1.4111423
   1.5537479   2.2298093   0.5116806   0.9374746   1.5176828   0.81002307
   0.5658159   0.15732586  0.59228736  1.8298533   2.0719485   2.5613408
   1.3533375   0.6853697   0.6880223   1.3527663   2.7582889   1.2770159
   2.4309187   0.25369382  1.6975217   2.561128    0.8375934   0.358733
   0.686528    1.3905075   0.6277137   0.9299905   2.9025416   1.2249047
   1.8801396   1.4293463   0.56225604  1.4368508   1.4496825   1.0285528
   0.32764792  0.5460515   0.94583565  3.264031    0.7505138   2.5411406
   0.44565398  1.2884374   2.5724912   1.6558241   1.2803984   2.139992
   2.4158044   0.26415658  0.7163834   1.0423273   2.0133827   2.1444478
   0.7835991   1.3770057   1.088101    1.0021937   0.91134816  2.053573
   1.048061    1.0027236   1.1930231   3.1077175   0.43808484  1.8624737
   2.7898865   0.49878109  1.1083472   1.975081    2.2582104   0.43854237
   5.657789   -5.0720744  -6.084234  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 16:13:04.447120
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.6136
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 16:13:04.451885
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9357.17
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 16:13:04.455729
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   97.0369
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 16:13:04.460538
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -837.001
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140025119695984
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140023892579048
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140023892579552
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140023892580056
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140023892580560
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140023892581064

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f5a37b22c18> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.470738
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.435208
grad_step = 000002, loss = 0.404287
grad_step = 000003, loss = 0.371581
grad_step = 000004, loss = 0.333268
grad_step = 000005, loss = 0.296359
grad_step = 000006, loss = 0.277772
grad_step = 000007, loss = 0.273281
grad_step = 000008, loss = 0.250827
grad_step = 000009, loss = 0.227326
grad_step = 000010, loss = 0.213603
grad_step = 000011, loss = 0.204231
grad_step = 000012, loss = 0.193866
grad_step = 000013, loss = 0.182746
grad_step = 000014, loss = 0.171064
grad_step = 000015, loss = 0.159071
grad_step = 000016, loss = 0.147806
grad_step = 000017, loss = 0.138299
grad_step = 000018, loss = 0.130025
grad_step = 000019, loss = 0.121692
grad_step = 000020, loss = 0.113216
grad_step = 000021, loss = 0.105137
grad_step = 000022, loss = 0.096788
grad_step = 000023, loss = 0.088445
grad_step = 000024, loss = 0.081406
grad_step = 000025, loss = 0.075996
grad_step = 000026, loss = 0.071065
grad_step = 000027, loss = 0.065312
grad_step = 000028, loss = 0.059058
grad_step = 000029, loss = 0.053600
grad_step = 000030, loss = 0.049380
grad_step = 000031, loss = 0.045444
grad_step = 000032, loss = 0.041124
grad_step = 000033, loss = 0.036849
grad_step = 000034, loss = 0.033085
grad_step = 000035, loss = 0.029704
grad_step = 000036, loss = 0.026625
grad_step = 000037, loss = 0.023991
grad_step = 000038, loss = 0.021643
grad_step = 000039, loss = 0.019301
grad_step = 000040, loss = 0.017060
grad_step = 000041, loss = 0.015190
grad_step = 000042, loss = 0.013573
grad_step = 000043, loss = 0.011954
grad_step = 000044, loss = 0.010459
grad_step = 000045, loss = 0.009273
grad_step = 000046, loss = 0.008278
grad_step = 000047, loss = 0.007353
grad_step = 000048, loss = 0.006547
grad_step = 000049, loss = 0.005863
grad_step = 000050, loss = 0.005270
grad_step = 000051, loss = 0.004810
grad_step = 000052, loss = 0.004440
grad_step = 000053, loss = 0.004052
grad_step = 000054, loss = 0.003687
grad_step = 000055, loss = 0.003457
grad_step = 000056, loss = 0.003318
grad_step = 000057, loss = 0.003160
grad_step = 000058, loss = 0.003017
grad_step = 000059, loss = 0.002925
grad_step = 000060, loss = 0.002846
grad_step = 000061, loss = 0.002765
grad_step = 000062, loss = 0.002705
grad_step = 000063, loss = 0.002654
grad_step = 000064, loss = 0.002606
grad_step = 000065, loss = 0.002588
grad_step = 000066, loss = 0.002581
grad_step = 000067, loss = 0.002564
grad_step = 000068, loss = 0.002551
grad_step = 000069, loss = 0.002549
grad_step = 000070, loss = 0.002541
grad_step = 000071, loss = 0.002524
grad_step = 000072, loss = 0.002515
grad_step = 000073, loss = 0.002511
grad_step = 000074, loss = 0.002509
grad_step = 000075, loss = 0.002506
grad_step = 000076, loss = 0.002500
grad_step = 000077, loss = 0.002489
grad_step = 000078, loss = 0.002482
grad_step = 000079, loss = 0.002472
grad_step = 000080, loss = 0.002456
grad_step = 000081, loss = 0.002442
grad_step = 000082, loss = 0.002430
grad_step = 000083, loss = 0.002415
grad_step = 000084, loss = 0.002399
grad_step = 000085, loss = 0.002386
grad_step = 000086, loss = 0.002372
grad_step = 000087, loss = 0.002356
grad_step = 000088, loss = 0.002341
grad_step = 000089, loss = 0.002327
grad_step = 000090, loss = 0.002315
grad_step = 000091, loss = 0.002304
grad_step = 000092, loss = 0.002294
grad_step = 000093, loss = 0.002284
grad_step = 000094, loss = 0.002276
grad_step = 000095, loss = 0.002268
grad_step = 000096, loss = 0.002261
grad_step = 000097, loss = 0.002255
grad_step = 000098, loss = 0.002250
grad_step = 000099, loss = 0.002245
grad_step = 000100, loss = 0.002241
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002238
grad_step = 000102, loss = 0.002235
grad_step = 000103, loss = 0.002232
grad_step = 000104, loss = 0.002229
grad_step = 000105, loss = 0.002226
grad_step = 000106, loss = 0.002225
grad_step = 000107, loss = 0.002223
grad_step = 000108, loss = 0.002220
grad_step = 000109, loss = 0.002219
grad_step = 000110, loss = 0.002217
grad_step = 000111, loss = 0.002215
grad_step = 000112, loss = 0.002213
grad_step = 000113, loss = 0.002212
grad_step = 000114, loss = 0.002210
grad_step = 000115, loss = 0.002208
grad_step = 000116, loss = 0.002206
grad_step = 000117, loss = 0.002205
grad_step = 000118, loss = 0.002203
grad_step = 000119, loss = 0.002201
grad_step = 000120, loss = 0.002199
grad_step = 000121, loss = 0.002197
grad_step = 000122, loss = 0.002195
grad_step = 000123, loss = 0.002193
grad_step = 000124, loss = 0.002191
grad_step = 000125, loss = 0.002189
grad_step = 000126, loss = 0.002187
grad_step = 000127, loss = 0.002185
grad_step = 000128, loss = 0.002183
grad_step = 000129, loss = 0.002182
grad_step = 000130, loss = 0.002180
grad_step = 000131, loss = 0.002178
grad_step = 000132, loss = 0.002176
grad_step = 000133, loss = 0.002174
grad_step = 000134, loss = 0.002172
grad_step = 000135, loss = 0.002170
grad_step = 000136, loss = 0.002168
grad_step = 000137, loss = 0.002166
grad_step = 000138, loss = 0.002164
grad_step = 000139, loss = 0.002162
grad_step = 000140, loss = 0.002160
grad_step = 000141, loss = 0.002158
grad_step = 000142, loss = 0.002156
grad_step = 000143, loss = 0.002154
grad_step = 000144, loss = 0.002152
grad_step = 000145, loss = 0.002150
grad_step = 000146, loss = 0.002148
grad_step = 000147, loss = 0.002146
grad_step = 000148, loss = 0.002144
grad_step = 000149, loss = 0.002142
grad_step = 000150, loss = 0.002140
grad_step = 000151, loss = 0.002138
grad_step = 000152, loss = 0.002136
grad_step = 000153, loss = 0.002134
grad_step = 000154, loss = 0.002131
grad_step = 000155, loss = 0.002129
grad_step = 000156, loss = 0.002127
grad_step = 000157, loss = 0.002125
grad_step = 000158, loss = 0.002123
grad_step = 000159, loss = 0.002121
grad_step = 000160, loss = 0.002119
grad_step = 000161, loss = 0.002116
grad_step = 000162, loss = 0.002114
grad_step = 000163, loss = 0.002112
grad_step = 000164, loss = 0.002110
grad_step = 000165, loss = 0.002108
grad_step = 000166, loss = 0.002105
grad_step = 000167, loss = 0.002103
grad_step = 000168, loss = 0.002101
grad_step = 000169, loss = 0.002098
grad_step = 000170, loss = 0.002096
grad_step = 000171, loss = 0.002093
grad_step = 000172, loss = 0.002091
grad_step = 000173, loss = 0.002088
grad_step = 000174, loss = 0.002086
grad_step = 000175, loss = 0.002083
grad_step = 000176, loss = 0.002080
grad_step = 000177, loss = 0.002078
grad_step = 000178, loss = 0.002075
grad_step = 000179, loss = 0.002072
grad_step = 000180, loss = 0.002068
grad_step = 000181, loss = 0.002065
grad_step = 000182, loss = 0.002061
grad_step = 000183, loss = 0.002058
grad_step = 000184, loss = 0.002056
grad_step = 000185, loss = 0.002054
grad_step = 000186, loss = 0.002050
grad_step = 000187, loss = 0.002047
grad_step = 000188, loss = 0.002042
grad_step = 000189, loss = 0.002037
grad_step = 000190, loss = 0.002033
grad_step = 000191, loss = 0.002029
grad_step = 000192, loss = 0.002026
grad_step = 000193, loss = 0.002023
grad_step = 000194, loss = 0.002021
grad_step = 000195, loss = 0.002019
grad_step = 000196, loss = 0.002017
grad_step = 000197, loss = 0.002020
grad_step = 000198, loss = 0.002022
grad_step = 000199, loss = 0.002025
grad_step = 000200, loss = 0.002021
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002012
grad_step = 000202, loss = 0.001998
grad_step = 000203, loss = 0.001985
grad_step = 000204, loss = 0.001978
grad_step = 000205, loss = 0.001974
grad_step = 000206, loss = 0.001974
grad_step = 000207, loss = 0.001978
grad_step = 000208, loss = 0.001982
grad_step = 000209, loss = 0.001983
grad_step = 000210, loss = 0.001983
grad_step = 000211, loss = 0.001980
grad_step = 000212, loss = 0.001972
grad_step = 000213, loss = 0.001958
grad_step = 000214, loss = 0.001945
grad_step = 000215, loss = 0.001939
grad_step = 000216, loss = 0.001934
grad_step = 000217, loss = 0.001927
grad_step = 000218, loss = 0.001923
grad_step = 000219, loss = 0.001920
grad_step = 000220, loss = 0.001917
grad_step = 000221, loss = 0.001915
grad_step = 000222, loss = 0.001915
grad_step = 000223, loss = 0.001923
grad_step = 000224, loss = 0.001959
grad_step = 000225, loss = 0.002062
grad_step = 000226, loss = 0.002248
grad_step = 000227, loss = 0.002256
grad_step = 000228, loss = 0.002012
grad_step = 000229, loss = 0.001901
grad_step = 000230, loss = 0.002077
grad_step = 000231, loss = 0.002080
grad_step = 000232, loss = 0.001889
grad_step = 000233, loss = 0.001990
grad_step = 000234, loss = 0.002051
grad_step = 000235, loss = 0.001910
grad_step = 000236, loss = 0.001931
grad_step = 000237, loss = 0.002013
grad_step = 000238, loss = 0.001907
grad_step = 000239, loss = 0.001905
grad_step = 000240, loss = 0.001953
grad_step = 000241, loss = 0.001895
grad_step = 000242, loss = 0.001891
grad_step = 000243, loss = 0.001904
grad_step = 000244, loss = 0.001884
grad_step = 000245, loss = 0.001879
grad_step = 000246, loss = 0.001879
grad_step = 000247, loss = 0.001862
grad_step = 000248, loss = 0.001876
grad_step = 000249, loss = 0.001860
grad_step = 000250, loss = 0.001846
grad_step = 000251, loss = 0.001862
grad_step = 000252, loss = 0.001853
grad_step = 000253, loss = 0.001832
grad_step = 000254, loss = 0.001843
grad_step = 000255, loss = 0.001847
grad_step = 000256, loss = 0.001824
grad_step = 000257, loss = 0.001826
grad_step = 000258, loss = 0.001832
grad_step = 000259, loss = 0.001821
grad_step = 000260, loss = 0.001815
grad_step = 000261, loss = 0.001816
grad_step = 000262, loss = 0.001812
grad_step = 000263, loss = 0.001807
grad_step = 000264, loss = 0.001807
grad_step = 000265, loss = 0.001801
grad_step = 000266, loss = 0.001794
grad_step = 000267, loss = 0.001795
grad_step = 000268, loss = 0.001795
grad_step = 000269, loss = 0.001794
grad_step = 000270, loss = 0.001787
grad_step = 000271, loss = 0.001780
grad_step = 000272, loss = 0.001779
grad_step = 000273, loss = 0.001776
grad_step = 000274, loss = 0.001771
grad_step = 000275, loss = 0.001768
grad_step = 000276, loss = 0.001766
grad_step = 000277, loss = 0.001766
grad_step = 000278, loss = 0.001767
grad_step = 000279, loss = 0.001770
grad_step = 000280, loss = 0.001782
grad_step = 000281, loss = 0.001807
grad_step = 000282, loss = 0.001838
grad_step = 000283, loss = 0.001872
grad_step = 000284, loss = 0.001869
grad_step = 000285, loss = 0.001826
grad_step = 000286, loss = 0.001762
grad_step = 000287, loss = 0.001733
grad_step = 000288, loss = 0.001749
grad_step = 000289, loss = 0.001779
grad_step = 000290, loss = 0.001786
grad_step = 000291, loss = 0.001763
grad_step = 000292, loss = 0.001738
grad_step = 000293, loss = 0.001728
grad_step = 000294, loss = 0.001730
grad_step = 000295, loss = 0.001733
grad_step = 000296, loss = 0.001726
grad_step = 000297, loss = 0.001716
grad_step = 000298, loss = 0.001714
grad_step = 000299, loss = 0.001720
grad_step = 000300, loss = 0.001725
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001721
grad_step = 000302, loss = 0.001709
grad_step = 000303, loss = 0.001693
grad_step = 000304, loss = 0.001681
grad_step = 000305, loss = 0.001676
grad_step = 000306, loss = 0.001677
grad_step = 000307, loss = 0.001681
grad_step = 000308, loss = 0.001683
grad_step = 000309, loss = 0.001685
grad_step = 000310, loss = 0.001688
grad_step = 000311, loss = 0.001695
grad_step = 000312, loss = 0.001712
grad_step = 000313, loss = 0.001729
grad_step = 000314, loss = 0.001756
grad_step = 000315, loss = 0.001753
grad_step = 000316, loss = 0.001744
grad_step = 000317, loss = 0.001749
grad_step = 000318, loss = 0.001778
grad_step = 000319, loss = 0.001823
grad_step = 000320, loss = 0.001870
grad_step = 000321, loss = 0.001823
grad_step = 000322, loss = 0.001746
grad_step = 000323, loss = 0.001643
grad_step = 000324, loss = 0.001651
grad_step = 000325, loss = 0.001726
grad_step = 000326, loss = 0.001720
grad_step = 000327, loss = 0.001661
grad_step = 000328, loss = 0.001626
grad_step = 000329, loss = 0.001652
grad_step = 000330, loss = 0.001690
grad_step = 000331, loss = 0.001687
grad_step = 000332, loss = 0.001665
grad_step = 000333, loss = 0.001643
grad_step = 000334, loss = 0.001628
grad_step = 000335, loss = 0.001634
grad_step = 000336, loss = 0.001642
grad_step = 000337, loss = 0.001640
grad_step = 000338, loss = 0.001616
grad_step = 000339, loss = 0.001592
grad_step = 000340, loss = 0.001592
grad_step = 000341, loss = 0.001609
grad_step = 000342, loss = 0.001618
grad_step = 000343, loss = 0.001606
grad_step = 000344, loss = 0.001587
grad_step = 000345, loss = 0.001579
grad_step = 000346, loss = 0.001585
grad_step = 000347, loss = 0.001597
grad_step = 000348, loss = 0.001604
grad_step = 000349, loss = 0.001600
grad_step = 000350, loss = 0.001593
grad_step = 000351, loss = 0.001586
grad_step = 000352, loss = 0.001586
grad_step = 000353, loss = 0.001592
grad_step = 000354, loss = 0.001604
grad_step = 000355, loss = 0.001618
grad_step = 000356, loss = 0.001639
grad_step = 000357, loss = 0.001648
grad_step = 000358, loss = 0.001659
grad_step = 000359, loss = 0.001639
grad_step = 000360, loss = 0.001623
grad_step = 000361, loss = 0.001602
grad_step = 000362, loss = 0.001589
grad_step = 000363, loss = 0.001567
grad_step = 000364, loss = 0.001546
grad_step = 000365, loss = 0.001536
grad_step = 000366, loss = 0.001542
grad_step = 000367, loss = 0.001555
grad_step = 000368, loss = 0.001561
grad_step = 000369, loss = 0.001564
grad_step = 000370, loss = 0.001564
grad_step = 000371, loss = 0.001578
grad_step = 000372, loss = 0.001595
grad_step = 000373, loss = 0.001621
grad_step = 000374, loss = 0.001612
grad_step = 000375, loss = 0.001605
grad_step = 000376, loss = 0.001597
grad_step = 000377, loss = 0.001599
grad_step = 000378, loss = 0.001580
grad_step = 000379, loss = 0.001542
grad_step = 000380, loss = 0.001511
grad_step = 000381, loss = 0.001504
grad_step = 000382, loss = 0.001509
grad_step = 000383, loss = 0.001511
grad_step = 000384, loss = 0.001513
grad_step = 000385, loss = 0.001520
grad_step = 000386, loss = 0.001533
grad_step = 000387, loss = 0.001537
grad_step = 000388, loss = 0.001545
grad_step = 000389, loss = 0.001552
grad_step = 000390, loss = 0.001580
grad_step = 000391, loss = 0.001591
grad_step = 000392, loss = 0.001587
grad_step = 000393, loss = 0.001527
grad_step = 000394, loss = 0.001484
grad_step = 000395, loss = 0.001477
grad_step = 000396, loss = 0.001491
grad_step = 000397, loss = 0.001490
grad_step = 000398, loss = 0.001467
grad_step = 000399, loss = 0.001458
grad_step = 000400, loss = 0.001463
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001460
grad_step = 000402, loss = 0.001452
grad_step = 000403, loss = 0.001455
grad_step = 000404, loss = 0.001464
grad_step = 000405, loss = 0.001466
grad_step = 000406, loss = 0.001471
grad_step = 000407, loss = 0.001503
grad_step = 000408, loss = 0.001561
grad_step = 000409, loss = 0.001671
grad_step = 000410, loss = 0.001765
grad_step = 000411, loss = 0.001893
grad_step = 000412, loss = 0.001832
grad_step = 000413, loss = 0.001650
grad_step = 000414, loss = 0.001443
grad_step = 000415, loss = 0.001434
grad_step = 000416, loss = 0.001569
grad_step = 000417, loss = 0.001598
grad_step = 000418, loss = 0.001468
grad_step = 000419, loss = 0.001395
grad_step = 000420, loss = 0.001474
grad_step = 000421, loss = 0.001535
grad_step = 000422, loss = 0.001456
grad_step = 000423, loss = 0.001382
grad_step = 000424, loss = 0.001408
grad_step = 000425, loss = 0.001459
grad_step = 000426, loss = 0.001452
grad_step = 000427, loss = 0.001395
grad_step = 000428, loss = 0.001370
grad_step = 000429, loss = 0.001390
grad_step = 000430, loss = 0.001409
grad_step = 000431, loss = 0.001398
grad_step = 000432, loss = 0.001363
grad_step = 000433, loss = 0.001348
grad_step = 000434, loss = 0.001363
grad_step = 000435, loss = 0.001376
grad_step = 000436, loss = 0.001367
grad_step = 000437, loss = 0.001340
grad_step = 000438, loss = 0.001330
grad_step = 000439, loss = 0.001344
grad_step = 000440, loss = 0.001360
grad_step = 000441, loss = 0.001356
grad_step = 000442, loss = 0.001345
grad_step = 000443, loss = 0.001342
grad_step = 000444, loss = 0.001369
grad_step = 000445, loss = 0.001413
grad_step = 000446, loss = 0.001494
grad_step = 000447, loss = 0.001532
grad_step = 000448, loss = 0.001573
grad_step = 000449, loss = 0.001471
grad_step = 000450, loss = 0.001362
grad_step = 000451, loss = 0.001294
grad_step = 000452, loss = 0.001316
grad_step = 000453, loss = 0.001374
grad_step = 000454, loss = 0.001366
grad_step = 000455, loss = 0.001333
grad_step = 000456, loss = 0.001307
grad_step = 000457, loss = 0.001307
grad_step = 000458, loss = 0.001311
grad_step = 000459, loss = 0.001294
grad_step = 000460, loss = 0.001288
grad_step = 000461, loss = 0.001292
grad_step = 000462, loss = 0.001293
grad_step = 000463, loss = 0.001281
grad_step = 000464, loss = 0.001257
grad_step = 000465, loss = 0.001250
grad_step = 000466, loss = 0.001261
grad_step = 000467, loss = 0.001271
grad_step = 000468, loss = 0.001267
grad_step = 000469, loss = 0.001248
grad_step = 000470, loss = 0.001231
grad_step = 000471, loss = 0.001227
grad_step = 000472, loss = 0.001231
grad_step = 000473, loss = 0.001237
grad_step = 000474, loss = 0.001231
grad_step = 000475, loss = 0.001220
grad_step = 000476, loss = 0.001209
grad_step = 000477, loss = 0.001205
grad_step = 000478, loss = 0.001205
grad_step = 000479, loss = 0.001207
grad_step = 000480, loss = 0.001210
grad_step = 000481, loss = 0.001212
grad_step = 000482, loss = 0.001217
grad_step = 000483, loss = 0.001233
grad_step = 000484, loss = 0.001280
grad_step = 000485, loss = 0.001432
grad_step = 000486, loss = 0.001702
grad_step = 000487, loss = 0.002324
grad_step = 000488, loss = 0.002106
grad_step = 000489, loss = 0.001711
grad_step = 000490, loss = 0.001277
grad_step = 000491, loss = 0.001506
grad_step = 000492, loss = 0.001683
grad_step = 000493, loss = 0.001313
grad_step = 000494, loss = 0.001475
grad_step = 000495, loss = 0.001531
grad_step = 000496, loss = 0.001263
grad_step = 000497, loss = 0.001522
grad_step = 000498, loss = 0.001353
grad_step = 000499, loss = 0.001292
grad_step = 000500, loss = 0.001425
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001210
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

  date_run                              2020-05-13 16:13:27.928297
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.212935
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 16:13:27.934559
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.107734
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 16:13:27.941436
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.132092
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 16:13:27.947854
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.637053
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
0   2020-05-13 16:12:55.363130  ...    mean_absolute_error
1   2020-05-13 16:12:55.367311  ...     mean_squared_error
2   2020-05-13 16:12:55.370761  ...  median_absolute_error
3   2020-05-13 16:12:55.374455  ...               r2_score
4   2020-05-13 16:13:04.447120  ...    mean_absolute_error
5   2020-05-13 16:13:04.451885  ...     mean_squared_error
6   2020-05-13 16:13:04.455729  ...  median_absolute_error
7   2020-05-13 16:13:04.460538  ...               r2_score
8   2020-05-13 16:13:27.928297  ...    mean_absolute_error
9   2020-05-13 16:13:27.934559  ...     mean_squared_error
10  2020-05-13 16:13:27.941436  ...  median_absolute_error
11  2020-05-13 16:13:27.947854  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa29e89ccf8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:18, 125707.89it/s] 52%|█████▏    | 5160960/9912422 [00:00<00:26, 179394.83it/s]9920512it [00:00, 34774011.42it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1036610.13it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 158515.86it/s]1654784it [00:00, 11177551.63it/s]                         
0it [00:00, ?it/s]8192it [00:00, 132670.77it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa251255eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2508830f0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa24e017518> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa29e85ff28> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa251255eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa29e85ff28> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa24e017518> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa29e85ff28> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa251255eb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa29e8a7940> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7ff7c6d44240> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=6cc3152b5ec6a59b40c6a96f41d4dda9bf84401f2bb9810a265b2873262061e8
  Stored in directory: /tmp/pip-ephem-wheel-cache-ojup4xwp/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ff7bceaf048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3039232/17464789 [====>.........................] - ETA: 0s
10944512/17464789 [=================>............] - ETA: 0s
16883712/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 16:14:55.259648: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 16:14:55.264252: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 16:14:55.264444: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558ff8460410 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 16:14:55.264459: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8046 - accuracy: 0.4910
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7586 - accuracy: 0.4940 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6564 - accuracy: 0.5007
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6398 - accuracy: 0.5017
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6329 - accuracy: 0.5022
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6845 - accuracy: 0.4988
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7082 - accuracy: 0.4973
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6206 - accuracy: 0.5030
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6314 - accuracy: 0.5023
11000/25000 [============>.................] - ETA: 4s - loss: 7.6150 - accuracy: 0.5034
12000/25000 [=============>................] - ETA: 4s - loss: 7.6487 - accuracy: 0.5012
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6360 - accuracy: 0.5020
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6261 - accuracy: 0.5026
15000/25000 [=================>............] - ETA: 3s - loss: 7.6247 - accuracy: 0.5027
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6187 - accuracy: 0.5031
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6206 - accuracy: 0.5030
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6104 - accuracy: 0.5037
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6448 - accuracy: 0.5014
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6413 - accuracy: 0.5016
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6440 - accuracy: 0.5015
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6471 - accuracy: 0.5013
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6560 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
25000/25000 [==============================] - 10s 387us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 16:15:12.664911
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 16:15:12.664911  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<52:12:02, 4.59kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<36:46:17, 6.51kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:02<25:47:31, 9.28kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:02<18:03:16, 13.3kB/s].vector_cache/glove.6B.zip:   0%|          | 2.69M/862M [00:02<12:36:57, 18.9kB/s].vector_cache/glove.6B.zip:   1%|          | 6.46M/862M [00:02<8:47:40, 27.0kB/s] .vector_cache/glove.6B.zip:   1%|          | 10.7M/862M [00:02<6:07:38, 38.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.9M/862M [00:02<4:16:10, 55.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.0M/862M [00:02<2:58:33, 78.7kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.0M/862M [00:02<2:04:29, 112kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 27.4M/862M [00:02<1:26:47, 160kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.0M/862M [00:02<1:00:30, 229kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.9M/862M [00:03<42:15, 326kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 40.5M/862M [00:03<29:30, 464kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.0M/862M [00:03<20:40, 659kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.5M/862M [00:03<14:29, 936kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:03<10:40, 1.26MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<07:32, 1.78MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:05<34:20, 391kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:06<26:14, 512kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:06<18:53, 710kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:07<15:31, 861kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:08<12:17, 1.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:08<08:53, 1.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<06:23, 2.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:09<51:35, 258kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:10<38:50, 342kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:10<27:51, 477kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<19:35, 675kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:11<12:47:34, 17.2kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<8:58:26, 24.5kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:12<6:16:30, 35.0kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:13<4:25:55, 49.5kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.3M/862M [00:13<3:07:10, 70.2kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:14<2:11:13, 100kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<1:31:42, 143kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:15<2:03:52, 106kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:16<1:29:24, 146kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:16<1:03:11, 207kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.8M/862M [00:16<44:20, 294kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:17<35:46, 364kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:18<26:38, 489kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.8M/862M [00:18<18:59, 684kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<13:34, 955kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:19<17:22, 745kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:20<14:26, 897kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:20<10:39, 1.21MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:22<10:01, 1.28MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:22<17:43, 726kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:22<14:23, 894kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:22<11:20, 1.13MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:22<08:12, 1.56MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:24<08:23, 1.53MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:24<08:07, 1.58MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:24<06:10, 2.07MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.3M/862M [00:24<04:33, 2.80MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:26<07:36, 1.67MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:26<07:34, 1.68MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:26<05:47, 2.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:26<04:16, 2.97MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<07:40, 1.65MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<07:40, 1.65MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:28<05:56, 2.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<04:16, 2.95MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<2:30:12, 83.9kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<1:47:19, 117kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:30<1:15:31, 167kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<52:58, 237kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<41:30, 302kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<31:19, 400kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:32<22:24, 559kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<15:48, 789kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<17:48, 700kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<14:38, 851kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:34<10:43, 1.16MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<07:44, 1.60MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<10:28, 1.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<08:40, 1.42MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<08:14, 1.50MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:38<06:15, 1.97MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:38<04:36, 2.67MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<07:18, 1.68MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<07:19, 1.67MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:40<05:36, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<04:06, 2.98MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<08:35, 1.42MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<08:08, 1.50MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:42<06:10, 1.97MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<04:30, 2.69MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<08:05, 1.50MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<07:51, 1.54MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:44<05:57, 2.03MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<04:23, 2.75MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<06:58, 1.73MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<06:59, 1.72MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:46<05:25, 2.22MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<05:49, 2.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<06:09, 1.94MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:48<04:46, 2.51MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<03:33, 3.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<06:31, 1.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<06:39, 1.79MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<05:07, 2.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:50<03:44, 3.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<08:30, 1.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<08:02, 1.47MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:52<06:08, 1.93MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<06:17, 1.87MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<06:31, 1.81MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:54<05:05, 2.31MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<03:41, 3.17MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<12:21, 947kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<10:40, 1.10MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<07:54, 1.48MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<05:43, 2.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<08:30, 1.37MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<08:03, 1.44MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<06:04, 1.92MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<04:23, 2.64MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<09:40, 1.20MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:59<08:47, 1.32MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<06:39, 1.73MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<04:48, 2.40MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<11:27, 1.00MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<10:02, 1.15MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<07:31, 1.53MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<05:27, 2.10MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<07:59, 1.43MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<07:34, 1.51MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<05:43, 1.99MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<04:15, 2.68MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<06:24, 1.77MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<06:30, 1.75MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<05:03, 2.25MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<05:26, 2.08MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:07<05:52, 1.92MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:07<04:36, 2.45MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<03:21, 3.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<11:27, 979kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<10:00, 1.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:09<07:27, 1.50MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:19, 2.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<54:24, 205kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<40:05, 278kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:11<28:26, 392kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<19:59, 555kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<22:00, 504kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<17:20, 639kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:13<12:33, 882kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<08:59, 1.23MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<09:43, 1.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<08:44, 1.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:15<06:36, 1.67MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<04:44, 2.31MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:17<14:09, 774kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:17<11:48, 927kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:17<08:40, 1.26MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<06:13, 1.75MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<09:09, 1.19MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<08:18, 1.31MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:19<06:16, 1.73MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:30, 2.40MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<12:32, 863kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:21<10:41, 1.01MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:21<07:56, 1.36MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:40, 1.89MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<10:44, 1.00MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:23<09:24, 1.14MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:23<07:02, 1.52MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:01, 2.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:25<18:05, 590kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:25<14:35, 731kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:25<10:36, 1.01MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<07:31, 1.41MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:27<12:56, 820kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:27<10:54, 973kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<08:01, 1.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:43, 1.84MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<12:07, 870kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<10:24, 1.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:29<07:40, 1.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:28, 1.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<12:42, 824kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<12:32, 835kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<09:43, 1.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<07:00, 1.49MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<07:34, 1.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<07:07, 1.46MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:33<05:21, 1.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<03:52, 2.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<10:33, 978kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<09:11, 1.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:35<06:48, 1.51MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<04:53, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<08:57, 1.15MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:37<08:09, 1.26MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:37<06:09, 1.66MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<06:01, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:39<05:59, 1.70MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:39<04:38, 2.19MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<04:57, 2.04MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:41<05:13, 1.94MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:41<04:05, 2.47MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<04:34, 2.20MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:43<04:58, 2.02MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:43<03:54, 2.57MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<02:49, 3.53MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<1:07:04, 149kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:45<48:40, 205kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:45<34:24, 290kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<24:08, 412kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<22:04, 449kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<17:15, 575kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:47<12:25, 797kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<08:48, 1.12MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<11:12, 879kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<09:33, 1.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:49<07:06, 1.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<06:37, 1.48MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<06:15, 1.56MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:51<05:16, 1.85MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:51<03:58, 2.45MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<02:54, 3.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<15:14, 637kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<13:07, 740kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:52<10:15, 946kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:53<07:40, 1.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<05:30, 1.75MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<07:25, 1.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<07:04, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<05:46, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:55<04:16, 2.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<05:36, 1.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<10:41, 895kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<09:15, 1.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:57<06:50, 1.40MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<04:55, 1.93MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<07:05, 1.34MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<06:37, 1.43MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:59<04:59, 1.90MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<03:35, 2.63MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<10:38, 887kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<09:11, 1.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:01<06:50, 1.38MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<06:21, 1.47MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<06:05, 1.54MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:03<04:39, 2.01MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<04:50, 1.92MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<05:01, 1.85MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:05<03:55, 2.36MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<04:18, 2.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:07<06:29, 1.42MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:07<05:22, 1.71MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<03:55, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:08<05:06, 1.79MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<05:09, 1.77MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<04:00, 2.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:10<04:20, 2.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:11<04:37, 1.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:11<03:34, 2.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<02:36, 3.47MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<09:41, 931kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:13<08:21, 1.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:13<06:10, 1.46MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<04:25, 2.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<09:16, 966kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<08:06, 1.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:15<06:00, 1.49MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<04:19, 2.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<07:18, 1.22MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<06:39, 1.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:17<04:59, 1.78MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:17<03:38, 2.43MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:18<06:01, 1.46MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:18<05:49, 1.51MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:19<04:24, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<03:11, 2.74MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:20<07:07, 1.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:20<08:03, 1.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<06:19, 1.38MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:21<04:35, 1.90MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<05:13, 1.66MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<05:15, 1.65MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<03:57, 2.18MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<02:59, 2.90MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:24<04:26, 1.94MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<04:37, 1.86MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<03:33, 2.42MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:25<02:36, 3.28MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:26<05:42, 1.50MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:26<05:29, 1.55MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<04:09, 2.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:27<03:04, 2.77MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:28<05:05, 1.66MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:28<05:04, 1.67MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:28<03:51, 2.19MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<02:51, 2.95MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:30<04:40, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:30<04:45, 1.77MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<03:38, 2.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<02:43, 3.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<04:22, 1.90MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<04:36, 1.81MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:32<03:33, 2.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<03:53, 2.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<04:10, 1.98MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:34<03:16, 2.51MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:36<03:41, 2.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:36<04:04, 2.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:36<03:09, 2.58MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<02:20, 3.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<04:49, 1.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<04:48, 1.69MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:38<03:41, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<02:42, 2.99MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<05:32, 1.45MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:40<05:17, 1.53MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:40<03:59, 2.01MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<02:52, 2.78MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<10:47, 741kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<10:33, 757kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<08:06, 984kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<05:48, 1.37MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<06:05, 1.30MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<05:39, 1.40MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:44<04:18, 1.84MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<04:20, 1.81MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<04:05, 1.92MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:46<03:04, 2.55MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:48<03:40, 2.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:48<05:24, 1.44MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:48<04:28, 1.74MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<03:15, 2.38MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:50<04:18, 1.79MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:50<04:19, 1.78MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:50<03:19, 2.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<02:27, 3.11MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:52<04:18, 1.78MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:52<04:20, 1.76MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:52<03:22, 2.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:54<03:38, 2.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:54<03:50, 1.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:54<02:57, 2.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<02:11, 3.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:56<04:22, 1.72MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:56<04:23, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:56<03:23, 2.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<03:38, 2.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<03:51, 1.92MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:58<03:01, 2.45MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<02:11, 3.36MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:00<08:58, 822kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:00<07:33, 976kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:00<05:33, 1.32MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<03:58, 1.84MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:02<06:33, 1.11MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:02<05:52, 1.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:02<04:25, 1.65MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<04:18, 1.68MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<04:14, 1.70MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:04<03:13, 2.23MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<02:22, 3.02MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:06<04:24, 1.63MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:06<04:21, 1.65MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<03:20, 2.14MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:06<02:29, 2.86MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:08<03:41, 1.92MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:08<03:49, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:08<02:58, 2.38MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<02:13, 3.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:09<03:31, 1.99MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:10<03:22, 2.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:10<02:33, 2.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<01:51, 3.74MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:11<15:11, 458kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:12<11:51, 587kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:12<08:32, 812kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<06:04, 1.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<06:53, 1.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:14<06:01, 1.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:14<04:28, 1.54MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<03:13, 2.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:15<05:19, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:16<04:35, 1.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:16<03:23, 2.00MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<02:28, 2.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:17<05:58, 1.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:18<06:33, 1.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:18<05:05, 1.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:18<03:41, 1.82MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:19<04:08, 1.62MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:19<04:02, 1.65MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<03:05, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<02:15, 2.95MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:21<04:49, 1.37MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:21<04:29, 1.47MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:22<03:25, 1.92MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<02:28, 2.65MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:23<05:55, 1.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:23<05:18, 1.23MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:24<03:59, 1.64MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<02:52, 2.26MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:25<05:19, 1.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:25<04:48, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:26<03:38, 1.77MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<02:37, 2.45MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:27<06:49, 939kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:27<05:36, 1.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<04:05, 1.56MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<02:57, 2.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:29<05:30, 1.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:29<04:56, 1.28MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<03:40, 1.72MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:30<02:40, 2.35MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:31<04:05, 1.54MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:31<03:54, 1.60MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:31<03:00, 2.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<02:10, 2.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:33<06:06, 1.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:33<05:22, 1.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<03:58, 1.56MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:52, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:35<04:17, 1.43MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:35<04:01, 1.52MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<03:03, 2.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:12, 2.76MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<06:26, 942kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<05:32, 1.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<04:07, 1.46MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:56, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<15:02, 399kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:39<11:31, 520kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<08:16, 722kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<05:52, 1.01MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<06:17, 943kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:41<05:25, 1.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<04:02, 1.46MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:54, 2.03MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:43<05:05, 1.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:43<04:33, 1.29MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<03:25, 1.71MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:27, 2.36MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:45<05:54, 980kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:45<05:06, 1.13MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<03:48, 1.51MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:43, 2.10MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<06:29, 882kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<05:30, 1.04MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:47<04:02, 1.41MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:55, 1.94MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:49<04:19, 1.31MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:49<03:59, 1.42MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:49<03:00, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:09, 2.59MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<06:20, 880kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<05:25, 1.03MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:51<04:01, 1.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:51, 1.93MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:53<14:25, 382kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:53<11:01, 500kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:53<07:55, 693kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:55<06:26, 846kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:55<05:25, 1.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:55<04:00, 1.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<02:51, 1.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:57<05:53, 912kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:57<05:01, 1.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:57<03:42, 1.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<02:40, 1.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<03:46, 1.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<03:34, 1.49MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:59<02:40, 1.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<01:56, 2.70MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:01<04:33, 1.15MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:01<06:52, 761kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:01<05:46, 907kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:02<04:15, 1.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:03<03:46, 1.37MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:03<03:22, 1.53MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:03<02:53, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<02:08, 2.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:05<02:31, 2.02MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:05<02:31, 2.02MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:05<02:16, 2.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:05<01:50, 2.75MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:06<01:27, 3.47MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<01:09, 4.37MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:07<03:12, 1.57MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:07<03:22, 1.49MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:07<03:09, 1.59MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:07<02:36, 1.93MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:07<02:01, 2.48MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<01:31, 3.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:09<02:39, 1.86MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:09<02:14, 2.22MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:09<01:46, 2.78MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<01:20, 3.68MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<01:03, 4.65MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:11<09:00, 544kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:11<07:12, 680kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:11<05:22, 910kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:11<03:55, 1.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:11<02:53, 1.68MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<02:09, 2.24MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<06:27, 748kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<08:29, 569kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<07:04, 683kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:14<05:10, 930kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<03:42, 1.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<03:44, 1.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<03:25, 1.39MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<02:35, 1.83MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<02:36, 1.80MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<02:36, 1.80MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:18<02:00, 2.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:19<02:11, 2.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<03:14, 1.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<03:06, 1.48MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<02:30, 1.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:20<01:51, 2.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:21<02:16, 2.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<02:47, 1.63MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<02:37, 1.73MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<02:08, 2.11MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:22<01:35, 2.82MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:23<02:08, 2.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:24<02:17, 1.96MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<01:47, 2.50MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<01:17, 3.42MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:25<04:49, 916kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:25<04:11, 1.05MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:26<03:19, 1.32MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:26<02:30, 1.76MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<01:47, 2.43MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:27<04:33, 952kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:27<03:55, 1.10MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<03:00, 1.44MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<02:11, 1.97MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:29<02:34, 1.67MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:29<02:34, 1.66MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<02:00, 2.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:30<01:28, 2.87MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:31<02:11, 1.91MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:31<02:05, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:31<01:43, 2.44MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<01:16, 3.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:33<02:03, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:33<02:08, 1.94MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:33<01:51, 2.23MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:34<01:23, 2.95MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:35<01:49, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:35<01:59, 2.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:36<01:34, 2.58MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:36<01:07, 3.55MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:37<08:34, 467kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:37<06:40, 599kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<05:00, 798kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:38<03:34, 1.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:38<02:32, 1.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<09:16, 424kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<07:09, 549kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:39<05:09, 758kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:40<03:37, 1.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:41<05:44, 672kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:41<04:25, 872kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:41<03:22, 1.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:41<02:27, 1.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:42<01:47, 2.13MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:43<03:23, 1.12MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:43<03:01, 1.25MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:43<02:16, 1.66MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<01:38, 2.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:45<02:39, 1.40MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:45<02:31, 1.48MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:45<01:53, 1.96MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:45<01:22, 2.68MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<02:54, 1.26MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<02:39, 1.38MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:47<02:00, 1.81MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<01:25, 2.51MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<04:53, 734kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<04:01, 890kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:49<02:57, 1.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:49<02:04, 1.69MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:51<16:11, 217kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:51<11:55, 295kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:51<08:27, 414kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<05:52, 587kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<14:43, 234kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<10:53, 317kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:53<07:43, 444kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<05:22, 629kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<09:34, 353kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<07:11, 470kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:55<05:13, 645kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:55<03:41, 905kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<03:27, 959kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<02:50, 1.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:57<02:03, 1.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<02:04, 1.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<02:05, 1.55MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<01:43, 1.88MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:59<01:17, 2.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:01<01:32, 2.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:01<01:36, 1.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:01<01:15, 2.51MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<01:24, 2.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<01:30, 2.06MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:03<01:11, 2.60MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<01:25, 2.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<02:04, 1.47MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<01:41, 1.79MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<01:17, 2.34MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<01:24, 2.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<01:30, 1.95MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:07<01:09, 2.53MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<00:52, 3.34MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<01:26, 2.01MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<01:39, 1.75MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<01:40, 1.73MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<01:25, 2.02MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<01:09, 2.49MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:10<00:54, 3.18MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:10<00:40, 4.23MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<07:51, 360kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<05:46, 489kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<04:11, 673kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:11<02:59, 937kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<02:09, 1.29MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:11<01:35, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<10:25, 265kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<08:08, 339kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<06:10, 447kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<04:27, 616kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:13<03:11, 857kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:13<02:16, 1.19MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<02:40, 1.01MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<02:19, 1.16MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:15<01:43, 1.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:15<01:15, 2.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<01:36, 1.63MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<01:33, 1.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<01:14, 2.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<00:57, 2.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:17<00:43, 3.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<01:24, 1.82MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<01:27, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<01:06, 2.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<00:49, 3.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<01:22, 1.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<01:23, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<01:03, 2.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:21<00:47, 3.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<01:13, 1.97MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<01:16, 1.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<00:58, 2.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:23<00:43, 3.25MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<01:10, 2.00MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:25<01:14, 1.90MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<00:56, 2.46MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:25<00:40, 3.36MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<01:57, 1.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<01:39, 1.37MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:27<01:14, 1.83MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:27<00:53, 2.51MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<01:43, 1.28MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<01:35, 1.39MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<01:11, 1.83MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<01:11, 1.80MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<01:11, 1.80MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:31<00:54, 2.35MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:31<00:39, 3.20MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<01:40, 1.23MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<01:31, 1.35MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:33<01:08, 1.81MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:33<00:49, 2.45MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<01:11, 1.69MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<01:10, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:35<00:54, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<00:57, 2.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<00:59, 1.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:37<00:46, 2.50MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<00:50, 2.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<01:14, 1.50MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<01:00, 1.84MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<00:43, 2.51MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<00:57, 1.88MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<00:58, 1.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:41<00:44, 2.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:41<00:31, 3.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<25:37, 67.4kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<18:12, 94.7kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:43<12:42, 134kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:44<08:55, 186kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:45<06:26, 257kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:45<04:30, 363kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:46<03:22, 470kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<02:53, 549kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<02:07, 741kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:47<01:31, 1.03MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<01:19, 1.14MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:49<01:11, 1.28MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:49<00:53, 1.69MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<00:51, 1.71MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:51<00:50, 1.73MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:51<00:38, 2.23MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:51<00:26, 3.08MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:52<09:18, 148kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:53<06:44, 204kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:53<04:43, 288kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:54<03:25, 385kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<02:36, 503kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:55<01:51, 697kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:56<01:27, 849kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<01:13, 1.01MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:57<00:53, 1.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:57<00:38, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:50, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:46, 1.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:59<00:34, 1.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:59<00:24, 2.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<00:45, 1.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<00:42, 1.55MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:31, 2.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:01<00:22, 2.77MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:38, 1.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:37, 1.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<00:28, 2.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:29, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:30, 1.91MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<00:23, 2.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:06<00:24, 2.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:06<00:36, 1.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<00:29, 1.83MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:07<00:21, 2.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:25, 1.98MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:20, 2.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:16, 2.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:10<00:18, 2.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:21, 2.14MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<00:16, 2.75MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:11, 3.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:36, 1.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:38, 1.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:29, 1.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<00:20, 1.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:23, 1.61MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:22, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:17, 2.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:16, 2.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:17, 1.92MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:16<00:12, 2.48MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:16<00:08, 3.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<00:33, 871kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<00:33, 863kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<00:25, 1.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:17, 1.55MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:16, 1.49MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:15, 1.56MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<00:11, 2.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:22<00:10, 1.93MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:22<00:10, 1.88MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<00:08, 2.41MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:07, 2.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:08, 2.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:06, 2.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<00:05, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<00:06, 2.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<00:04, 2.62MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:02, 3.59MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:18, 451kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:14, 580kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<00:09, 804kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:28<00:05, 1.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:30<00:04, 1.07MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:30<00:03, 1.21MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:02, 1.61MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:32<00:00, 1.65MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:32<00:00, 1.76MB/s].vector_cache/glove.6B.zip: 862MB [06:32, 2.20MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 733/400000 [00:00<00:54, 7322.95it/s]  0%|          | 1495/400000 [00:00<00:53, 7408.94it/s]  1%|          | 2254/400000 [00:00<00:53, 7461.39it/s]  1%|          | 3050/400000 [00:00<00:52, 7604.04it/s]  1%|          | 3834/400000 [00:00<00:51, 7672.59it/s]  1%|          | 4552/400000 [00:00<00:52, 7514.89it/s]  1%|▏         | 5302/400000 [00:00<00:52, 7509.87it/s]  2%|▏         | 6065/400000 [00:00<00:52, 7544.03it/s]  2%|▏         | 6858/400000 [00:00<00:51, 7653.52it/s]  2%|▏         | 7658/400000 [00:01<00:50, 7753.31it/s]  2%|▏         | 8451/400000 [00:01<00:50, 7805.39it/s]  2%|▏         | 9244/400000 [00:01<00:49, 7840.93it/s]  3%|▎         | 10025/400000 [00:01<00:49, 7829.22it/s]  3%|▎         | 10826/400000 [00:01<00:49, 7882.50it/s]  3%|▎         | 11627/400000 [00:01<00:49, 7918.04it/s]  3%|▎         | 12423/400000 [00:01<00:48, 7928.71it/s]  3%|▎         | 13214/400000 [00:01<00:49, 7885.43it/s]  4%|▎         | 14002/400000 [00:01<00:49, 7854.10it/s]  4%|▎         | 14819/400000 [00:01<00:48, 7946.16it/s]  4%|▍         | 15614/400000 [00:02<00:48, 7924.33it/s]  4%|▍         | 16411/400000 [00:02<00:48, 7937.71it/s]  4%|▍         | 17214/400000 [00:02<00:48, 7962.46it/s]  5%|▍         | 18011/400000 [00:02<00:49, 7795.61it/s]  5%|▍         | 18792/400000 [00:02<00:49, 7696.61it/s]  5%|▍         | 19563/400000 [00:02<00:49, 7655.88it/s]  5%|▌         | 20366/400000 [00:02<00:48, 7762.24it/s]  5%|▌         | 21166/400000 [00:02<00:48, 7830.56it/s]  5%|▌         | 21950/400000 [00:02<00:48, 7784.56it/s]  6%|▌         | 22739/400000 [00:02<00:48, 7814.73it/s]  6%|▌         | 23542/400000 [00:03<00:47, 7874.09it/s]  6%|▌         | 24339/400000 [00:03<00:47, 7901.37it/s]  6%|▋         | 25130/400000 [00:03<00:47, 7838.11it/s]  6%|▋         | 25915/400000 [00:03<00:48, 7667.29it/s]  7%|▋         | 26703/400000 [00:03<00:48, 7728.29it/s]  7%|▋         | 27487/400000 [00:03<00:48, 7759.58it/s]  7%|▋         | 28296/400000 [00:03<00:47, 7853.80it/s]  7%|▋         | 29103/400000 [00:03<00:46, 7915.84it/s]  7%|▋         | 29896/400000 [00:03<00:47, 7799.74it/s]  8%|▊         | 30708/400000 [00:03<00:46, 7891.54it/s]  8%|▊         | 31498/400000 [00:04<00:47, 7729.76it/s]  8%|▊         | 32273/400000 [00:04<00:48, 7635.20it/s]  8%|▊         | 33064/400000 [00:04<00:47, 7713.42it/s]  8%|▊         | 33845/400000 [00:04<00:47, 7739.55it/s]  9%|▊         | 34620/400000 [00:04<00:47, 7735.38it/s]  9%|▉         | 35411/400000 [00:04<00:46, 7785.12it/s]  9%|▉         | 36207/400000 [00:04<00:46, 7836.26it/s]  9%|▉         | 37009/400000 [00:04<00:46, 7890.40it/s]  9%|▉         | 37799/400000 [00:04<00:46, 7758.43it/s] 10%|▉         | 38587/400000 [00:04<00:46, 7792.79it/s] 10%|▉         | 39396/400000 [00:05<00:45, 7877.23it/s] 10%|█         | 40201/400000 [00:05<00:45, 7927.63it/s] 10%|█         | 41000/400000 [00:05<00:45, 7946.18it/s] 10%|█         | 41795/400000 [00:05<00:45, 7877.94it/s] 11%|█         | 42584/400000 [00:05<00:45, 7858.52it/s] 11%|█         | 43380/400000 [00:05<00:45, 7887.82it/s] 11%|█         | 44170/400000 [00:05<00:45, 7831.68it/s] 11%|█         | 44954/400000 [00:05<00:45, 7810.61it/s] 11%|█▏        | 45736/400000 [00:05<00:45, 7775.85it/s] 12%|█▏        | 46534/400000 [00:05<00:45, 7835.25it/s] 12%|█▏        | 47345/400000 [00:06<00:44, 7913.46it/s] 12%|█▏        | 48166/400000 [00:06<00:43, 7998.48it/s] 12%|█▏        | 48967/400000 [00:06<00:44, 7928.60it/s] 12%|█▏        | 49761/400000 [00:06<00:44, 7847.05it/s] 13%|█▎        | 50557/400000 [00:06<00:44, 7878.13it/s] 13%|█▎        | 51346/400000 [00:06<00:44, 7837.03it/s] 13%|█▎        | 52142/400000 [00:06<00:44, 7872.34it/s] 13%|█▎        | 52945/400000 [00:06<00:43, 7916.53it/s] 13%|█▎        | 53742/400000 [00:06<00:43, 7930.04it/s] 14%|█▎        | 54549/400000 [00:06<00:43, 7970.77it/s] 14%|█▍        | 55366/400000 [00:07<00:42, 8029.19it/s] 14%|█▍        | 56188/400000 [00:07<00:42, 8083.86it/s] 14%|█▍        | 57007/400000 [00:07<00:42, 8112.70it/s] 14%|█▍        | 57819/400000 [00:07<00:43, 7840.79it/s] 15%|█▍        | 58606/400000 [00:07<00:44, 7626.85it/s] 15%|█▍        | 59392/400000 [00:07<00:44, 7694.77it/s] 15%|█▌        | 60220/400000 [00:07<00:43, 7858.19it/s] 15%|█▌        | 61009/400000 [00:07<00:45, 7515.71it/s] 15%|█▌        | 61832/400000 [00:07<00:43, 7714.24it/s] 16%|█▌        | 62667/400000 [00:08<00:42, 7892.80it/s] 16%|█▌        | 63494/400000 [00:08<00:42, 8001.24it/s] 16%|█▌        | 64331/400000 [00:08<00:41, 8105.06it/s] 16%|█▋        | 65145/400000 [00:08<00:41, 8109.98it/s] 16%|█▋        | 65958/400000 [00:08<00:41, 7998.25it/s] 17%|█▋        | 66763/400000 [00:08<00:41, 8013.01it/s] 17%|█▋        | 67578/400000 [00:08<00:41, 8052.54it/s] 17%|█▋        | 68385/400000 [00:08<00:41, 8007.82it/s] 17%|█▋        | 69191/400000 [00:08<00:41, 8022.43it/s] 17%|█▋        | 69994/400000 [00:08<00:42, 7832.77it/s] 18%|█▊        | 70779/400000 [00:09<00:42, 7788.86it/s] 18%|█▊        | 71584/400000 [00:09<00:41, 7864.92it/s] 18%|█▊        | 72405/400000 [00:09<00:41, 7964.18it/s] 18%|█▊        | 73232/400000 [00:09<00:40, 8053.43it/s] 19%|█▊        | 74039/400000 [00:09<00:40, 8017.23it/s] 19%|█▊        | 74869/400000 [00:09<00:40, 8099.32it/s] 19%|█▉        | 75680/400000 [00:09<00:40, 8068.78it/s] 19%|█▉        | 76512/400000 [00:09<00:39, 8140.20it/s] 19%|█▉        | 77327/400000 [00:09<00:39, 8105.24it/s] 20%|█▉        | 78138/400000 [00:09<00:39, 8101.87it/s] 20%|█▉        | 78960/400000 [00:10<00:39, 8136.27it/s] 20%|█▉        | 79787/400000 [00:10<00:39, 8173.67it/s] 20%|██        | 80620/400000 [00:10<00:38, 8219.39it/s] 20%|██        | 81445/400000 [00:10<00:38, 8227.15it/s] 21%|██        | 82268/400000 [00:10<00:39, 8045.71it/s] 21%|██        | 83101/400000 [00:10<00:38, 8126.41it/s] 21%|██        | 83915/400000 [00:10<00:40, 7892.05it/s] 21%|██        | 84707/400000 [00:10<00:40, 7832.84it/s] 21%|██▏       | 85499/400000 [00:10<00:40, 7854.68it/s] 22%|██▏       | 86296/400000 [00:10<00:39, 7885.78it/s] 22%|██▏       | 87122/400000 [00:11<00:39, 7993.87it/s] 22%|██▏       | 87932/400000 [00:11<00:38, 8024.75it/s] 22%|██▏       | 88756/400000 [00:11<00:38, 8087.07it/s] 22%|██▏       | 89566/400000 [00:11<00:38, 8070.26it/s] 23%|██▎       | 90374/400000 [00:11<00:38, 8011.00it/s] 23%|██▎       | 91188/400000 [00:11<00:38, 8048.35it/s] 23%|██▎       | 91994/400000 [00:11<00:39, 7877.00it/s] 23%|██▎       | 92783/400000 [00:11<00:39, 7858.70it/s] 23%|██▎       | 93570/400000 [00:11<00:39, 7834.93it/s] 24%|██▎       | 94355/400000 [00:11<00:39, 7835.03it/s] 24%|██▍       | 95176/400000 [00:12<00:38, 7943.33it/s] 24%|██▍       | 95992/400000 [00:12<00:37, 8004.92it/s] 24%|██▍       | 96794/400000 [00:12<00:37, 7988.37it/s] 24%|██▍       | 97594/400000 [00:12<00:39, 7682.51it/s] 25%|██▍       | 98388/400000 [00:12<00:38, 7756.34it/s] 25%|██▍       | 99166/400000 [00:12<00:38, 7727.63it/s] 25%|██▍       | 99993/400000 [00:12<00:38, 7882.01it/s] 25%|██▌       | 100793/400000 [00:12<00:37, 7916.55it/s] 25%|██▌       | 101603/400000 [00:12<00:37, 7969.98it/s] 26%|██▌       | 102401/400000 [00:12<00:37, 7917.16it/s] 26%|██▌       | 103211/400000 [00:13<00:37, 7964.13it/s] 26%|██▌       | 104009/400000 [00:13<00:37, 7931.70it/s] 26%|██▌       | 104803/400000 [00:13<00:37, 7904.29it/s] 26%|██▋       | 105598/400000 [00:13<00:37, 7915.74it/s] 27%|██▋       | 106390/400000 [00:13<00:37, 7801.43it/s] 27%|██▋       | 107171/400000 [00:13<00:37, 7771.77it/s] 27%|██▋       | 107962/400000 [00:13<00:37, 7810.23it/s] 27%|██▋       | 108770/400000 [00:13<00:36, 7887.44it/s] 27%|██▋       | 109594/400000 [00:13<00:36, 7987.90it/s] 28%|██▊       | 110394/400000 [00:14<00:36, 7855.61it/s] 28%|██▊       | 111184/400000 [00:14<00:36, 7866.69it/s] 28%|██▊       | 111993/400000 [00:14<00:36, 7929.35it/s] 28%|██▊       | 112814/400000 [00:14<00:35, 8009.93it/s] 28%|██▊       | 113635/400000 [00:14<00:35, 8067.89it/s] 29%|██▊       | 114443/400000 [00:14<00:35, 8013.99it/s] 29%|██▉       | 115245/400000 [00:14<00:35, 7948.70it/s] 29%|██▉       | 116044/400000 [00:14<00:35, 7960.18it/s] 29%|██▉       | 116841/400000 [00:14<00:35, 7879.70it/s] 29%|██▉       | 117655/400000 [00:14<00:35, 7953.48it/s] 30%|██▉       | 118455/400000 [00:15<00:35, 7967.21it/s] 30%|██▉       | 119287/400000 [00:15<00:34, 8068.31it/s] 30%|███       | 120115/400000 [00:15<00:34, 8128.60it/s] 30%|███       | 120947/400000 [00:15<00:34, 8184.83it/s] 30%|███       | 121768/400000 [00:15<00:33, 8192.06it/s] 31%|███       | 122588/400000 [00:15<00:34, 8146.38it/s] 31%|███       | 123403/400000 [00:15<00:34, 8091.06it/s] 31%|███       | 124213/400000 [00:15<00:34, 7918.39it/s] 31%|███▏      | 125006/400000 [00:15<00:35, 7774.10it/s] 31%|███▏      | 125795/400000 [00:15<00:35, 7800.83it/s] 32%|███▏      | 126582/400000 [00:16<00:34, 7819.64it/s] 32%|███▏      | 127366/400000 [00:16<00:34, 7824.62it/s] 32%|███▏      | 128172/400000 [00:16<00:34, 7893.31it/s] 32%|███▏      | 128986/400000 [00:16<00:34, 7965.37it/s] 32%|███▏      | 129784/400000 [00:16<00:33, 7956.02it/s] 33%|███▎      | 130580/400000 [00:16<00:33, 7939.27it/s] 33%|███▎      | 131407/400000 [00:16<00:33, 8033.65it/s] 33%|███▎      | 132232/400000 [00:16<00:33, 8095.95it/s] 33%|███▎      | 133055/400000 [00:16<00:32, 8135.08it/s] 33%|███▎      | 133869/400000 [00:16<00:32, 8094.06it/s] 34%|███▎      | 134680/400000 [00:17<00:32, 8097.07it/s] 34%|███▍      | 135490/400000 [00:17<00:32, 8059.02it/s] 34%|███▍      | 136319/400000 [00:17<00:32, 8124.44it/s] 34%|███▍      | 137132/400000 [00:17<00:33, 7915.09it/s] 34%|███▍      | 137925/400000 [00:17<00:33, 7733.94it/s] 35%|███▍      | 138719/400000 [00:17<00:33, 7792.61it/s] 35%|███▍      | 139553/400000 [00:17<00:32, 7949.06it/s] 35%|███▌      | 140350/400000 [00:17<00:33, 7845.69it/s] 35%|███▌      | 141164/400000 [00:17<00:32, 7930.51it/s] 35%|███▌      | 141961/400000 [00:17<00:32, 7940.90it/s] 36%|███▌      | 142756/400000 [00:18<00:32, 7843.80it/s] 36%|███▌      | 143570/400000 [00:18<00:32, 7928.44it/s] 36%|███▌      | 144403/400000 [00:18<00:31, 8042.39it/s] 36%|███▋      | 145209/400000 [00:18<00:31, 7977.40it/s] 37%|███▋      | 146035/400000 [00:18<00:31, 8057.17it/s] 37%|███▋      | 146856/400000 [00:18<00:31, 8101.37it/s] 37%|███▋      | 147672/400000 [00:18<00:31, 8116.10it/s] 37%|███▋      | 148485/400000 [00:18<00:31, 8067.82it/s] 37%|███▋      | 149293/400000 [00:18<00:31, 8022.19it/s] 38%|███▊      | 150096/400000 [00:18<00:31, 7971.82it/s] 38%|███▊      | 150894/400000 [00:19<00:31, 7889.00it/s] 38%|███▊      | 151721/400000 [00:19<00:31, 7997.93it/s] 38%|███▊      | 152552/400000 [00:19<00:30, 8087.51it/s] 38%|███▊      | 153362/400000 [00:19<00:30, 8027.45it/s] 39%|███▊      | 154188/400000 [00:19<00:30, 8095.70it/s] 39%|███▊      | 154999/400000 [00:19<00:30, 8058.01it/s] 39%|███▉      | 155831/400000 [00:19<00:30, 8134.52it/s] 39%|███▉      | 156663/400000 [00:19<00:29, 8188.06it/s] 39%|███▉      | 157483/400000 [00:19<00:29, 8087.13it/s] 40%|███▉      | 158307/400000 [00:19<00:29, 8132.18it/s] 40%|███▉      | 159121/400000 [00:20<00:29, 8081.95it/s] 40%|███▉      | 159959/400000 [00:20<00:29, 8166.61it/s] 40%|████      | 160788/400000 [00:20<00:29, 8202.43it/s] 40%|████      | 161609/400000 [00:20<00:29, 8146.03it/s] 41%|████      | 162424/400000 [00:20<00:29, 8146.39it/s] 41%|████      | 163249/400000 [00:20<00:28, 8176.21it/s] 41%|████      | 164067/400000 [00:20<00:28, 8138.77it/s] 41%|████      | 164882/400000 [00:20<00:29, 8022.42it/s] 41%|████▏     | 165715/400000 [00:20<00:28, 8111.08it/s] 42%|████▏     | 166527/400000 [00:20<00:28, 8083.17it/s] 42%|████▏     | 167361/400000 [00:21<00:28, 8156.29it/s] 42%|████▏     | 168184/400000 [00:21<00:28, 8176.92it/s] 42%|████▏     | 169003/400000 [00:21<00:28, 8122.51it/s] 42%|████▏     | 169843/400000 [00:21<00:28, 8202.05it/s] 43%|████▎     | 170670/400000 [00:21<00:27, 8221.55it/s] 43%|████▎     | 171497/400000 [00:21<00:27, 8235.87it/s] 43%|████▎     | 172331/400000 [00:21<00:27, 8266.26it/s] 43%|████▎     | 173158/400000 [00:21<00:27, 8217.25it/s] 43%|████▎     | 173982/400000 [00:21<00:27, 8223.88it/s] 44%|████▎     | 174805/400000 [00:22<00:27, 8170.70it/s] 44%|████▍     | 175623/400000 [00:22<00:27, 8113.95it/s] 44%|████▍     | 176447/400000 [00:22<00:27, 8150.01it/s] 44%|████▍     | 177263/400000 [00:22<00:27, 8110.60it/s] 45%|████▍     | 178075/400000 [00:22<00:27, 8086.17it/s] 45%|████▍     | 178884/400000 [00:22<00:27, 8065.42it/s] 45%|████▍     | 179700/400000 [00:22<00:27, 8092.47it/s] 45%|████▌     | 180544/400000 [00:22<00:26, 8193.68it/s] 45%|████▌     | 181373/400000 [00:22<00:26, 8220.73it/s] 46%|████▌     | 182226/400000 [00:22<00:26, 8309.80it/s] 46%|████▌     | 183065/400000 [00:23<00:26, 8330.36it/s] 46%|████▌     | 183899/400000 [00:23<00:26, 8268.32it/s] 46%|████▌     | 184757/400000 [00:23<00:25, 8356.73it/s] 46%|████▋     | 185594/400000 [00:23<00:25, 8318.02it/s] 47%|████▋     | 186438/400000 [00:23<00:25, 8353.78it/s] 47%|████▋     | 187274/400000 [00:23<00:25, 8329.75it/s] 47%|████▋     | 188108/400000 [00:23<00:25, 8232.10it/s] 47%|████▋     | 188945/400000 [00:23<00:25, 8271.87it/s] 47%|████▋     | 189773/400000 [00:23<00:25, 8198.38it/s] 48%|████▊     | 190594/400000 [00:23<00:26, 7924.66it/s] 48%|████▊     | 191389/400000 [00:24<00:26, 7746.33it/s] 48%|████▊     | 192194/400000 [00:24<00:26, 7834.83it/s] 48%|████▊     | 193001/400000 [00:24<00:26, 7901.43it/s] 48%|████▊     | 193822/400000 [00:24<00:25, 7989.37it/s] 49%|████▊     | 194627/400000 [00:24<00:25, 8006.77it/s] 49%|████▉     | 195435/400000 [00:24<00:25, 8025.68it/s] 49%|████▉     | 196250/400000 [00:24<00:25, 8060.55it/s] 49%|████▉     | 197069/400000 [00:24<00:25, 8098.70it/s] 49%|████▉     | 197899/400000 [00:24<00:24, 8156.15it/s] 50%|████▉     | 198715/400000 [00:24<00:24, 8119.51it/s] 50%|████▉     | 199532/400000 [00:25<00:24, 8133.04it/s] 50%|█████     | 200346/400000 [00:25<00:24, 8124.32it/s] 50%|█████     | 201174/400000 [00:25<00:24, 8168.27it/s] 50%|█████     | 201991/400000 [00:25<00:24, 8049.43it/s] 51%|█████     | 202797/400000 [00:25<00:24, 8041.47it/s] 51%|█████     | 203620/400000 [00:25<00:24, 8095.19it/s] 51%|█████     | 204430/400000 [00:25<00:24, 7919.71it/s] 51%|█████▏    | 205224/400000 [00:25<00:24, 7887.35it/s] 52%|█████▏    | 206055/400000 [00:25<00:24, 8006.56it/s] 52%|█████▏    | 206857/400000 [00:25<00:24, 7914.54it/s] 52%|█████▏    | 207650/400000 [00:26<00:24, 7913.78it/s] 52%|█████▏    | 208459/400000 [00:26<00:24, 7963.29it/s] 52%|█████▏    | 209283/400000 [00:26<00:23, 8042.64it/s] 53%|█████▎    | 210088/400000 [00:26<00:23, 8041.06it/s] 53%|█████▎    | 210910/400000 [00:26<00:23, 8093.27it/s] 53%|█████▎    | 211725/400000 [00:26<00:23, 8109.05it/s] 53%|█████▎    | 212537/400000 [00:26<00:23, 8013.67it/s] 53%|█████▎    | 213355/400000 [00:26<00:23, 8062.67it/s] 54%|█████▎    | 214180/400000 [00:26<00:22, 8117.13it/s] 54%|█████▍    | 215004/400000 [00:26<00:22, 8152.02it/s] 54%|█████▍    | 215833/400000 [00:27<00:22, 8192.27it/s] 54%|█████▍    | 216653/400000 [00:27<00:22, 8150.45it/s] 54%|█████▍    | 217487/400000 [00:27<00:22, 8205.22it/s] 55%|█████▍    | 218308/400000 [00:27<00:22, 7984.14it/s] 55%|█████▍    | 219108/400000 [00:27<00:22, 7979.82it/s] 55%|█████▍    | 219909/400000 [00:27<00:22, 7986.57it/s] 55%|█████▌    | 220734/400000 [00:27<00:22, 8062.56it/s] 55%|█████▌    | 221565/400000 [00:27<00:21, 8135.19it/s] 56%|█████▌    | 222380/400000 [00:27<00:21, 8131.71it/s] 56%|█████▌    | 223206/400000 [00:27<00:21, 8169.45it/s] 56%|█████▌    | 224024/400000 [00:28<00:21, 8160.16it/s] 56%|█████▌    | 224841/400000 [00:28<00:21, 8135.38it/s] 56%|█████▋    | 225672/400000 [00:28<00:21, 8186.22it/s] 57%|█████▋    | 226491/400000 [00:28<00:21, 8170.07it/s] 57%|█████▋    | 227317/400000 [00:28<00:21, 8196.34it/s] 57%|█████▋    | 228145/400000 [00:28<00:20, 8219.64it/s] 57%|█████▋    | 228968/400000 [00:28<00:21, 8051.24it/s] 57%|█████▋    | 229780/400000 [00:28<00:21, 8069.85it/s] 58%|█████▊    | 230604/400000 [00:28<00:20, 8116.54it/s] 58%|█████▊    | 231417/400000 [00:28<00:21, 7947.37it/s] 58%|█████▊    | 232213/400000 [00:29<00:21, 7737.72it/s] 58%|█████▊    | 232989/400000 [00:29<00:21, 7720.80it/s] 58%|█████▊    | 233763/400000 [00:29<00:21, 7715.85it/s] 59%|█████▊    | 234558/400000 [00:29<00:21, 7784.60it/s] 59%|█████▉    | 235367/400000 [00:29<00:20, 7871.92it/s] 59%|█████▉    | 236166/400000 [00:29<00:20, 7904.90it/s] 59%|█████▉    | 236966/400000 [00:29<00:20, 7932.74it/s] 59%|█████▉    | 237772/400000 [00:29<00:20, 7968.76it/s] 60%|█████▉    | 238576/400000 [00:29<00:20, 7988.10it/s] 60%|█████▉    | 239388/400000 [00:30<00:20, 8025.98it/s] 60%|██████    | 240209/400000 [00:30<00:19, 8078.60it/s] 60%|██████    | 241018/400000 [00:30<00:19, 8069.99it/s] 60%|██████    | 241843/400000 [00:30<00:19, 8122.97it/s] 61%|██████    | 242672/400000 [00:30<00:19, 8169.95it/s] 61%|██████    | 243504/400000 [00:30<00:19, 8214.37it/s] 61%|██████    | 244326/400000 [00:30<00:19, 8168.35it/s] 61%|██████▏   | 245144/400000 [00:30<00:19, 7994.23it/s] 61%|██████▏   | 245955/400000 [00:30<00:19, 8026.94it/s] 62%|██████▏   | 246778/400000 [00:30<00:18, 8085.34it/s] 62%|██████▏   | 247610/400000 [00:31<00:18, 8152.92it/s] 62%|██████▏   | 248440/400000 [00:31<00:18, 8193.47it/s] 62%|██████▏   | 249260/400000 [00:31<00:18, 8180.64it/s] 63%|██████▎   | 250079/400000 [00:31<00:18, 8142.46it/s] 63%|██████▎   | 250894/400000 [00:31<00:18, 8144.30it/s] 63%|██████▎   | 251722/400000 [00:31<00:18, 8181.04it/s] 63%|██████▎   | 252541/400000 [00:31<00:18, 8100.43it/s] 63%|██████▎   | 253352/400000 [00:31<00:18, 7822.16it/s] 64%|██████▎   | 254171/400000 [00:31<00:18, 7925.76it/s] 64%|██████▍   | 255010/400000 [00:31<00:17, 8057.41it/s] 64%|██████▍   | 255819/400000 [00:32<00:17, 8067.14it/s] 64%|██████▍   | 256655/400000 [00:32<00:17, 8152.62it/s] 64%|██████▍   | 257472/400000 [00:32<00:18, 7871.35it/s] 65%|██████▍   | 258263/400000 [00:32<00:20, 7079.54it/s] 65%|██████▍   | 259056/400000 [00:32<00:19, 7313.26it/s] 65%|██████▍   | 259874/400000 [00:32<00:18, 7553.00it/s] 65%|██████▌   | 260674/400000 [00:32<00:18, 7680.12it/s] 65%|██████▌   | 261467/400000 [00:32<00:17, 7753.20it/s] 66%|██████▌   | 262286/400000 [00:32<00:17, 7878.88it/s] 66%|██████▌   | 263109/400000 [00:32<00:17, 7980.65it/s] 66%|██████▌   | 263933/400000 [00:33<00:16, 8056.39it/s] 66%|██████▌   | 264742/400000 [00:33<00:16, 8053.83it/s] 66%|██████▋   | 265562/400000 [00:33<00:16, 8095.07it/s] 67%|██████▋   | 266373/400000 [00:33<00:16, 8068.95it/s] 67%|██████▋   | 267211/400000 [00:33<00:16, 8157.82it/s] 67%|██████▋   | 268028/400000 [00:33<00:16, 7975.53it/s] 67%|██████▋   | 268828/400000 [00:33<00:16, 7965.36it/s] 67%|██████▋   | 269650/400000 [00:33<00:16, 8038.44it/s] 68%|██████▊   | 270455/400000 [00:33<00:16, 8041.69it/s] 68%|██████▊   | 271260/400000 [00:33<00:16, 8027.61it/s] 68%|██████▊   | 272064/400000 [00:34<00:15, 8013.51it/s] 68%|██████▊   | 272866/400000 [00:34<00:15, 7995.93it/s] 68%|██████▊   | 273666/400000 [00:34<00:15, 7974.39it/s] 69%|██████▊   | 274464/400000 [00:34<00:15, 7972.89it/s] 69%|██████▉   | 275284/400000 [00:34<00:15, 8038.41it/s] 69%|██████▉   | 276098/400000 [00:34<00:15, 8065.54it/s] 69%|██████▉   | 276905/400000 [00:34<00:15, 7985.12it/s] 69%|██████▉   | 277704/400000 [00:34<00:15, 7968.53it/s] 70%|██████▉   | 278519/400000 [00:34<00:15, 8018.12it/s] 70%|██████▉   | 279342/400000 [00:35<00:14, 8079.49it/s] 70%|███████   | 280170/400000 [00:35<00:14, 8135.98it/s] 70%|███████   | 280984/400000 [00:35<00:14, 8083.59it/s] 70%|███████   | 281793/400000 [00:35<00:14, 8032.66it/s] 71%|███████   | 282602/400000 [00:35<00:14, 8049.11it/s] 71%|███████   | 283428/400000 [00:35<00:14, 8109.87it/s] 71%|███████   | 284241/400000 [00:35<00:14, 8114.91it/s] 71%|███████▏  | 285053/400000 [00:35<00:14, 8067.37it/s] 71%|███████▏  | 285860/400000 [00:35<00:14, 7984.03it/s] 72%|███████▏  | 286699/400000 [00:35<00:13, 8099.85it/s] 72%|███████▏  | 287547/400000 [00:36<00:13, 8208.43it/s] 72%|███████▏  | 288369/400000 [00:36<00:13, 8207.18it/s] 72%|███████▏  | 289191/400000 [00:36<00:13, 8119.43it/s] 73%|███████▎  | 290004/400000 [00:36<00:13, 8103.21it/s] 73%|███████▎  | 290815/400000 [00:36<00:13, 8099.59it/s] 73%|███████▎  | 291638/400000 [00:36<00:13, 8137.49it/s] 73%|███████▎  | 292459/400000 [00:36<00:13, 8158.02it/s] 73%|███████▎  | 293276/400000 [00:36<00:13, 8159.95it/s] 74%|███████▎  | 294093/400000 [00:36<00:13, 8080.48it/s] 74%|███████▎  | 294923/400000 [00:36<00:12, 8142.44it/s] 74%|███████▍  | 295738/400000 [00:37<00:12, 8113.13it/s] 74%|███████▍  | 296550/400000 [00:37<00:12, 8110.44it/s] 74%|███████▍  | 297362/400000 [00:37<00:12, 8004.85it/s] 75%|███████▍  | 298163/400000 [00:37<00:12, 7977.64it/s] 75%|███████▍  | 298962/400000 [00:37<00:12, 7904.50it/s] 75%|███████▍  | 299774/400000 [00:37<00:12, 7965.15it/s] 75%|███████▌  | 300600/400000 [00:37<00:12, 8049.75it/s] 75%|███████▌  | 301406/400000 [00:37<00:12, 8050.38it/s] 76%|███████▌  | 302213/400000 [00:37<00:12, 8053.77it/s] 76%|███████▌  | 303044/400000 [00:37<00:11, 8127.95it/s] 76%|███████▌  | 303870/400000 [00:38<00:11, 8166.43it/s] 76%|███████▌  | 304687/400000 [00:38<00:11, 8128.63it/s] 76%|███████▋  | 305501/400000 [00:38<00:11, 8103.65it/s] 77%|███████▋  | 306312/400000 [00:38<00:11, 8060.04it/s] 77%|███████▋  | 307134/400000 [00:38<00:11, 8103.37it/s] 77%|███████▋  | 307948/400000 [00:38<00:11, 8112.74it/s] 77%|███████▋  | 308775/400000 [00:38<00:11, 8157.35it/s] 77%|███████▋  | 309591/400000 [00:38<00:11, 8123.42it/s] 78%|███████▊  | 310404/400000 [00:38<00:11, 8021.40it/s] 78%|███████▊  | 311207/400000 [00:38<00:11, 8022.01it/s] 78%|███████▊  | 312010/400000 [00:39<00:11, 7997.46it/s] 78%|███████▊  | 312815/400000 [00:39<00:10, 8012.04it/s] 78%|███████▊  | 313627/400000 [00:39<00:10, 8043.43it/s] 79%|███████▊  | 314432/400000 [00:39<00:10, 8008.43it/s] 79%|███████▉  | 315233/400000 [00:39<00:10, 7827.73it/s] 79%|███████▉  | 316017/400000 [00:39<00:10, 7645.43it/s] 79%|███████▉  | 316845/400000 [00:39<00:10, 7824.95it/s] 79%|███████▉  | 317648/400000 [00:39<00:10, 7882.81it/s] 80%|███████▉  | 318456/400000 [00:39<00:10, 7939.08it/s] 80%|███████▉  | 319252/400000 [00:39<00:10, 7910.88it/s] 80%|████████  | 320082/400000 [00:40<00:09, 8023.64it/s] 80%|████████  | 320911/400000 [00:40<00:09, 8099.96it/s] 80%|████████  | 321722/400000 [00:40<00:09, 8090.61it/s] 81%|████████  | 322532/400000 [00:40<00:09, 8035.17it/s] 81%|████████  | 323365/400000 [00:40<00:09, 8119.33it/s] 81%|████████  | 324182/400000 [00:40<00:09, 8131.74it/s] 81%|████████  | 324996/400000 [00:40<00:09, 7885.08it/s] 81%|████████▏ | 325787/400000 [00:40<00:09, 7810.68it/s] 82%|████████▏ | 326577/400000 [00:40<00:09, 7836.57it/s] 82%|████████▏ | 327416/400000 [00:40<00:09, 7992.88it/s] 82%|████████▏ | 328246/400000 [00:41<00:08, 8080.89it/s] 82%|████████▏ | 329079/400000 [00:41<00:08, 8152.85it/s] 82%|████████▏ | 329896/400000 [00:41<00:08, 8122.51it/s] 83%|████████▎ | 330710/400000 [00:41<00:08, 8076.10it/s] 83%|████████▎ | 331542/400000 [00:41<00:08, 8145.49it/s] 83%|████████▎ | 332364/400000 [00:41<00:08, 8164.82it/s] 83%|████████▎ | 333181/400000 [00:41<00:08, 8067.96it/s] 83%|████████▎ | 333989/400000 [00:41<00:08, 7950.25it/s] 84%|████████▎ | 334801/400000 [00:41<00:08, 7998.89it/s] 84%|████████▍ | 335621/400000 [00:41<00:07, 8057.64it/s] 84%|████████▍ | 336466/400000 [00:42<00:07, 8170.99it/s] 84%|████████▍ | 337293/400000 [00:42<00:07, 8199.50it/s] 85%|████████▍ | 338114/400000 [00:42<00:07, 8133.86it/s] 85%|████████▍ | 338928/400000 [00:42<00:07, 8128.56it/s] 85%|████████▍ | 339742/400000 [00:42<00:07, 8117.78it/s] 85%|████████▌ | 340571/400000 [00:42<00:07, 8164.97it/s] 85%|████████▌ | 341396/400000 [00:42<00:07, 8190.28it/s] 86%|████████▌ | 342216/400000 [00:42<00:07, 8072.80it/s] 86%|████████▌ | 343024/400000 [00:42<00:07, 8016.11it/s] 86%|████████▌ | 343827/400000 [00:43<00:07, 7914.01it/s] 86%|████████▌ | 344649/400000 [00:43<00:06, 8001.65it/s] 86%|████████▋ | 345450/400000 [00:43<00:06, 7939.84it/s] 87%|████████▋ | 346245/400000 [00:43<00:06, 7856.16it/s] 87%|████████▋ | 347039/400000 [00:43<00:06, 7880.93it/s] 87%|████████▋ | 347828/400000 [00:43<00:06, 7820.81it/s] 87%|████████▋ | 348611/400000 [00:43<00:06, 7815.83it/s] 87%|████████▋ | 349404/400000 [00:43<00:06, 7847.45it/s] 88%|████████▊ | 350193/400000 [00:43<00:06, 7858.78it/s] 88%|████████▊ | 351000/400000 [00:43<00:06, 7919.58it/s] 88%|████████▊ | 351793/400000 [00:44<00:06, 7837.14it/s] 88%|████████▊ | 352593/400000 [00:44<00:06, 7884.53it/s] 88%|████████▊ | 353391/400000 [00:44<00:05, 7912.54it/s] 89%|████████▊ | 354183/400000 [00:44<00:05, 7806.51it/s] 89%|████████▊ | 354965/400000 [00:44<00:05, 7720.65it/s] 89%|████████▉ | 355738/400000 [00:44<00:05, 7660.42it/s] 89%|████████▉ | 356505/400000 [00:44<00:05, 7356.62it/s] 89%|████████▉ | 357275/400000 [00:44<00:05, 7455.46it/s] 90%|████████▉ | 358037/400000 [00:44<00:05, 7502.96it/s] 90%|████████▉ | 358848/400000 [00:44<00:05, 7672.69it/s] 90%|████████▉ | 359643/400000 [00:45<00:05, 7751.67it/s] 90%|█████████ | 360449/400000 [00:45<00:05, 7840.92it/s] 90%|█████████ | 361256/400000 [00:45<00:04, 7907.09it/s] 91%|█████████ | 362048/400000 [00:45<00:04, 7868.00it/s] 91%|█████████ | 362868/400000 [00:45<00:04, 7962.38it/s] 91%|█████████ | 363672/400000 [00:45<00:04, 7982.44it/s] 91%|█████████ | 364480/400000 [00:45<00:04, 8009.69it/s] 91%|█████████▏| 365282/400000 [00:45<00:04, 7952.27it/s] 92%|█████████▏| 366078/400000 [00:45<00:04, 7832.10it/s] 92%|█████████▏| 366884/400000 [00:45<00:04, 7896.84it/s] 92%|█████████▏| 367686/400000 [00:46<00:04, 7933.34it/s] 92%|█████████▏| 368487/400000 [00:46<00:03, 7953.65it/s] 92%|█████████▏| 369283/400000 [00:46<00:03, 7866.90it/s] 93%|█████████▎| 370071/400000 [00:46<00:03, 7802.49it/s] 93%|█████████▎| 370879/400000 [00:46<00:03, 7882.35it/s] 93%|█████████▎| 371720/400000 [00:46<00:03, 8031.28it/s] 93%|█████████▎| 372552/400000 [00:46<00:03, 8115.37it/s] 93%|█████████▎| 373365/400000 [00:46<00:03, 8079.25it/s] 94%|█████████▎| 374174/400000 [00:46<00:03, 7981.37it/s] 94%|█████████▎| 374973/400000 [00:46<00:03, 7962.04it/s] 94%|█████████▍| 375793/400000 [00:47<00:03, 8029.61it/s] 94%|█████████▍| 376627/400000 [00:47<00:02, 8119.41it/s] 94%|█████████▍| 377440/400000 [00:47<00:02, 8118.59it/s] 95%|█████████▍| 378253/400000 [00:47<00:02, 7965.97it/s] 95%|█████████▍| 379051/400000 [00:47<00:02, 7901.23it/s] 95%|█████████▍| 379878/400000 [00:47<00:02, 8007.46it/s] 95%|█████████▌| 380702/400000 [00:47<00:02, 8074.87it/s] 95%|█████████▌| 381511/400000 [00:47<00:02, 8005.70it/s] 96%|█████████▌| 382323/400000 [00:47<00:02, 8037.81it/s] 96%|█████████▌| 383147/400000 [00:47<00:02, 8097.00it/s] 96%|█████████▌| 383958/400000 [00:48<00:01, 8067.36it/s] 96%|█████████▌| 384778/400000 [00:48<00:01, 8105.66it/s] 96%|█████████▋| 385611/400000 [00:48<00:01, 8168.94it/s] 97%|█████████▋| 386431/400000 [00:48<00:01, 8176.17it/s] 97%|█████████▋| 387250/400000 [00:48<00:01, 8179.12it/s] 97%|█████████▋| 388077/400000 [00:48<00:01, 8204.14it/s] 97%|█████████▋| 388898/400000 [00:48<00:01, 8168.78it/s] 97%|█████████▋| 389716/400000 [00:48<00:01, 8137.37it/s] 98%|█████████▊| 390530/400000 [00:48<00:01, 8042.64it/s] 98%|█████████▊| 391347/400000 [00:48<00:01, 8079.94it/s] 98%|█████████▊| 392156/400000 [00:49<00:00, 8069.29it/s] 98%|█████████▊| 392975/400000 [00:49<00:00, 8103.36it/s] 98%|█████████▊| 393795/400000 [00:49<00:00, 8131.21it/s] 99%|█████████▊| 394609/400000 [00:49<00:00, 8063.23it/s] 99%|█████████▉| 395416/400000 [00:49<00:00, 8036.53it/s] 99%|█████████▉| 396233/400000 [00:49<00:00, 8074.70it/s] 99%|█████████▉| 397049/400000 [00:49<00:00, 8096.49it/s] 99%|█████████▉| 397859/400000 [00:49<00:00, 8036.63it/s]100%|█████████▉| 398663/400000 [00:49<00:00, 7968.64it/s]100%|█████████▉| 399461/400000 [00:50<00:00, 7845.62it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 7986.83it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f8a2a1dea58> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01096811911919939 	 Accuracy: 55
Train Epoch: 1 	 Loss: 0.01114677366205681 	 Accuracy: 51

  model saves at 51% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15937 out of table with 15915 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15937 out of table with 15915 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-13 16:24:20.010672: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 16:24:20.014995: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 16:24:20.015780: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56497973a660 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 16:24:20.015798: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f89d6bbcd68> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 8.0346 - accuracy: 0.4760
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7586 - accuracy: 0.4940
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7586 - accuracy: 0.4940 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7625 - accuracy: 0.4938
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7648 - accuracy: 0.4936
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6794 - accuracy: 0.4992
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6710 - accuracy: 0.4997
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6724 - accuracy: 0.4996
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6223 - accuracy: 0.5029
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6314 - accuracy: 0.5023
11000/25000 [============>.................] - ETA: 4s - loss: 7.6722 - accuracy: 0.4996
12000/25000 [=============>................] - ETA: 4s - loss: 7.6462 - accuracy: 0.5013
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6619 - accuracy: 0.5003
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6677 - accuracy: 0.4999
15000/25000 [=================>............] - ETA: 3s - loss: 7.6441 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6484 - accuracy: 0.5012
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6441 - accuracy: 0.5015
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6360 - accuracy: 0.5020
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6610 - accuracy: 0.5004
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6544 - accuracy: 0.5008
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6586 - accuracy: 0.5005
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6694 - accuracy: 0.4998
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6753 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 10s 386us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f898e8398d0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f899be30160> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.3500 - crf_viterbi_accuracy: 0.1200 - val_loss: 1.3264 - val_crf_viterbi_accuracy: 0.1600

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

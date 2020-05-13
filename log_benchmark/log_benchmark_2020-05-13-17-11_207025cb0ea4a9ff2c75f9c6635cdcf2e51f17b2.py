
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f8946342f60> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 17:12:06.748619
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 17:12:06.752359
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 17:12:06.755378
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 17:12:06.758493
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f895210c400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352827.2812
Epoch 2/10

1/1 [==============================] - 0s 99ms/step - loss: 264011.1250
Epoch 3/10

1/1 [==============================] - 0s 94ms/step - loss: 170378.2031
Epoch 4/10

1/1 [==============================] - 0s 113ms/step - loss: 96072.9453
Epoch 5/10

1/1 [==============================] - 0s 92ms/step - loss: 53167.5391
Epoch 6/10

1/1 [==============================] - 0s 101ms/step - loss: 30394.5273
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 18630.7773
Epoch 8/10

1/1 [==============================] - 0s 102ms/step - loss: 12402.5400
Epoch 9/10

1/1 [==============================] - 0s 99ms/step - loss: 8911.4961
Epoch 10/10

1/1 [==============================] - 0s 101ms/step - loss: 6790.0562

  #### Inference Need return ypred, ytrue ######################### 
[[  0.7911445    0.7431621    0.1064661   -0.5491321    1.2081827
    0.04007453  -1.7650743   -0.9291799    1.1123773   -0.73419833
    0.14552218  -1.1702805    0.4576005   -1.0739484    0.32453573
   -0.0136565   -2.7277784   -0.550383     0.59786767  -0.64601314
   -0.03491998  -0.8207889   -1.2193328    0.11541003   0.02074081
    1.7482735   -1.234117    -0.8608283   -0.10226215   0.47412702
    1.0684587    0.55507016   0.23320153  -0.05057895   0.4765574
   -0.32987276   1.7118915   -0.73848635  -0.31537735   0.39667517
    0.31556794  -0.25420317  -0.20267737   1.0514369    0.786674
   -1.4143975    1.5583136    0.6690055    0.61362356  -1.4812849
    2.0060143   -0.43172407   0.76153433  -2.6698005    1.3371596
   -1.5210416    0.8349767    0.98186177   0.77611166  -0.30391234
   -0.10304187   6.8186526    5.762118     8.609989     7.4383597
   10.444923     8.290708     8.4376135    7.330514     6.949833
    9.5573225    7.9377203    7.47192      8.478483     7.9573135
    6.277785     6.170493     9.266852     7.961037     6.1004844
    9.305776     9.1759405    9.34667      8.692143     9.00362
    9.474934     8.015581     8.243242     8.353676     7.8396797
    8.5700245    8.001529     7.397155     6.77374      7.4557643
    8.254279     9.863672     7.33344      8.799741     8.641024
    6.4518795    9.3896475    8.806612     9.177593     6.879417
    9.075315     7.2548413    6.137838     9.867941     7.3225527
    8.87287      8.706844     8.28629      7.856819     6.6728377
    7.92204      6.2471633    6.300498     6.2188025    7.6677723
    0.97476417  -0.77903557   1.0119023   -0.3115886    0.75289994
    1.3215054    0.23743337  -2.1727126    0.84347236   0.3245328
   -0.5155091   -1.2543509    0.13451853  -0.9485169   -0.31579572
   -0.7709654    0.4298362   -1.1819685    0.42795798  -0.45322675
    1.0430305   -0.33954537   0.23237544   0.08561608  -1.8365258
   -2.1649687   -1.0266618    0.19957848  -0.40121242   2.0184987
    0.55502963   0.6869863   -0.25137746   0.1387456   -0.962553
   -2.3005457   -1.4394245    0.29115373   1.392354    -0.67997783
    0.06849533   1.3622228   -1.0496409    0.70180523   0.96217585
   -1.3790145    0.5759162    0.98665816   0.28473687  -0.9306732
   -0.66414255   0.22713858   0.9775288   -0.4561451    1.552476
   -0.26440877  -1.6314284    0.39941907  -1.8521826    1.6316744
    1.1294103    0.5224011    2.6765113    0.41995025   0.51057506
    0.96650803   2.303131     1.7642928    2.2604284    1.2631588
    1.5558748    2.6559544    0.93414485   0.23863459   0.16213965
    1.6611443    3.782967     0.69520074   0.12674928   2.9694905
    0.14430743   0.91312784   0.3796727    0.48950905   1.5475786
    1.3966639    0.5926577    0.6934806    1.8883634    0.8156736
    1.1107361    0.3649181    1.4461056    0.6470358    1.0469956
    0.8389314    1.5850614    0.90965873   0.1200586    1.3652092
    1.1698875    0.47578192   0.09582186   2.5029216    1.2141594
    0.91542757   0.52471733   0.73939943   1.0814488    1.0578322
    1.0215161    3.35984      0.20363456   1.9805517    1.3762662
    0.5704804    2.8796468    1.4123493    0.5256645    0.1657626
    0.12284511   9.490555     8.889022     7.5024543    7.3308725
    8.533124     8.089041     9.243236     8.461936     7.519027
    6.5113626    8.801072     9.93724      7.503342     8.332799
    7.9900703    8.103774     6.8165345    8.710578     9.887473
    7.1281395    8.502468     9.421738     6.7521677    8.369862
    8.788104     7.236377     9.633871     8.956098     7.802497
    9.836536     7.524818     8.297681     7.815578     7.6142607
    8.6592865    7.723712     6.448033     9.253476     8.470523
    9.072747     9.266181     8.275846     8.053671     9.031905
    8.058434     9.239344     7.948706     7.5599294    9.448661
    8.43144      7.6408134    8.25395      7.7147145    6.134002
    9.687668    10.048541     7.195585     8.928426     8.7118435
    1.0602376    2.659676     0.8100762    0.59024864   1.7380017
    2.090385     0.39866447   2.1200705    0.4749577    2.719528
    1.1942586    2.2964363    0.99344677   1.8097751    1.1341828
    0.29373115   1.7390485    0.13341725   0.2536906    1.9098022
    0.28099507   1.207555     0.22291678   0.7919174    1.7796395
    1.429044     1.802341     1.6105473    0.3448807    0.75000405
    0.19362104   1.4154801    1.4853587    2.8617468    3.89299
    1.6594015    0.13605869   1.1550795    1.0563092    1.1826212
    0.23342025   1.2525339    3.137404     1.4852706    2.4569678
    2.4634886    0.23140824   0.507805     0.6797579    0.20956624
    0.35120934   0.3126005    0.9407402    2.2155657    0.09651482
    0.9038219    0.90571445   1.5908774    0.5588445    2.320334
  -10.46347      6.2089763   -4.7935553 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 17:12:16.940821
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.5303
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 17:12:16.944509
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8772.22
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 17:12:16.947613
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.2964
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 17:12:16.950698
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -784.614
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140227205939720
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140226264560192
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140226264560696
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140226264561200
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140226264561704
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140226264562208

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f894df8deb8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.639538
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.597440
grad_step = 000002, loss = 0.562540
grad_step = 000003, loss = 0.524788
grad_step = 000004, loss = 0.484909
grad_step = 000005, loss = 0.451696
grad_step = 000006, loss = 0.436570
grad_step = 000007, loss = 0.432390
grad_step = 000008, loss = 0.420728
grad_step = 000009, loss = 0.399231
grad_step = 000010, loss = 0.381810
grad_step = 000011, loss = 0.371167
grad_step = 000012, loss = 0.363368
grad_step = 000013, loss = 0.354867
grad_step = 000014, loss = 0.344330
grad_step = 000015, loss = 0.331801
grad_step = 000016, loss = 0.318187
grad_step = 000017, loss = 0.305144
grad_step = 000018, loss = 0.294332
grad_step = 000019, loss = 0.285767
grad_step = 000020, loss = 0.277189
grad_step = 000021, loss = 0.266748
grad_step = 000022, loss = 0.255169
grad_step = 000023, loss = 0.244376
grad_step = 000024, loss = 0.235300
grad_step = 000025, loss = 0.227349
grad_step = 000026, loss = 0.219280
grad_step = 000027, loss = 0.210401
grad_step = 000028, loss = 0.201055
grad_step = 000029, loss = 0.192026
grad_step = 000030, loss = 0.183817
grad_step = 000031, loss = 0.176307
grad_step = 000032, loss = 0.168977
grad_step = 000033, loss = 0.161442
grad_step = 000034, loss = 0.153806
grad_step = 000035, loss = 0.146475
grad_step = 000036, loss = 0.139735
grad_step = 000037, loss = 0.133410
grad_step = 000038, loss = 0.127095
grad_step = 000039, loss = 0.120692
grad_step = 000040, loss = 0.114495
grad_step = 000041, loss = 0.108724
grad_step = 000042, loss = 0.103251
grad_step = 000043, loss = 0.097907
grad_step = 000044, loss = 0.092696
grad_step = 000045, loss = 0.087665
grad_step = 000046, loss = 0.082813
grad_step = 000047, loss = 0.078181
grad_step = 000048, loss = 0.073778
grad_step = 000049, loss = 0.069547
grad_step = 000050, loss = 0.065480
grad_step = 000051, loss = 0.061572
grad_step = 000052, loss = 0.057814
grad_step = 000053, loss = 0.054265
grad_step = 000054, loss = 0.050943
grad_step = 000055, loss = 0.047735
grad_step = 000056, loss = 0.044609
grad_step = 000057, loss = 0.041673
grad_step = 000058, loss = 0.038951
grad_step = 000059, loss = 0.036367
grad_step = 000060, loss = 0.033879
grad_step = 000061, loss = 0.031507
grad_step = 000062, loss = 0.029317
grad_step = 000063, loss = 0.027298
grad_step = 000064, loss = 0.025359
grad_step = 000065, loss = 0.023507
grad_step = 000066, loss = 0.021811
grad_step = 000067, loss = 0.020250
grad_step = 000068, loss = 0.018776
grad_step = 000069, loss = 0.017376
grad_step = 000070, loss = 0.016088
grad_step = 000071, loss = 0.014922
grad_step = 000072, loss = 0.013830
grad_step = 000073, loss = 0.012805
grad_step = 000074, loss = 0.011865
grad_step = 000075, loss = 0.011015
grad_step = 000076, loss = 0.010233
grad_step = 000077, loss = 0.009497
grad_step = 000078, loss = 0.008829
grad_step = 000079, loss = 0.008229
grad_step = 000080, loss = 0.007677
grad_step = 000081, loss = 0.007164
grad_step = 000082, loss = 0.006701
grad_step = 000083, loss = 0.006289
grad_step = 000084, loss = 0.005908
grad_step = 000085, loss = 0.005558
grad_step = 000086, loss = 0.005245
grad_step = 000087, loss = 0.004964
grad_step = 000088, loss = 0.004704
grad_step = 000089, loss = 0.004469
grad_step = 000090, loss = 0.004258
grad_step = 000091, loss = 0.004068
grad_step = 000092, loss = 0.003892
grad_step = 000093, loss = 0.003734
grad_step = 000094, loss = 0.003592
grad_step = 000095, loss = 0.003463
grad_step = 000096, loss = 0.003343
grad_step = 000097, loss = 0.003236
grad_step = 000098, loss = 0.003139
grad_step = 000099, loss = 0.003049
grad_step = 000100, loss = 0.002967
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002893
grad_step = 000102, loss = 0.002825
grad_step = 000103, loss = 0.002762
grad_step = 000104, loss = 0.002704
grad_step = 000105, loss = 0.002652
grad_step = 000106, loss = 0.002603
grad_step = 000107, loss = 0.002558
grad_step = 000108, loss = 0.002517
grad_step = 000109, loss = 0.002479
grad_step = 000110, loss = 0.002444
grad_step = 000111, loss = 0.002412
grad_step = 000112, loss = 0.002382
grad_step = 000113, loss = 0.002354
grad_step = 000114, loss = 0.002329
grad_step = 000115, loss = 0.002305
grad_step = 000116, loss = 0.002283
grad_step = 000117, loss = 0.002262
grad_step = 000118, loss = 0.002243
grad_step = 000119, loss = 0.002226
grad_step = 000120, loss = 0.002210
grad_step = 000121, loss = 0.002196
grad_step = 000122, loss = 0.002183
grad_step = 000123, loss = 0.002170
grad_step = 000124, loss = 0.002156
grad_step = 000125, loss = 0.002143
grad_step = 000126, loss = 0.002132
grad_step = 000127, loss = 0.002124
grad_step = 000128, loss = 0.002115
grad_step = 000129, loss = 0.002107
grad_step = 000130, loss = 0.002097
grad_step = 000131, loss = 0.002088
grad_step = 000132, loss = 0.002079
grad_step = 000133, loss = 0.002072
grad_step = 000134, loss = 0.002066
grad_step = 000135, loss = 0.002061
grad_step = 000136, loss = 0.002058
grad_step = 000137, loss = 0.002055
grad_step = 000138, loss = 0.002053
grad_step = 000139, loss = 0.002048
grad_step = 000140, loss = 0.002040
grad_step = 000141, loss = 0.002029
grad_step = 000142, loss = 0.002020
grad_step = 000143, loss = 0.002014
grad_step = 000144, loss = 0.002012
grad_step = 000145, loss = 0.002012
grad_step = 000146, loss = 0.002014
grad_step = 000147, loss = 0.002020
grad_step = 000148, loss = 0.002025
grad_step = 000149, loss = 0.002030
grad_step = 000150, loss = 0.002018
grad_step = 000151, loss = 0.002000
grad_step = 000152, loss = 0.001982
grad_step = 000153, loss = 0.001975
grad_step = 000154, loss = 0.001980
grad_step = 000155, loss = 0.001988
grad_step = 000156, loss = 0.001996
grad_step = 000157, loss = 0.001995
grad_step = 000158, loss = 0.001987
grad_step = 000159, loss = 0.001971
grad_step = 000160, loss = 0.001958
grad_step = 000161, loss = 0.001950
grad_step = 000162, loss = 0.001949
grad_step = 000163, loss = 0.001953
grad_step = 000164, loss = 0.001959
grad_step = 000165, loss = 0.001967
grad_step = 000166, loss = 0.001974
grad_step = 000167, loss = 0.001982
grad_step = 000168, loss = 0.001979
grad_step = 000169, loss = 0.001969
grad_step = 000170, loss = 0.001948
grad_step = 000171, loss = 0.001930
grad_step = 000172, loss = 0.001920
grad_step = 000173, loss = 0.001920
grad_step = 000174, loss = 0.001927
grad_step = 000175, loss = 0.001936
grad_step = 000176, loss = 0.001948
grad_step = 000177, loss = 0.001955
grad_step = 000178, loss = 0.001960
grad_step = 000179, loss = 0.001947
grad_step = 000180, loss = 0.001929
grad_step = 000181, loss = 0.001908
grad_step = 000182, loss = 0.001897
grad_step = 000183, loss = 0.001897
grad_step = 000184, loss = 0.001905
grad_step = 000185, loss = 0.001915
grad_step = 000186, loss = 0.001921
grad_step = 000187, loss = 0.001924
grad_step = 000188, loss = 0.001917
grad_step = 000189, loss = 0.001907
grad_step = 000190, loss = 0.001893
grad_step = 000191, loss = 0.001883
grad_step = 000192, loss = 0.001876
grad_step = 000193, loss = 0.001873
grad_step = 000194, loss = 0.001874
grad_step = 000195, loss = 0.001877
grad_step = 000196, loss = 0.001883
grad_step = 000197, loss = 0.001891
grad_step = 000198, loss = 0.001904
grad_step = 000199, loss = 0.001917
grad_step = 000200, loss = 0.001937
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001942
grad_step = 000202, loss = 0.001936
grad_step = 000203, loss = 0.001901
grad_step = 000204, loss = 0.001868
grad_step = 000205, loss = 0.001851
grad_step = 000206, loss = 0.001858
grad_step = 000207, loss = 0.001875
grad_step = 000208, loss = 0.001883
grad_step = 000209, loss = 0.001879
grad_step = 000210, loss = 0.001861
grad_step = 000211, loss = 0.001844
grad_step = 000212, loss = 0.001836
grad_step = 000213, loss = 0.001840
grad_step = 000214, loss = 0.001849
grad_step = 000215, loss = 0.001855
grad_step = 000216, loss = 0.001854
grad_step = 000217, loss = 0.001846
grad_step = 000218, loss = 0.001837
grad_step = 000219, loss = 0.001829
grad_step = 000220, loss = 0.001823
grad_step = 000221, loss = 0.001819
grad_step = 000222, loss = 0.001819
grad_step = 000223, loss = 0.001821
grad_step = 000224, loss = 0.001823
grad_step = 000225, loss = 0.001827
grad_step = 000226, loss = 0.001828
grad_step = 000227, loss = 0.001830
grad_step = 000228, loss = 0.001825
grad_step = 000229, loss = 0.001822
grad_step = 000230, loss = 0.001812
grad_step = 000231, loss = 0.001805
grad_step = 000232, loss = 0.001798
grad_step = 000233, loss = 0.001794
grad_step = 000234, loss = 0.001793
grad_step = 000235, loss = 0.001794
grad_step = 000236, loss = 0.001795
grad_step = 000237, loss = 0.001795
grad_step = 000238, loss = 0.001799
grad_step = 000239, loss = 0.001803
grad_step = 000240, loss = 0.001817
grad_step = 000241, loss = 0.001827
grad_step = 000242, loss = 0.001855
grad_step = 000243, loss = 0.001846
grad_step = 000244, loss = 0.001838
grad_step = 000245, loss = 0.001792
grad_step = 000246, loss = 0.001770
grad_step = 000247, loss = 0.001784
grad_step = 000248, loss = 0.001802
grad_step = 000249, loss = 0.001804
grad_step = 000250, loss = 0.001778
grad_step = 000251, loss = 0.001757
grad_step = 000252, loss = 0.001752
grad_step = 000253, loss = 0.001759
grad_step = 000254, loss = 0.001783
grad_step = 000255, loss = 0.001828
grad_step = 000256, loss = 0.001943
grad_step = 000257, loss = 0.002014
grad_step = 000258, loss = 0.002111
grad_step = 000259, loss = 0.001880
grad_step = 000260, loss = 0.001746
grad_step = 000261, loss = 0.001788
grad_step = 000262, loss = 0.001839
grad_step = 000263, loss = 0.001781
grad_step = 000264, loss = 0.001747
grad_step = 000265, loss = 0.001793
grad_step = 000266, loss = 0.001783
grad_step = 000267, loss = 0.001729
grad_step = 000268, loss = 0.001745
grad_step = 000269, loss = 0.001777
grad_step = 000270, loss = 0.001756
grad_step = 000271, loss = 0.001732
grad_step = 000272, loss = 0.001706
grad_step = 000273, loss = 0.001698
grad_step = 000274, loss = 0.001710
grad_step = 000275, loss = 0.001728
grad_step = 000276, loss = 0.001731
grad_step = 000277, loss = 0.001719
grad_step = 000278, loss = 0.001704
grad_step = 000279, loss = 0.001685
grad_step = 000280, loss = 0.001670
grad_step = 000281, loss = 0.001667
grad_step = 000282, loss = 0.001671
grad_step = 000283, loss = 0.001674
grad_step = 000284, loss = 0.001678
grad_step = 000285, loss = 0.001688
grad_step = 000286, loss = 0.001691
grad_step = 000287, loss = 0.001697
grad_step = 000288, loss = 0.001696
grad_step = 000289, loss = 0.001702
grad_step = 000290, loss = 0.001694
grad_step = 000291, loss = 0.001692
grad_step = 000292, loss = 0.001679
grad_step = 000293, loss = 0.001673
grad_step = 000294, loss = 0.001651
grad_step = 000295, loss = 0.001635
grad_step = 000296, loss = 0.001621
grad_step = 000297, loss = 0.001612
grad_step = 000298, loss = 0.001604
grad_step = 000299, loss = 0.001600
grad_step = 000300, loss = 0.001597
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001594
grad_step = 000302, loss = 0.001593
grad_step = 000303, loss = 0.001594
grad_step = 000304, loss = 0.001603
grad_step = 000305, loss = 0.001630
grad_step = 000306, loss = 0.001724
grad_step = 000307, loss = 0.001859
grad_step = 000308, loss = 0.002123
grad_step = 000309, loss = 0.002000
grad_step = 000310, loss = 0.001783
grad_step = 000311, loss = 0.001596
grad_step = 000312, loss = 0.001720
grad_step = 000313, loss = 0.001842
grad_step = 000314, loss = 0.001681
grad_step = 000315, loss = 0.001595
grad_step = 000316, loss = 0.001690
grad_step = 000317, loss = 0.001700
grad_step = 000318, loss = 0.001611
grad_step = 000319, loss = 0.001570
grad_step = 000320, loss = 0.001647
grad_step = 000321, loss = 0.001675
grad_step = 000322, loss = 0.001596
grad_step = 000323, loss = 0.001568
grad_step = 000324, loss = 0.001594
grad_step = 000325, loss = 0.001612
grad_step = 000326, loss = 0.001597
grad_step = 000327, loss = 0.001564
grad_step = 000328, loss = 0.001555
grad_step = 000329, loss = 0.001580
grad_step = 000330, loss = 0.001580
grad_step = 000331, loss = 0.001546
grad_step = 000332, loss = 0.001550
grad_step = 000333, loss = 0.001571
grad_step = 000334, loss = 0.001560
grad_step = 000335, loss = 0.001542
grad_step = 000336, loss = 0.001540
grad_step = 000337, loss = 0.001545
grad_step = 000338, loss = 0.001548
grad_step = 000339, loss = 0.001540
grad_step = 000340, loss = 0.001527
grad_step = 000341, loss = 0.001529
grad_step = 000342, loss = 0.001536
grad_step = 000343, loss = 0.001531
grad_step = 000344, loss = 0.001520
grad_step = 000345, loss = 0.001518
grad_step = 000346, loss = 0.001521
grad_step = 000347, loss = 0.001521
grad_step = 000348, loss = 0.001519
grad_step = 000349, loss = 0.001515
grad_step = 000350, loss = 0.001510
grad_step = 000351, loss = 0.001506
grad_step = 000352, loss = 0.001506
grad_step = 000353, loss = 0.001508
grad_step = 000354, loss = 0.001506
grad_step = 000355, loss = 0.001503
grad_step = 000356, loss = 0.001500
grad_step = 000357, loss = 0.001496
grad_step = 000358, loss = 0.001493
grad_step = 000359, loss = 0.001492
grad_step = 000360, loss = 0.001492
grad_step = 000361, loss = 0.001491
grad_step = 000362, loss = 0.001490
grad_step = 000363, loss = 0.001488
grad_step = 000364, loss = 0.001487
grad_step = 000365, loss = 0.001484
grad_step = 000366, loss = 0.001482
grad_step = 000367, loss = 0.001479
grad_step = 000368, loss = 0.001477
grad_step = 000369, loss = 0.001474
grad_step = 000370, loss = 0.001472
grad_step = 000371, loss = 0.001470
grad_step = 000372, loss = 0.001468
grad_step = 000373, loss = 0.001466
grad_step = 000374, loss = 0.001464
grad_step = 000375, loss = 0.001462
grad_step = 000376, loss = 0.001461
grad_step = 000377, loss = 0.001459
grad_step = 000378, loss = 0.001458
grad_step = 000379, loss = 0.001456
grad_step = 000380, loss = 0.001456
grad_step = 000381, loss = 0.001457
grad_step = 000382, loss = 0.001463
grad_step = 000383, loss = 0.001476
grad_step = 000384, loss = 0.001506
grad_step = 000385, loss = 0.001560
grad_step = 000386, loss = 0.001634
grad_step = 000387, loss = 0.001713
grad_step = 000388, loss = 0.001705
grad_step = 000389, loss = 0.001622
grad_step = 000390, loss = 0.001500
grad_step = 000391, loss = 0.001448
grad_step = 000392, loss = 0.001489
grad_step = 000393, loss = 0.001541
grad_step = 000394, loss = 0.001528
grad_step = 000395, loss = 0.001467
grad_step = 000396, loss = 0.001449
grad_step = 000397, loss = 0.001477
grad_step = 000398, loss = 0.001484
grad_step = 000399, loss = 0.001464
grad_step = 000400, loss = 0.001445
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001448
grad_step = 000402, loss = 0.001453
grad_step = 000403, loss = 0.001440
grad_step = 000404, loss = 0.001430
grad_step = 000405, loss = 0.001432
grad_step = 000406, loss = 0.001432
grad_step = 000407, loss = 0.001428
grad_step = 000408, loss = 0.001422
grad_step = 000409, loss = 0.001419
grad_step = 000410, loss = 0.001414
grad_step = 000411, loss = 0.001408
grad_step = 000412, loss = 0.001408
grad_step = 000413, loss = 0.001411
grad_step = 000414, loss = 0.001411
grad_step = 000415, loss = 0.001404
grad_step = 000416, loss = 0.001393
grad_step = 000417, loss = 0.001388
grad_step = 000418, loss = 0.001390
grad_step = 000419, loss = 0.001395
grad_step = 000420, loss = 0.001396
grad_step = 000421, loss = 0.001392
grad_step = 000422, loss = 0.001387
grad_step = 000423, loss = 0.001385
grad_step = 000424, loss = 0.001385
grad_step = 000425, loss = 0.001384
grad_step = 000426, loss = 0.001380
grad_step = 000427, loss = 0.001374
grad_step = 000428, loss = 0.001369
grad_step = 000429, loss = 0.001367
grad_step = 000430, loss = 0.001365
grad_step = 000431, loss = 0.001364
grad_step = 000432, loss = 0.001361
grad_step = 000433, loss = 0.001358
grad_step = 000434, loss = 0.001355
grad_step = 000435, loss = 0.001354
grad_step = 000436, loss = 0.001353
grad_step = 000437, loss = 0.001352
grad_step = 000438, loss = 0.001352
grad_step = 000439, loss = 0.001351
grad_step = 000440, loss = 0.001350
grad_step = 000441, loss = 0.001352
grad_step = 000442, loss = 0.001358
grad_step = 000443, loss = 0.001373
grad_step = 000444, loss = 0.001407
grad_step = 000445, loss = 0.001469
grad_step = 000446, loss = 0.001594
grad_step = 000447, loss = 0.001705
grad_step = 000448, loss = 0.001799
grad_step = 000449, loss = 0.001612
grad_step = 000450, loss = 0.001415
grad_step = 000451, loss = 0.001361
grad_step = 000452, loss = 0.001447
grad_step = 000453, loss = 0.001492
grad_step = 000454, loss = 0.001399
grad_step = 000455, loss = 0.001356
grad_step = 000456, loss = 0.001421
grad_step = 000457, loss = 0.001422
grad_step = 000458, loss = 0.001351
grad_step = 000459, loss = 0.001333
grad_step = 000460, loss = 0.001383
grad_step = 000461, loss = 0.001387
grad_step = 000462, loss = 0.001322
grad_step = 000463, loss = 0.001316
grad_step = 000464, loss = 0.001355
grad_step = 000465, loss = 0.001345
grad_step = 000466, loss = 0.001310
grad_step = 000467, loss = 0.001302
grad_step = 000468, loss = 0.001323
grad_step = 000469, loss = 0.001326
grad_step = 000470, loss = 0.001301
grad_step = 000471, loss = 0.001292
grad_step = 000472, loss = 0.001304
grad_step = 000473, loss = 0.001310
grad_step = 000474, loss = 0.001300
grad_step = 000475, loss = 0.001284
grad_step = 000476, loss = 0.001281
grad_step = 000477, loss = 0.001288
grad_step = 000478, loss = 0.001293
grad_step = 000479, loss = 0.001288
grad_step = 000480, loss = 0.001277
grad_step = 000481, loss = 0.001270
grad_step = 000482, loss = 0.001270
grad_step = 000483, loss = 0.001273
grad_step = 000484, loss = 0.001273
grad_step = 000485, loss = 0.001270
grad_step = 000486, loss = 0.001264
grad_step = 000487, loss = 0.001260
grad_step = 000488, loss = 0.001257
grad_step = 000489, loss = 0.001257
grad_step = 000490, loss = 0.001255
grad_step = 000491, loss = 0.001253
grad_step = 000492, loss = 0.001251
grad_step = 000493, loss = 0.001248
grad_step = 000494, loss = 0.001246
grad_step = 000495, loss = 0.001244
grad_step = 000496, loss = 0.001242
grad_step = 000497, loss = 0.001240
grad_step = 000498, loss = 0.001237
grad_step = 000499, loss = 0.001235
grad_step = 000500, loss = 0.001234
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001232
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

  date_run                              2020-05-13 17:12:34.974264
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.233838
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 17:12:34.980030
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.146042
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 17:12:34.987241
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.134233
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 17:12:34.992322
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.21916
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
0   2020-05-13 17:12:06.748619  ...    mean_absolute_error
1   2020-05-13 17:12:06.752359  ...     mean_squared_error
2   2020-05-13 17:12:06.755378  ...  median_absolute_error
3   2020-05-13 17:12:06.758493  ...               r2_score
4   2020-05-13 17:12:16.940821  ...    mean_absolute_error
5   2020-05-13 17:12:16.944509  ...     mean_squared_error
6   2020-05-13 17:12:16.947613  ...  median_absolute_error
7   2020-05-13 17:12:16.950698  ...               r2_score
8   2020-05-13 17:12:34.974264  ...    mean_absolute_error
9   2020-05-13 17:12:34.980030  ...     mean_squared_error
10  2020-05-13 17:12:34.987241  ...  median_absolute_error
11  2020-05-13 17:12:34.992322  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc470006cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 158540.72it/s] 65%|██████▍   | 6414336/9912422 [00:00<00:15, 226244.39it/s]9920512it [00:00, 42662642.51it/s]                           
0it [00:00, ?it/s]32768it [00:00, 555519.89it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:09, 163818.28it/s]1654784it [00:00, 11638677.53it/s]                         
0it [00:00, ?it/s]8192it [00:00, 272268.49it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc470011940> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc421fed0f0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc4229bfeb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc421f460f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc470011940> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc41f76dc18> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc4229bfeb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc421f03748> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc470011940> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc421dbe5c0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f59a9c27240> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=2aecf76c45c5d90b5a32f7a808a9fabaa0d97e806ddb0b049cfa562ed6368d0c
  Stored in directory: /tmp/pip-ephem-wheel-cache-n6bqxxa3/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f5941a22710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 2s
 1769472/17464789 [==>...........................] - ETA: 0s
 7921664/17464789 [============>.................] - ETA: 0s
14786560/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 17:14:01.372313: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 17:14:01.376246: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-13 17:14:01.376441: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564c2a1e8930 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 17:14:01.376457: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6206 - accuracy: 0.5030
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6666 - accuracy: 0.5000 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5746 - accuracy: 0.5060
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.4941 - accuracy: 0.5113
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.4489 - accuracy: 0.5142
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5337 - accuracy: 0.5087
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5418 - accuracy: 0.5081
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5286 - accuracy: 0.5090
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5474 - accuracy: 0.5078
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5608 - accuracy: 0.5069
11000/25000 [============>.................] - ETA: 3s - loss: 7.5370 - accuracy: 0.5085
12000/25000 [=============>................] - ETA: 3s - loss: 7.5529 - accuracy: 0.5074
13000/25000 [==============>...............] - ETA: 2s - loss: 7.5569 - accuracy: 0.5072
14000/25000 [===============>..............] - ETA: 2s - loss: 7.5779 - accuracy: 0.5058
15000/25000 [=================>............] - ETA: 2s - loss: 7.5756 - accuracy: 0.5059
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5967 - accuracy: 0.5046
17000/25000 [===================>..........] - ETA: 1s - loss: 7.5990 - accuracy: 0.5044
18000/25000 [====================>.........] - ETA: 1s - loss: 7.5917 - accuracy: 0.5049
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6158 - accuracy: 0.5033
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6344 - accuracy: 0.5021
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6462 - accuracy: 0.5013
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6631 - accuracy: 0.5002
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6560 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6570 - accuracy: 0.5006
25000/25000 [==============================] - 7s 282us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 17:14:15.005007
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 17:14:15.005007  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<26:16:44, 9.11kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<18:37:43, 12.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<13:05:38, 18.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<9:10:15, 26.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.89M/862M [00:01<6:24:30, 37.2kB/s].vector_cache/glove.6B.zip:   1%|          | 6.64M/862M [00:01<4:28:05, 53.2kB/s].vector_cache/glove.6B.zip:   1%|          | 10.3M/862M [00:01<3:06:59, 75.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.4M/862M [00:01<2:10:11, 108kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.8M/862M [00:01<1:30:53, 155kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.1M/862M [00:01<1:03:24, 221kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.2M/862M [00:02<44:16, 314kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.4M/862M [00:02<30:59, 447kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.3M/862M [00:02<21:52, 632kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.5M/862M [00:02<15:17, 898kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.3M/862M [00:02<10:46, 1.27MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.0M/862M [00:02<07:35, 1.79MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.0M/862M [00:02<05:23, 2.51MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<04:07, 3.28MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:04<04:47, 2.80MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<05:22, 2.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:04<04:10, 3.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<03:04, 4.36MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<55:08, 243kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<40:11, 333kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:06<28:27, 469kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:08<22:37, 588kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<18:33, 717kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<13:33, 980kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.3M/862M [00:09<09:37, 1.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:10<15:42, 842kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<12:13, 1.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<09:16, 1.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.5M/862M [00:10<06:37, 1.99MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:13<18:28, 713kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:13<18:21, 717kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:13<14:02, 938kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:13<10:21, 1.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:13<07:22, 1.78MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<28:00, 468kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:15<20:43, 632kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:15<14:59, 873kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:15<10:35, 1.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<59:52, 218kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:17<43:13, 301kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:17<30:31, 426kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<24:22, 532kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:19<19:44, 656kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:19<14:28, 894kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:19<10:14, 1.26MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:20<12:29:56, 17.2kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:21<8:46:02, 24.5kB/s] .vector_cache/glove.6B.zip:  11%|█         | 90.5M/862M [00:21<6:07:47, 35.0kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:22<4:19:43, 49.4kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<3:04:23, 69.5kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:23<2:09:31, 98.9kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.0M/862M [00:23<1:30:33, 141kB/s] .vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:24<1:10:21, 181kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<50:32, 252kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:25<35:38, 357kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<27:50, 456kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<20:48, 609kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<14:52, 851kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<13:32, 932kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<12:05, 1.04MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<09:00, 1.40MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:29<06:32, 1.92MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<08:30, 1.48MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:16, 1.72MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<05:24, 2.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<06:42, 1.86MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:59, 2.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<04:30, 2.76MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<06:05, 2.04MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:33, 2.23MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<04:12, 2.94MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:50, 2.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:22, 2.30MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<04:04, 3.02MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:44, 2.14MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:16, 2.33MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:00, 3.06MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:40, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:30, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:34, 2.19MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<04:13, 2.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<05:28, 2.22MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<05:07, 2.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<03:50, 3.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<02:51, 4.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<46:42, 258kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<33:54, 356kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<23:56, 503kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<16:52, 711kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:46<1:41:06, 119kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:46<1:13:14, 164kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<51:45, 232kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<36:16, 329kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<32:18, 369kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<23:51, 500kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<16:55, 703kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:50<14:36, 812kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<12:43, 932kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<09:30, 1.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:52<08:31, 1.38MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<07:12, 1.64MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:20, 2.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<06:27, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<05:43, 2.05MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<04:18, 2.72MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<06:03, 1.92MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<06:41, 1.74MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:11, 2.24MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<03:50, 3.02MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<06:18, 1.84MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<05:36, 2.06MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:13, 2.74MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<05:38, 2.04MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<05:06, 2.26MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<03:51, 2.97MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<05:24, 2.12MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<04:57, 2.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<03:43, 3.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:04<05:18, 2.14MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<04:53, 2.33MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<03:42, 3.06MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:06<05:14, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<04:50, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<03:40, 3.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:08<05:11, 2.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<04:47, 2.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<03:38, 3.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:10<05:44, 1.95MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<08:37, 1.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<07:01, 1.59MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<05:23, 2.07MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<03:52, 2.86MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<33:28, 332kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<23:58, 463kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<16:51, 655kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<12:12, 903kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<08:55, 1.23MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<21:48:29, 8.38kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<15:26:46, 11.8kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:16<10:51:08, 16.8kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<7:35:58, 24.0kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<5:18:01, 34.3kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<3:52:05, 46.9kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<2:43:37, 66.6kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<1:54:36, 94.8kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<1:22:16, 132kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<59:49, 181kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<42:19, 255kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<29:37, 363kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<30:46, 350kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<22:39, 475kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<16:05, 666kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<13:44, 778kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<10:42, 998kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<07:45, 1.37MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<07:54, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<06:36, 1.60MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<04:53, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<05:53, 1.79MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<05:11, 2.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<03:53, 2.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<05:11, 2.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<04:43, 2.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<03:33, 2.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:32<04:56, 2.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<04:31, 2.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<03:23, 3.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:34<04:49, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<05:29, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<04:22, 2.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:36<04:42, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<04:23, 2.34MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<03:20, 3.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<04:41, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<04:21, 2.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<03:18, 3.08MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<04:41, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<04:19, 2.34MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<03:16, 3.09MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<04:40, 2.15MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<05:20, 1.88MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<04:10, 2.41MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<03:03, 3.27MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<06:11, 1.62MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<05:22, 1.86MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<03:59, 2.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:46<05:06, 1.94MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:46<05:37, 1.77MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<04:22, 2.27MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<03:09, 3.12MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<12:35, 784kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<09:39, 1.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<06:56, 1.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<05:05, 1.93MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<08:19, 1.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<06:49, 1.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<05:01, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<05:48, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<05:03, 1.92MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:52<03:47, 2.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:55, 1.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<04:26, 2.17MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<03:20, 2.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:36, 2.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:12, 2.27MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:56<03:08, 3.04MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<04:26, 2.15MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<04:06, 2.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<03:06, 3.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<04:23, 2.15MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<04:02, 2.34MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<03:03, 3.07MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<04:21, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<04:54, 1.91MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<03:55, 2.39MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<02:49, 3.29MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<17:02, 547kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<12:45, 730kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<09:09, 1.01MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<08:32, 1.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<06:56, 1.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:03, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<05:42, 1.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<04:56, 1.85MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<03:41, 2.48MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:09<04:43, 1.93MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<04:13, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<03:09, 2.88MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<04:21, 2.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<03:59, 2.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<02:59, 3.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<04:12, 2.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<04:48, 1.87MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<03:49, 2.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<02:46, 3.22MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<07:53, 1.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<06:27, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:41, 1.89MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<03:24, 2.60MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:17<20:31, 431kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:17<16:07, 548kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<11:43, 752kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<09:34, 915kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<07:37, 1.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:31, 1.58MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<05:52, 1.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<05:53, 1.48MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<04:29, 1.93MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<03:17, 2.63MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<05:15, 1.64MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:23<04:35, 1.88MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<03:25, 2.51MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:25<04:23, 1.95MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<04:50, 1.77MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:25<03:49, 2.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:27<04:02, 2.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<03:42, 2.29MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<02:48, 3.01MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<03:55, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<03:28, 2.43MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:29<02:43, 3.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<01:59, 4.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<32:57, 254kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<23:54, 349kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<16:52, 493kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<13:44, 603kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<10:28, 790kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<07:31, 1.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:35<07:11, 1.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:35<05:52, 1.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:18, 1.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:37<04:55, 1.65MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:37<04:17, 1.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<03:10, 2.56MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:39<04:08, 1.95MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<03:43, 2.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<02:46, 2.89MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<03:50, 2.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:41<04:22, 1.83MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<03:28, 2.31MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<03:41, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<03:24, 2.33MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<02:35, 3.06MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<03:38, 2.16MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<03:21, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<02:31, 3.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<03:35, 2.17MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:07, 1.89MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<03:13, 2.42MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<02:21, 3.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<04:53, 1.58MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<04:14, 1.82MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<03:09, 2.44MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<04:21, 1.76MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<06:14, 1.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<05:09, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<03:45, 2.03MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:53<04:21, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:53<03:50, 1.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<02:50, 2.66MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:55<03:45, 2.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<03:24, 2.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<02:34, 2.92MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:57<03:33, 2.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:57<03:57, 1.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<03:06, 2.39MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<02:14, 3.31MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:58<55:53, 132kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:59<39:52, 185kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<28:00, 263kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<21:13, 345kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:01<15:36, 469kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<11:04, 659kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:02<09:25, 771kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:03<08:06, 896kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<05:59, 1.21MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:16, 1.69MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<06:47, 1.06MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<05:29, 1.31MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<04:00, 1.78MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<04:28, 1.59MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<03:51, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<02:50, 2.49MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<03:38, 1.94MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<03:15, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<02:27, 2.86MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<03:21, 2.08MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<03:04, 2.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<02:17, 3.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<03:15, 2.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<02:57, 2.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<02:13, 3.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<01:38, 4.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<59:09, 116kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<41:55, 163kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<29:42, 230kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<20:43, 327kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<1:35:05, 71.3kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<1:07:12, 101kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<47:02, 143kB/s]  .vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:18<34:20, 195kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<24:42, 271kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<17:23, 384kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<13:40, 486kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<10:15, 647kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<07:19, 903kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<06:39, 988kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<05:20, 1.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<03:53, 1.68MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<04:14, 1.53MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<03:38, 1.78MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<02:41, 2.40MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<03:24, 1.89MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<03:02, 2.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<02:17, 2.80MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:28<03:06, 2.05MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:28<02:48, 2.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<02:06, 3.02MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:30<02:57, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:30<02:43, 2.32MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<02:02, 3.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:32<02:54, 2.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:32<03:18, 1.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<02:34, 2.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<01:54, 3.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:34<03:17, 1.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<02:56, 2.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<02:12, 2.77MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<02:57, 2.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<02:41, 2.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<02:00, 3.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<02:50, 2.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<02:36, 2.31MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<01:58, 3.04MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:40<02:47, 2.13MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:40<02:39, 2.24MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<02:01, 2.92MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:42<02:38, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:42<02:30, 2.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<01:53, 3.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<01:26, 4.07MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:44<04:19, 1.35MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:44<03:40, 1.58MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<02:42, 2.14MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:46<03:08, 1.83MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<02:47, 2.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<02:05, 2.73MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:48<02:47, 2.03MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<02:37, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<01:58, 2.86MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:50<02:45, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:50<04:29, 1.25MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:50<03:47, 1.48MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<02:48, 1.99MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:52<03:03, 1.81MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:52<02:45, 2.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<02:02, 2.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:54<02:39, 2.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:54<03:00, 1.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<02:22, 2.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<01:42, 3.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<05:38, 959kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<04:30, 1.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<03:16, 1.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:58<03:31, 1.51MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:58<03:02, 1.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<02:17, 2.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<01:39, 3.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:00<14:43, 358kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:00<11:24, 461kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:00<08:15, 637kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<05:53, 889kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:02<05:18, 981kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:02<04:14, 1.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<03:05, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:04<03:21, 1.53MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<02:52, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:04<02:09, 2.37MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<01:33, 3.25MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:06<34:22, 147kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<24:33, 206kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<17:15, 292kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<12:02, 415kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<45:28, 110kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<32:18, 154kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:08<22:37, 219kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<16:53, 292kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<12:50, 383kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:10<09:11, 534kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<06:27, 755kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:11<07:00, 693kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:12<05:24, 896kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:12<03:51, 1.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:45, 1.74MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<29:59, 160kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<24:59, 192kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:15<18:31, 258kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:15<13:09, 363kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:15<09:17, 512kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<07:32, 625kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<05:46, 817kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<04:08, 1.13MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:18<03:57, 1.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:18<03:14, 1.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<02:22, 1.94MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:20<02:44, 1.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<02:23, 1.92MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:20<01:46, 2.56MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:22<02:18, 1.96MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<02:04, 2.18MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:22<01:33, 2.88MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<02:07, 2.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<01:56, 2.28MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<01:28, 3.01MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:26<02:03, 2.13MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:26<01:53, 2.32MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<01:25, 3.05MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<02:00, 2.15MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<01:50, 2.33MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<01:23, 3.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:30<01:58, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:30<01:48, 2.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<01:22, 3.08MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<01:56, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<01:46, 2.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:32<01:20, 3.08MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:34<01:54, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:34<02:10, 1.89MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<01:43, 2.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:36<01:50, 2.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:36<01:42, 2.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:36<01:16, 3.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<00:56, 4.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:38<16:22, 242kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:38<12:13, 324kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:38<08:42, 453kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<06:05, 642kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:40<06:28, 601kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:40<04:56, 787kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:40<03:31, 1.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<02:29, 1.53MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:42<15:55, 240kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:42<11:31, 331kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:42<08:06, 468kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<06:30, 577kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<04:56, 759kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:44<03:31, 1.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:46<03:19, 1.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:46<02:42, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:46<01:58, 1.85MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:48<02:13, 1.63MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:48<01:55, 1.87MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:48<01:25, 2.51MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:50<01:50, 1.94MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:50<01:39, 2.14MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:50<01:14, 2.84MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<00:54, 3.85MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:52<31:04, 112kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:52<23:11, 150kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:52<16:33, 210kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:52<11:34, 298kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:54<08:48, 388kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:54<06:30, 523kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:54<04:36, 734kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:56<03:58, 843kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:56<03:07, 1.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:56<02:15, 1.47MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:58<02:20, 1.40MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:58<01:58, 1.66MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:58<01:26, 2.25MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<01:02, 3.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:00<24:32, 131kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:00<17:49, 180kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:00<12:35, 253kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<08:42, 361kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:02<28:40, 110kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:02<20:21, 154kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:02<14:13, 219kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:04<10:32, 291kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:04<07:41, 399kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:04<05:24, 563kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:04<03:49, 790kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:06<04:17, 701kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:06<03:21, 894kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:06<02:25, 1.23MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:08<02:16, 1.29MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:08<01:56, 1.51MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:08<01:26, 2.02MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:10<01:35, 1.81MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:10<01:26, 1.99MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:10<01:03, 2.66MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:10<00:46, 3.61MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:12<04:57, 563kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:12<03:47, 736kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:12<02:42, 1.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:14<02:27, 1.11MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:14<01:59, 1.36MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:14<01:27, 1.85MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:16<01:38, 1.63MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:16<01:24, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:16<01:02, 2.52MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:18<01:20, 1.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:18<01:11, 2.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:18<00:53, 2.85MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:20<01:12, 2.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:20<01:06, 2.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:20<00:49, 3.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:22<01:08, 2.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:22<01:00, 2.44MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:22<00:50, 2.90MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:22<00:36, 3.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:24<04:40, 511kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:24<03:30, 678kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:24<02:29, 945kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<02:15, 1.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:26<01:48, 1.28MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:26<01:22, 1.67MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:26<00:58, 2.32MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:27<02:11, 1.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:28<01:45, 1.27MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:28<01:16, 1.73MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<01:23, 1.56MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:30<01:12, 1.81MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:30<00:52, 2.43MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:31<01:06, 1.91MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:32<00:59, 2.13MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:32<00:44, 2.82MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:33<00:59, 2.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:33<00:54, 2.25MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:34<00:40, 2.99MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:36<00:59, 1.98MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:36<01:06, 1.79MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:36<00:51, 2.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:36<00:37, 3.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:37<01:06, 1.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:38<00:57, 1.96MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:38<00:42, 2.61MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<00:55, 1.99MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:40<00:50, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:40<00:37, 2.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:41<00:50, 2.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:42<00:46, 2.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:42<00:34, 3.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:43<00:47, 2.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:44<00:43, 2.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:44<00:32, 3.06MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:45<00:45, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:45<00:41, 2.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:46<00:31, 3.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:47<00:43, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:47<00:39, 2.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:48<00:29, 3.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:49<00:41, 2.15MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:49<00:38, 2.33MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<00:28, 3.07MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:51<00:39, 2.16MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:51<00:36, 2.33MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:51<00:26, 3.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:53<00:37, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:53<00:43, 1.87MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:53<00:34, 2.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<00:24, 3.21MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:55<00:52, 1.47MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:55<00:45, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:55<00:33, 2.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:57<00:38, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:57<00:42, 1.73MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:57<00:32, 2.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:57<00:23, 2.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:59<00:35, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:59<00:32, 2.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:59<00:23, 2.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<00:31, 2.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:01<00:34, 1.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:01<00:27, 2.32MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<00:28, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:03<00:25, 2.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:03<00:19, 3.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:05<00:26, 2.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:05<00:24, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:05<00:17, 3.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:07<00:24, 2.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:07<00:22, 2.35MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:07<00:16, 3.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:09<00:22, 2.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:09<00:25, 1.88MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:09<00:20, 2.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:11<00:20, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:11<00:18, 2.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:11<00:13, 3.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:13<00:18, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:13<00:16, 2.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:13<00:12, 3.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:15<00:16, 2.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:15<00:15, 2.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:15<00:11, 3.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:17<00:14, 2.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:17<00:13, 2.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:17<00:09, 3.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:12, 2.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:19<00:11, 2.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:19<00:08, 3.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:21<00:10, 2.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:21<00:12, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:21<00:09, 2.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:23<00:08, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:23<00:07, 2.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:23<00:05, 3.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:23<00:03, 4.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:25<01:04, 239kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:25<00:45, 330kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:25<00:28, 467kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:25<00:17, 661kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:27<01:36, 118kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:27<01:05, 166kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:27<00:43, 235kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:27<00:21, 334kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:29<00:40, 177kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:29<00:29, 240kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:29<00:18, 338kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:29<00:10, 477kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:05, 589kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:31<00:03, 772kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:31<00:01, 1.07MB/s].vector_cache/glove.6B.zip: 862MB [06:31, 2.20MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 828/400000 [00:00<00:48, 8277.70it/s]  0%|          | 1694/400000 [00:00<00:47, 8387.56it/s]  1%|          | 2571/400000 [00:00<00:46, 8497.79it/s]  1%|          | 3411/400000 [00:00<00:46, 8467.45it/s]  1%|          | 4272/400000 [00:00<00:46, 8507.92it/s]  1%|▏         | 5153/400000 [00:00<00:45, 8595.76it/s]  2%|▏         | 6039/400000 [00:00<00:45, 8670.61it/s]  2%|▏         | 6903/400000 [00:00<00:45, 8659.84it/s]  2%|▏         | 7777/400000 [00:00<00:45, 8681.57it/s]  2%|▏         | 8652/400000 [00:01<00:44, 8701.83it/s]  2%|▏         | 9499/400000 [00:01<00:45, 8628.89it/s]  3%|▎         | 10371/400000 [00:01<00:45, 8653.69it/s]  3%|▎         | 11244/400000 [00:01<00:44, 8675.45it/s]  3%|▎         | 12104/400000 [00:01<00:44, 8647.07it/s]  3%|▎         | 12986/400000 [00:01<00:44, 8696.46it/s]  3%|▎         | 13857/400000 [00:01<00:44, 8700.18it/s]  4%|▎         | 14725/400000 [00:01<00:44, 8662.40it/s]  4%|▍         | 15614/400000 [00:01<00:44, 8727.06it/s]  4%|▍         | 16498/400000 [00:01<00:43, 8759.76it/s]  4%|▍         | 17389/400000 [00:02<00:43, 8802.51it/s]  5%|▍         | 18269/400000 [00:02<00:43, 8706.34it/s]  5%|▍         | 19149/400000 [00:02<00:43, 8733.31it/s]  5%|▌         | 20026/400000 [00:02<00:43, 8741.46it/s]  5%|▌         | 20901/400000 [00:02<00:43, 8721.06it/s]  5%|▌         | 21792/400000 [00:02<00:43, 8776.77it/s]  6%|▌         | 22675/400000 [00:02<00:42, 8789.84it/s]  6%|▌         | 23565/400000 [00:02<00:42, 8820.90it/s]  6%|▌         | 24455/400000 [00:02<00:42, 8841.64it/s]  6%|▋         | 25340/400000 [00:02<00:42, 8839.90it/s]  7%|▋         | 26225/400000 [00:03<00:42, 8817.88it/s]  7%|▋         | 27108/400000 [00:03<00:42, 8819.37it/s]  7%|▋         | 27990/400000 [00:03<00:42, 8686.82it/s]  7%|▋         | 28863/400000 [00:03<00:42, 8698.41it/s]  7%|▋         | 29749/400000 [00:03<00:42, 8743.61it/s]  8%|▊         | 30630/400000 [00:03<00:42, 8760.23it/s]  8%|▊         | 31507/400000 [00:03<00:42, 8716.16it/s]  8%|▊         | 32379/400000 [00:03<00:42, 8712.12it/s]  8%|▊         | 33257/400000 [00:03<00:42, 8729.70it/s]  9%|▊         | 34131/400000 [00:03<00:42, 8629.96it/s]  9%|▉         | 35010/400000 [00:04<00:42, 8676.26it/s]  9%|▉         | 35890/400000 [00:04<00:41, 8712.21it/s]  9%|▉         | 36772/400000 [00:04<00:41, 8744.12it/s]  9%|▉         | 37647/400000 [00:04<00:41, 8738.23it/s] 10%|▉         | 38521/400000 [00:04<00:41, 8716.75it/s] 10%|▉         | 39400/400000 [00:04<00:41, 8737.67it/s] 10%|█         | 40280/400000 [00:04<00:41, 8754.59it/s] 10%|█         | 41156/400000 [00:04<00:41, 8639.04it/s] 11%|█         | 42044/400000 [00:04<00:41, 8708.27it/s] 11%|█         | 42918/400000 [00:04<00:40, 8717.16it/s] 11%|█         | 43791/400000 [00:05<00:40, 8717.49it/s] 11%|█         | 44672/400000 [00:05<00:40, 8743.51it/s] 11%|█▏        | 45556/400000 [00:05<00:40, 8770.35it/s] 12%|█▏        | 46443/400000 [00:05<00:40, 8799.13it/s] 12%|█▏        | 47324/400000 [00:05<00:40, 8632.61it/s] 12%|█▏        | 48189/400000 [00:05<00:41, 8426.40it/s] 12%|█▏        | 49034/400000 [00:05<00:42, 8338.06it/s] 12%|█▏        | 49879/400000 [00:05<00:41, 8370.59it/s] 13%|█▎        | 50743/400000 [00:05<00:41, 8449.39it/s] 13%|█▎        | 51619/400000 [00:05<00:40, 8539.42it/s] 13%|█▎        | 52500/400000 [00:06<00:40, 8616.88it/s] 13%|█▎        | 53363/400000 [00:06<00:40, 8598.32it/s] 14%|█▎        | 54224/400000 [00:06<00:40, 8541.04it/s] 14%|█▍        | 55082/400000 [00:06<00:40, 8550.21it/s] 14%|█▍        | 55952/400000 [00:06<00:40, 8592.65it/s] 14%|█▍        | 56837/400000 [00:06<00:39, 8667.61it/s] 14%|█▍        | 57706/400000 [00:06<00:39, 8673.81it/s] 15%|█▍        | 58589/400000 [00:06<00:39, 8718.57it/s] 15%|█▍        | 59462/400000 [00:06<00:39, 8550.95it/s] 15%|█▌        | 60348/400000 [00:06<00:39, 8638.54it/s] 15%|█▌        | 61228/400000 [00:07<00:39, 8681.74it/s] 16%|█▌        | 62097/400000 [00:07<00:39, 8660.32it/s] 16%|█▌        | 62964/400000 [00:07<00:38, 8653.35it/s] 16%|█▌        | 63839/400000 [00:07<00:38, 8681.68it/s] 16%|█▌        | 64710/400000 [00:07<00:38, 8687.85it/s] 16%|█▋        | 65588/400000 [00:07<00:38, 8713.30it/s] 17%|█▋        | 66460/400000 [00:07<00:39, 8381.95it/s] 17%|█▋        | 67301/400000 [00:07<00:39, 8324.63it/s] 17%|█▋        | 68162/400000 [00:07<00:39, 8405.67it/s] 17%|█▋        | 69038/400000 [00:07<00:38, 8506.40it/s] 17%|█▋        | 69921/400000 [00:08<00:38, 8598.41it/s] 18%|█▊        | 70795/400000 [00:08<00:38, 8637.80it/s] 18%|█▊        | 71666/400000 [00:08<00:37, 8657.22it/s] 18%|█▊        | 72536/400000 [00:08<00:37, 8669.64it/s] 18%|█▊        | 73406/400000 [00:08<00:37, 8678.54it/s] 19%|█▊        | 74275/400000 [00:08<00:37, 8576.23it/s] 19%|█▉        | 75150/400000 [00:08<00:37, 8624.80it/s] 19%|█▉        | 76013/400000 [00:08<00:37, 8600.89it/s] 19%|█▉        | 76894/400000 [00:08<00:37, 8659.80it/s] 19%|█▉        | 77765/400000 [00:08<00:37, 8673.24it/s] 20%|█▉        | 78633/400000 [00:09<00:37, 8665.83it/s] 20%|█▉        | 79500/400000 [00:09<00:37, 8654.83it/s] 20%|██        | 80366/400000 [00:09<00:37, 8600.70it/s] 20%|██        | 81227/400000 [00:09<00:37, 8548.16it/s] 21%|██        | 82096/400000 [00:09<00:37, 8583.49it/s] 21%|██        | 82955/400000 [00:09<00:38, 8192.89it/s] 21%|██        | 83834/400000 [00:09<00:37, 8362.37it/s] 21%|██        | 84714/400000 [00:09<00:37, 8487.97it/s] 21%|██▏       | 85566/400000 [00:09<00:37, 8467.37it/s] 22%|██▏       | 86439/400000 [00:10<00:36, 8542.87it/s] 22%|██▏       | 87310/400000 [00:10<00:36, 8590.61it/s] 22%|██▏       | 88171/400000 [00:10<00:36, 8595.62it/s] 22%|██▏       | 89039/400000 [00:10<00:36, 8619.65it/s] 22%|██▏       | 89916/400000 [00:10<00:35, 8662.02it/s] 23%|██▎       | 90799/400000 [00:10<00:35, 8711.71it/s] 23%|██▎       | 91679/400000 [00:10<00:35, 8736.81it/s] 23%|██▎       | 92556/400000 [00:10<00:35, 8744.01it/s] 23%|██▎       | 93431/400000 [00:10<00:36, 8353.68it/s] 24%|██▎       | 94301/400000 [00:10<00:36, 8454.50it/s] 24%|██▍       | 95150/400000 [00:11<00:37, 8216.89it/s] 24%|██▍       | 96003/400000 [00:11<00:36, 8307.20it/s] 24%|██▍       | 96866/400000 [00:11<00:36, 8400.17it/s] 24%|██▍       | 97745/400000 [00:11<00:35, 8512.35it/s] 25%|██▍       | 98602/400000 [00:11<00:35, 8527.26it/s] 25%|██▍       | 99457/400000 [00:11<00:35, 8523.93it/s] 25%|██▌       | 100337/400000 [00:11<00:34, 8602.68it/s] 25%|██▌       | 101213/400000 [00:11<00:34, 8646.74it/s] 26%|██▌       | 102079/400000 [00:11<00:34, 8582.15it/s] 26%|██▌       | 102938/400000 [00:11<00:35, 8464.84it/s] 26%|██▌       | 103814/400000 [00:12<00:34, 8549.48it/s] 26%|██▌       | 104684/400000 [00:12<00:34, 8592.65it/s] 26%|██▋       | 105544/400000 [00:12<00:34, 8591.42it/s] 27%|██▋       | 106404/400000 [00:12<00:34, 8581.69it/s] 27%|██▋       | 107263/400000 [00:12<00:34, 8535.21it/s] 27%|██▋       | 108136/400000 [00:12<00:33, 8591.51it/s] 27%|██▋       | 109007/400000 [00:12<00:33, 8625.44it/s] 27%|██▋       | 109870/400000 [00:12<00:33, 8613.69it/s] 28%|██▊       | 110732/400000 [00:12<00:33, 8595.85it/s] 28%|██▊       | 111605/400000 [00:12<00:33, 8632.76it/s] 28%|██▊       | 112481/400000 [00:13<00:33, 8670.47it/s] 28%|██▊       | 113354/400000 [00:13<00:32, 8686.41it/s] 29%|██▊       | 114224/400000 [00:13<00:32, 8690.17it/s] 29%|██▉       | 115094/400000 [00:13<00:33, 8585.76it/s] 29%|██▉       | 115953/400000 [00:13<00:33, 8573.77it/s] 29%|██▉       | 116811/400000 [00:13<00:33, 8470.34it/s] 29%|██▉       | 117692/400000 [00:13<00:32, 8565.15it/s] 30%|██▉       | 118568/400000 [00:13<00:32, 8620.17it/s] 30%|██▉       | 119431/400000 [00:13<00:32, 8606.93it/s] 30%|███       | 120312/400000 [00:13<00:32, 8665.56it/s] 30%|███       | 121180/400000 [00:14<00:32, 8667.58it/s] 31%|███       | 122047/400000 [00:14<00:32, 8660.71it/s] 31%|███       | 122926/400000 [00:14<00:31, 8697.67it/s] 31%|███       | 123805/400000 [00:14<00:31, 8723.70it/s] 31%|███       | 124679/400000 [00:14<00:31, 8726.33it/s] 31%|███▏      | 125559/400000 [00:14<00:31, 8745.68it/s] 32%|███▏      | 126434/400000 [00:14<00:31, 8659.60it/s] 32%|███▏      | 127307/400000 [00:14<00:31, 8680.39it/s] 32%|███▏      | 128176/400000 [00:14<00:31, 8603.96it/s] 32%|███▏      | 129037/400000 [00:14<00:31, 8501.80it/s] 32%|███▏      | 129909/400000 [00:15<00:31, 8564.90it/s] 33%|███▎      | 130789/400000 [00:15<00:31, 8632.26it/s] 33%|███▎      | 131657/400000 [00:15<00:31, 8646.24it/s] 33%|███▎      | 132532/400000 [00:15<00:30, 8676.31it/s] 33%|███▎      | 133403/400000 [00:15<00:30, 8683.41it/s] 34%|███▎      | 134282/400000 [00:15<00:30, 8714.46it/s] 34%|███▍      | 135155/400000 [00:15<00:30, 8716.56it/s] 34%|███▍      | 136027/400000 [00:15<00:30, 8690.01it/s] 34%|███▍      | 136897/400000 [00:15<00:30, 8639.05it/s] 34%|███▍      | 137774/400000 [00:15<00:30, 8677.51it/s] 35%|███▍      | 138644/400000 [00:16<00:30, 8681.64it/s] 35%|███▍      | 139513/400000 [00:16<00:30, 8408.08it/s] 35%|███▌      | 140378/400000 [00:16<00:30, 8478.39it/s] 35%|███▌      | 141254/400000 [00:16<00:30, 8558.54it/s] 36%|███▌      | 142112/400000 [00:16<00:30, 8509.99it/s] 36%|███▌      | 142964/400000 [00:16<00:30, 8471.40it/s] 36%|███▌      | 143836/400000 [00:16<00:29, 8543.99it/s] 36%|███▌      | 144692/400000 [00:16<00:30, 8451.43it/s] 36%|███▋      | 145539/400000 [00:16<00:30, 8454.51it/s] 37%|███▋      | 146411/400000 [00:16<00:29, 8530.82it/s] 37%|███▋      | 147277/400000 [00:17<00:29, 8566.52it/s] 37%|███▋      | 148154/400000 [00:17<00:29, 8624.77it/s] 37%|███▋      | 149023/400000 [00:17<00:29, 8641.54it/s] 37%|███▋      | 149905/400000 [00:17<00:28, 8692.60it/s] 38%|███▊      | 150775/400000 [00:17<00:28, 8683.17it/s] 38%|███▊      | 151644/400000 [00:17<00:28, 8676.07it/s] 38%|███▊      | 152512/400000 [00:17<00:28, 8592.92it/s] 38%|███▊      | 153372/400000 [00:17<00:29, 8376.19it/s] 39%|███▊      | 154212/400000 [00:17<00:29, 8317.89it/s] 39%|███▉      | 155094/400000 [00:17<00:28, 8461.97it/s] 39%|███▉      | 155975/400000 [00:18<00:28, 8562.04it/s] 39%|███▉      | 156855/400000 [00:18<00:28, 8630.25it/s] 39%|███▉      | 157734/400000 [00:18<00:27, 8676.69it/s] 40%|███▉      | 158605/400000 [00:18<00:27, 8685.15it/s] 40%|███▉      | 159486/400000 [00:18<00:27, 8720.37it/s] 40%|████      | 160364/400000 [00:18<00:27, 8736.63it/s] 40%|████      | 161240/400000 [00:18<00:27, 8742.74it/s] 41%|████      | 162115/400000 [00:18<00:27, 8721.59it/s] 41%|████      | 162997/400000 [00:18<00:27, 8748.83it/s] 41%|████      | 163884/400000 [00:19<00:26, 8783.57it/s] 41%|████      | 164767/400000 [00:19<00:26, 8797.24it/s] 41%|████▏     | 165647/400000 [00:19<00:26, 8792.62it/s] 42%|████▏     | 166531/400000 [00:19<00:26, 8804.68it/s] 42%|████▏     | 167412/400000 [00:19<00:27, 8555.70it/s] 42%|████▏     | 168270/400000 [00:19<00:27, 8549.41it/s] 42%|████▏     | 169145/400000 [00:19<00:26, 8606.08it/s] 43%|████▎     | 170018/400000 [00:19<00:26, 8641.63it/s] 43%|████▎     | 170902/400000 [00:19<00:26, 8699.60it/s] 43%|████▎     | 171773/400000 [00:19<00:26, 8640.04it/s] 43%|████▎     | 172638/400000 [00:20<00:26, 8423.70it/s] 43%|████▎     | 173508/400000 [00:20<00:26, 8502.24it/s] 44%|████▎     | 174383/400000 [00:20<00:26, 8574.20it/s] 44%|████▍     | 175257/400000 [00:20<00:26, 8620.95it/s] 44%|████▍     | 176123/400000 [00:20<00:25, 8631.89it/s] 44%|████▍     | 176996/400000 [00:20<00:25, 8659.94it/s] 44%|████▍     | 177866/400000 [00:20<00:25, 8671.29it/s] 45%|████▍     | 178747/400000 [00:20<00:25, 8711.06it/s] 45%|████▍     | 179619/400000 [00:20<00:25, 8585.64it/s] 45%|████▌     | 180479/400000 [00:20<00:25, 8543.09it/s] 45%|████▌     | 181334/400000 [00:21<00:25, 8517.34it/s] 46%|████▌     | 182187/400000 [00:21<00:26, 8306.86it/s] 46%|████▌     | 183067/400000 [00:21<00:25, 8446.30it/s] 46%|████▌     | 183914/400000 [00:21<00:25, 8358.19it/s] 46%|████▌     | 184785/400000 [00:21<00:25, 8458.95it/s] 46%|████▋     | 185661/400000 [00:21<00:25, 8546.88it/s] 47%|████▋     | 186539/400000 [00:21<00:24, 8614.35it/s] 47%|████▋     | 187411/400000 [00:21<00:24, 8643.91it/s] 47%|████▋     | 188277/400000 [00:21<00:24, 8609.51it/s] 47%|████▋     | 189139/400000 [00:21<00:25, 8329.80it/s] 48%|████▊     | 190012/400000 [00:22<00:24, 8445.58it/s] 48%|████▊     | 190859/400000 [00:22<00:24, 8385.96it/s] 48%|████▊     | 191739/400000 [00:22<00:24, 8504.14it/s] 48%|████▊     | 192621/400000 [00:22<00:24, 8594.18it/s] 48%|████▊     | 193493/400000 [00:22<00:23, 8630.21it/s] 49%|████▊     | 194357/400000 [00:22<00:23, 8592.92it/s] 49%|████▉     | 195217/400000 [00:22<00:24, 8498.34it/s] 49%|████▉     | 196068/400000 [00:22<00:24, 8432.87it/s] 49%|████▉     | 196944/400000 [00:22<00:23, 8526.97it/s] 49%|████▉     | 197806/400000 [00:22<00:23, 8554.22it/s] 50%|████▉     | 198686/400000 [00:23<00:23, 8624.76it/s] 50%|████▉     | 199573/400000 [00:23<00:23, 8694.43it/s] 50%|█████     | 200444/400000 [00:23<00:22, 8697.22it/s] 50%|█████     | 201318/400000 [00:23<00:22, 8708.30it/s] 51%|█████     | 202190/400000 [00:23<00:22, 8703.64it/s] 51%|█████     | 203061/400000 [00:23<00:22, 8649.95it/s] 51%|█████     | 203946/400000 [00:23<00:22, 8708.60it/s] 51%|█████     | 204826/400000 [00:23<00:22, 8734.25it/s] 51%|█████▏    | 205700/400000 [00:23<00:22, 8708.27it/s] 52%|█████▏    | 206571/400000 [00:23<00:22, 8672.33it/s] 52%|█████▏    | 207455/400000 [00:24<00:22, 8720.69it/s] 52%|█████▏    | 208342/400000 [00:24<00:21, 8762.20it/s] 52%|█████▏    | 209221/400000 [00:24<00:21, 8746.25it/s] 53%|█████▎    | 210096/400000 [00:24<00:21, 8672.64it/s] 53%|█████▎    | 210964/400000 [00:24<00:21, 8655.00it/s] 53%|█████▎    | 211846/400000 [00:24<00:21, 8703.09it/s] 53%|█████▎    | 212729/400000 [00:24<00:21, 8739.55it/s] 53%|█████▎    | 213613/400000 [00:24<00:21, 8766.98it/s] 54%|█████▎    | 214493/400000 [00:24<00:21, 8774.90it/s] 54%|█████▍    | 215371/400000 [00:24<00:21, 8775.21it/s] 54%|█████▍    | 216249/400000 [00:25<00:21, 8693.15it/s] 54%|█████▍    | 217133/400000 [00:25<00:20, 8733.84it/s] 55%|█████▍    | 218019/400000 [00:25<00:20, 8768.99it/s] 55%|█████▍    | 218897/400000 [00:25<00:20, 8748.43it/s] 55%|█████▍    | 219775/400000 [00:25<00:20, 8756.90it/s] 55%|█████▌    | 220657/400000 [00:25<00:20, 8774.95it/s] 55%|█████▌    | 221542/400000 [00:25<00:20, 8794.88it/s] 56%|█████▌    | 222427/400000 [00:25<00:20, 8809.07it/s] 56%|█████▌    | 223315/400000 [00:25<00:20, 8829.96it/s] 56%|█████▌    | 224199/400000 [00:25<00:19, 8816.10it/s] 56%|█████▋    | 225081/400000 [00:26<00:20, 8711.50it/s] 56%|█████▋    | 225965/400000 [00:26<00:19, 8747.27it/s] 57%|█████▋    | 226850/400000 [00:26<00:19, 8776.27it/s] 57%|█████▋    | 227728/400000 [00:26<00:19, 8738.49it/s] 57%|█████▋    | 228603/400000 [00:26<00:19, 8702.88it/s] 57%|█████▋    | 229478/400000 [00:26<00:19, 8716.74it/s] 58%|█████▊    | 230361/400000 [00:26<00:19, 8747.64it/s] 58%|█████▊    | 231236/400000 [00:26<00:19, 8728.04it/s] 58%|█████▊    | 232110/400000 [00:26<00:19, 8730.46it/s] 58%|█████▊    | 232990/400000 [00:26<00:19, 8750.20it/s] 58%|█████▊    | 233875/400000 [00:27<00:18, 8779.31it/s] 59%|█████▊    | 234761/400000 [00:27<00:18, 8801.82it/s] 59%|█████▉    | 235651/400000 [00:27<00:18, 8828.56it/s] 59%|█████▉    | 236537/400000 [00:27<00:18, 8835.10it/s] 59%|█████▉    | 237421/400000 [00:27<00:18, 8698.16it/s] 60%|█████▉    | 238304/400000 [00:27<00:18, 8736.22it/s] 60%|█████▉    | 239181/400000 [00:27<00:18, 8745.99it/s] 60%|██████    | 240066/400000 [00:27<00:18, 8774.79it/s] 60%|██████    | 240947/400000 [00:27<00:18, 8783.25it/s] 60%|██████    | 241826/400000 [00:27<00:18, 8761.94it/s] 61%|██████    | 242703/400000 [00:28<00:18, 8709.30it/s] 61%|██████    | 243575/400000 [00:28<00:17, 8697.16it/s] 61%|██████    | 244445/400000 [00:28<00:18, 8369.55it/s] 61%|██████▏   | 245305/400000 [00:28<00:18, 8435.37it/s] 62%|██████▏   | 246167/400000 [00:28<00:18, 8487.01it/s] 62%|██████▏   | 247018/400000 [00:28<00:18, 8463.90it/s] 62%|██████▏   | 247905/400000 [00:28<00:17, 8580.67it/s] 62%|██████▏   | 248765/400000 [00:28<00:17, 8535.40it/s] 62%|██████▏   | 249620/400000 [00:28<00:17, 8401.03it/s] 63%|██████▎   | 250494/400000 [00:29<00:17, 8498.12it/s] 63%|██████▎   | 251358/400000 [00:29<00:17, 8538.98it/s] 63%|██████▎   | 252238/400000 [00:29<00:17, 8614.69it/s] 63%|██████▎   | 253101/400000 [00:29<00:17, 8471.12it/s] 63%|██████▎   | 253978/400000 [00:29<00:17, 8557.07it/s] 64%|██████▎   | 254848/400000 [00:29<00:16, 8598.34it/s] 64%|██████▍   | 255709/400000 [00:29<00:16, 8490.53it/s] 64%|██████▍   | 256569/400000 [00:29<00:16, 8521.84it/s] 64%|██████▍   | 257446/400000 [00:29<00:16, 8591.71it/s] 65%|██████▍   | 258328/400000 [00:29<00:16, 8656.02it/s] 65%|██████▍   | 259195/400000 [00:30<00:16, 8650.31it/s] 65%|██████▌   | 260061/400000 [00:30<00:16, 8566.88it/s] 65%|██████▌   | 260948/400000 [00:30<00:16, 8653.39it/s] 65%|██████▌   | 261833/400000 [00:30<00:15, 8711.02it/s] 66%|██████▌   | 262718/400000 [00:30<00:15, 8751.23it/s] 66%|██████▌   | 263600/400000 [00:30<00:15, 8771.18it/s] 66%|██████▌   | 264483/400000 [00:30<00:15, 8786.46it/s] 66%|██████▋   | 265362/400000 [00:30<00:15, 8673.93it/s] 67%|██████▋   | 266247/400000 [00:30<00:15, 8725.58it/s] 67%|██████▋   | 267120/400000 [00:30<00:15, 8626.93it/s] 67%|██████▋   | 267984/400000 [00:31<00:15, 8532.70it/s] 67%|██████▋   | 268839/400000 [00:31<00:15, 8535.26it/s] 67%|██████▋   | 269699/400000 [00:31<00:15, 8551.77it/s] 68%|██████▊   | 270576/400000 [00:31<00:15, 8613.78it/s] 68%|██████▊   | 271446/400000 [00:31<00:14, 8639.14it/s] 68%|██████▊   | 272311/400000 [00:31<00:14, 8594.72it/s] 68%|██████▊   | 273178/400000 [00:31<00:14, 8614.96it/s] 69%|██████▊   | 274049/400000 [00:31<00:14, 8641.10it/s] 69%|██████▊   | 274914/400000 [00:31<00:14, 8620.64it/s] 69%|██████▉   | 275791/400000 [00:31<00:14, 8662.09it/s] 69%|██████▉   | 276658/400000 [00:32<00:14, 8587.07it/s] 69%|██████▉   | 277517/400000 [00:32<00:14, 8583.19it/s] 70%|██████▉   | 278380/400000 [00:32<00:14, 8596.89it/s] 70%|██████▉   | 279267/400000 [00:32<00:13, 8674.69it/s] 70%|███████   | 280156/400000 [00:32<00:13, 8735.88it/s] 70%|███████   | 281039/400000 [00:32<00:13, 8762.34it/s] 70%|███████   | 281926/400000 [00:32<00:13, 8793.00it/s] 71%|███████   | 282815/400000 [00:32<00:13, 8821.66it/s] 71%|███████   | 283698/400000 [00:32<00:13, 8752.09it/s] 71%|███████   | 284574/400000 [00:32<00:13, 8632.21it/s] 71%|███████▏  | 285440/400000 [00:33<00:13, 8638.29it/s] 72%|███████▏  | 286305/400000 [00:33<00:13, 8630.71it/s] 72%|███████▏  | 287189/400000 [00:33<00:12, 8689.76it/s] 72%|███████▏  | 288059/400000 [00:33<00:12, 8657.81it/s] 72%|███████▏  | 288946/400000 [00:33<00:12, 8719.67it/s] 72%|███████▏  | 289825/400000 [00:33<00:12, 8739.66it/s] 73%|███████▎  | 290709/400000 [00:33<00:12, 8768.68it/s] 73%|███████▎  | 291596/400000 [00:33<00:12, 8796.76it/s] 73%|███████▎  | 292482/400000 [00:33<00:12, 8814.62it/s] 73%|███████▎  | 293364/400000 [00:33<00:12, 8815.39it/s] 74%|███████▎  | 294246/400000 [00:34<00:12, 8810.89it/s] 74%|███████▍  | 295132/400000 [00:34<00:11, 8825.03it/s] 74%|███████▍  | 296015/400000 [00:34<00:11, 8804.80it/s] 74%|███████▍  | 296900/400000 [00:34<00:11, 8815.44it/s] 74%|███████▍  | 297782/400000 [00:34<00:11, 8751.22it/s] 75%|███████▍  | 298658/400000 [00:34<00:11, 8591.16it/s] 75%|███████▍  | 299518/400000 [00:34<00:11, 8535.51it/s] 75%|███████▌  | 300406/400000 [00:34<00:11, 8633.76it/s] 75%|███████▌  | 301293/400000 [00:34<00:11, 8702.07it/s] 76%|███████▌  | 302170/400000 [00:34<00:11, 8720.99it/s] 76%|███████▌  | 303043/400000 [00:35<00:11, 8707.06it/s] 76%|███████▌  | 303921/400000 [00:35<00:11, 8727.67it/s] 76%|███████▌  | 304807/400000 [00:35<00:10, 8766.68it/s] 76%|███████▋  | 305695/400000 [00:35<00:10, 8798.61it/s] 77%|███████▋  | 306581/400000 [00:35<00:10, 8814.09it/s] 77%|███████▋  | 307465/400000 [00:35<00:10, 8820.10it/s] 77%|███████▋  | 308348/400000 [00:35<00:10, 8808.80it/s] 77%|███████▋  | 309231/400000 [00:35<00:10, 8814.32it/s] 78%|███████▊  | 310113/400000 [00:35<00:10, 8782.28it/s] 78%|███████▊  | 310995/400000 [00:35<00:10, 8792.74it/s] 78%|███████▊  | 311875/400000 [00:36<00:10, 8639.13it/s] 78%|███████▊  | 312740/400000 [00:36<00:10, 8580.36it/s] 78%|███████▊  | 313599/400000 [00:36<00:10, 8564.87it/s] 79%|███████▊  | 314469/400000 [00:36<00:09, 8604.70it/s] 79%|███████▉  | 315352/400000 [00:36<00:09, 8669.19it/s] 79%|███████▉  | 316239/400000 [00:36<00:09, 8727.58it/s] 79%|███████▉  | 317115/400000 [00:36<00:09, 8735.99it/s] 79%|███████▉  | 317998/400000 [00:36<00:09, 8761.27it/s] 80%|███████▉  | 318875/400000 [00:36<00:09, 8578.14it/s] 80%|███████▉  | 319756/400000 [00:36<00:09, 8644.58it/s] 80%|████████  | 320638/400000 [00:37<00:09, 8693.90it/s] 80%|████████  | 321509/400000 [00:37<00:09, 8657.43it/s] 81%|████████  | 322389/400000 [00:37<00:08, 8699.44it/s] 81%|████████  | 323264/400000 [00:37<00:08, 8712.23it/s] 81%|████████  | 324146/400000 [00:37<00:08, 8742.23it/s] 81%|████████▏ | 325021/400000 [00:37<00:08, 8742.44it/s] 81%|████████▏ | 325896/400000 [00:37<00:08, 8658.67it/s] 82%|████████▏ | 326763/400000 [00:37<00:08, 8642.32it/s] 82%|████████▏ | 327645/400000 [00:37<00:08, 8694.46it/s] 82%|████████▏ | 328529/400000 [00:38<00:08, 8736.70it/s] 82%|████████▏ | 329409/400000 [00:38<00:08, 8752.82it/s] 83%|████████▎ | 330285/400000 [00:38<00:07, 8718.99it/s] 83%|████████▎ | 331168/400000 [00:38<00:07, 8751.31it/s] 83%|████████▎ | 332044/400000 [00:38<00:07, 8724.28it/s] 83%|████████▎ | 332926/400000 [00:38<00:07, 8750.24it/s] 83%|████████▎ | 333805/400000 [00:38<00:07, 8762.02it/s] 84%|████████▎ | 334682/400000 [00:38<00:07, 8699.56it/s] 84%|████████▍ | 335563/400000 [00:38<00:07, 8730.06it/s] 84%|████████▍ | 336437/400000 [00:38<00:07, 8720.52it/s] 84%|████████▍ | 337317/400000 [00:39<00:07, 8743.73it/s] 85%|████████▍ | 338198/400000 [00:39<00:07, 8762.33it/s] 85%|████████▍ | 339075/400000 [00:39<00:07, 8607.75it/s] 85%|████████▍ | 339951/400000 [00:39<00:06, 8651.64it/s] 85%|████████▌ | 340817/400000 [00:39<00:06, 8601.65it/s] 85%|████████▌ | 341678/400000 [00:39<00:06, 8373.86it/s] 86%|████████▌ | 342518/400000 [00:39<00:06, 8344.71it/s] 86%|████████▌ | 343381/400000 [00:39<00:06, 8427.49it/s] 86%|████████▌ | 344264/400000 [00:39<00:06, 8543.80it/s] 86%|████████▋ | 345144/400000 [00:39<00:06, 8618.44it/s] 87%|████████▋ | 346016/400000 [00:40<00:06, 8648.57it/s] 87%|████████▋ | 346900/400000 [00:40<00:06, 8704.00it/s] 87%|████████▋ | 347771/400000 [00:40<00:06, 8667.74it/s] 87%|████████▋ | 348647/400000 [00:40<00:05, 8694.76it/s] 87%|████████▋ | 349532/400000 [00:40<00:05, 8740.24it/s] 88%|████████▊ | 350407/400000 [00:40<00:05, 8721.47it/s] 88%|████████▊ | 351293/400000 [00:40<00:05, 8761.30it/s] 88%|████████▊ | 352178/400000 [00:40<00:05, 8786.95it/s] 88%|████████▊ | 353059/400000 [00:40<00:05, 8793.79it/s] 88%|████████▊ | 353945/400000 [00:40<00:05, 8813.13it/s] 89%|████████▊ | 354827/400000 [00:41<00:05, 8726.51it/s] 89%|████████▉ | 355705/400000 [00:41<00:05, 8741.93it/s] 89%|████████▉ | 356580/400000 [00:41<00:04, 8721.50it/s] 89%|████████▉ | 357453/400000 [00:41<00:04, 8707.34it/s] 90%|████████▉ | 358327/400000 [00:41<00:04, 8715.50it/s] 90%|████████▉ | 359200/400000 [00:41<00:04, 8717.99it/s] 90%|█████████ | 360081/400000 [00:41<00:04, 8744.74it/s] 90%|█████████ | 360967/400000 [00:41<00:04, 8778.99it/s] 90%|█████████ | 361856/400000 [00:41<00:04, 8809.03it/s] 91%|█████████ | 362737/400000 [00:41<00:04, 8778.22it/s] 91%|█████████ | 363615/400000 [00:42<00:04, 8685.02it/s] 91%|█████████ | 364501/400000 [00:42<00:04, 8735.80it/s] 91%|█████████▏| 365388/400000 [00:42<00:03, 8773.60it/s] 92%|█████████▏| 366266/400000 [00:42<00:03, 8771.59it/s] 92%|█████████▏| 367153/400000 [00:42<00:03, 8799.28it/s] 92%|█████████▏| 368042/400000 [00:42<00:03, 8824.23it/s] 92%|█████████▏| 368926/400000 [00:42<00:03, 8828.50it/s] 92%|█████████▏| 369814/400000 [00:42<00:03, 8842.01it/s] 93%|█████████▎| 370699/400000 [00:42<00:03, 8705.53it/s] 93%|█████████▎| 371587/400000 [00:42<00:03, 8756.05it/s] 93%|█████████▎| 372472/400000 [00:43<00:03, 8780.93it/s] 93%|█████████▎| 373355/400000 [00:43<00:03, 8794.42it/s] 94%|█████████▎| 374235/400000 [00:43<00:02, 8721.79it/s] 94%|█████████▍| 375108/400000 [00:43<00:02, 8618.95it/s] 94%|█████████▍| 375971/400000 [00:43<00:02, 8610.16it/s] 94%|█████████▍| 376833/400000 [00:43<00:02, 8575.09it/s] 94%|█████████▍| 377708/400000 [00:43<00:02, 8625.85it/s] 95%|█████████▍| 378587/400000 [00:43<00:02, 8672.84it/s] 95%|█████████▍| 379456/400000 [00:43<00:02, 8676.16it/s] 95%|█████████▌| 380344/400000 [00:43<00:02, 8735.08it/s] 95%|█████████▌| 381225/400000 [00:44<00:02, 8755.93it/s] 96%|█████████▌| 382101/400000 [00:44<00:02, 8745.52it/s] 96%|█████████▌| 382976/400000 [00:44<00:01, 8576.20it/s] 96%|█████████▌| 383844/400000 [00:44<00:01, 8605.06it/s] 96%|█████████▌| 384706/400000 [00:44<00:01, 8562.99it/s] 96%|█████████▋| 385591/400000 [00:44<00:01, 8645.02it/s] 97%|█████████▋| 386457/400000 [00:44<00:01, 8630.45it/s] 97%|█████████▋| 387321/400000 [00:44<00:01, 8610.41it/s] 97%|█████████▋| 388183/400000 [00:44<00:01, 8556.04it/s] 97%|█████████▋| 389050/400000 [00:44<00:01, 8588.78it/s] 97%|█████████▋| 389910/400000 [00:45<00:01, 8464.95it/s] 98%|█████████▊| 390788/400000 [00:45<00:01, 8555.82it/s] 98%|█████████▊| 391670/400000 [00:45<00:00, 8633.05it/s] 98%|█████████▊| 392554/400000 [00:45<00:00, 8693.42it/s] 98%|█████████▊| 393433/400000 [00:45<00:00, 8720.73it/s] 99%|█████████▊| 394319/400000 [00:45<00:00, 8760.33it/s] 99%|█████████▉| 395208/400000 [00:45<00:00, 8797.05it/s] 99%|█████████▉| 396088/400000 [00:45<00:00, 8767.88it/s] 99%|█████████▉| 396973/400000 [00:45<00:00, 8791.09it/s] 99%|█████████▉| 397862/400000 [00:45<00:00, 8819.39it/s]100%|█████████▉| 398745/400000 [00:46<00:00, 8763.00it/s]100%|█████████▉| 399622/400000 [00:46<00:00, 8710.76it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8653.47it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7feb507d5d30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011460625647816365 	 Accuracy: 48
Train Epoch: 1 	 Loss: 0.011033610954731205 	 Accuracy: 54

  model saves at 54% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15906 out of table with 15828 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15906 out of table with 15828 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-13 17:23:22.239653: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 17:23:22.243930: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-13 17:23:22.244080: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b0f2943d90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 17:23:22.244097: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7feb5a859048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.0533 - accuracy: 0.5400
 2000/25000 [=>............................] - ETA: 8s - loss: 7.3446 - accuracy: 0.5210 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.4826 - accuracy: 0.5120
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5133 - accuracy: 0.5100
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5378 - accuracy: 0.5084
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5721 - accuracy: 0.5062
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5943 - accuracy: 0.5047
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6436 - accuracy: 0.5015
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6411 - accuracy: 0.5017
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6758 - accuracy: 0.4994
11000/25000 [============>.................] - ETA: 3s - loss: 7.6555 - accuracy: 0.5007
12000/25000 [=============>................] - ETA: 3s - loss: 7.6653 - accuracy: 0.5001
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6973 - accuracy: 0.4980
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6611 - accuracy: 0.5004
15000/25000 [=================>............] - ETA: 2s - loss: 7.6554 - accuracy: 0.5007
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6532 - accuracy: 0.5009
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6414 - accuracy: 0.5016
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6615 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6707 - accuracy: 0.4997
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6896 - accuracy: 0.4985
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6980 - accuracy: 0.4980
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6760 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6583 - accuracy: 0.5005
25000/25000 [==============================] - 7s 283us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7feab4dfa8d0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7feb57f33160> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.5545 - crf_viterbi_accuracy: 0.1600 - val_loss: 1.4662 - val_crf_viterbi_accuracy: 0.1867

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

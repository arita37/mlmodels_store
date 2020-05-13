
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f557d94bfd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 18:13:18.795503
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 18:13:18.799112
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 18:13:18.802253
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 18:13:18.805474
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f5589715470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355751.2500
Epoch 2/10

1/1 [==============================] - 0s 96ms/step - loss: 263888.2812
Epoch 3/10

1/1 [==============================] - 0s 90ms/step - loss: 165969.0625
Epoch 4/10

1/1 [==============================] - 0s 95ms/step - loss: 89567.5703
Epoch 5/10

1/1 [==============================] - 0s 90ms/step - loss: 48117.3398
Epoch 6/10

1/1 [==============================] - 0s 98ms/step - loss: 27746.9473
Epoch 7/10

1/1 [==============================] - 0s 93ms/step - loss: 17394.0488
Epoch 8/10

1/1 [==============================] - 0s 95ms/step - loss: 11934.5713
Epoch 9/10

1/1 [==============================] - 0s 94ms/step - loss: 8482.7520
Epoch 10/10

1/1 [==============================] - 0s 92ms/step - loss: 6636.6538

  #### Inference Need return ypred, ytrue ######################### 
[[ 3.18369985e-01  9.24206638e+00  8.95705891e+00  7.78701591e+00
   7.80604839e+00  9.05688763e+00  9.16261673e+00  7.43158436e+00
   8.84266949e+00  8.95285892e+00  7.75738335e+00  8.58993053e+00
   8.03543472e+00  8.67102337e+00  7.86522102e+00  6.59288502e+00
   7.86731195e+00  7.85459185e+00  8.70764065e+00  7.08348322e+00
   9.43638706e+00  7.97504663e+00  7.69022894e+00  8.50948524e+00
   9.90952110e+00  7.10486269e+00  7.91714096e+00  8.22513294e+00
   8.82848358e+00  1.00405741e+01  7.67086267e+00  8.02578926e+00
   8.51933670e+00  9.44419765e+00  8.14887238e+00  7.51892900e+00
   8.02785206e+00  9.68299866e+00  9.62973976e+00  7.59938955e+00
   8.70625496e+00  8.42687416e+00  7.83790398e+00  8.24940968e+00
   7.97859049e+00  7.90718889e+00  9.10276413e+00  8.80816269e+00
   8.06800842e+00  1.00779715e+01  7.66919947e+00  8.75255394e+00
   7.62612247e+00  7.53687334e+00  7.73479652e+00  8.33609772e+00
   7.65910721e+00  7.04695463e+00  8.77014637e+00  8.14782047e+00
   1.74489522e+00 -1.07594180e+00 -2.95488626e-01 -7.07621872e-01
   8.06919575e-01  1.38751996e+00 -1.55360788e-01 -2.09632814e-01
   1.21915650e+00 -3.49770010e-01  2.06856817e-01  5.19700170e-01
   3.36768806e-01  1.14873683e+00 -1.94828391e-01 -1.47196621e-01
   1.56150818e-01  1.53340995e-02 -5.94509125e-01 -5.45249581e-01
   1.17694414e+00 -1.49582553e+00 -2.70103574e-01  5.72110951e-01
   3.84365886e-01 -2.60281444e-01  7.56469488e-01  6.98526442e-01
  -1.10020816e+00 -9.42084491e-02 -4.03409988e-01  2.05838799e-01
   8.73302579e-01 -8.45823288e-01 -7.90375590e-01  6.93696260e-01
   3.04737389e-01 -9.50328827e-01 -2.93594807e-01 -2.10864925e+00
   4.30809647e-01  3.13075900e-01  4.16458368e-01 -8.94602776e-01
  -1.49260759e+00 -1.44169474e+00 -1.27052021e+00  4.78082657e-01
   9.82928395e-01 -1.73142409e+00 -5.21961033e-01 -1.29652619e-02
  -3.80952716e-01  5.14922976e-01  5.11895269e-02 -4.61172193e-01
  -1.28054547e+00 -1.32101607e+00  7.77682900e-01 -3.96763682e-02
   4.21314955e-01 -7.31918812e-01 -2.11348325e-01 -4.26270306e-01
   2.01069444e-01 -3.19458187e-01 -9.90454137e-01  1.21090436e+00
  -1.75333691e+00 -2.11572552e+00  5.68129838e-01 -1.01403677e+00
   4.71804619e-01  1.30881298e+00  1.04279351e+00 -8.54187369e-01
  -1.57441139e+00 -1.52709484e-01  6.07828438e-01  3.56135339e-01
   9.50698018e-01  4.20648813e-01  6.27595723e-01 -6.69610739e-01
  -8.71827722e-01 -6.90989375e-01  5.64631402e-01  3.49369138e-01
  -7.30649471e-01 -5.68447471e-01 -4.91430938e-01  3.54691833e-01
  -2.54806495e+00 -1.35070086e-03  2.67937243e-01 -1.56904244e+00
   7.25674808e-01 -6.69340491e-02  1.02906680e+00  1.61167771e-01
  -5.72938025e-01  8.02934051e-01  2.14930534e-01  4.74458337e-02
   1.61245549e+00  1.28801370e+00  3.14048558e-01  3.55876237e-02
  -1.87144315e+00  5.57707474e-02  1.09516311e+00 -1.83515221e-01
   1.25339285e-01 -1.40374541e+00  1.36401951e-01  1.49198961e+00
   1.48703754e+00  7.49172807e-01 -1.74623859e+00 -1.27269194e-01
   3.24111581e-02  9.41481495e+00  8.35902119e+00  7.75481224e+00
   7.12652206e+00  8.90334606e+00  8.13214874e+00  1.03241520e+01
   8.23834324e+00  7.07306051e+00  8.87407875e+00  8.36041641e+00
   7.36989498e+00  8.25250340e+00  9.82005310e+00  8.27874184e+00
   8.81472874e+00  9.38250732e+00  8.78640270e+00  7.24567509e+00
   6.98990536e+00  7.34378481e+00  8.44700050e+00  8.31117344e+00
   7.76348209e+00  8.51422024e+00  8.67478180e+00  7.36112404e+00
   7.29549932e+00  9.19481182e+00  7.82794189e+00  9.83547211e+00
   9.21473217e+00  8.92661190e+00  8.91983414e+00  8.59955311e+00
   8.52507877e+00  8.44315815e+00  7.94529343e+00  7.96128225e+00
   8.59305191e+00  8.40932846e+00  8.97516918e+00  8.54659367e+00
   8.60830784e+00  7.66155243e+00  7.96768570e+00  8.47225285e+00
   8.03473663e+00  8.64626312e+00  8.49334240e+00  9.56890488e+00
   9.57061863e+00  9.05846405e+00  9.10656071e+00  8.19981861e+00
   8.44268799e+00  8.28170109e+00  8.50419807e+00  8.28075600e+00
   2.80523968e+00  6.48493171e-01  2.14739418e+00  1.45199001e+00
   2.98869491e-01  2.93419504e+00  2.15027630e-01  5.34519434e-01
   8.86362076e-01  5.57545424e-01  9.09279644e-01  1.83097804e+00
   1.35602736e+00  6.85313463e-01  6.78733706e-01  6.41385853e-01
   1.27014017e+00  7.23719418e-01  2.01634312e+00  2.34046698e+00
   1.61327767e+00  1.11847138e+00  1.09222174e+00  1.19807577e+00
   9.73705947e-01  1.79801798e+00  1.96430397e+00  2.13749981e+00
   2.18634462e+00  5.07457614e-01  1.12874579e+00  1.20898223e+00
   1.46991849e+00  1.29445004e+00  7.15119064e-01  1.48446119e+00
   8.02588403e-01  1.54487491e+00  1.44173455e+00  1.74914289e+00
   2.13238859e+00  1.56354296e+00  7.47990966e-01  1.32801008e+00
   8.81134748e-01  1.12234163e+00  9.82554495e-01  1.01566839e+00
   5.12054265e-01  8.61941040e-01  6.35862708e-01  2.64142871e-01
   3.09310675e-01  2.17169619e+00  2.92492867e-01  1.22798920e-01
   4.34768558e-01  3.18567395e-01  8.22955489e-01  2.80420303e-01
   2.05130816e-01  1.55313373e-01  5.10647237e-01  1.89063823e+00
   1.27310395e-01  1.65561771e+00  1.36388075e+00  2.07644272e+00
   1.33153772e+00  5.82062185e-01  1.61831498e+00  4.28209543e-01
   2.74108362e+00  4.64672685e-01  8.86288941e-01  3.37212229e+00
   1.93607676e+00  7.45239079e-01  1.87536645e+00  1.70645595e-01
   8.52924943e-01  1.97905278e+00  3.23182642e-01  1.80989861e-01
   2.12473154e+00  3.57692242e-01  7.25265622e-01  7.82546401e-01
   2.70125914e+00  2.33890772e-01  1.31025052e+00  1.94874942e-01
   1.21473289e+00  1.16821682e+00  8.03900898e-01  1.62040293e-01
   8.90570998e-01  7.16417432e-01  4.16747689e-01  6.57118440e-01
   1.23331714e+00  3.06218672e+00  2.58073032e-01  9.15552378e-01
   9.01001275e-01  8.14486444e-01  1.67450762e+00  2.18830132e+00
   4.24377978e-01  1.18499076e+00  1.26785111e+00  1.05280912e+00
   1.69499111e+00  2.72944212e+00  1.44143939e+00  2.27638340e+00
   2.12414193e+00  1.58766389e-01  2.65442669e-01  1.68136358e+00
   6.22748232e+00 -6.14664125e+00 -9.18872547e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 18:13:27.105741
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.9766
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 18:13:27.109642
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8852.59
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 18:13:27.112725
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.8129
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 18:13:27.117221
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -791.812
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140004796782128
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140003855192696
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140003855193200
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140003855193704
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140003855194208
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140003855194712

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f5585594f28> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.498486
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.469452
grad_step = 000002, loss = 0.445454
grad_step = 000003, loss = 0.420759
grad_step = 000004, loss = 0.393715
grad_step = 000005, loss = 0.365737
grad_step = 000006, loss = 0.344124
grad_step = 000007, loss = 0.325874
grad_step = 000008, loss = 0.308288
grad_step = 000009, loss = 0.298685
grad_step = 000010, loss = 0.283450
grad_step = 000011, loss = 0.268208
grad_step = 000012, loss = 0.257830
grad_step = 000013, loss = 0.249360
grad_step = 000014, loss = 0.240499
grad_step = 000015, loss = 0.231264
grad_step = 000016, loss = 0.221356
grad_step = 000017, loss = 0.210943
grad_step = 000018, loss = 0.200805
grad_step = 000019, loss = 0.191223
grad_step = 000020, loss = 0.182026
grad_step = 000021, loss = 0.173100
grad_step = 000022, loss = 0.164691
grad_step = 000023, loss = 0.156838
grad_step = 000024, loss = 0.148541
grad_step = 000025, loss = 0.139930
grad_step = 000026, loss = 0.131997
grad_step = 000027, loss = 0.124796
grad_step = 000028, loss = 0.117798
grad_step = 000029, loss = 0.110843
grad_step = 000030, loss = 0.104085
grad_step = 000031, loss = 0.097569
grad_step = 000032, loss = 0.091264
grad_step = 000033, loss = 0.085163
grad_step = 000034, loss = 0.079271
grad_step = 000035, loss = 0.073711
grad_step = 000036, loss = 0.068441
grad_step = 000037, loss = 0.063311
grad_step = 000038, loss = 0.058430
grad_step = 000039, loss = 0.053984
grad_step = 000040, loss = 0.049773
grad_step = 000041, loss = 0.045655
grad_step = 000042, loss = 0.041750
grad_step = 000043, loss = 0.038141
grad_step = 000044, loss = 0.034871
grad_step = 000045, loss = 0.031828
grad_step = 000046, loss = 0.029022
grad_step = 000047, loss = 0.026367
grad_step = 000048, loss = 0.023835
grad_step = 000049, loss = 0.021580
grad_step = 000050, loss = 0.019582
grad_step = 000051, loss = 0.017757
grad_step = 000052, loss = 0.016039
grad_step = 000053, loss = 0.014463
grad_step = 000054, loss = 0.013042
grad_step = 000055, loss = 0.011761
grad_step = 000056, loss = 0.010606
grad_step = 000057, loss = 0.009593
grad_step = 000058, loss = 0.008653
grad_step = 000059, loss = 0.007818
grad_step = 000060, loss = 0.007074
grad_step = 000061, loss = 0.006406
grad_step = 000062, loss = 0.005818
grad_step = 000063, loss = 0.005308
grad_step = 000064, loss = 0.004866
grad_step = 000065, loss = 0.004462
grad_step = 000066, loss = 0.004113
grad_step = 000067, loss = 0.003815
grad_step = 000068, loss = 0.003566
grad_step = 000069, loss = 0.003364
grad_step = 000070, loss = 0.003184
grad_step = 000071, loss = 0.003030
grad_step = 000072, loss = 0.002904
grad_step = 000073, loss = 0.002807
grad_step = 000074, loss = 0.002729
grad_step = 000075, loss = 0.002665
grad_step = 000076, loss = 0.002607
grad_step = 000077, loss = 0.002565
grad_step = 000078, loss = 0.002535
grad_step = 000079, loss = 0.002513
grad_step = 000080, loss = 0.002494
grad_step = 000081, loss = 0.002475
grad_step = 000082, loss = 0.002461
grad_step = 000083, loss = 0.002452
grad_step = 000084, loss = 0.002443
grad_step = 000085, loss = 0.002434
grad_step = 000086, loss = 0.002424
grad_step = 000087, loss = 0.002414
grad_step = 000088, loss = 0.002404
grad_step = 000089, loss = 0.002395
grad_step = 000090, loss = 0.002385
grad_step = 000091, loss = 0.002375
grad_step = 000092, loss = 0.002363
grad_step = 000093, loss = 0.002351
grad_step = 000094, loss = 0.002340
grad_step = 000095, loss = 0.002326
grad_step = 000096, loss = 0.002312
grad_step = 000097, loss = 0.002298
grad_step = 000098, loss = 0.002283
grad_step = 000099, loss = 0.002267
grad_step = 000100, loss = 0.002252
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002238
grad_step = 000102, loss = 0.002225
grad_step = 000103, loss = 0.002213
grad_step = 000104, loss = 0.002198
grad_step = 000105, loss = 0.002187
grad_step = 000106, loss = 0.002176
grad_step = 000107, loss = 0.002166
grad_step = 000108, loss = 0.002154
grad_step = 000109, loss = 0.002140
grad_step = 000110, loss = 0.002130
grad_step = 000111, loss = 0.002124
grad_step = 000112, loss = 0.002123
grad_step = 000113, loss = 0.002116
grad_step = 000114, loss = 0.002097
grad_step = 000115, loss = 0.002083
grad_step = 000116, loss = 0.002076
grad_step = 000117, loss = 0.002078
grad_step = 000118, loss = 0.002079
grad_step = 000119, loss = 0.002073
grad_step = 000120, loss = 0.002064
grad_step = 000121, loss = 0.002050
grad_step = 000122, loss = 0.002039
grad_step = 000123, loss = 0.002032
grad_step = 000124, loss = 0.002029
grad_step = 000125, loss = 0.002034
grad_step = 000126, loss = 0.002044
grad_step = 000127, loss = 0.002053
grad_step = 000128, loss = 0.002041
grad_step = 000129, loss = 0.002018
grad_step = 000130, loss = 0.002002
grad_step = 000131, loss = 0.002012
grad_step = 000132, loss = 0.002022
grad_step = 000133, loss = 0.002018
grad_step = 000134, loss = 0.001984
grad_step = 000135, loss = 0.001959
grad_step = 000136, loss = 0.001958
grad_step = 000137, loss = 0.001970
grad_step = 000138, loss = 0.001977
grad_step = 000139, loss = 0.001967
grad_step = 000140, loss = 0.001946
grad_step = 000141, loss = 0.001930
grad_step = 000142, loss = 0.001924
grad_step = 000143, loss = 0.001928
grad_step = 000144, loss = 0.001934
grad_step = 000145, loss = 0.001932
grad_step = 000146, loss = 0.001922
grad_step = 000147, loss = 0.001909
grad_step = 000148, loss = 0.001897
grad_step = 000149, loss = 0.001894
grad_step = 000150, loss = 0.001907
grad_step = 000151, loss = 0.001965
grad_step = 000152, loss = 0.002175
grad_step = 000153, loss = 0.002392
grad_step = 000154, loss = 0.002495
grad_step = 000155, loss = 0.002050
grad_step = 000156, loss = 0.002045
grad_step = 000157, loss = 0.002288
grad_step = 000158, loss = 0.002063
grad_step = 000159, loss = 0.001996
grad_step = 000160, loss = 0.002159
grad_step = 000161, loss = 0.001956
grad_step = 000162, loss = 0.001972
grad_step = 000163, loss = 0.002033
grad_step = 000164, loss = 0.001915
grad_step = 000165, loss = 0.001933
grad_step = 000166, loss = 0.001965
grad_step = 000167, loss = 0.001895
grad_step = 000168, loss = 0.001911
grad_step = 000169, loss = 0.001896
grad_step = 000170, loss = 0.001890
grad_step = 000171, loss = 0.001877
grad_step = 000172, loss = 0.001853
grad_step = 000173, loss = 0.001876
grad_step = 000174, loss = 0.001840
grad_step = 000175, loss = 0.001830
grad_step = 000176, loss = 0.001851
grad_step = 000177, loss = 0.001823
grad_step = 000178, loss = 0.001802
grad_step = 000179, loss = 0.001836
grad_step = 000180, loss = 0.001798
grad_step = 000181, loss = 0.001789
grad_step = 000182, loss = 0.001808
grad_step = 000183, loss = 0.001784
grad_step = 000184, loss = 0.001772
grad_step = 000185, loss = 0.001782
grad_step = 000186, loss = 0.001770
grad_step = 000187, loss = 0.001753
grad_step = 000188, loss = 0.001763
grad_step = 000189, loss = 0.001750
grad_step = 000190, loss = 0.001737
grad_step = 000191, loss = 0.001743
grad_step = 000192, loss = 0.001732
grad_step = 000193, loss = 0.001720
grad_step = 000194, loss = 0.001720
grad_step = 000195, loss = 0.001717
grad_step = 000196, loss = 0.001704
grad_step = 000197, loss = 0.001698
grad_step = 000198, loss = 0.001697
grad_step = 000199, loss = 0.001685
grad_step = 000200, loss = 0.001678
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001676
grad_step = 000202, loss = 0.001669
grad_step = 000203, loss = 0.001661
grad_step = 000204, loss = 0.001656
grad_step = 000205, loss = 0.001654
grad_step = 000206, loss = 0.001656
grad_step = 000207, loss = 0.001663
grad_step = 000208, loss = 0.001679
grad_step = 000209, loss = 0.001726
grad_step = 000210, loss = 0.001772
grad_step = 000211, loss = 0.001820
grad_step = 000212, loss = 0.001759
grad_step = 000213, loss = 0.001663
grad_step = 000214, loss = 0.001595
grad_step = 000215, loss = 0.001615
grad_step = 000216, loss = 0.001672
grad_step = 000217, loss = 0.001679
grad_step = 000218, loss = 0.001643
grad_step = 000219, loss = 0.001585
grad_step = 000220, loss = 0.001563
grad_step = 000221, loss = 0.001577
grad_step = 000222, loss = 0.001610
grad_step = 000223, loss = 0.001650
grad_step = 000224, loss = 0.001633
grad_step = 000225, loss = 0.001604
grad_step = 000226, loss = 0.001546
grad_step = 000227, loss = 0.001511
grad_step = 000228, loss = 0.001509
grad_step = 000229, loss = 0.001529
grad_step = 000230, loss = 0.001556
grad_step = 000231, loss = 0.001566
grad_step = 000232, loss = 0.001580
grad_step = 000233, loss = 0.001569
grad_step = 000234, loss = 0.001563
grad_step = 000235, loss = 0.001527
grad_step = 000236, loss = 0.001490
grad_step = 000237, loss = 0.001465
grad_step = 000238, loss = 0.001464
grad_step = 000239, loss = 0.001485
grad_step = 000240, loss = 0.001517
grad_step = 000241, loss = 0.001561
grad_step = 000242, loss = 0.001563
grad_step = 000243, loss = 0.001556
grad_step = 000244, loss = 0.001502
grad_step = 000245, loss = 0.001456
grad_step = 000246, loss = 0.001423
grad_step = 000247, loss = 0.001420
grad_step = 000248, loss = 0.001439
grad_step = 000249, loss = 0.001461
grad_step = 000250, loss = 0.001483
grad_step = 000251, loss = 0.001477
grad_step = 000252, loss = 0.001468
grad_step = 000253, loss = 0.001437
grad_step = 000254, loss = 0.001411
grad_step = 000255, loss = 0.001391
grad_step = 000256, loss = 0.001381
grad_step = 000257, loss = 0.001380
grad_step = 000258, loss = 0.001384
grad_step = 000259, loss = 0.001396
grad_step = 000260, loss = 0.001413
grad_step = 000261, loss = 0.001449
grad_step = 000262, loss = 0.001484
grad_step = 000263, loss = 0.001557
grad_step = 000264, loss = 0.001550
grad_step = 000265, loss = 0.001544
grad_step = 000266, loss = 0.001437
grad_step = 000267, loss = 0.001363
grad_step = 000268, loss = 0.001354
grad_step = 000269, loss = 0.001396
grad_step = 000270, loss = 0.001436
grad_step = 000271, loss = 0.001409
grad_step = 000272, loss = 0.001365
grad_step = 000273, loss = 0.001336
grad_step = 000274, loss = 0.001346
grad_step = 000275, loss = 0.001373
grad_step = 000276, loss = 0.001384
grad_step = 000277, loss = 0.001381
grad_step = 000278, loss = 0.001354
grad_step = 000279, loss = 0.001335
grad_step = 000280, loss = 0.001330
grad_step = 000281, loss = 0.001346
grad_step = 000282, loss = 0.001382
grad_step = 000283, loss = 0.001429
grad_step = 000284, loss = 0.001489
grad_step = 000285, loss = 0.001487
grad_step = 000286, loss = 0.001443
grad_step = 000287, loss = 0.001355
grad_step = 000288, loss = 0.001336
grad_step = 000289, loss = 0.001368
grad_step = 000290, loss = 0.001361
grad_step = 000291, loss = 0.001324
grad_step = 000292, loss = 0.001310
grad_step = 000293, loss = 0.001340
grad_step = 000294, loss = 0.001363
grad_step = 000295, loss = 0.001339
grad_step = 000296, loss = 0.001302
grad_step = 000297, loss = 0.001292
grad_step = 000298, loss = 0.001306
grad_step = 000299, loss = 0.001317
grad_step = 000300, loss = 0.001304
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001284
grad_step = 000302, loss = 0.001280
grad_step = 000303, loss = 0.001292
grad_step = 000304, loss = 0.001302
grad_step = 000305, loss = 0.001301
grad_step = 000306, loss = 0.001308
grad_step = 000307, loss = 0.001336
grad_step = 000308, loss = 0.001409
grad_step = 000309, loss = 0.001462
grad_step = 000310, loss = 0.001534
grad_step = 000311, loss = 0.001429
grad_step = 000312, loss = 0.001339
grad_step = 000313, loss = 0.001298
grad_step = 000314, loss = 0.001317
grad_step = 000315, loss = 0.001333
grad_step = 000316, loss = 0.001308
grad_step = 000317, loss = 0.001298
grad_step = 000318, loss = 0.001296
grad_step = 000319, loss = 0.001273
grad_step = 000320, loss = 0.001280
grad_step = 000321, loss = 0.001297
grad_step = 000322, loss = 0.001264
grad_step = 000323, loss = 0.001237
grad_step = 000324, loss = 0.001256
grad_step = 000325, loss = 0.001272
grad_step = 000326, loss = 0.001256
grad_step = 000327, loss = 0.001240
grad_step = 000328, loss = 0.001248
grad_step = 000329, loss = 0.001256
grad_step = 000330, loss = 0.001240
grad_step = 000331, loss = 0.001237
grad_step = 000332, loss = 0.001252
grad_step = 000333, loss = 0.001264
grad_step = 000334, loss = 0.001263
grad_step = 000335, loss = 0.001268
grad_step = 000336, loss = 0.001284
grad_step = 000337, loss = 0.001314
grad_step = 000338, loss = 0.001319
grad_step = 000339, loss = 0.001331
grad_step = 000340, loss = 0.001304
grad_step = 000341, loss = 0.001269
grad_step = 000342, loss = 0.001220
grad_step = 000343, loss = 0.001198
grad_step = 000344, loss = 0.001212
grad_step = 000345, loss = 0.001238
grad_step = 000346, loss = 0.001249
grad_step = 000347, loss = 0.001230
grad_step = 000348, loss = 0.001207
grad_step = 000349, loss = 0.001192
grad_step = 000350, loss = 0.001190
grad_step = 000351, loss = 0.001194
grad_step = 000352, loss = 0.001200
grad_step = 000353, loss = 0.001209
grad_step = 000354, loss = 0.001213
grad_step = 000355, loss = 0.001214
grad_step = 000356, loss = 0.001205
grad_step = 000357, loss = 0.001199
grad_step = 000358, loss = 0.001198
grad_step = 000359, loss = 0.001204
grad_step = 000360, loss = 0.001211
grad_step = 000361, loss = 0.001220
grad_step = 000362, loss = 0.001225
grad_step = 000363, loss = 0.001225
grad_step = 000364, loss = 0.001203
grad_step = 000365, loss = 0.001179
grad_step = 000366, loss = 0.001167
grad_step = 000367, loss = 0.001179
grad_step = 000368, loss = 0.001199
grad_step = 000369, loss = 0.001205
grad_step = 000370, loss = 0.001198
grad_step = 000371, loss = 0.001197
grad_step = 000372, loss = 0.001207
grad_step = 000373, loss = 0.001236
grad_step = 000374, loss = 0.001251
grad_step = 000375, loss = 0.001270
grad_step = 000376, loss = 0.001253
grad_step = 000377, loss = 0.001229
grad_step = 000378, loss = 0.001181
grad_step = 000379, loss = 0.001146
grad_step = 000380, loss = 0.001133
grad_step = 000381, loss = 0.001143
grad_step = 000382, loss = 0.001160
grad_step = 000383, loss = 0.001167
grad_step = 000384, loss = 0.001167
grad_step = 000385, loss = 0.001159
grad_step = 000386, loss = 0.001157
grad_step = 000387, loss = 0.001150
grad_step = 000388, loss = 0.001143
grad_step = 000389, loss = 0.001126
grad_step = 000390, loss = 0.001114
grad_step = 000391, loss = 0.001110
grad_step = 000392, loss = 0.001114
grad_step = 000393, loss = 0.001121
grad_step = 000394, loss = 0.001123
grad_step = 000395, loss = 0.001124
grad_step = 000396, loss = 0.001127
grad_step = 000397, loss = 0.001135
grad_step = 000398, loss = 0.001141
grad_step = 000399, loss = 0.001155
grad_step = 000400, loss = 0.001170
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001209
grad_step = 000402, loss = 0.001233
grad_step = 000403, loss = 0.001273
grad_step = 000404, loss = 0.001238
grad_step = 000405, loss = 0.001198
grad_step = 000406, loss = 0.001133
grad_step = 000407, loss = 0.001100
grad_step = 000408, loss = 0.001106
grad_step = 000409, loss = 0.001119
grad_step = 000410, loss = 0.001129
grad_step = 000411, loss = 0.001123
grad_step = 000412, loss = 0.001111
grad_step = 000413, loss = 0.001093
grad_step = 000414, loss = 0.001078
grad_step = 000415, loss = 0.001081
grad_step = 000416, loss = 0.001096
grad_step = 000417, loss = 0.001107
grad_step = 000418, loss = 0.001099
grad_step = 000419, loss = 0.001081
grad_step = 000420, loss = 0.001066
grad_step = 000421, loss = 0.001064
grad_step = 000422, loss = 0.001068
grad_step = 000423, loss = 0.001070
grad_step = 000424, loss = 0.001065
grad_step = 000425, loss = 0.001061
grad_step = 000426, loss = 0.001062
grad_step = 000427, loss = 0.001068
grad_step = 000428, loss = 0.001075
grad_step = 000429, loss = 0.001074
grad_step = 000430, loss = 0.001074
grad_step = 000431, loss = 0.001074
grad_step = 000432, loss = 0.001075
grad_step = 000433, loss = 0.001072
grad_step = 000434, loss = 0.001063
grad_step = 000435, loss = 0.001051
grad_step = 000436, loss = 0.001043
grad_step = 000437, loss = 0.001045
grad_step = 000438, loss = 0.001062
grad_step = 000439, loss = 0.001091
grad_step = 000440, loss = 0.001154
grad_step = 000441, loss = 0.001231
grad_step = 000442, loss = 0.001390
grad_step = 000443, loss = 0.001422
grad_step = 000444, loss = 0.001436
grad_step = 000445, loss = 0.001228
grad_step = 000446, loss = 0.001079
grad_step = 000447, loss = 0.001073
grad_step = 000448, loss = 0.001138
grad_step = 000449, loss = 0.001187
grad_step = 000450, loss = 0.001147
grad_step = 000451, loss = 0.001083
grad_step = 000452, loss = 0.001081
grad_step = 000453, loss = 0.001087
grad_step = 000454, loss = 0.001114
grad_step = 000455, loss = 0.001109
grad_step = 000456, loss = 0.001056
grad_step = 000457, loss = 0.001026
grad_step = 000458, loss = 0.001046
grad_step = 000459, loss = 0.001070
grad_step = 000460, loss = 0.001044
grad_step = 000461, loss = 0.001000
grad_step = 000462, loss = 0.001008
grad_step = 000463, loss = 0.001041
grad_step = 000464, loss = 0.001034
grad_step = 000465, loss = 0.001009
grad_step = 000466, loss = 0.001002
grad_step = 000467, loss = 0.001010
grad_step = 000468, loss = 0.001004
grad_step = 000469, loss = 0.000991
grad_step = 000470, loss = 0.000992
grad_step = 000471, loss = 0.001004
grad_step = 000472, loss = 0.000998
grad_step = 000473, loss = 0.000982
grad_step = 000474, loss = 0.000974
grad_step = 000475, loss = 0.000980
grad_step = 000476, loss = 0.000984
grad_step = 000477, loss = 0.000978
grad_step = 000478, loss = 0.000975
grad_step = 000479, loss = 0.000985
grad_step = 000480, loss = 0.000990
grad_step = 000481, loss = 0.000990
grad_step = 000482, loss = 0.000988
grad_step = 000483, loss = 0.001002
grad_step = 000484, loss = 0.001022
grad_step = 000485, loss = 0.001046
grad_step = 000486, loss = 0.001059
grad_step = 000487, loss = 0.001094
grad_step = 000488, loss = 0.001082
grad_step = 000489, loss = 0.001076
grad_step = 000490, loss = 0.001016
grad_step = 000491, loss = 0.000975
grad_step = 000492, loss = 0.000960
grad_step = 000493, loss = 0.000959
grad_step = 000494, loss = 0.000963
grad_step = 000495, loss = 0.000967
grad_step = 000496, loss = 0.000976
grad_step = 000497, loss = 0.000974
grad_step = 000498, loss = 0.000963
grad_step = 000499, loss = 0.000949
grad_step = 000500, loss = 0.000940
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000935
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

  date_run                              2020-05-13 18:13:44.726934
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.222537
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 18:13:44.733535
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.116638
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 18:13:44.741053
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.132748
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 18:13:44.745932
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.772357
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
0   2020-05-13 18:13:18.795503  ...    mean_absolute_error
1   2020-05-13 18:13:18.799112  ...     mean_squared_error
2   2020-05-13 18:13:18.802253  ...  median_absolute_error
3   2020-05-13 18:13:18.805474  ...               r2_score
4   2020-05-13 18:13:27.105741  ...    mean_absolute_error
5   2020-05-13 18:13:27.109642  ...     mean_squared_error
6   2020-05-13 18:13:27.112725  ...  median_absolute_error
7   2020-05-13 18:13:27.117221  ...               r2_score
8   2020-05-13 18:13:44.726934  ...    mean_absolute_error
9   2020-05-13 18:13:44.733535  ...     mean_squared_error
10  2020-05-13 18:13:44.741053  ...  median_absolute_error
11  2020-05-13 18:13:44.745932  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4efd6efd30> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 311628.46it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 405079.61it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 560721.69it/s] 30%|███       | 3022848/9912422 [00:00<00:08, 790241.22it/s] 58%|█████▊    | 5734400/9912422 [00:00<00:03, 1112025.30it/s] 88%|████████▊ | 8732672/9912422 [00:00<00:00, 1556961.93it/s]9920512it [00:00, 9956367.99it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 147227.21it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 310749.60it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 402466.36it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 557354.01it/s]1654784it [00:00, 2632055.49it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 53597.06it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4eb00aae80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4eaf6db0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4eb00aae80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4eaf6300f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4eace6c4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4eace56c50> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4eb00aae80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4eaf5ee710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4eace6c4e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4eaf6db128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f3179fe7240> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=b3c9bab0711636f489549148095ec7e5bc1488a0b576e80f71fe8cd6718fc8e9
  Stored in directory: /tmp/pip-ephem-wheel-cache-_32wuoxj/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f3111de2710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 45s
   57344/17464789 [..............................] - ETA: 39s
   90112/17464789 [..............................] - ETA: 37s
  180224/17464789 [..............................] - ETA: 24s
  335872/17464789 [..............................] - ETA: 16s
  647168/17464789 [>.............................] - ETA: 10s
 1294336/17464789 [=>............................] - ETA: 5s 
 2539520/17464789 [===>..........................] - ETA: 3s
 5029888/17464789 [=======>......................] - ETA: 1s
 8110080/17464789 [============>.................] - ETA: 0s
11223040/17464789 [==================>...........] - ETA: 0s
14172160/17464789 [=======================>......] - ETA: 0s
17252352/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 18:15:14.497389: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 18:15:14.501264: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-13 18:15:14.501390: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5568702d3f50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 18:15:14.501403: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.6513 - accuracy: 0.5010
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5593 - accuracy: 0.5070 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6462 - accuracy: 0.5013
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7050 - accuracy: 0.4975
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6973 - accuracy: 0.4980
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6491 - accuracy: 0.5011
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6632 - accuracy: 0.5002
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6467 - accuracy: 0.5013
11000/25000 [============>.................] - ETA: 3s - loss: 7.6471 - accuracy: 0.5013
12000/25000 [=============>................] - ETA: 3s - loss: 7.6296 - accuracy: 0.5024
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6312 - accuracy: 0.5023
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6195 - accuracy: 0.5031
15000/25000 [=================>............] - ETA: 2s - loss: 7.6196 - accuracy: 0.5031
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6369 - accuracy: 0.5019
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6351 - accuracy: 0.5021
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6428 - accuracy: 0.5016
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6537 - accuracy: 0.5008
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6674 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6640 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6756 - accuracy: 0.4994
25000/25000 [==============================] - 7s 276us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 18:15:27.876688
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 18:15:27.876688  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<33:36:00, 7.13kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<23:32:54, 10.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.33M/862M [00:01<16:25:36, 14.5kB/s].vector_cache/glove.6B.zip:   1%|          | 6.34M/862M [00:01<11:28:16, 20.7kB/s].vector_cache/glove.6B.zip:   1%|          | 10.6M/862M [00:01<7:59:30, 29.6kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 16.9M/862M [00:02<5:33:14, 42.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.5M/862M [00:02<3:52:04, 60.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.7M/862M [00:02<2:41:45, 86.2kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.3M/862M [00:02<1:52:50, 123kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 30.7M/862M [00:02<1:19:09, 175kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.0M/862M [00:02<55:38, 249kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 33.3M/862M [00:02<39:14, 352kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.4M/862M [00:02<27:49, 496kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.7M/862M [00:02<19:45, 697kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.8M/862M [00:02<13:51, 989kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.9M/862M [00:03<09:42, 1.40MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:03<07:01, 1.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:06<07:21, 1.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:06<07:09, 1.87MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.8M/862M [00:06<05:09, 2.59MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:07<07:04, 1.89MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:07<05:53, 2.27MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.8M/862M [00:07<04:18, 3.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.6M/862M [00:07<03:32, 3.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:09<10:20, 1.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:09<08:12, 1.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.3M/862M [00:09<06:01, 2.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.2M/862M [00:09<04:39, 2.84MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:11<08:05, 1.64MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:11<06:57, 1.90MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.2M/862M [00:11<05:11, 2.54MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.3M/862M [00:11<03:58, 3.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:13<07:31, 1.75MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:13<06:50, 1.92MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:13<05:04, 2.59MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.6M/862M [00:13<03:51, 3.40MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:15<07:58, 1.64MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:15<07:30, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:15<05:38, 2.32MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:15<04:15, 3.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:17<06:54, 1.88MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:17<05:56, 2.19MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:17<04:21, 2.98MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:17<03:17, 3.93MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:19<20:42, 626kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:19<16:30, 784kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:19<11:55, 1.08MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.8M/862M [00:19<08:35, 1.50MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:21<10:39, 1.21MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:21<09:06, 1.41MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:21<06:39, 1.93MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:21<04:48, 2.67MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:23<42:32, 301kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:23<31:49, 402kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:23<22:45, 563kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.5M/862M [00:23<16:03, 795kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:25<17:28, 730kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:25<13:49, 921kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:25<09:54, 1.28MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<07:05, 1.79MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<14:04:31, 15.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<9:53:23, 21.4kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<6:54:35, 30.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<4:52:30, 43.1kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<3:27:28, 60.8kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<2:25:11, 86.6kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<1:44:00, 121kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<1:14:52, 167kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<52:28, 238kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<39:51, 313kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<31:43, 393kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<22:25, 555kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<18:11, 682kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<14:39, 846kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<10:26, 1.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<10:17, 1.20MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<08:59, 1.37MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<06:25, 1.91MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<08:51, 1.39MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<09:27, 1.30MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<06:58, 1.76MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:39<05:03, 2.41MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<08:59, 1.36MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<08:08, 1.50MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<05:48, 2.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<09:04, 1.34MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<11:31, 1.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<08:52, 1.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<06:23, 1.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<07:52, 1.53MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<07:57, 1.51MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<05:43, 2.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<07:24, 1.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<07:38, 1.57MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<05:31, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<06:54, 1.73MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<08:50, 1.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<06:24, 1.85MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<06:58, 1.70MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<08:16, 1.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<06:03, 1.95MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<08:34, 1.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<07:39, 1.54MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:54<05:29, 2.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:55<06:05, 1.92MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<04:30, 2.60MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<03:19, 3.51MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:57<11:09, 1.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<09:27, 1.23MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<06:43, 1.72MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<09:16, 1.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<08:07, 1.42MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:47, 1.99MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<09:02, 1.27MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<07:22, 1.56MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<05:16, 2.17MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<09:33, 1.20MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<07:37, 1.50MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<07:04, 1.61MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<05:46, 1.97MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<04:09, 2.72MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<08:14, 1.37MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<07:01, 1.61MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<05:01, 2.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<09:05, 1.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<07:22, 1.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<05:16, 2.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:11<08:39, 1.29MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<07:00, 1.59MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<06:32, 1.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<05:53, 1.88MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<04:13, 2.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<09:15, 1.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<07:35, 1.45MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<05:25, 2.02MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:17<08:05, 1.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:17<06:57, 1.57MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<04:57, 2.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<08:46, 1.24MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<07:03, 1.54MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:02, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<09:39, 1.12MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<07:40, 1.41MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<05:27, 1.97MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<12:06, 888kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<09:39, 1.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<06:51, 1.56MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<09:35, 1.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:25<07:33, 1.41MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:22, 1.98MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<10:21, 1.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:27<08:38, 1.23MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<06:08, 1.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<09:13, 1.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<07:25, 1.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<05:18, 1.98MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<07:57, 1.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<06:26, 1.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<04:36, 2.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<08:05, 1.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<06:39, 1.56MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:44, 2.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<10:37, 973kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<08:30, 1.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<06:02, 1.70MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<09:08, 1.12MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<07:47, 1.32MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<05:32, 1.84MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<08:26, 1.21MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:39<06:46, 1.50MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<04:51, 2.09MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:41<07:22, 1.37MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:41<06:35, 1.54MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<04:42, 2.14MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:43<08:10, 1.23MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:43<07:02, 1.43MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:43<05:01, 2.00MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:45<08:32, 1.17MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:45<07:01, 1.42MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<04:59, 1.99MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<08:45, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<07:31, 1.32MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:47<05:23, 1.83MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<06:36, 1.49MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<05:36, 1.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<03:59, 2.45MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<10:48, 906kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<08:30, 1.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<06:01, 1.62MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<09:29, 1.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<08:01, 1.21MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<05:42, 1.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<08:23, 1.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<07:13, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:55<05:08, 1.86MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<07:42, 1.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<06:45, 1.42MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:57<04:50, 1.97MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:58<06:30, 1.46MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<05:45, 1.65MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:59<04:08, 2.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<05:53, 1.60MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<05:10, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<03:42, 2.54MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<07:14, 1.29MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<06:27, 1.45MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:36, 2.03MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<07:03, 1.32MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<06:18, 1.48MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:29, 2.06MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<08:11, 1.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<07:04, 1.31MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<05:01, 1.83MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:08<07:42, 1.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:08<06:42, 1.37MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:45, 1.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:10<08:18, 1.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:10<07:07, 1.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<05:02, 1.79MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:12<09:19, 969kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<07:59, 1.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<05:42, 1.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<06:20, 1.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<05:45, 1.56MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<04:06, 2.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<06:56, 1.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<06:06, 1.46MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<04:23, 2.02MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:18<05:28, 1.61MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:18<05:07, 1.72MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<03:39, 2.40MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:20<07:00, 1.25MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:20<06:09, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:22, 1.99MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<08:10, 1.06MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<06:59, 1.24MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<05:03, 1.71MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:24<05:25, 1.59MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:24<05:44, 1.50MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<04:09, 2.06MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:26<04:50, 1.76MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:26<04:41, 1.82MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<03:31, 2.42MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<04:03, 2.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<04:55, 1.72MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:28<03:44, 2.26MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<02:43, 3.09MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:31<10:24, 808kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:32<07:58, 1.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:32<05:39, 1.48MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:33<07:06, 1.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<07:01, 1.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:34<05:04, 1.64MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:34<03:41, 2.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:35<07:42, 1.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:35<06:15, 1.32MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:36<04:27, 1.85MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:37<06:14, 1.32MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:37<04:58, 1.65MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:38<03:33, 2.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:39<05:52, 1.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:39<04:49, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<03:25, 2.36MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:41<23:52, 338kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:41<17:41, 456kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:41<12:25, 646kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:43<11:39, 686kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:43<09:04, 881kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:43<06:24, 1.24MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:45<07:36, 1.04MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<06:27, 1.23MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:45<04:35, 1.72MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:47<06:09, 1.28MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:47<05:25, 1.45MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:47<03:52, 2.02MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:49<05:44, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:49<05:11, 1.50MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:49<03:42, 2.09MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:51<05:14, 1.47MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:51<04:45, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<03:24, 2.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:53<05:22, 1.42MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:53<04:52, 1.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:53<03:30, 2.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:55<04:46, 1.59MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:55<04:27, 1.70MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:55<03:10, 2.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:57<05:39, 1.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:57<05:02, 1.49MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:57<03:36, 2.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:59<05:34, 1.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:59<05:01, 1.48MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:59<03:34, 2.07MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:01<05:39, 1.30MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:01<05:00, 1.47MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:01<03:35, 2.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:03<04:49, 1.52MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:03<04:14, 1.72MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:03<03:01, 2.40MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:05<05:30, 1.31MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:05<04:37, 1.57MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:05<03:18, 2.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:07<05:19, 1.35MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:07<04:46, 1.50MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<03:24, 2.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<05:03, 1.40MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<04:33, 1.56MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<03:14, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:11<05:13, 1.35MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:11<04:23, 1.60MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:11<03:09, 2.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:13<04:33, 1.53MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<03:47, 1.83MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:13<02:43, 2.54MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:15<04:57, 1.39MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<03:57, 1.74MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:15<02:50, 2.41MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<04:37, 1.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:17<04:01, 1.70MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:17<02:53, 2.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:19<04:18, 1.57MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:19<03:34, 1.89MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:19<02:34, 2.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:21<03:54, 1.72MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:21<03:26, 1.94MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:21<02:28, 2.68MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:23<04:18, 1.54MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:23<03:37, 1.83MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:23<02:37, 2.51MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:23<01:56, 3.37MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<37:39, 174kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<26:57, 243kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:25<18:51, 345kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:27<15:10, 427kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:27<11:36, 558kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:27<08:10, 789kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:29<07:37, 842kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<05:52, 1.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<04:08, 1.54MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:30<08:33, 743kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:31<06:52, 922kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:31<04:52, 1.29MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:32<05:14, 1.20MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:33<04:36, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:33<03:17, 1.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:34<04:14, 1.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<03:51, 1.61MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:35<02:46, 2.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:36<03:42, 1.65MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<03:27, 1.77MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:37<02:29, 2.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<03:31, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<03:01, 2.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:39<02:10, 2.77MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<03:59, 1.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<03:25, 1.75MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:41<02:30, 2.39MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:42<03:05, 1.92MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:42<03:03, 1.94MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:43<02:13, 2.64MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:44<02:57, 1.99MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:44<02:56, 2.00MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<02:10, 2.68MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:46<02:41, 2.15MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<02:43, 2.12MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<02:00, 2.86MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:48<02:41, 2.13MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<02:43, 2.10MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:48<01:59, 2.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:50<02:44, 2.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:50<02:44, 2.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<02:02, 2.76MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<02:34, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<02:39, 2.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:52<02:00, 2.78MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:52<01:29, 3.71MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:54<03:34, 1.54MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:54<03:13, 1.71MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:54<02:21, 2.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<01:42, 3.18MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<19:38, 278kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<14:32, 375kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<10:18, 527kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<07:15, 744kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:58<07:49, 688kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:58<06:18, 852kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<04:34, 1.17MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<03:15, 1.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<05:36, 947kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:00<04:39, 1.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:00<03:29, 1.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:00<02:29, 2.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:02<04:56, 1.06MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<04:22, 1.20MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<03:19, 1.57MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:02<02:24, 2.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<03:09, 1.64MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<02:47, 1.86MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:04<02:07, 2.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<01:31, 3.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:06<12:49, 398kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:06<09:57, 513kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:06<07:07, 715kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<04:59, 1.01MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:08<2:36:18, 32.2kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:08<1:50:13, 45.7kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<1:16:54, 65.2kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<54:25, 91.3kB/s]  .vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<38:48, 128kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<26:57, 183kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:12<21:10, 231kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:12<15:26, 317kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:12<10:46, 451kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<07:27, 641kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:15<23:35, 202kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:16<17:07, 279kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<11:57, 396kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:17<09:54, 475kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:18<07:26, 632kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<05:16, 886kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:18<03:47, 1.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:19<04:39, 997kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:20<04:42, 986kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:20<03:33, 1.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<02:51, 1.62MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<02:25, 1.90MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<02:06, 2.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<01:48, 2.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<01:36, 2.87MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:20<01:26, 3.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:20<01:18, 3.50MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<01:12, 3.78MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:21<06:10, 740kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:21<05:04, 901kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<03:51, 1.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<03:07, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<02:29, 1.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<02:01, 2.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<01:41, 2.69MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:22<01:25, 3.18MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:22<01:15, 3.60MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<01:06, 4.04MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:23<29:33, 152kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:23<21:47, 206kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:24<15:49, 284kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:24<11:27, 392kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:24<08:15, 543kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<06:01, 742kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<04:26, 1.00MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:24<03:23, 1.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:24<02:36, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<02:08, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:25<06:38, 668kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:25<05:44, 772kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:26<06:02, 732kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:26<04:29, 983kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:26<03:26, 1.28MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:26<02:40, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:26<02:07, 2.06MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:26<01:45, 2.50MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:27<01:32, 2.83MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:27<01:35, 2.76MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:27<01:26, 3.03MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:27<01:17, 3.39MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:28<08:45, 497kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:28<06:28, 671kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<04:46, 909kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<03:35, 1.21MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:28<02:47, 1.55MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:28<02:12, 1.96MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<01:45, 2.44MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<01:27, 2.95MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<01:18, 3.28MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:30<21:06, 203kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:30<16:14, 264kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:30<11:55, 359kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<08:34, 498kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:30<06:11, 688kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:30<04:31, 939kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:30<03:21, 1.26MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:30<02:32, 1.67MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:30<01:59, 2.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:31<13:15, 318kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:32<10:14, 411kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:32<07:49, 538kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<05:57, 707kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<04:25, 949kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:32<03:17, 1.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<02:28, 1.68MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<01:54, 2.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:32<01:30, 2.74MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:33<05:20, 776kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:34<04:32, 912kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<03:57, 1.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<03:30, 1.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<02:41, 1.54MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:34<02:02, 2.02MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:34<01:35, 2.58MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<01:16, 3.21MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<01:03, 3.86MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:36<33:44, 121kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:36<25:08, 162kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<19:10, 213kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<14:28, 282kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<10:23, 391kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:36<07:24, 547kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:36<05:19, 759kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:36<03:51, 1.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:36<02:50, 1.41MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<12:21, 325kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<09:28, 423kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<07:11, 556kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<05:44, 697kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<04:33, 876kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:38<03:25, 1.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:38<02:32, 1.57MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:38<01:54, 2.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:38<01:28, 2.69MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<05:31, 714kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<04:31, 871kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<03:21, 1.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:40<02:37, 1.50MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:40<01:57, 2.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:40<01:29, 2.61MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:40<01:09, 3.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:41<18:18, 211kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:42<13:22, 289kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<09:29, 406kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<06:48, 565kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:42<05:00, 767kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:42<03:36, 1.06MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:42<02:38, 1.44MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:43<20:44, 184kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:44<15:06, 252kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<10:43, 354kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<07:41, 491kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:44<05:30, 684kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<03:56, 950kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<04:40, 800kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:46<04:33, 820kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:46<04:07, 906kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:46<03:46, 986kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:46<03:41, 1.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<02:50, 1.31MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:46<02:04, 1.78MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:46<01:33, 2.37MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:46<01:11, 3.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:47<39:15, 93.4kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:48<28:03, 131kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<19:49, 184kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<14:12, 257kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<10:19, 353kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:48<07:19, 495kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:48<05:12, 694kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:49<05:19, 675kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:49<04:14, 847kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:50<03:04, 1.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:50<02:15, 1.58MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:50<01:40, 2.11MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<02:43, 1.30MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<03:14, 1.09MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:52<02:51, 1.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:52<02:35, 1.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:52<02:19, 1.51MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:52<01:48, 1.94MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:52<01:21, 2.58MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:52<01:01, 3.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<03:19, 1.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<02:48, 1.23MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:54<02:02, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:54<01:29, 2.29MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<02:20, 1.45MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<02:08, 1.58MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:55<01:38, 2.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:56<01:19, 2.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:56<01:06, 3.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:56<00:56, 3.55MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<01:37, 2.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<01:37, 2.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<01:11, 2.75MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:58<00:53, 3.66MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:59<02:16, 1.43MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:59<02:03, 1.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:59<01:30, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:00<01:05, 2.92MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:02<03:35, 888kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:02<02:57, 1.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:02<02:06, 1.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:02<01:30, 2.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:04<05:33, 561kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:04<04:20, 717kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:04<03:04, 1.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:04<02:09, 1.41MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:06<3:07:26, 16.3kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:06<2:11:30, 23.1kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:06<1:31:13, 33.1kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:07<1:03:49, 46.7kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:08<45:04, 66.0kB/s]  .vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:08<31:16, 94.2kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<22:22, 130kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:10<15:53, 183kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:10<11:02, 260kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<08:32, 333kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:12<06:23, 444kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:12<04:27, 628kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<03:51, 719kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:14<03:05, 897kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:14<02:10, 1.26MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<02:18, 1.17MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<02:00, 1.35MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:16<01:25, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<01:48, 1.46MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<01:39, 1.59MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:18<01:10, 2.21MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<01:35, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<01:48, 1.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:20<01:23, 1.83MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:20<01:10, 2.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:20<00:59, 2.55MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:20<00:51, 2.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:20<00:47, 3.21MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:20<00:43, 3.50MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:20<00:39, 3.86MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:20<00:38, 3.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:21<03:33, 702kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<02:52, 868kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<02:09, 1.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:22<01:39, 1.50MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:22<01:17, 1.91MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:22<01:02, 2.38MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:22<00:51, 2.84MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:22<00:44, 3.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:22<00:38, 3.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:23<05:57, 407kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<04:41, 517kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<03:26, 703kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:24<02:31, 956kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:24<01:52, 1.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:24<01:26, 1.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:24<01:10, 2.04MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:24<00:58, 2.42MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:24<00:50, 2.84MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<02:34, 914kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<02:09, 1.09MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<01:39, 1.42MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:26<01:17, 1.81MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:26<01:02, 2.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:26<00:51, 2.70MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:26<00:44, 3.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:26<00:38, 3.59MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:26<00:33, 4.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<03:31, 649kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<02:48, 813kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<02:05, 1.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:28<01:34, 1.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:28<01:12, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:28<00:57, 2.35MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:28<00:46, 2.88MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 729M/862M [05:28<00:38, 3.45MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<03:09, 705kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<03:37, 612kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<02:39, 834kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<01:57, 1.13MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:30<01:27, 1.50MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:30<01:06, 1.96MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:30<00:52, 2.49MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:30<00:41, 3.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:32<12:33, 171kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:32<09:06, 236kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:32<06:25, 332kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:32<04:33, 466kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:32<03:15, 648kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:32<02:21, 892kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:32<01:43, 1.21MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:34<04:31, 460kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:34<03:28, 600kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:34<02:29, 829kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:34<01:48, 1.13MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:34<01:20, 1.53MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:34<01:00, 2.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:36<01:56, 1.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:36<01:39, 1.21MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:36<01:14, 1.62MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:36<00:55, 2.14MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:36<00:42, 2.75MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:36<00:33, 3.47MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:37<01:58, 987kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:38<01:49, 1.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:38<01:22, 1.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:38<01:01, 1.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:38<00:46, 2.45MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:38<00:36, 3.13MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:40<01:37, 1.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:40<01:23, 1.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:40<01:02, 1.80MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:40<00:46, 2.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:40<00:35, 3.06MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:40<00:28, 3.85MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:42<06:19, 286kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:42<05:01, 360kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:42<03:46, 478kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:42<02:48, 642kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:42<02:04, 861kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:42<01:35, 1.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:42<01:11, 1.50MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:42<00:52, 2.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:44<01:16, 1.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:44<01:13, 1.42MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:44<00:54, 1.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:44<00:40, 2.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:44<00:31, 3.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:44<00:26, 3.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:46<03:12, 523kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:46<02:26, 684kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:46<01:44, 950kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:47<01:14, 1.31MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:47<00:54, 1.78MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:48<01:55, 834kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:48<01:42, 934kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:48<01:14, 1.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:48<00:55, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:49<00:42, 2.19MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:49<00:41, 2.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:49<00:36, 2.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:49<00:28, 3.24MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:50<09:04, 169kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:50<06:33, 233kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:50<04:35, 330kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:50<03:12, 466kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:51<02:15, 653kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:52<02:42, 543kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:52<02:05, 698kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:52<01:29, 969kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:52<01:04, 1.34MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:53<00:47, 1.80MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:53<00:39, 2.13MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:54<02:07, 660kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:54<01:39, 839kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:54<01:10, 1.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:54<00:51, 1.59MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:54<00:37, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:56<02:01, 658kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:56<01:35, 829kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:56<01:08, 1.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:56<00:48, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:56<00:35, 2.14MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:58<02:38, 476kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:58<02:02, 617kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:58<01:26, 860kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:58<01:01, 1.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:59<00:43, 1.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:00<02:19, 513kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:00<01:47, 664kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:01<01:15, 925kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:01<00:53, 1.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:01<00:38, 1.76MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:02<04:30, 249kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:02<03:18, 338kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:03<02:18, 477kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:03<01:36, 672kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:03<01:08, 933kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:04<02:23, 440kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:04<01:48, 579kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:04<01:16, 810kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:05<00:53, 1.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:05<00:38, 1.55MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:06<03:57, 249kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:06<02:57, 332kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:06<02:03, 469kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:07<01:26, 655kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:07<01:00, 917kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:08<01:31, 599kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:08<01:13, 747kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:08<00:51, 1.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:09<00:36, 1.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:10<00:43, 1.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:10<00:38, 1.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:10<00:27, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:10<00:19, 2.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:12<00:38, 1.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:12<00:32, 1.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:12<00:22, 1.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:12<00:15, 2.69MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:14<01:10, 608kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:14<00:54, 773kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:14<00:38, 1.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:14<00:25, 1.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:16<01:00, 636kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:16<00:45, 837kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:16<00:31, 1.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<00:21, 1.63MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:18<02:25, 236kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:18<01:46, 321kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:18<01:10, 456kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:20<00:53, 560kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:20<00:39, 752kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:20<00:26, 1.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:22<00:24, 1.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:22<00:18, 1.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:22<00:12, 1.90MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:24<00:14, 1.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:24<00:12, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:24<00:08, 2.38MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:26<00:09, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:26<00:09, 1.88MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:26<00:05, 2.59MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:28<00:07, 1.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:28<00:06, 2.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:28<00:04, 2.83MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:30<00:05, 1.86MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:30<00:04, 1.92MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:30<00:02, 2.65MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:32<00:03, 1.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:32<00:02, 1.85MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:32<00:01, 2.56MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:34<00:00, 1.52MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:34<00:00, 1.65MB/s].vector_cache/glove.6B.zip: 862MB [06:34, 2.19MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 857/400000 [00:00<00:46, 8561.60it/s]  0%|          | 1617/400000 [00:00<00:48, 8248.25it/s]  1%|          | 2501/400000 [00:00<00:47, 8415.40it/s]  1%|          | 3386/400000 [00:00<00:46, 8541.07it/s]  1%|          | 4270/400000 [00:00<00:45, 8626.49it/s]  1%|▏         | 5085/400000 [00:00<00:46, 8477.62it/s]  1%|▏         | 5954/400000 [00:00<00:46, 8537.70it/s]  2%|▏         | 6822/400000 [00:00<00:45, 8577.30it/s]  2%|▏         | 7699/400000 [00:00<00:45, 8634.17it/s]  2%|▏         | 8576/400000 [00:01<00:45, 8673.28it/s]  2%|▏         | 9457/400000 [00:01<00:44, 8711.88it/s]  3%|▎         | 10345/400000 [00:01<00:44, 8760.80it/s]  3%|▎         | 11215/400000 [00:01<00:44, 8742.00it/s]  3%|▎         | 12082/400000 [00:01<00:44, 8640.64it/s]  3%|▎         | 12953/400000 [00:01<00:44, 8659.18it/s]  3%|▎         | 13841/400000 [00:01<00:44, 8722.70it/s]  4%|▎         | 14724/400000 [00:01<00:44, 8752.20it/s]  4%|▍         | 15609/400000 [00:01<00:43, 8779.28it/s]  4%|▍         | 16498/400000 [00:01<00:43, 8809.11it/s]  4%|▍         | 17385/400000 [00:02<00:43, 8824.91it/s]  5%|▍         | 18275/400000 [00:02<00:43, 8845.92it/s]  5%|▍         | 19162/400000 [00:02<00:43, 8850.72it/s]  5%|▌         | 20050/400000 [00:02<00:42, 8858.54it/s]  5%|▌         | 20937/400000 [00:02<00:42, 8861.04it/s]  5%|▌         | 21823/400000 [00:02<00:42, 8832.11it/s]  6%|▌         | 22710/400000 [00:02<00:42, 8842.46it/s]  6%|▌         | 23600/400000 [00:02<00:42, 8856.99it/s]  6%|▌         | 24486/400000 [00:02<00:42, 8810.50it/s]  6%|▋         | 25372/400000 [00:02<00:42, 8823.16it/s]  7%|▋         | 26261/400000 [00:03<00:42, 8842.73it/s]  7%|▋         | 27146/400000 [00:03<00:42, 8810.24it/s]  7%|▋         | 28036/400000 [00:03<00:42, 8835.21it/s]  7%|▋         | 28920/400000 [00:03<00:42, 8806.69it/s]  7%|▋         | 29806/400000 [00:03<00:41, 8820.35it/s]  8%|▊         | 30693/400000 [00:03<00:41, 8834.80it/s]  8%|▊         | 31579/400000 [00:03<00:41, 8841.19it/s]  8%|▊         | 32468/400000 [00:03<00:41, 8854.75it/s]  8%|▊         | 33354/400000 [00:03<00:41, 8815.66it/s]  9%|▊         | 34236/400000 [00:03<00:41, 8803.76it/s]  9%|▉         | 35128/400000 [00:04<00:41, 8836.25it/s]  9%|▉         | 36013/400000 [00:04<00:41, 8839.78it/s]  9%|▉         | 36898/400000 [00:04<00:41, 8747.35it/s]  9%|▉         | 37773/400000 [00:04<00:41, 8735.70it/s] 10%|▉         | 38658/400000 [00:04<00:41, 8768.09it/s] 10%|▉         | 39535/400000 [00:04<00:41, 8748.80it/s] 10%|█         | 40421/400000 [00:04<00:40, 8780.52it/s] 10%|█         | 41311/400000 [00:04<00:40, 8813.29it/s] 11%|█         | 42193/400000 [00:04<00:40, 8812.79it/s] 11%|█         | 43082/400000 [00:04<00:40, 8833.95it/s] 11%|█         | 43970/400000 [00:05<00:40, 8845.44it/s] 11%|█         | 44855/400000 [00:05<00:40, 8839.46it/s] 11%|█▏        | 45743/400000 [00:05<00:40, 8851.47it/s] 12%|█▏        | 46629/400000 [00:05<00:39, 8851.09it/s] 12%|█▏        | 47515/400000 [00:05<00:40, 8658.40it/s] 12%|█▏        | 48401/400000 [00:05<00:40, 8716.90it/s] 12%|█▏        | 49285/400000 [00:05<00:40, 8751.49it/s] 13%|█▎        | 50164/400000 [00:05<00:39, 8762.34it/s] 13%|█▎        | 51041/400000 [00:05<00:39, 8753.26it/s] 13%|█▎        | 51917/400000 [00:05<00:39, 8703.12it/s] 13%|█▎        | 52803/400000 [00:06<00:39, 8749.10it/s] 13%|█▎        | 53693/400000 [00:06<00:39, 8793.23it/s] 14%|█▎        | 54579/400000 [00:06<00:39, 8810.30it/s] 14%|█▍        | 55466/400000 [00:06<00:39, 8826.63it/s] 14%|█▍        | 56349/400000 [00:06<00:39, 8677.93it/s] 14%|█▍        | 57240/400000 [00:06<00:39, 8745.95it/s] 15%|█▍        | 58133/400000 [00:06<00:38, 8799.95it/s] 15%|█▍        | 59018/400000 [00:06<00:38, 8813.07it/s] 15%|█▍        | 59906/400000 [00:06<00:38, 8830.83it/s] 15%|█▌        | 60797/400000 [00:06<00:38, 8854.17it/s] 15%|█▌        | 61683/400000 [00:07<00:38, 8846.21it/s] 16%|█▌        | 62569/400000 [00:07<00:38, 8847.94it/s] 16%|█▌        | 63458/400000 [00:07<00:37, 8858.93it/s] 16%|█▌        | 64344/400000 [00:07<00:37, 8845.00it/s] 16%|█▋        | 65233/400000 [00:07<00:37, 8852.40it/s] 17%|█▋        | 66119/400000 [00:07<00:38, 8661.91it/s] 17%|█▋        | 66987/400000 [00:07<00:38, 8662.33it/s] 17%|█▋        | 67854/400000 [00:07<00:38, 8659.86it/s] 17%|█▋        | 68738/400000 [00:07<00:38, 8712.58it/s] 17%|█▋        | 69622/400000 [00:07<00:37, 8748.24it/s] 18%|█▊        | 70505/400000 [00:08<00:37, 8771.43it/s] 18%|█▊        | 71383/400000 [00:08<00:37, 8761.43it/s] 18%|█▊        | 72276/400000 [00:08<00:37, 8811.25it/s] 18%|█▊        | 73165/400000 [00:08<00:36, 8834.38it/s] 19%|█▊        | 74049/400000 [00:08<00:37, 8808.41it/s] 19%|█▊        | 74939/400000 [00:08<00:36, 8831.09it/s] 19%|█▉        | 75829/400000 [00:08<00:36, 8850.66it/s] 19%|█▉        | 76715/400000 [00:08<00:36, 8852.77it/s] 19%|█▉        | 77601/400000 [00:08<00:36, 8834.30it/s] 20%|█▉        | 78493/400000 [00:08<00:36, 8858.43it/s] 20%|█▉        | 79386/400000 [00:09<00:36, 8878.08it/s] 20%|██        | 80274/400000 [00:09<00:36, 8847.76it/s] 20%|██        | 81163/400000 [00:09<00:35, 8858.72it/s] 21%|██        | 82051/400000 [00:09<00:35, 8862.61it/s] 21%|██        | 82938/400000 [00:09<00:35, 8826.58it/s] 21%|██        | 83826/400000 [00:09<00:35, 8839.97it/s] 21%|██        | 84713/400000 [00:09<00:35, 8848.16it/s] 21%|██▏       | 85601/400000 [00:09<00:35, 8854.90it/s] 22%|██▏       | 86487/400000 [00:09<00:35, 8791.92it/s] 22%|██▏       | 87367/400000 [00:09<00:35, 8766.83it/s] 22%|██▏       | 88245/400000 [00:10<00:35, 8768.28it/s] 22%|██▏       | 89126/400000 [00:10<00:35, 8780.71it/s] 23%|██▎       | 90012/400000 [00:10<00:35, 8801.93it/s] 23%|██▎       | 90893/400000 [00:10<00:35, 8798.12it/s] 23%|██▎       | 91773/400000 [00:10<00:35, 8793.91it/s] 23%|██▎       | 92657/400000 [00:10<00:34, 8805.90it/s] 23%|██▎       | 93548/400000 [00:10<00:34, 8834.52it/s] 24%|██▎       | 94433/400000 [00:10<00:34, 8838.70it/s] 24%|██▍       | 95321/400000 [00:10<00:34, 8848.74it/s] 24%|██▍       | 96206/400000 [00:10<00:34, 8795.44it/s] 24%|██▍       | 97093/400000 [00:11<00:34, 8815.56it/s] 24%|██▍       | 97975/400000 [00:11<00:34, 8800.21it/s] 25%|██▍       | 98856/400000 [00:11<00:34, 8800.28it/s] 25%|██▍       | 99741/400000 [00:11<00:34, 8812.48it/s] 25%|██▌       | 100623/400000 [00:11<00:34, 8636.37it/s] 25%|██▌       | 101506/400000 [00:11<00:34, 8691.93it/s] 26%|██▌       | 102394/400000 [00:11<00:34, 8745.94it/s] 26%|██▌       | 103270/400000 [00:11<00:34, 8719.22it/s] 26%|██▌       | 104154/400000 [00:11<00:33, 8753.79it/s] 26%|██▋       | 105037/400000 [00:11<00:33, 8775.39it/s] 26%|██▋       | 105924/400000 [00:12<00:33, 8802.31it/s] 27%|██▋       | 106811/400000 [00:12<00:33, 8822.17it/s] 27%|██▋       | 107694/400000 [00:12<00:33, 8764.06it/s] 27%|██▋       | 108584/400000 [00:12<00:33, 8802.83it/s] 27%|██▋       | 109465/400000 [00:12<00:33, 8793.69it/s] 28%|██▊       | 110352/400000 [00:12<00:32, 8815.21it/s] 28%|██▊       | 111241/400000 [00:12<00:32, 8837.03it/s] 28%|██▊       | 112125/400000 [00:12<00:32, 8834.52it/s] 28%|██▊       | 113013/400000 [00:12<00:32, 8845.23it/s] 28%|██▊       | 113898/400000 [00:12<00:32, 8806.30it/s] 29%|██▊       | 114785/400000 [00:13<00:32, 8823.98it/s] 29%|██▉       | 115674/400000 [00:13<00:32, 8841.94it/s] 29%|██▉       | 116559/400000 [00:13<00:32, 8838.96it/s] 29%|██▉       | 117443/400000 [00:13<00:32, 8750.04it/s] 30%|██▉       | 118319/400000 [00:13<00:32, 8742.81it/s] 30%|██▉       | 119204/400000 [00:13<00:32, 8772.41it/s] 30%|███       | 120094/400000 [00:13<00:31, 8808.89it/s] 30%|███       | 120977/400000 [00:13<00:31, 8814.33it/s] 30%|███       | 121860/400000 [00:13<00:31, 8818.71it/s] 31%|███       | 122742/400000 [00:13<00:31, 8807.36it/s] 31%|███       | 123626/400000 [00:14<00:31, 8817.10it/s] 31%|███       | 124509/400000 [00:14<00:31, 8818.84it/s] 31%|███▏      | 125391/400000 [00:14<00:31, 8780.45it/s] 32%|███▏      | 126278/400000 [00:14<00:31, 8805.28it/s] 32%|███▏      | 127159/400000 [00:14<00:31, 8792.83it/s] 32%|███▏      | 128048/400000 [00:14<00:30, 8821.03it/s] 32%|███▏      | 128938/400000 [00:14<00:30, 8844.21it/s] 32%|███▏      | 129827/400000 [00:14<00:30, 8855.73it/s] 33%|███▎      | 130713/400000 [00:14<00:30, 8818.73it/s] 33%|███▎      | 131595/400000 [00:14<00:31, 8469.79it/s] 33%|███▎      | 132445/400000 [00:15<00:31, 8472.33it/s] 33%|███▎      | 133331/400000 [00:15<00:31, 8583.60it/s] 34%|███▎      | 134192/400000 [00:15<00:31, 8566.07it/s] 34%|███▍      | 135074/400000 [00:15<00:30, 8639.46it/s] 34%|███▍      | 135942/400000 [00:15<00:30, 8650.62it/s] 34%|███▍      | 136833/400000 [00:15<00:30, 8725.63it/s] 34%|███▍      | 137722/400000 [00:15<00:29, 8772.33it/s] 35%|███▍      | 138605/400000 [00:15<00:29, 8787.61it/s] 35%|███▍      | 139489/400000 [00:15<00:29, 8801.30it/s] 35%|███▌      | 140370/400000 [00:15<00:29, 8793.01it/s] 35%|███▌      | 141250/400000 [00:16<00:29, 8780.71it/s] 36%|███▌      | 142134/400000 [00:16<00:29, 8795.51it/s] 36%|███▌      | 143026/400000 [00:16<00:29, 8832.00it/s] 36%|███▌      | 143910/400000 [00:16<00:29, 8783.47it/s] 36%|███▌      | 144796/400000 [00:16<00:28, 8805.20it/s] 36%|███▋      | 145686/400000 [00:16<00:28, 8832.69it/s] 37%|███▋      | 146579/400000 [00:16<00:28, 8860.04it/s] 37%|███▋      | 147466/400000 [00:16<00:28, 8858.57it/s] 37%|███▋      | 148352/400000 [00:16<00:28, 8835.84it/s] 37%|███▋      | 149236/400000 [00:16<00:28, 8801.17it/s] 38%|███▊      | 150128/400000 [00:17<00:28, 8836.00it/s] 38%|███▊      | 151012/400000 [00:17<00:28, 8834.01it/s] 38%|███▊      | 151901/400000 [00:17<00:28, 8848.18it/s] 38%|███▊      | 152789/400000 [00:17<00:27, 8856.28it/s] 38%|███▊      | 153675/400000 [00:17<00:27, 8851.25it/s] 39%|███▊      | 154567/400000 [00:17<00:27, 8871.45it/s] 39%|███▉      | 155457/400000 [00:17<00:27, 8878.85it/s] 39%|███▉      | 156345/400000 [00:17<00:27, 8854.17it/s] 39%|███▉      | 157231/400000 [00:17<00:27, 8831.47it/s] 40%|███▉      | 158118/400000 [00:18<00:27, 8840.58it/s] 40%|███▉      | 159003/400000 [00:18<00:27, 8833.81it/s] 40%|███▉      | 159891/400000 [00:18<00:27, 8846.21it/s] 40%|████      | 160776/400000 [00:18<00:27, 8839.03it/s] 40%|████      | 161660/400000 [00:18<00:27, 8711.76it/s] 41%|████      | 162542/400000 [00:18<00:27, 8743.34it/s] 41%|████      | 163433/400000 [00:18<00:26, 8791.56it/s] 41%|████      | 164324/400000 [00:18<00:26, 8825.20it/s] 41%|████▏     | 165217/400000 [00:18<00:26, 8853.92it/s] 42%|████▏     | 166103/400000 [00:18<00:26, 8851.33it/s] 42%|████▏     | 166989/400000 [00:19<00:26, 8844.11it/s] 42%|████▏     | 167880/400000 [00:19<00:26, 8861.27it/s] 42%|████▏     | 168771/400000 [00:19<00:26, 8874.98it/s] 42%|████▏     | 169659/400000 [00:19<00:25, 8875.25it/s] 43%|████▎     | 170547/400000 [00:19<00:26, 8700.39it/s] 43%|████▎     | 171424/400000 [00:19<00:26, 8721.11it/s] 43%|████▎     | 172312/400000 [00:19<00:25, 8768.17it/s] 43%|████▎     | 173205/400000 [00:19<00:25, 8813.75it/s] 44%|████▎     | 174094/400000 [00:19<00:25, 8835.86it/s] 44%|████▎     | 174978/400000 [00:19<00:25, 8836.78it/s] 44%|████▍     | 175868/400000 [00:20<00:25, 8855.34it/s] 44%|████▍     | 176754/400000 [00:20<00:25, 8723.70it/s] 44%|████▍     | 177644/400000 [00:20<00:25, 8773.37it/s] 45%|████▍     | 178530/400000 [00:20<00:25, 8797.36it/s] 45%|████▍     | 179411/400000 [00:20<00:25, 8781.30it/s] 45%|████▌     | 180295/400000 [00:20<00:24, 8796.11it/s] 45%|████▌     | 181184/400000 [00:20<00:24, 8822.99it/s] 46%|████▌     | 182071/400000 [00:20<00:24, 8836.22it/s] 46%|████▌     | 182959/400000 [00:20<00:24, 8848.04it/s] 46%|████▌     | 183844/400000 [00:20<00:24, 8846.90it/s] 46%|████▌     | 184729/400000 [00:21<00:24, 8833.96it/s] 46%|████▋     | 185613/400000 [00:21<00:24, 8738.46it/s] 47%|████▋     | 186503/400000 [00:21<00:24, 8785.25it/s] 47%|████▋     | 187386/400000 [00:21<00:24, 8795.81it/s] 47%|████▋     | 188266/400000 [00:21<00:24, 8760.79it/s] 47%|████▋     | 189143/400000 [00:21<00:24, 8645.75it/s] 48%|████▊     | 190032/400000 [00:21<00:24, 8715.85it/s] 48%|████▊     | 190923/400000 [00:21<00:23, 8772.72it/s] 48%|████▊     | 191811/400000 [00:21<00:23, 8804.52it/s] 48%|████▊     | 192692/400000 [00:21<00:23, 8805.47it/s] 48%|████▊     | 193584/400000 [00:22<00:23, 8839.16it/s] 49%|████▊     | 194469/400000 [00:22<00:23, 8829.30it/s] 49%|████▉     | 195361/400000 [00:22<00:23, 8854.11it/s] 49%|████▉     | 196247/400000 [00:22<00:23, 8841.98it/s] 49%|████▉     | 197134/400000 [00:22<00:22, 8848.81it/s] 50%|████▉     | 198019/400000 [00:22<00:22, 8843.58it/s] 50%|████▉     | 198908/400000 [00:22<00:22, 8855.99it/s] 50%|████▉     | 199801/400000 [00:22<00:22, 8876.24it/s] 50%|█████     | 200694/400000 [00:22<00:22, 8889.93it/s] 50%|█████     | 201584/400000 [00:22<00:22, 8859.98it/s] 51%|█████     | 202471/400000 [00:23<00:22, 8855.21it/s] 51%|█████     | 203357/400000 [00:23<00:22, 8846.65it/s] 51%|█████     | 204242/400000 [00:23<00:22, 8846.71it/s] 51%|█████▏    | 205127/400000 [00:23<00:22, 8820.96it/s] 52%|█████▏    | 206014/400000 [00:23<00:21, 8834.49it/s] 52%|█████▏    | 206898/400000 [00:23<00:21, 8811.70it/s] 52%|█████▏    | 207787/400000 [00:23<00:21, 8834.92it/s] 52%|█████▏    | 208671/400000 [00:23<00:21, 8795.14it/s] 52%|█████▏    | 209551/400000 [00:23<00:21, 8770.44it/s] 53%|█████▎    | 210429/400000 [00:23<00:21, 8740.44it/s] 53%|█████▎    | 211304/400000 [00:24<00:21, 8738.18it/s] 53%|█████▎    | 212185/400000 [00:24<00:21, 8758.42it/s] 53%|█████▎    | 213061/400000 [00:24<00:21, 8664.57it/s] 53%|█████▎    | 213940/400000 [00:24<00:21, 8699.16it/s] 54%|█████▎    | 214811/400000 [00:24<00:21, 8634.04it/s] 54%|█████▍    | 215695/400000 [00:24<00:21, 8694.00it/s] 54%|█████▍    | 216579/400000 [00:24<00:20, 8736.49it/s] 54%|█████▍    | 217469/400000 [00:24<00:20, 8782.65it/s] 55%|█████▍    | 218356/400000 [00:24<00:20, 8808.05it/s] 55%|█████▍    | 219249/400000 [00:24<00:20, 8841.35it/s] 55%|█████▌    | 220134/400000 [00:25<00:20, 8697.41it/s] 55%|█████▌    | 221014/400000 [00:25<00:20, 8726.95it/s] 55%|█████▌    | 221890/400000 [00:25<00:20, 8734.04it/s] 56%|█████▌    | 222764/400000 [00:25<00:20, 8637.69it/s] 56%|█████▌    | 223651/400000 [00:25<00:20, 8704.00it/s] 56%|█████▌    | 224541/400000 [00:25<00:20, 8759.27it/s] 56%|█████▋    | 225426/400000 [00:25<00:19, 8783.43it/s] 57%|█████▋    | 226318/400000 [00:25<00:19, 8821.64it/s] 57%|█████▋    | 227205/400000 [00:25<00:19, 8835.78it/s] 57%|█████▋    | 228096/400000 [00:25<00:19, 8857.21it/s] 57%|█████▋    | 228982/400000 [00:26<00:19, 8805.71it/s] 57%|█████▋    | 229863/400000 [00:26<00:19, 8717.83it/s] 58%|█████▊    | 230741/400000 [00:26<00:19, 8734.86it/s] 58%|█████▊    | 231624/400000 [00:26<00:19, 8762.35it/s] 58%|█████▊    | 232501/400000 [00:26<00:19, 8762.54it/s] 58%|█████▊    | 233378/400000 [00:26<00:19, 8668.02it/s] 59%|█████▊    | 234268/400000 [00:26<00:18, 8735.14it/s] 59%|█████▉    | 235161/400000 [00:26<00:18, 8791.90it/s] 59%|█████▉    | 236050/400000 [00:26<00:18, 8820.81it/s] 59%|█████▉    | 236933/400000 [00:26<00:19, 8411.17it/s] 59%|█████▉    | 237786/400000 [00:27<00:19, 8444.52it/s] 60%|█████▉    | 238671/400000 [00:27<00:18, 8560.66it/s] 60%|█████▉    | 239561/400000 [00:27<00:18, 8657.89it/s] 60%|██████    | 240453/400000 [00:27<00:18, 8732.41it/s] 60%|██████    | 241328/400000 [00:27<00:18, 8735.91it/s] 61%|██████    | 242209/400000 [00:27<00:18, 8756.84it/s] 61%|██████    | 243095/400000 [00:27<00:17, 8784.76it/s] 61%|██████    | 243986/400000 [00:27<00:17, 8820.47it/s] 61%|██████    | 244872/400000 [00:27<00:17, 8832.10it/s] 61%|██████▏   | 245763/400000 [00:27<00:17, 8853.10it/s] 62%|██████▏   | 246654/400000 [00:28<00:17, 8868.80it/s] 62%|██████▏   | 247545/400000 [00:28<00:17, 8880.32it/s] 62%|██████▏   | 248434/400000 [00:28<00:17, 8840.37it/s] 62%|██████▏   | 249319/400000 [00:28<00:17, 8795.40it/s] 63%|██████▎   | 250199/400000 [00:28<00:17, 8713.36it/s] 63%|██████▎   | 251093/400000 [00:28<00:16, 8779.91it/s] 63%|██████▎   | 251975/400000 [00:28<00:16, 8791.09it/s] 63%|██████▎   | 252868/400000 [00:28<00:16, 8831.31it/s] 63%|██████▎   | 253758/400000 [00:28<00:16, 8851.15it/s] 64%|██████▎   | 254649/400000 [00:28<00:16, 8866.04it/s] 64%|██████▍   | 255541/400000 [00:29<00:16, 8879.95it/s] 64%|██████▍   | 256430/400000 [00:29<00:16, 8876.64it/s] 64%|██████▍   | 257318/400000 [00:29<00:16, 8813.22it/s] 65%|██████▍   | 258200/400000 [00:29<00:16, 8796.64it/s] 65%|██████▍   | 259080/400000 [00:29<00:16, 8781.52it/s] 65%|██████▍   | 259959/400000 [00:29<00:15, 8777.90it/s] 65%|██████▌   | 260845/400000 [00:29<00:15, 8800.95it/s] 65%|██████▌   | 261731/400000 [00:29<00:15, 8817.46it/s] 66%|██████▌   | 262617/400000 [00:29<00:15, 8829.54it/s] 66%|██████▌   | 263501/400000 [00:29<00:15, 8830.26it/s] 66%|██████▌   | 264385/400000 [00:30<00:15, 8712.37it/s] 66%|██████▋   | 265257/400000 [00:30<00:15, 8656.28it/s] 67%|██████▋   | 266128/400000 [00:30<00:15, 8670.80it/s] 67%|██████▋   | 267010/400000 [00:30<00:15, 8712.72it/s] 67%|██████▋   | 267882/400000 [00:30<00:15, 8650.94it/s] 67%|██████▋   | 268748/400000 [00:30<00:15, 8624.03it/s] 67%|██████▋   | 269623/400000 [00:30<00:15, 8659.32it/s] 68%|██████▊   | 270507/400000 [00:30<00:14, 8710.44it/s] 68%|██████▊   | 271393/400000 [00:30<00:14, 8753.70it/s] 68%|██████▊   | 272285/400000 [00:31<00:14, 8802.13it/s] 68%|██████▊   | 273167/400000 [00:31<00:14, 8806.36it/s] 69%|██████▊   | 274056/400000 [00:31<00:14, 8830.05it/s] 69%|██████▊   | 274940/400000 [00:31<00:14, 8790.31it/s] 69%|██████▉   | 275821/400000 [00:31<00:14, 8796.09it/s] 69%|██████▉   | 276713/400000 [00:31<00:13, 8829.98it/s] 69%|██████▉   | 277603/400000 [00:31<00:13, 8850.46it/s] 70%|██████▉   | 278494/400000 [00:31<00:13, 8866.68it/s] 70%|██████▉   | 279381/400000 [00:31<00:13, 8853.13it/s] 70%|███████   | 280273/400000 [00:31<00:13, 8872.63it/s] 70%|███████   | 281163/400000 [00:32<00:13, 8880.35it/s] 71%|███████   | 282053/400000 [00:32<00:13, 8883.40it/s] 71%|███████   | 282944/400000 [00:32<00:13, 8890.11it/s] 71%|███████   | 283834/400000 [00:32<00:13, 8873.30it/s] 71%|███████   | 284725/400000 [00:32<00:12, 8881.54it/s] 71%|███████▏  | 285614/400000 [00:32<00:12, 8858.93it/s] 72%|███████▏  | 286505/400000 [00:32<00:12, 8873.02it/s] 72%|███████▏  | 287393/400000 [00:32<00:12, 8867.31it/s] 72%|███████▏  | 288280/400000 [00:32<00:12, 8856.51it/s] 72%|███████▏  | 289171/400000 [00:32<00:12, 8869.74it/s] 73%|███████▎  | 290058/400000 [00:33<00:12, 8840.86it/s] 73%|███████▎  | 290943/400000 [00:33<00:12, 8830.12it/s] 73%|███████▎  | 291827/400000 [00:33<00:12, 8822.97it/s] 73%|███████▎  | 292710/400000 [00:33<00:12, 8821.17it/s] 73%|███████▎  | 293595/400000 [00:33<00:12, 8827.15it/s] 74%|███████▎  | 294484/400000 [00:33<00:11, 8845.80it/s] 74%|███████▍  | 295376/400000 [00:33<00:11, 8865.22it/s] 74%|███████▍  | 296267/400000 [00:33<00:11, 8877.64it/s] 74%|███████▍  | 297155/400000 [00:33<00:11, 8856.18it/s] 75%|███████▍  | 298041/400000 [00:33<00:11, 8853.96it/s] 75%|███████▍  | 298931/400000 [00:34<00:11, 8867.23it/s] 75%|███████▍  | 299818/400000 [00:34<00:11, 8867.24it/s] 75%|███████▌  | 300710/400000 [00:34<00:11, 8881.14it/s] 75%|███████▌  | 301599/400000 [00:34<00:11, 8746.09it/s] 76%|███████▌  | 302475/400000 [00:34<00:11, 8745.45it/s] 76%|███████▌  | 303350/400000 [00:34<00:11, 8735.44it/s] 76%|███████▌  | 304224/400000 [00:34<00:11, 8656.69it/s] 76%|███████▋  | 305112/400000 [00:34<00:10, 8721.55it/s] 76%|███████▋  | 305985/400000 [00:34<00:10, 8658.30it/s] 77%|███████▋  | 306868/400000 [00:34<00:10, 8707.58it/s] 77%|███████▋  | 307754/400000 [00:35<00:10, 8749.69it/s] 77%|███████▋  | 308640/400000 [00:35<00:10, 8781.72it/s] 77%|███████▋  | 309519/400000 [00:35<00:10, 8615.45it/s] 78%|███████▊  | 310382/400000 [00:35<00:10, 8292.91it/s] 78%|███████▊  | 311215/400000 [00:35<00:10, 8291.56it/s] 78%|███████▊  | 312104/400000 [00:35<00:10, 8461.59it/s] 78%|███████▊  | 312994/400000 [00:35<00:10, 8586.30it/s] 78%|███████▊  | 313885/400000 [00:35<00:09, 8678.53it/s] 79%|███████▊  | 314766/400000 [00:35<00:09, 8715.12it/s] 79%|███████▉  | 315657/400000 [00:35<00:09, 8771.51it/s] 79%|███████▉  | 316548/400000 [00:36<00:09, 8811.17it/s] 79%|███████▉  | 317440/400000 [00:36<00:09, 8841.36it/s] 80%|███████▉  | 318326/400000 [00:36<00:09, 8845.10it/s] 80%|███████▉  | 319211/400000 [00:36<00:09, 8838.69it/s] 80%|████████  | 320101/400000 [00:36<00:09, 8855.34it/s] 80%|████████  | 320987/400000 [00:36<00:08, 8851.26it/s] 80%|████████  | 321876/400000 [00:36<00:08, 8860.68it/s] 81%|████████  | 322763/400000 [00:36<00:08, 8861.24it/s] 81%|████████  | 323650/400000 [00:36<00:08, 8830.62it/s] 81%|████████  | 324539/400000 [00:36<00:08, 8846.79it/s] 81%|████████▏ | 325424/400000 [00:37<00:08, 8838.48it/s] 82%|████████▏ | 326312/400000 [00:37<00:08, 8848.47it/s] 82%|████████▏ | 327203/400000 [00:37<00:08, 8866.29it/s] 82%|████████▏ | 328090/400000 [00:37<00:08, 8807.75it/s] 82%|████████▏ | 328971/400000 [00:37<00:08, 8749.34it/s] 82%|████████▏ | 329859/400000 [00:37<00:07, 8785.48it/s] 83%|████████▎ | 330738/400000 [00:37<00:07, 8784.05it/s] 83%|████████▎ | 331628/400000 [00:37<00:07, 8818.13it/s] 83%|████████▎ | 332515/400000 [00:37<00:07, 8832.91it/s] 83%|████████▎ | 333405/400000 [00:37<00:07, 8851.56it/s] 84%|████████▎ | 334291/400000 [00:38<00:07, 8849.13it/s] 84%|████████▍ | 335176/400000 [00:38<00:07, 8841.45it/s] 84%|████████▍ | 336061/400000 [00:38<00:07, 8838.45it/s] 84%|████████▍ | 336947/400000 [00:38<00:07, 8843.69it/s] 84%|████████▍ | 337832/400000 [00:38<00:07, 8792.06it/s] 85%|████████▍ | 338724/400000 [00:38<00:06, 8829.48it/s] 85%|████████▍ | 339616/400000 [00:38<00:06, 8854.75it/s] 85%|████████▌ | 340508/400000 [00:38<00:06, 8874.00it/s] 85%|████████▌ | 341396/400000 [00:38<00:06, 8862.52it/s] 86%|████████▌ | 342285/400000 [00:38<00:06, 8869.26it/s] 86%|████████▌ | 343176/400000 [00:39<00:06, 8880.77it/s] 86%|████████▌ | 344068/400000 [00:39<00:06, 8890.48it/s] 86%|████████▌ | 344958/400000 [00:39<00:06, 8875.27it/s] 86%|████████▋ | 345846/400000 [00:39<00:06, 8788.45it/s] 87%|████████▋ | 346726/400000 [00:39<00:06, 8761.36it/s] 87%|████████▋ | 347616/400000 [00:39<00:05, 8799.75it/s] 87%|████████▋ | 348506/400000 [00:39<00:05, 8828.07it/s] 87%|████████▋ | 349391/400000 [00:39<00:05, 8832.82it/s] 88%|████████▊ | 350277/400000 [00:39<00:05, 8840.44it/s] 88%|████████▊ | 351168/400000 [00:39<00:05, 8859.13it/s] 88%|████████▊ | 352059/400000 [00:40<00:05, 8872.15it/s] 88%|████████▊ | 352947/400000 [00:40<00:05, 8850.23it/s] 88%|████████▊ | 353840/400000 [00:40<00:05, 8872.39it/s] 89%|████████▊ | 354729/400000 [00:40<00:05, 8874.96it/s] 89%|████████▉ | 355617/400000 [00:40<00:05, 8835.85it/s] 89%|████████▉ | 356507/400000 [00:40<00:04, 8853.55it/s] 89%|████████▉ | 357399/400000 [00:40<00:04, 8872.60it/s] 90%|████████▉ | 358287/400000 [00:40<00:04, 8790.57it/s] 90%|████████▉ | 359173/400000 [00:40<00:04, 8809.45it/s] 90%|█████████ | 360055/400000 [00:40<00:04, 8501.74it/s] 90%|█████████ | 360908/400000 [00:41<00:04, 8483.04it/s] 90%|█████████ | 361799/400000 [00:41<00:04, 8604.48it/s] 91%|█████████ | 362692/400000 [00:41<00:04, 8696.78it/s] 91%|█████████ | 363576/400000 [00:41<00:04, 8737.33it/s] 91%|█████████ | 364451/400000 [00:41<00:04, 8593.39it/s] 91%|█████████▏| 365327/400000 [00:41<00:04, 8640.26it/s] 92%|█████████▏| 366192/400000 [00:41<00:03, 8585.59it/s] 92%|█████████▏| 367081/400000 [00:41<00:03, 8672.87it/s] 92%|█████████▏| 367970/400000 [00:41<00:03, 8736.23it/s] 92%|█████████▏| 368845/400000 [00:41<00:03, 8546.75it/s] 92%|█████████▏| 369730/400000 [00:42<00:03, 8633.48it/s] 93%|█████████▎| 370616/400000 [00:42<00:03, 8700.07it/s] 93%|█████████▎| 371498/400000 [00:42<00:03, 8734.06it/s] 93%|█████████▎| 372380/400000 [00:42<00:03, 8757.94it/s] 93%|█████████▎| 373257/400000 [00:42<00:03, 8749.60it/s] 94%|█████████▎| 374148/400000 [00:42<00:02, 8796.69it/s] 94%|█████████▍| 375037/400000 [00:42<00:02, 8823.89it/s] 94%|█████████▍| 375929/400000 [00:42<00:02, 8850.54it/s] 94%|█████████▍| 376820/400000 [00:42<00:02, 8867.25it/s] 94%|█████████▍| 377713/400000 [00:43<00:02, 8885.06it/s] 95%|█████████▍| 378602/400000 [00:43<00:02, 8850.94it/s] 95%|█████████▍| 379488/400000 [00:43<00:02, 8785.57it/s] 95%|█████████▌| 380373/400000 [00:43<00:02, 8803.81it/s] 95%|█████████▌| 381254/400000 [00:43<00:02, 8800.63it/s] 96%|█████████▌| 382138/400000 [00:43<00:02, 8809.86it/s] 96%|█████████▌| 383029/400000 [00:43<00:01, 8837.36it/s] 96%|█████████▌| 383918/400000 [00:43<00:01, 8849.64it/s] 96%|█████████▌| 384807/400000 [00:43<00:01, 8860.46it/s] 96%|█████████▋| 385695/400000 [00:43<00:01, 8863.81it/s] 97%|█████████▋| 386582/400000 [00:44<00:01, 8843.86it/s] 97%|█████████▋| 387476/400000 [00:44<00:01, 8870.76it/s] 97%|█████████▋| 388365/400000 [00:44<00:01, 8874.63it/s] 97%|█████████▋| 389253/400000 [00:44<00:01, 8776.85it/s] 98%|█████████▊| 390131/400000 [00:44<00:01, 8772.76it/s] 98%|█████████▊| 391017/400000 [00:44<00:01, 8797.38it/s] 98%|█████████▊| 391899/400000 [00:44<00:00, 8803.90it/s] 98%|█████████▊| 392790/400000 [00:44<00:00, 8834.65it/s] 98%|█████████▊| 393678/400000 [00:44<00:00, 8846.09it/s] 99%|█████████▊| 394563/400000 [00:44<00:00, 8836.61it/s] 99%|█████████▉| 395447/400000 [00:45<00:00, 8784.99it/s] 99%|█████████▉| 396326/400000 [00:45<00:00, 8691.19it/s] 99%|█████████▉| 397219/400000 [00:45<00:00, 8758.67it/s]100%|█████████▉| 398096/400000 [00:45<00:00, 8642.62it/s]100%|█████████▉| 398984/400000 [00:45<00:00, 8709.88it/s]100%|█████████▉| 399870/400000 [00:45<00:00, 8754.21it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8784.16it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fcd659dda58> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011334212979192557 	 Accuracy: 49
Train Epoch: 1 	 Loss: 0.011072814065875815 	 Accuracy: 70

  model saves at 70% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15965 out of table with 15806 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15965 out of table with 15806 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-13 18:24:39.551841: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 18:24:39.556915: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-13 18:24:39.557072: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56518fb30e80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 18:24:39.557086: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fcd18f10160> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 8.2646 - accuracy: 0.4610
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8276 - accuracy: 0.4895 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7075 - accuracy: 0.4973
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6590 - accuracy: 0.5005
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6973 - accuracy: 0.4980
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7101 - accuracy: 0.4972
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7805 - accuracy: 0.4926
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6992 - accuracy: 0.4979
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6956 - accuracy: 0.4981
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7234 - accuracy: 0.4963
11000/25000 [============>.................] - ETA: 3s - loss: 7.7266 - accuracy: 0.4961
12000/25000 [=============>................] - ETA: 3s - loss: 7.7216 - accuracy: 0.4964
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7315 - accuracy: 0.4958
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7499 - accuracy: 0.4946
15000/25000 [=================>............] - ETA: 2s - loss: 7.7310 - accuracy: 0.4958
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7567 - accuracy: 0.4941
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7379 - accuracy: 0.4954
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7288 - accuracy: 0.4959
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7263 - accuracy: 0.4961
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7142 - accuracy: 0.4969
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6849 - accuracy: 0.4988
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6820 - accuracy: 0.4990
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6753 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6737 - accuracy: 0.4995
25000/25000 [==============================] - 7s 276us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fcce18d4f60> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fcd659dd4a8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4058 - crf_viterbi_accuracy: 0.2800 - val_loss: 1.3109 - val_crf_viterbi_accuracy: 0.3333

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

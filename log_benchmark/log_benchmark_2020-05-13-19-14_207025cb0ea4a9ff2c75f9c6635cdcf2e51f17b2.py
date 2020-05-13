
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7efd808eef60> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 19:14:17.266010
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 19:14:17.270598
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 19:14:17.273824
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 19:14:17.276483
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7efd8c6b8438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354104.8750
Epoch 2/10

1/1 [==============================] - 0s 95ms/step - loss: 261312.2812
Epoch 3/10

1/1 [==============================] - 0s 89ms/step - loss: 164586.1562
Epoch 4/10

1/1 [==============================] - 0s 87ms/step - loss: 86140.3125
Epoch 5/10

1/1 [==============================] - 0s 90ms/step - loss: 45199.5625
Epoch 6/10

1/1 [==============================] - 0s 89ms/step - loss: 26214.8945
Epoch 7/10

1/1 [==============================] - 0s 90ms/step - loss: 16679.2031
Epoch 8/10

1/1 [==============================] - 0s 95ms/step - loss: 11469.2051
Epoch 9/10

1/1 [==============================] - 0s 89ms/step - loss: 8395.9990
Epoch 10/10

1/1 [==============================] - 0s 90ms/step - loss: 6489.0098

  #### Inference Need return ypred, ytrue ######################### 
[[ 6.83317900e-01  1.75679922e+00 -1.55669630e+00  9.24552321e-01
  -2.01644278e+00 -4.54967260e-01 -7.28847563e-01  1.20357800e+00
   7.39016652e-01 -1.27394629e+00  5.07136881e-02 -8.07379842e-01
   5.58716178e-01 -1.03376412e+00 -5.36150396e-01 -2.06516337e+00
   4.25610572e-01 -1.05503035e+00  1.47566843e+00  1.16622686e+00
  -5.87619841e-02 -8.18688989e-01  5.22399664e-01 -2.14725539e-01
  -2.65506595e-01 -2.06751496e-01 -3.80910933e-01 -6.74267411e-01
   1.05117762e+00 -4.41947877e-01  2.61185765e-02 -5.20748258e-01
   1.37549567e+00  6.18685603e-01 -4.41696048e-01  1.18970668e+00
  -7.36548901e-01 -6.96317792e-01 -8.11764657e-01 -1.63172692e-01
  -9.06549275e-01  5.95439553e-01  6.53090835e-01 -1.23834610e-03
  -2.81127125e-01 -6.73449576e-01 -1.25703216e+00 -4.15717900e-01
   1.99554384e-01  3.77250075e-01 -8.24958444e-01 -1.13441920e+00
   3.25398326e-01  6.19008899e-01 -5.79224408e-01  9.99481440e-01
   5.64036369e-01  5.44109821e-01  3.91153395e-01 -2.54919201e-01
  -3.74458104e-01  7.63889980e+00  8.77571201e+00  9.29912376e+00
   6.96337843e+00  7.95930338e+00  8.65139389e+00  7.95561886e+00
   8.08093452e+00  5.82707834e+00  6.96819782e+00  7.86273670e+00
   6.89744663e+00  6.26150751e+00  8.06150246e+00  7.66215754e+00
   7.70212221e+00  7.36011696e+00  7.72847939e+00  9.08303547e+00
   9.50693607e+00  5.94374180e+00  7.51667786e+00  7.88932562e+00
   7.80661297e+00  7.22852707e+00  7.07443523e+00  7.33031082e+00
   9.30538750e+00  7.24606609e+00  1.03362570e+01  7.16448355e+00
   9.04213715e+00  8.38178349e+00  6.99272633e+00  9.96827221e+00
   7.30838299e+00  7.86108780e+00  7.99552679e+00  8.60913277e+00
   9.86876583e+00  8.13831520e+00  8.64204693e+00  9.19756126e+00
   8.42220879e+00  1.02370319e+01  8.16383362e+00  7.69799614e+00
   7.86795616e+00  7.64987707e+00  6.69208813e+00  7.07158995e+00
   8.04369354e+00  8.48270798e+00  6.67093515e+00  7.23910952e+00
   7.33295631e+00  8.00070000e+00  9.17763519e+00  9.27370262e+00
   4.86877531e-01 -9.51757431e-01 -1.07049614e-01  3.23145054e-02
   1.08603001e+00 -1.63377261e+00 -3.31318378e-02 -9.48269725e-01
  -8.80827725e-01  1.31644773e+00  1.76348388e-02  1.43301055e-01
  -1.69760036e+00 -6.63743973e-01 -6.98782921e-01 -1.39806283e+00
  -6.96159601e-02  9.04538989e-01  2.12087691e-01  8.82110178e-01
  -4.21805531e-01 -1.15488738e-01 -1.90875554e+00  2.45917544e-01
  -4.96869028e-01  1.51506257e+00 -5.28195798e-01  8.93611014e-02
   1.67285693e+00 -1.54215872e+00 -4.56735849e-01 -2.69158959e-01
   3.47399324e-01 -4.65455711e-01  2.87743866e-01 -2.16732591e-01
  -8.61657381e-01 -1.16817522e+00 -1.07060742e+00 -5.78368127e-01
   7.23382771e-01 -1.14201236e+00  9.82059002e-01 -1.71754217e+00
   1.17749155e+00 -1.32505643e+00  4.21767563e-01  1.62400335e-01
   3.64451826e-01 -7.74797201e-01  9.77892995e-01 -2.09023029e-01
   4.09747064e-01 -1.57898411e-01 -1.49541664e+00 -2.68403798e-01
   2.23984718e+00  9.68783736e-01  1.33707583e-01  5.63088655e-01
   1.20698440e+00  3.34484935e-01  6.48379445e-01  1.22561431e+00
   1.64409614e+00  4.27434623e-01  7.16607928e-01  5.63158870e-01
   1.35929513e+00  7.89426804e-01  2.75216913e+00  1.58310127e+00
   2.26998711e+00  1.53800297e+00  6.80188000e-01  2.82960987e+00
   7.85238028e-01  1.23162353e+00  1.61998272e+00  5.28137147e-01
   7.84391105e-01  1.38949192e+00  5.51372886e-01  9.00842190e-01
   2.03884506e+00  4.29989636e-01  3.10392261e-01  1.55863440e+00
   2.49377155e+00  9.88987684e-01  1.27896249e+00  2.37456417e+00
   2.38721919e+00  6.92683518e-01  4.58037853e-01  4.03184772e-01
   2.01622963e-01  6.54882550e-01  2.08192170e-01  1.13284588e+00
   5.89189410e-01  5.03180206e-01  1.62905037e+00  2.39584541e+00
   9.15265501e-01  1.64350998e+00  5.49300909e-01  1.24533486e+00
   1.95721304e+00  7.89752007e-01  6.40297890e-01  1.47693872e+00
   2.49564910e+00  2.69785643e-01  5.81358731e-01  4.89122391e-01
   3.90466273e-01  1.16584718e+00  4.44663346e-01  5.32718301e-01
   1.86044753e-01  8.63717365e+00  8.12194920e+00  9.10631180e+00
   9.33441353e+00  8.47955132e+00  8.94963169e+00  8.65576077e+00
   8.69920826e+00  8.04378319e+00  8.69698811e+00  8.76648140e+00
   1.01386385e+01  8.58104610e+00  7.24908161e+00  9.00592136e+00
   7.67718029e+00  9.69623184e+00  7.59421062e+00  7.61938334e+00
   1.04273796e+01  9.22193336e+00  8.40590668e+00  1.05391140e+01
   7.87146997e+00  7.38370132e+00  7.54108381e+00  9.07727242e+00
   7.26709843e+00  7.93849993e+00  9.04796600e+00  9.19757271e+00
   9.14445210e+00  7.64281321e+00  8.25776577e+00  8.99352646e+00
   7.18204546e+00  8.32051277e+00  7.56754494e+00  7.46708107e+00
   8.61787510e+00  8.63728333e+00  8.01841450e+00  7.41318083e+00
   8.34676170e+00  8.52324295e+00  9.18044090e+00  9.37274170e+00
   8.39456749e+00  1.02986507e+01  7.02476215e+00  7.59846544e+00
   7.27222204e+00  8.28575325e+00  9.37087250e+00  8.90515041e+00
   9.64919662e+00  7.17532873e+00  1.00485840e+01  7.84395552e+00
   3.48396361e-01  7.57004619e-01  6.32301927e-01  5.75820148e-01
   1.54201508e-01  8.48101020e-01  7.34406114e-01  1.60376906e+00
   2.92296171e-01  3.12691689e-01  2.91260123e-01  4.61859703e-01
   8.12932789e-01  1.83825910e-01  3.89385641e-01  1.48647249e-01
   2.25995874e+00  3.58504868e+00  1.05814743e+00  6.28070295e-01
   2.19598770e-01  1.61520839e+00  1.33224154e+00  3.48085999e-01
   1.05406809e+00  5.48809111e-01  7.24601805e-01  1.36548209e+00
   5.36574543e-01  6.26713276e-01  1.43684149e-01  2.43531644e-01
   1.72964382e+00  1.09784484e+00  1.38345015e+00  1.03197205e+00
   6.15496159e-01  6.12476707e-01  1.33162856e+00  2.40213418e+00
   8.01381171e-01  8.61906350e-01  1.78114486e+00  4.50955749e-01
   2.47718191e+00  2.72743225e+00  3.87215674e-01  8.90008152e-01
   1.09401274e+00  8.73124361e-01  4.12362695e-01  2.33562851e+00
   1.20122981e+00  1.16848814e+00  8.05160940e-01  2.25914860e+00
   2.51618385e+00  2.23751736e+00  7.29518771e-01  2.17949104e+00
  -7.26575375e+00  2.78779459e+00 -6.53297853e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 19:14:25.714561
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.4952
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 19:14:25.718240
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8950.75
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 19:14:25.720965
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.2362
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 19:14:25.723587
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -800.603
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139626889500096
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139625679770120
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139625679770624
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139625679771128
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139625679771632
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139625679772136

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7efd6c2c9e10> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.726651
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.697298
grad_step = 000002, loss = 0.672440
grad_step = 000003, loss = 0.644842
grad_step = 000004, loss = 0.615638
grad_step = 000005, loss = 0.588072
grad_step = 000006, loss = 0.564947
grad_step = 000007, loss = 0.541890
grad_step = 000008, loss = 0.525063
grad_step = 000009, loss = 0.509744
grad_step = 000010, loss = 0.491964
grad_step = 000011, loss = 0.476319
grad_step = 000012, loss = 0.462818
grad_step = 000013, loss = 0.450151
grad_step = 000014, loss = 0.437010
grad_step = 000015, loss = 0.422306
grad_step = 000016, loss = 0.407543
grad_step = 000017, loss = 0.393301
grad_step = 000018, loss = 0.379385
grad_step = 000019, loss = 0.365749
grad_step = 000020, loss = 0.351895
grad_step = 000021, loss = 0.337820
grad_step = 000022, loss = 0.324631
grad_step = 000023, loss = 0.312407
grad_step = 000024, loss = 0.300200
grad_step = 000025, loss = 0.287990
grad_step = 000026, loss = 0.275832
grad_step = 000027, loss = 0.263340
grad_step = 000028, loss = 0.251487
grad_step = 000029, loss = 0.240281
grad_step = 000030, loss = 0.229249
grad_step = 000031, loss = 0.218717
grad_step = 000032, loss = 0.208320
grad_step = 000033, loss = 0.198131
grad_step = 000034, loss = 0.188731
grad_step = 000035, loss = 0.179455
grad_step = 000036, loss = 0.170497
grad_step = 000037, loss = 0.161852
grad_step = 000038, loss = 0.153584
grad_step = 000039, loss = 0.145770
grad_step = 000040, loss = 0.138057
grad_step = 000041, loss = 0.130775
grad_step = 000042, loss = 0.123916
grad_step = 000043, loss = 0.117362
grad_step = 000044, loss = 0.111184
grad_step = 000045, loss = 0.105274
grad_step = 000046, loss = 0.099655
grad_step = 000047, loss = 0.094262
grad_step = 000048, loss = 0.089209
grad_step = 000049, loss = 0.084332
grad_step = 000050, loss = 0.079744
grad_step = 000051, loss = 0.075405
grad_step = 000052, loss = 0.071321
grad_step = 000053, loss = 0.067432
grad_step = 000054, loss = 0.063751
grad_step = 000055, loss = 0.060268
grad_step = 000056, loss = 0.056981
grad_step = 000057, loss = 0.053803
grad_step = 000058, loss = 0.050786
grad_step = 000059, loss = 0.047957
grad_step = 000060, loss = 0.045287
grad_step = 000061, loss = 0.042748
grad_step = 000062, loss = 0.040343
grad_step = 000063, loss = 0.038047
grad_step = 000064, loss = 0.035888
grad_step = 000065, loss = 0.033822
grad_step = 000066, loss = 0.031872
grad_step = 000067, loss = 0.030021
grad_step = 000068, loss = 0.028275
grad_step = 000069, loss = 0.026626
grad_step = 000070, loss = 0.025060
grad_step = 000071, loss = 0.023588
grad_step = 000072, loss = 0.022199
grad_step = 000073, loss = 0.020885
grad_step = 000074, loss = 0.019641
grad_step = 000075, loss = 0.018472
grad_step = 000076, loss = 0.017368
grad_step = 000077, loss = 0.016329
grad_step = 000078, loss = 0.015357
grad_step = 000079, loss = 0.014443
grad_step = 000080, loss = 0.013581
grad_step = 000081, loss = 0.012770
grad_step = 000082, loss = 0.012012
grad_step = 000083, loss = 0.011302
grad_step = 000084, loss = 0.010635
grad_step = 000085, loss = 0.010015
grad_step = 000086, loss = 0.009436
grad_step = 000087, loss = 0.008894
grad_step = 000088, loss = 0.008388
grad_step = 000089, loss = 0.007916
grad_step = 000090, loss = 0.007477
grad_step = 000091, loss = 0.007067
grad_step = 000092, loss = 0.006686
grad_step = 000093, loss = 0.006332
grad_step = 000094, loss = 0.006003
grad_step = 000095, loss = 0.005696
grad_step = 000096, loss = 0.005412
grad_step = 000097, loss = 0.005147
grad_step = 000098, loss = 0.004902
grad_step = 000099, loss = 0.004674
grad_step = 000100, loss = 0.004464
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.004268
grad_step = 000102, loss = 0.004088
grad_step = 000103, loss = 0.003921
grad_step = 000104, loss = 0.003767
grad_step = 000105, loss = 0.003624
grad_step = 000106, loss = 0.003493
grad_step = 000107, loss = 0.003372
grad_step = 000108, loss = 0.003261
grad_step = 000109, loss = 0.003159
grad_step = 000110, loss = 0.003065
grad_step = 000111, loss = 0.002979
grad_step = 000112, loss = 0.002900
grad_step = 000113, loss = 0.002828
grad_step = 000114, loss = 0.002763
grad_step = 000115, loss = 0.002703
grad_step = 000116, loss = 0.002649
grad_step = 000117, loss = 0.002600
grad_step = 000118, loss = 0.002555
grad_step = 000119, loss = 0.002515
grad_step = 000120, loss = 0.002479
grad_step = 000121, loss = 0.002446
grad_step = 000122, loss = 0.002416
grad_step = 000123, loss = 0.002390
grad_step = 000124, loss = 0.002366
grad_step = 000125, loss = 0.002344
grad_step = 000126, loss = 0.002325
grad_step = 000127, loss = 0.002307
grad_step = 000128, loss = 0.002292
grad_step = 000129, loss = 0.002278
grad_step = 000130, loss = 0.002266
grad_step = 000131, loss = 0.002254
grad_step = 000132, loss = 0.002245
grad_step = 000133, loss = 0.002236
grad_step = 000134, loss = 0.002228
grad_step = 000135, loss = 0.002221
grad_step = 000136, loss = 0.002214
grad_step = 000137, loss = 0.002208
grad_step = 000138, loss = 0.002203
grad_step = 000139, loss = 0.002198
grad_step = 000140, loss = 0.002193
grad_step = 000141, loss = 0.002189
grad_step = 000142, loss = 0.002185
grad_step = 000143, loss = 0.002182
grad_step = 000144, loss = 0.002179
grad_step = 000145, loss = 0.002177
grad_step = 000146, loss = 0.002176
grad_step = 000147, loss = 0.002178
grad_step = 000148, loss = 0.002188
grad_step = 000149, loss = 0.002207
grad_step = 000150, loss = 0.002231
grad_step = 000151, loss = 0.002254
grad_step = 000152, loss = 0.002255
grad_step = 000153, loss = 0.002230
grad_step = 000154, loss = 0.002190
grad_step = 000155, loss = 0.002158
grad_step = 000156, loss = 0.002153
grad_step = 000157, loss = 0.002169
grad_step = 000158, loss = 0.002189
grad_step = 000159, loss = 0.002196
grad_step = 000160, loss = 0.002187
grad_step = 000161, loss = 0.002165
grad_step = 000162, loss = 0.002145
grad_step = 000163, loss = 0.002138
grad_step = 000164, loss = 0.002142
grad_step = 000165, loss = 0.002152
grad_step = 000166, loss = 0.002160
grad_step = 000167, loss = 0.002160
grad_step = 000168, loss = 0.002153
grad_step = 000169, loss = 0.002144
grad_step = 000170, loss = 0.002134
grad_step = 000171, loss = 0.002126
grad_step = 000172, loss = 0.002121
grad_step = 000173, loss = 0.002119
grad_step = 000174, loss = 0.002119
grad_step = 000175, loss = 0.002120
grad_step = 000176, loss = 0.002123
grad_step = 000177, loss = 0.002126
grad_step = 000178, loss = 0.002132
grad_step = 000179, loss = 0.002140
grad_step = 000180, loss = 0.002152
grad_step = 000181, loss = 0.002170
grad_step = 000182, loss = 0.002193
grad_step = 000183, loss = 0.002219
grad_step = 000184, loss = 0.002240
grad_step = 000185, loss = 0.002244
grad_step = 000186, loss = 0.002220
grad_step = 000187, loss = 0.002173
grad_step = 000188, loss = 0.002122
grad_step = 000189, loss = 0.002091
grad_step = 000190, loss = 0.002091
grad_step = 000191, loss = 0.002111
grad_step = 000192, loss = 0.002137
grad_step = 000193, loss = 0.002152
grad_step = 000194, loss = 0.002149
grad_step = 000195, loss = 0.002130
grad_step = 000196, loss = 0.002104
grad_step = 000197, loss = 0.002082
grad_step = 000198, loss = 0.002073
grad_step = 000199, loss = 0.002075
grad_step = 000200, loss = 0.002085
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002097
grad_step = 000202, loss = 0.002105
grad_step = 000203, loss = 0.002109
grad_step = 000204, loss = 0.002106
grad_step = 000205, loss = 0.002099
grad_step = 000206, loss = 0.002088
grad_step = 000207, loss = 0.002077
grad_step = 000208, loss = 0.002067
grad_step = 000209, loss = 0.002059
grad_step = 000210, loss = 0.002054
grad_step = 000211, loss = 0.002051
grad_step = 000212, loss = 0.002049
grad_step = 000213, loss = 0.002048
grad_step = 000214, loss = 0.002048
grad_step = 000215, loss = 0.002050
grad_step = 000216, loss = 0.002054
grad_step = 000217, loss = 0.002064
grad_step = 000218, loss = 0.002084
grad_step = 000219, loss = 0.002122
grad_step = 000220, loss = 0.002197
grad_step = 000221, loss = 0.002315
grad_step = 000222, loss = 0.002493
grad_step = 000223, loss = 0.002589
grad_step = 000224, loss = 0.002509
grad_step = 000225, loss = 0.002221
grad_step = 000226, loss = 0.002035
grad_step = 000227, loss = 0.002109
grad_step = 000228, loss = 0.002277
grad_step = 000229, loss = 0.002283
grad_step = 000230, loss = 0.002112
grad_step = 000231, loss = 0.002025
grad_step = 000232, loss = 0.002111
grad_step = 000233, loss = 0.002193
grad_step = 000234, loss = 0.002136
grad_step = 000235, loss = 0.002031
grad_step = 000236, loss = 0.002035
grad_step = 000237, loss = 0.002110
grad_step = 000238, loss = 0.002117
grad_step = 000239, loss = 0.002051
grad_step = 000240, loss = 0.002011
grad_step = 000241, loss = 0.002042
grad_step = 000242, loss = 0.002080
grad_step = 000243, loss = 0.002059
grad_step = 000244, loss = 0.002015
grad_step = 000245, loss = 0.002007
grad_step = 000246, loss = 0.002034
grad_step = 000247, loss = 0.002047
grad_step = 000248, loss = 0.002026
grad_step = 000249, loss = 0.002002
grad_step = 000250, loss = 0.002002
grad_step = 000251, loss = 0.002018
grad_step = 000252, loss = 0.002024
grad_step = 000253, loss = 0.002010
grad_step = 000254, loss = 0.001994
grad_step = 000255, loss = 0.001992
grad_step = 000256, loss = 0.002001
grad_step = 000257, loss = 0.002007
grad_step = 000258, loss = 0.002001
grad_step = 000259, loss = 0.001990
grad_step = 000260, loss = 0.001984
grad_step = 000261, loss = 0.001985
grad_step = 000262, loss = 0.001990
grad_step = 000263, loss = 0.001991
grad_step = 000264, loss = 0.001988
grad_step = 000265, loss = 0.001984
grad_step = 000266, loss = 0.001979
grad_step = 000267, loss = 0.001975
grad_step = 000268, loss = 0.001973
grad_step = 000269, loss = 0.001973
grad_step = 000270, loss = 0.001973
grad_step = 000271, loss = 0.001974
grad_step = 000272, loss = 0.001974
grad_step = 000273, loss = 0.001974
grad_step = 000274, loss = 0.001972
grad_step = 000275, loss = 0.001970
grad_step = 000276, loss = 0.001968
grad_step = 000277, loss = 0.001965
grad_step = 000278, loss = 0.001963
grad_step = 000279, loss = 0.001961
grad_step = 000280, loss = 0.001959
grad_step = 000281, loss = 0.001958
grad_step = 000282, loss = 0.001957
grad_step = 000283, loss = 0.001956
grad_step = 000284, loss = 0.001957
grad_step = 000285, loss = 0.001959
grad_step = 000286, loss = 0.001963
grad_step = 000287, loss = 0.001973
grad_step = 000288, loss = 0.001993
grad_step = 000289, loss = 0.002036
grad_step = 000290, loss = 0.002112
grad_step = 000291, loss = 0.002260
grad_step = 000292, loss = 0.002419
grad_step = 000293, loss = 0.002590
grad_step = 000294, loss = 0.002496
grad_step = 000295, loss = 0.002236
grad_step = 000296, loss = 0.001976
grad_step = 000297, loss = 0.001984
grad_step = 000298, loss = 0.002174
grad_step = 000299, loss = 0.002236
grad_step = 000300, loss = 0.002092
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001945
grad_step = 000302, loss = 0.001987
grad_step = 000303, loss = 0.002103
grad_step = 000304, loss = 0.002088
grad_step = 000305, loss = 0.001975
grad_step = 000306, loss = 0.001930
grad_step = 000307, loss = 0.001996
grad_step = 000308, loss = 0.002051
grad_step = 000309, loss = 0.002003
grad_step = 000310, loss = 0.001933
grad_step = 000311, loss = 0.001932
grad_step = 000312, loss = 0.001979
grad_step = 000313, loss = 0.001995
grad_step = 000314, loss = 0.001953
grad_step = 000315, loss = 0.001918
grad_step = 000316, loss = 0.001930
grad_step = 000317, loss = 0.001958
grad_step = 000318, loss = 0.001957
grad_step = 000319, loss = 0.001928
grad_step = 000320, loss = 0.001911
grad_step = 000321, loss = 0.001921
grad_step = 000322, loss = 0.001937
grad_step = 000323, loss = 0.001935
grad_step = 000324, loss = 0.001917
grad_step = 000325, loss = 0.001905
grad_step = 000326, loss = 0.001908
grad_step = 000327, loss = 0.001918
grad_step = 000328, loss = 0.001919
grad_step = 000329, loss = 0.001910
grad_step = 000330, loss = 0.001900
grad_step = 000331, loss = 0.001897
grad_step = 000332, loss = 0.001902
grad_step = 000333, loss = 0.001905
grad_step = 000334, loss = 0.001903
grad_step = 000335, loss = 0.001897
grad_step = 000336, loss = 0.001891
grad_step = 000337, loss = 0.001889
grad_step = 000338, loss = 0.001891
grad_step = 000339, loss = 0.001893
grad_step = 000340, loss = 0.001892
grad_step = 000341, loss = 0.001889
grad_step = 000342, loss = 0.001885
grad_step = 000343, loss = 0.001881
grad_step = 000344, loss = 0.001880
grad_step = 000345, loss = 0.001880
grad_step = 000346, loss = 0.001880
grad_step = 000347, loss = 0.001880
grad_step = 000348, loss = 0.001878
grad_step = 000349, loss = 0.001876
grad_step = 000350, loss = 0.001873
grad_step = 000351, loss = 0.001871
grad_step = 000352, loss = 0.001869
grad_step = 000353, loss = 0.001867
grad_step = 000354, loss = 0.001866
grad_step = 000355, loss = 0.001865
grad_step = 000356, loss = 0.001865
grad_step = 000357, loss = 0.001864
grad_step = 000358, loss = 0.001864
grad_step = 000359, loss = 0.001863
grad_step = 000360, loss = 0.001863
grad_step = 000361, loss = 0.001864
grad_step = 000362, loss = 0.001865
grad_step = 000363, loss = 0.001867
grad_step = 000364, loss = 0.001871
grad_step = 000365, loss = 0.001878
grad_step = 000366, loss = 0.001892
grad_step = 000367, loss = 0.001914
grad_step = 000368, loss = 0.001953
grad_step = 000369, loss = 0.002006
grad_step = 000370, loss = 0.002091
grad_step = 000371, loss = 0.002166
grad_step = 000372, loss = 0.002237
grad_step = 000373, loss = 0.002186
grad_step = 000374, loss = 0.002061
grad_step = 000375, loss = 0.001905
grad_step = 000376, loss = 0.001841
grad_step = 000377, loss = 0.001886
grad_step = 000378, loss = 0.001975
grad_step = 000379, loss = 0.002027
grad_step = 000380, loss = 0.001988
grad_step = 000381, loss = 0.001906
grad_step = 000382, loss = 0.001840
grad_step = 000383, loss = 0.001838
grad_step = 000384, loss = 0.001882
grad_step = 000385, loss = 0.001921
grad_step = 000386, loss = 0.001925
grad_step = 000387, loss = 0.001886
grad_step = 000388, loss = 0.001841
grad_step = 000389, loss = 0.001821
grad_step = 000390, loss = 0.001832
grad_step = 000391, loss = 0.001856
grad_step = 000392, loss = 0.001870
grad_step = 000393, loss = 0.001866
grad_step = 000394, loss = 0.001844
grad_step = 000395, loss = 0.001821
grad_step = 000396, loss = 0.001809
grad_step = 000397, loss = 0.001813
grad_step = 000398, loss = 0.001824
grad_step = 000399, loss = 0.001832
grad_step = 000400, loss = 0.001833
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001825
grad_step = 000402, loss = 0.001813
grad_step = 000403, loss = 0.001802
grad_step = 000404, loss = 0.001796
grad_step = 000405, loss = 0.001795
grad_step = 000406, loss = 0.001797
grad_step = 000407, loss = 0.001801
grad_step = 000408, loss = 0.001804
grad_step = 000409, loss = 0.001805
grad_step = 000410, loss = 0.001803
grad_step = 000411, loss = 0.001801
grad_step = 000412, loss = 0.001796
grad_step = 000413, loss = 0.001791
grad_step = 000414, loss = 0.001786
grad_step = 000415, loss = 0.001781
grad_step = 000416, loss = 0.001777
grad_step = 000417, loss = 0.001773
grad_step = 000418, loss = 0.001770
grad_step = 000419, loss = 0.001768
grad_step = 000420, loss = 0.001765
grad_step = 000421, loss = 0.001763
grad_step = 000422, loss = 0.001761
grad_step = 000423, loss = 0.001759
grad_step = 000424, loss = 0.001758
grad_step = 000425, loss = 0.001756
grad_step = 000426, loss = 0.001754
grad_step = 000427, loss = 0.001754
grad_step = 000428, loss = 0.001754
grad_step = 000429, loss = 0.001757
grad_step = 000430, loss = 0.001767
grad_step = 000431, loss = 0.001790
grad_step = 000432, loss = 0.001844
grad_step = 000433, loss = 0.001947
grad_step = 000434, loss = 0.002168
grad_step = 000435, loss = 0.002446
grad_step = 000436, loss = 0.002822
grad_step = 000437, loss = 0.002817
grad_step = 000438, loss = 0.002466
grad_step = 000439, loss = 0.001884
grad_step = 000440, loss = 0.001784
grad_step = 000441, loss = 0.002139
grad_step = 000442, loss = 0.002285
grad_step = 000443, loss = 0.002001
grad_step = 000444, loss = 0.001744
grad_step = 000445, loss = 0.001872
grad_step = 000446, loss = 0.002049
grad_step = 000447, loss = 0.001949
grad_step = 000448, loss = 0.001790
grad_step = 000449, loss = 0.001783
grad_step = 000450, loss = 0.001877
grad_step = 000451, loss = 0.001900
grad_step = 000452, loss = 0.001784
grad_step = 000453, loss = 0.001736
grad_step = 000454, loss = 0.001811
grad_step = 000455, loss = 0.001836
grad_step = 000456, loss = 0.001750
grad_step = 000457, loss = 0.001716
grad_step = 000458, loss = 0.001781
grad_step = 000459, loss = 0.001792
grad_step = 000460, loss = 0.001720
grad_step = 000461, loss = 0.001715
grad_step = 000462, loss = 0.001760
grad_step = 000463, loss = 0.001750
grad_step = 000464, loss = 0.001707
grad_step = 000465, loss = 0.001709
grad_step = 000466, loss = 0.001732
grad_step = 000467, loss = 0.001726
grad_step = 000468, loss = 0.001702
grad_step = 000469, loss = 0.001702
grad_step = 000470, loss = 0.001711
grad_step = 000471, loss = 0.001708
grad_step = 000472, loss = 0.001697
grad_step = 000473, loss = 0.001694
grad_step = 000474, loss = 0.001694
grad_step = 000475, loss = 0.001693
grad_step = 000476, loss = 0.001690
grad_step = 000477, loss = 0.001687
grad_step = 000478, loss = 0.001683
grad_step = 000479, loss = 0.001681
grad_step = 000480, loss = 0.001681
grad_step = 000481, loss = 0.001681
grad_step = 000482, loss = 0.001677
grad_step = 000483, loss = 0.001672
grad_step = 000484, loss = 0.001672
grad_step = 000485, loss = 0.001673
grad_step = 000486, loss = 0.001672
grad_step = 000487, loss = 0.001668
grad_step = 000488, loss = 0.001664
grad_step = 000489, loss = 0.001664
grad_step = 000490, loss = 0.001665
grad_step = 000491, loss = 0.001664
grad_step = 000492, loss = 0.001660
grad_step = 000493, loss = 0.001657
grad_step = 000494, loss = 0.001657
grad_step = 000495, loss = 0.001657
grad_step = 000496, loss = 0.001657
grad_step = 000497, loss = 0.001654
grad_step = 000498, loss = 0.001652
grad_step = 000499, loss = 0.001651
grad_step = 000500, loss = 0.001650
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001649
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

  date_run                              2020-05-13 19:14:43.459103
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.254633
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 19:14:43.465776
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.162352
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 19:14:43.473564
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.154122
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 19:14:43.478372
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    -1.467
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
0   2020-05-13 19:14:17.266010  ...    mean_absolute_error
1   2020-05-13 19:14:17.270598  ...     mean_squared_error
2   2020-05-13 19:14:17.273824  ...  median_absolute_error
3   2020-05-13 19:14:17.276483  ...               r2_score
4   2020-05-13 19:14:25.714561  ...    mean_absolute_error
5   2020-05-13 19:14:25.718240  ...     mean_squared_error
6   2020-05-13 19:14:25.720965  ...  median_absolute_error
7   2020-05-13 19:14:25.723587  ...               r2_score
8   2020-05-13 19:14:43.459103  ...    mean_absolute_error
9   2020-05-13 19:14:43.465776  ...     mean_squared_error
10  2020-05-13 19:14:43.473564  ...  median_absolute_error
11  2020-05-13 19:14:43.478372  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f85402facf8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 315827.71it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 411684.13it/s]  9%|▉         | 876544/9912422 [00:00<00:15, 570282.40it/s] 36%|███▌      | 3522560/9912422 [00:00<00:07, 805878.37it/s] 77%|███████▋  | 7618560/9912422 [00:00<00:02, 1139476.17it/s]9920512it [00:00, 11474840.81it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 154540.22it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:04, 327690.21it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 421466.69it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 583522.14it/s]1654784it [00:00, 2848949.25it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 44047.61it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f84f2cb3e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f84f22e30b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8540305940> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f84f223a0b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f84f2cb3e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f84efa60be0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8540305940> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f84f21f96d8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f84f2cb3e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f85402bdf28> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fb5650fe240> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=b8da30098a93ccf8be44c9f01cf82024042af748e291a6a66b30761690764f93
  Stored in directory: /tmp/pip-ephem-wheel-cache-0y1axl4a/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fb4fcef9710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 1:38
   40960/17464789 [..............................] - ETA: 1:28
   57344/17464789 [..............................] - ETA: 1:24
  106496/17464789 [..............................] - ETA: 56s 
  139264/17464789 [..............................] - ETA: 1:00
  163840/17464789 [..............................] - ETA: 58s 
  180224/17464789 [..............................] - ETA: 59s
  212992/17464789 [..............................] - ETA: 55s
  229376/17464789 [..............................] - ETA: 56s
  245760/17464789 [..............................] - ETA: 57s
  262144/17464789 [..............................] - ETA: 58s
  303104/17464789 [..............................] - ETA: 54s
  319488/17464789 [..............................] - ETA: 55s
  335872/17464789 [..............................] - ETA: 55s
  368640/17464789 [..............................] - ETA: 54s
  385024/17464789 [..............................] - ETA: 54s
  417792/17464789 [..............................] - ETA: 53s
  442368/17464789 [..............................] - ETA: 52s
  475136/17464789 [..............................] - ETA: 51s
  491520/17464789 [..............................] - ETA: 52s
  524288/17464789 [..............................] - ETA: 51s
  557056/17464789 [..............................] - ETA: 50s
  598016/17464789 [>.............................] - ETA: 48s
  614400/17464789 [>.............................] - ETA: 49s
  647168/17464789 [>.............................] - ETA: 48s
  679936/17464789 [>.............................] - ETA: 47s
  720896/17464789 [>.............................] - ETA: 46s
  753664/17464789 [>.............................] - ETA: 45s
  786432/17464789 [>.............................] - ETA: 45s
  819200/17464789 [>.............................] - ETA: 44s
  860160/17464789 [>.............................] - ETA: 43s
  892928/17464789 [>.............................] - ETA: 43s
  925696/17464789 [>.............................] - ETA: 43s
  958464/17464789 [>.............................] - ETA: 42s
  999424/17464789 [>.............................] - ETA: 42s
 1048576/17464789 [>.............................] - ETA: 41s
 1081344/17464789 [>.............................] - ETA: 40s
 1114112/17464789 [>.............................] - ETA: 40s
 1155072/17464789 [>.............................] - ETA: 39s
 1204224/17464789 [=>............................] - ETA: 39s
 1236992/17464789 [=>............................] - ETA: 38s
 1294336/17464789 [=>............................] - ETA: 37s
 1327104/17464789 [=>............................] - ETA: 37s
 1376256/17464789 [=>............................] - ETA: 37s
 1433600/17464789 [=>............................] - ETA: 36s
 1466368/17464789 [=>............................] - ETA: 36s
 1515520/17464789 [=>............................] - ETA: 35s
 1572864/17464789 [=>............................] - ETA: 34s
 1622016/17464789 [=>............................] - ETA: 34s
 1654784/17464789 [=>............................] - ETA: 34s
 1712128/17464789 [=>............................] - ETA: 33s
 1761280/17464789 [==>...........................] - ETA: 33s
 1810432/17464789 [==>...........................] - ETA: 32s
 1867776/17464789 [==>...........................] - ETA: 32s
 1916928/17464789 [==>...........................] - ETA: 31s
 1990656/17464789 [==>...........................] - ETA: 31s
 2039808/17464789 [==>...........................] - ETA: 30s
 2088960/17464789 [==>...........................] - ETA: 30s
 2146304/17464789 [==>...........................] - ETA: 30s
 2195456/17464789 [==>...........................] - ETA: 29s
 2269184/17464789 [==>...........................] - ETA: 29s
 2318336/17464789 [==>...........................] - ETA: 28s
 2367488/17464789 [===>..........................] - ETA: 28s
 2441216/17464789 [===>..........................] - ETA: 28s
 2490368/17464789 [===>..........................] - ETA: 27s
 2547712/17464789 [===>..........................] - ETA: 27s
 2613248/17464789 [===>..........................] - ETA: 27s
 2670592/17464789 [===>..........................] - ETA: 26s
 2736128/17464789 [===>..........................] - ETA: 26s
 2801664/17464789 [===>..........................] - ETA: 25s
 2859008/17464789 [===>..........................] - ETA: 25s
 2924544/17464789 [====>.........................] - ETA: 25s
 2998272/17464789 [====>.........................] - ETA: 24s
 3047424/17464789 [====>.........................] - ETA: 24s
 3121152/17464789 [====>.........................] - ETA: 24s
 3186688/17464789 [====>.........................] - ETA: 24s
 3260416/17464789 [====>.........................] - ETA: 23s
 3325952/17464789 [====>.........................] - ETA: 23s
 3399680/17464789 [====>.........................] - ETA: 23s
 3465216/17464789 [====>.........................] - ETA: 22s
 3538944/17464789 [=====>........................] - ETA: 22s
 3604480/17464789 [=====>........................] - ETA: 22s
 3678208/17464789 [=====>........................] - ETA: 22s
 3743744/17464789 [=====>........................] - ETA: 21s
 3833856/17464789 [=====>........................] - ETA: 21s
 3899392/17464789 [=====>........................] - ETA: 21s
 3973120/17464789 [=====>........................] - ETA: 20s
 4038656/17464789 [=====>........................] - ETA: 20s
 4112384/17464789 [======>.......................] - ETA: 20s
 4194304/17464789 [======>.......................] - ETA: 20s
 4268032/17464789 [======>.......................] - ETA: 19s
 4341760/17464789 [======>.......................] - ETA: 19s
 4423680/17464789 [======>.......................] - ETA: 19s
 4497408/17464789 [======>.......................] - ETA: 19s
 4579328/17464789 [======>.......................] - ETA: 18s
 4653056/17464789 [======>.......................] - ETA: 18s
 4734976/17464789 [=======>......................] - ETA: 18s
 4808704/17464789 [=======>......................] - ETA: 18s
 4898816/17464789 [=======>......................] - ETA: 17s
 4964352/17464789 [=======>......................] - ETA: 17s
 5054464/17464789 [=======>......................] - ETA: 17s
 5136384/17464789 [=======>......................] - ETA: 17s
 5226496/17464789 [=======>......................] - ETA: 17s
 5316608/17464789 [========>.....................] - ETA: 16s
 5398528/17464789 [========>.....................] - ETA: 16s
 5488640/17464789 [========>.....................] - ETA: 16s
 5595136/17464789 [========>.....................] - ETA: 16s
 5693440/17464789 [========>.....................] - ETA: 15s
 5783552/17464789 [========>.....................] - ETA: 15s
 5890048/17464789 [=========>....................] - ETA: 15s
 6012928/17464789 [=========>....................] - ETA: 14s
 6111232/17464789 [=========>....................] - ETA: 14s
 6234112/17464789 [=========>....................] - ETA: 14s
 6356992/17464789 [=========>....................] - ETA: 14s
 6463488/17464789 [==========>...................] - ETA: 13s
 6602752/17464789 [==========>...................] - ETA: 13s
 6725632/17464789 [==========>...................] - ETA: 13s
 6864896/17464789 [==========>...................] - ETA: 12s
 7004160/17464789 [===========>..................] - ETA: 12s
 7143424/17464789 [===========>..................] - ETA: 12s
 7282688/17464789 [===========>..................] - ETA: 11s
 7438336/17464789 [===========>..................] - ETA: 11s
 7593984/17464789 [============>.................] - ETA: 11s
 7749632/17464789 [============>.................] - ETA: 10s
 7921664/17464789 [============>.................] - ETA: 10s
 8077312/17464789 [============>.................] - ETA: 10s
 8257536/17464789 [=============>................] - ETA: 9s 
 8445952/17464789 [=============>................] - ETA: 9s
 8617984/17464789 [=============>................] - ETA: 9s
 8814592/17464789 [==============>...............] - ETA: 8s
 9003008/17464789 [==============>...............] - ETA: 8s
 9216000/17464789 [==============>...............] - ETA: 8s
 9404416/17464789 [===============>..............] - ETA: 8s
 9609216/17464789 [===============>..............] - ETA: 7s
 9838592/17464789 [===============>..............] - ETA: 7s
10051584/17464789 [================>.............] - ETA: 7s
10272768/17464789 [================>.............] - ETA: 6s
10518528/17464789 [=================>............] - ETA: 6s
10764288/17464789 [=================>............] - ETA: 6s
11001856/17464789 [=================>............] - ETA: 5s
11247616/17464789 [==================>...........] - ETA: 5s
11419648/17464789 [==================>...........] - ETA: 5s
11681792/17464789 [===================>..........] - ETA: 4s
11943936/17464789 [===================>..........] - ETA: 4s
12222464/17464789 [===================>..........] - ETA: 4s
12500992/17464789 [====================>.........] - ETA: 4s
12795904/17464789 [====================>.........] - ETA: 3s
13090816/17464789 [=====================>........] - ETA: 3s
13393920/17464789 [======================>.......] - ETA: 3s
13705216/17464789 [======================>.......] - ETA: 2s
14032896/17464789 [=======================>......] - ETA: 2s
14368768/17464789 [=======================>......] - ETA: 2s
14696448/17464789 [========================>.....] - ETA: 2s
15040512/17464789 [========================>.....] - ETA: 1s
15392768/17464789 [=========================>....] - ETA: 1s
15736832/17464789 [==========================>...] - ETA: 1s
16121856/17464789 [==========================>...] - ETA: 0s
16490496/17464789 [===========================>..] - ETA: 0s
16891904/17464789 [============================>.] - ETA: 0s
17268736/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 11s 1us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 19:16:22.680169: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 19:16:22.683885: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-13 19:16:22.684572: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55624a5961a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 19:16:22.684589: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8813 - accuracy: 0.4860
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7663 - accuracy: 0.4935 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7586 - accuracy: 0.4940
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6728 - accuracy: 0.4996
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7280 - accuracy: 0.4960
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7104 - accuracy: 0.4971
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7356 - accuracy: 0.4955
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7433 - accuracy: 0.4950
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7770 - accuracy: 0.4928
11000/25000 [============>.................] - ETA: 3s - loss: 7.7754 - accuracy: 0.4929
12000/25000 [=============>................] - ETA: 3s - loss: 7.7650 - accuracy: 0.4936
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7928 - accuracy: 0.4918
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7641 - accuracy: 0.4936
15000/25000 [=================>............] - ETA: 2s - loss: 7.7341 - accuracy: 0.4956
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7318 - accuracy: 0.4958
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7090 - accuracy: 0.4972
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7160 - accuracy: 0.4968
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6908 - accuracy: 0.4984
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6743 - accuracy: 0.4995
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6674 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6780 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6756 - accuracy: 0.4994
25000/25000 [==============================] - 7s 269us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 19:16:35.697575
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 19:16:35.697575  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:06:27, 11.3kB/s].vector_cache/glove.6B.zip:   0%|          | 106k/862M [00:00<14:52:35, 16.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.06M/862M [00:00<10:23:37, 23.0kB/s].vector_cache/glove.6B.zip:   1%|          | 10.6M/862M [00:01<7:12:13, 32.8kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 15.8M/862M [00:01<5:00:48, 46.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.5M/862M [00:01<3:29:46, 67.0kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.7M/862M [00:01<2:26:39, 95.5kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.6M/862M [00:01<1:42:37, 136kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 25.7M/862M [00:01<1:11:51, 194kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.9M/862M [00:01<50:22, 276kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.3M/862M [00:01<35:20, 392kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.3M/862M [00:01<24:52, 556kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.1M/862M [00:01<17:36, 784kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.2M/862M [00:02<12:29, 1.10MB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.1M/862M [00:02<08:56, 1.53MB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.6M/862M [00:02<06:24, 2.14MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.0M/862M [00:02<04:35, 2.97MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.0M/862M [00:02<03:17, 4.12MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:02<02:43, 4.96MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<03:51, 3.48MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<03:45, 3.58MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.6M/862M [00:05<02:49, 4.75MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<06:11, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<05:27, 2.45MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.1M/862M [00:07<03:58, 3.35MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<08:15, 1.61MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<06:33, 2.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<04:41, 2.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<17:06, 773kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<13:11, 1.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:11<09:23, 1.40MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<10:54, 1.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<08:56, 1.47MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.4M/862M [00:12<06:24, 2.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<09:22, 1.40MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<08:13, 1.59MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:14<05:54, 2.21MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<08:17, 1.57MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<07:08, 1.82MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<06:52, 1.89MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<05:53, 2.20MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<06:01, 2.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<05:26, 2.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.9M/862M [00:20<03:57, 3.24MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:23<09:41, 1.32MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:23<08:24, 1.52MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.1M/862M [00:23<06:01, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:25<09:10, 1.39MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:25<08:34, 1.48MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:25<06:09, 2.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<08:06, 1.56MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<07:16, 1.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<05:12, 2.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<09:16, 1.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<08:00, 1.57MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<05:42, 2.20MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<10:26, 1.20MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<08:39, 1.45MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<06:08, 2.03MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<18:14, 684kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<13:45, 906kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<11:25, 1.08MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:35<08:51, 1.40MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<08:02, 1.53MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<06:29, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<06:22, 1.92MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:07, 2.39MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<05:31, 2.21MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<04:40, 2.61MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<05:06, 2.38MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<04:16, 2.84MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<04:52, 2.47MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<04:09, 2.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<03:00, 3.98MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<14:15, 841kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<10:41, 1.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<09:18, 1.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<07:29, 1.59MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<05:19, 2.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<20:03, 591kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<14:49, 799kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<12:06, 973kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<09:06, 1.29MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:55<08:13, 1.42MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<06:34, 1.78MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<04:40, 2.49MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<30:14, 385kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<21:49, 533kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<17:03, 679kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:59<12:39, 914kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<08:54, 1.29MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<55:57, 206kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<39:49, 289kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<29:29, 388kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<21:43, 526kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<16:47, 677kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<12:27, 911kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<10:24, 1.09MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:07<07:52, 1.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<07:17, 1.54MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:41, 1.97MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:51, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<04:53, 2.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<03:32, 3.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<06:49, 1.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<05:35, 1.98MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<05:34, 1.98MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<04:25, 2.49MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<04:49, 2.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<03:58, 2.75MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<04:30, 2.41MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<03:44, 2.91MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<04:30, 2.40MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:21<04:02, 2.68MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<02:53, 3.71MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<1:27:51, 122kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:23<1:02:03, 173kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<43:23, 246kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<36:04, 296kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:25<25:58, 411kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<18:12, 583kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<20:37, 514kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:27<15:24, 688kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<10:50, 974kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<14:15, 739kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<10:50, 972kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<07:37, 1.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<3:51:29, 45.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<2:42:44, 64.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<1:54:48, 90.6kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<1:21:06, 128kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<57:59, 178kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<41:06, 251kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<30:15, 339kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<21:52, 469kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<15:18, 666kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<1:05:51, 155kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<46:35, 219kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<34:03, 297kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<24:28, 414kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<17:07, 588kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<28:41, 351kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<20:36, 487kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<15:56, 627kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<11:36, 859kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<09:42, 1.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<07:20, 1.35MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<06:40, 1.48MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:48<05:16, 1.86MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<05:10, 1.89MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<04:05, 2.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<04:26, 2.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:52<03:37, 2.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<02:36, 3.69MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<11:55, 809kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<08:54, 1.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<07:40, 1.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<06:09, 1.56MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:22, 2.18MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<14:28, 657kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<11:05, 857kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<07:49, 1.21MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<11:04, 852kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<08:46, 1.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<06:14, 1.51MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<08:43, 1.07MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<06:48, 1.38MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<04:51, 1.92MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<07:25, 1.25MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<06:26, 1.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:05<04:38, 2.00MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<05:39, 1.63MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:07<05:10, 1.78MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<03:43, 2.47MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:08<06:01, 1.52MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<05:22, 1.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:09<03:50, 2.37MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<06:49, 1.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<05:47, 1.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<04:07, 2.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<07:14, 1.25MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<06:08, 1.47MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<04:22, 2.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<07:13, 1.24MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<05:57, 1.50MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<04:14, 2.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<08:06, 1.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<06:13, 1.43MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<04:30, 1.96MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<03:16, 2.69MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:18<51:48, 170kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:19<36:57, 238kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:20<27:00, 324kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<19:33, 447kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<13:44, 634kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<13:11, 658kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<09:53, 876kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<06:59, 1.23MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:24<09:33, 901kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<07:31, 1.14MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:25<05:20, 1.60MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:26<07:26, 1.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:26<06:05, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<04:19, 1.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:28<09:23, 902kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:28<07:12, 1.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:30<06:16, 1.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:30<04:52, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:33<04:55, 1.70MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:33<04:17, 1.94MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:33<03:06, 2.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:35<04:41, 1.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:35<04:28, 1.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:35<03:12, 2.56MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:37<05:32, 1.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:37<04:50, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:37<03:28, 2.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<05:46, 1.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:39<04:55, 1.65MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<03:29, 2.31MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<15:29, 520kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:41<11:31, 699kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<09:11, 870kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<06:51, 1.16MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<06:00, 1.32MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<04:42, 1.68MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<03:21, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<08:42, 902kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<06:29, 1.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<04:35, 1.70MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:48<12:44, 611kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:48<09:19, 833kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:49<06:34, 1.18MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:50<11:33, 668kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:50<08:41, 886kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<06:07, 1.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:52<09:08, 837kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:52<06:56, 1.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:53, 1.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:54<15:51, 478kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:54<11:38, 650kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<08:08, 922kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:56<1:14:27, 101kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:56<52:27, 143kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<37:37, 198kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<26:38, 279kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:00<19:41, 374kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [03:00<14:09, 520kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:02<10:59, 664kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:02<08:05, 902kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<06:46, 1.07MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:04<05:07, 1.41MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:06<04:42, 1.52MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<03:51, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:08<03:44, 1.90MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:08<03:00, 2.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:10<03:11, 2.20MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:10<02:36, 2.68MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:12<02:55, 2.38MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:12<02:25, 2.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:14<02:46, 2.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:14<02:18, 2.98MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:16<02:41, 2.53MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:16<02:11, 3.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:18<02:36, 2.58MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:18<02:12, 3.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<01:35, 4.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:20<18:13, 367kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<13:09, 507kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:22<10:09, 651kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:22<07:25, 890kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:24<06:11, 1.06MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:24<04:41, 1.39MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:26<04:17, 1.51MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:26<03:22, 1.92MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:28<03:21, 1.91MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:28<02:41, 2.37MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:30<02:51, 2.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:30<02:16, 2.78MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:32<02:36, 2.41MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:32<02:09, 2.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:34<02:42, 2.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:34<02:21, 2.62MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<01:41, 3.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:36<10:24, 589kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:36<07:43, 792kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<06:16, 967kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:38<04:42, 1.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<04:13, 1.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<03:16, 1.83MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<03:13, 1.84MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<02:34, 2.30MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:44<02:43, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:44<02:22, 2.46MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<02:30, 2.31MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<02:04, 2.79MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<02:19, 2.46MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<02:06, 2.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:50<02:18, 2.45MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<01:55, 2.92MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<02:12, 2.53MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:52<01:55, 2.90MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<01:22, 4.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:54<36:42, 150kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:54<25:56, 212kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<18:50, 289kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<13:31, 402kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<10:10, 528kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<07:25, 723kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:00<05:57, 890kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:00<04:25, 1.19MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<03:53, 1.34MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<03:02, 1.72MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<02:54, 1.78MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:04<02:18, 2.23MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:06<02:23, 2.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:06<02:00, 2.53MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<01:26, 3.50MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:08<05:44, 877kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<04:18, 1.16MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<03:45, 1.32MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:10<02:53, 1.71MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<02:46, 1.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<02:12, 2.21MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<02:17, 2.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:14<01:51, 2.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:15<02:02, 2.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<01:42, 2.79MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:17<01:54, 2.46MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:18<01:34, 2.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:19<01:48, 2.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<01:38, 2.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:21<01:48, 2.51MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:21<01:33, 2.90MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:23<01:45, 2.54MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:23<01:34, 2.83MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<01:07, 3.91MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:25<08:08, 542kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:25<05:58, 737kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:27<04:47, 907kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:27<03:42, 1.17MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:29<03:11, 1.34MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:29<02:36, 1.64MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<01:50, 2.29MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:31<13:07, 320kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:31<09:31, 441kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<06:36, 627kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:33<12:19, 336kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:33<08:57, 461kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:35<06:46, 600kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:35<04:59, 813kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<03:28, 1.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:37<53:05, 75.3kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:38<37:25, 107kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:39<26:22, 149kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:40<18:38, 210kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:40<12:54, 299kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:41<15:26, 250kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<11:04, 348kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:42<07:39, 495kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:43<20:43, 183kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<14:40, 258kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:45<10:42, 348kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:45<07:43, 481kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<05:52, 622kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:47<04:18, 846kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:48<03:00, 1.20MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<06:56, 516kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:49<05:07, 699kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:51<04:03, 868kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:51<03:05, 1.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<02:38, 1.30MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:53<02:07, 1.62MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:55<01:57, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:55<01:32, 2.20MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<01:34, 2.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<01:25, 2.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<01:27, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:59<01:12, 2.68MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:01<01:19, 2.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:01<01:05, 2.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<01:14, 2.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<01:09, 2.69MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<01:14, 2.45MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<01:02, 2.92MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<01:10, 2.53MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:07<00:57, 3.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<00:40, 4.25MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<08:38, 335kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<06:11, 466kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<04:41, 601kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<03:25, 824kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<02:46, 992kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<02:04, 1.32MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<01:51, 1.45MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<01:28, 1.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<01:02, 2.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<17:41, 148kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<12:28, 209kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<08:56, 285kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<06:23, 398kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<04:23, 565kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<06:29, 382kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<04:40, 528kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<03:13, 750kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<10:56, 221kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<07:48, 308kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:25<05:44, 409kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:25<04:12, 556kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<02:53, 788kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<24:47, 91.7kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<17:31, 129kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:27<12:00, 185kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<09:54, 223kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<07:07, 309kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:29<04:54, 439kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<04:34, 466kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<03:21, 633kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:31<02:20, 894kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<02:25, 851kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<01:53, 1.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:33<01:18, 1.53MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<02:13, 899kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:35<01:40, 1.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<01:26, 1.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:37<01:06, 1.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<01:02, 1.78MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<00:49, 2.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<00:50, 2.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<00:41, 2.58MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<00:44, 2.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<00:35, 2.88MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:45<00:39, 2.48MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:45<00:32, 3.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<00:39, 2.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<00:35, 2.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:49<00:37, 2.45MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:49<00:31, 2.90MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:52<00:41, 2.09MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:52<00:36, 2.37MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:52<00:26, 3.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:54<00:47, 1.74MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:54<00:44, 1.84MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:54<00:32, 2.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:54<00:23, 3.40MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:56<03:13, 405kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:56<02:25, 540kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:56<01:39, 763kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:58<01:32, 802kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:58<01:16, 974kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:58<00:52, 1.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:00<01:02, 1.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:00<00:51, 1.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<00:34, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:02<03:30, 314kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:02<02:34, 426kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:02<01:45, 604kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:04<01:32, 672kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:04<01:12, 858kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:48, 1.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:06<01:10, 825kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:06<00:54, 1.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:37, 1.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:08<00:43, 1.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:08<00:33, 1.59MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:10<00:29, 1.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:10<00:24, 2.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:10<00:16, 2.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:12<00:29, 1.52MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:12<00:24, 1.85MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<00:16, 2.56MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:14<00:32, 1.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:14<00:25, 1.62MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:16<00:21, 1.71MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:16<00:17, 2.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<00:11, 2.92MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:18<00:25, 1.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:18<00:20, 1.61MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:18<00:13, 2.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:20<00:20, 1.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:20<00:16, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<00:10, 2.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:22<00:20, 1.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:22<00:16, 1.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:22<00:11, 2.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:24<00:14, 1.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:24<00:11, 1.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:07, 2.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:26<00:23, 718kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:26<00:17, 947kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:26<00:10, 1.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:28<00:13, 960kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:28<00:10, 1.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:28<00:05, 1.70MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:30<00:06, 1.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:30<00:05, 1.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:30<00:02, 2.02MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:32<00:03, 1.40MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:32<00:02, 1.60MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:32<00:00, 2.24MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:34<00:00, 866kB/s] .vector_cache/glove.6B.zip: 862MB [06:34, 2.19MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 874/400000 [00:00<00:45, 8734.18it/s]  0%|          | 1721/400000 [00:00<00:46, 8649.99it/s]  1%|          | 2493/400000 [00:00<00:47, 8347.39it/s]  1%|          | 3354/400000 [00:00<00:47, 8421.25it/s]  1%|          | 4220/400000 [00:00<00:46, 8490.49it/s]  1%|▏         | 5075/400000 [00:00<00:46, 8507.19it/s]  1%|▏         | 5918/400000 [00:00<00:46, 8482.15it/s]  2%|▏         | 6755/400000 [00:00<00:46, 8447.62it/s]  2%|▏         | 7643/400000 [00:00<00:45, 8569.45it/s]  2%|▏         | 8505/400000 [00:01<00:45, 8583.85it/s]  2%|▏         | 9427/400000 [00:01<00:44, 8762.46it/s]  3%|▎         | 10302/400000 [00:01<00:44, 8755.74it/s]  3%|▎         | 11167/400000 [00:01<00:44, 8641.48it/s]  3%|▎         | 12040/400000 [00:01<00:44, 8665.34it/s]  3%|▎         | 12962/400000 [00:01<00:43, 8824.09it/s]  3%|▎         | 13842/400000 [00:01<00:43, 8789.01it/s]  4%|▎         | 14728/400000 [00:01<00:43, 8808.96it/s]  4%|▍         | 15608/400000 [00:01<00:43, 8737.89it/s]  4%|▍         | 16481/400000 [00:01<00:44, 8642.21it/s]  4%|▍         | 17345/400000 [00:02<00:44, 8574.61it/s]  5%|▍         | 18241/400000 [00:02<00:43, 8686.11it/s]  5%|▍         | 19110/400000 [00:02<00:44, 8580.03it/s]  5%|▍         | 19981/400000 [00:02<00:44, 8615.50it/s]  5%|▌         | 20881/400000 [00:02<00:43, 8727.22it/s]  5%|▌         | 21797/400000 [00:02<00:42, 8849.91it/s]  6%|▌         | 22689/400000 [00:02<00:42, 8868.12it/s]  6%|▌         | 23577/400000 [00:02<00:42, 8785.01it/s]  6%|▌         | 24457/400000 [00:02<00:43, 8560.46it/s]  6%|▋         | 25315/400000 [00:02<00:44, 8461.85it/s]  7%|▋         | 26192/400000 [00:03<00:43, 8551.56it/s]  7%|▋         | 27102/400000 [00:03<00:42, 8706.97it/s]  7%|▋         | 27975/400000 [00:03<00:42, 8678.13it/s]  7%|▋         | 28858/400000 [00:03<00:42, 8721.52it/s]  7%|▋         | 29760/400000 [00:03<00:42, 8808.18it/s]  8%|▊         | 30642/400000 [00:03<00:42, 8617.56it/s]  8%|▊         | 31512/400000 [00:03<00:42, 8642.09it/s]  8%|▊         | 32383/400000 [00:03<00:42, 8662.30it/s]  8%|▊         | 33250/400000 [00:03<00:43, 8370.40it/s]  9%|▊         | 34164/400000 [00:03<00:42, 8586.44it/s]  9%|▉         | 35075/400000 [00:04<00:41, 8734.32it/s]  9%|▉         | 35987/400000 [00:04<00:41, 8846.33it/s]  9%|▉         | 36891/400000 [00:04<00:40, 8902.28it/s]  9%|▉         | 37783/400000 [00:04<00:41, 8805.58it/s] 10%|▉         | 38666/400000 [00:04<00:41, 8665.22it/s] 10%|▉         | 39553/400000 [00:04<00:41, 8724.06it/s] 10%|█         | 40451/400000 [00:04<00:40, 8796.43it/s] 10%|█         | 41346/400000 [00:04<00:40, 8839.37it/s] 11%|█         | 42231/400000 [00:04<00:40, 8787.86it/s] 11%|█         | 43163/400000 [00:04<00:39, 8940.19it/s] 11%|█         | 44082/400000 [00:05<00:39, 9012.61it/s] 11%|█         | 44985/400000 [00:05<00:39, 8927.09it/s] 11%|█▏        | 45879/400000 [00:05<00:40, 8815.16it/s] 12%|█▏        | 46762/400000 [00:05<00:40, 8814.49it/s] 12%|█▏        | 47662/400000 [00:05<00:39, 8866.78it/s] 12%|█▏        | 48550/400000 [00:05<00:39, 8853.10it/s] 12%|█▏        | 49455/400000 [00:05<00:39, 8910.95it/s] 13%|█▎        | 50349/400000 [00:05<00:39, 8918.48it/s] 13%|█▎        | 51242/400000 [00:05<00:39, 8858.09it/s] 13%|█▎        | 52130/400000 [00:05<00:39, 8863.69it/s] 13%|█▎        | 53017/400000 [00:06<00:39, 8794.65it/s] 13%|█▎        | 53930/400000 [00:06<00:38, 8892.44it/s] 14%|█▎        | 54820/400000 [00:06<00:39, 8813.67it/s] 14%|█▍        | 55728/400000 [00:06<00:38, 8888.21it/s] 14%|█▍        | 56618/400000 [00:06<00:38, 8859.17it/s] 14%|█▍        | 57540/400000 [00:06<00:38, 8962.54it/s] 15%|█▍        | 58465/400000 [00:06<00:37, 9045.65it/s] 15%|█▍        | 59371/400000 [00:06<00:37, 8971.97it/s] 15%|█▌        | 60269/400000 [00:06<00:39, 8625.99it/s] 15%|█▌        | 61135/400000 [00:07<00:39, 8630.84it/s] 16%|█▌        | 62001/400000 [00:07<00:40, 8446.06it/s] 16%|█▌        | 62849/400000 [00:07<00:40, 8408.73it/s] 16%|█▌        | 63692/400000 [00:07<00:40, 8299.89it/s] 16%|█▌        | 64547/400000 [00:07<00:40, 8372.98it/s] 16%|█▋        | 65438/400000 [00:07<00:39, 8524.67it/s] 17%|█▋        | 66293/400000 [00:07<00:39, 8507.14it/s] 17%|█▋        | 67175/400000 [00:07<00:38, 8596.87it/s] 17%|█▋        | 68036/400000 [00:07<00:38, 8583.80it/s] 17%|█▋        | 68912/400000 [00:07<00:38, 8632.99it/s] 17%|█▋        | 69789/400000 [00:08<00:38, 8671.72it/s] 18%|█▊        | 70714/400000 [00:08<00:37, 8834.86it/s] 18%|█▊        | 71642/400000 [00:08<00:36, 8963.39it/s] 18%|█▊        | 72540/400000 [00:08<00:36, 8942.41it/s] 18%|█▊        | 73436/400000 [00:08<00:36, 8908.12it/s] 19%|█▊        | 74328/400000 [00:08<00:36, 8856.67it/s] 19%|█▉        | 75215/400000 [00:08<00:37, 8600.98it/s] 19%|█▉        | 76112/400000 [00:08<00:37, 8707.28it/s] 19%|█▉        | 77015/400000 [00:08<00:36, 8800.70it/s] 19%|█▉        | 77908/400000 [00:08<00:36, 8837.79it/s] 20%|█▉        | 78822/400000 [00:09<00:35, 8926.20it/s] 20%|█▉        | 79730/400000 [00:09<00:35, 8969.37it/s] 20%|██        | 80628/400000 [00:09<00:35, 8947.64it/s] 20%|██        | 81558/400000 [00:09<00:35, 9048.49it/s] 21%|██        | 82468/400000 [00:09<00:35, 9062.08it/s] 21%|██        | 83410/400000 [00:09<00:34, 9166.06it/s] 21%|██        | 84328/400000 [00:09<00:34, 9094.85it/s] 21%|██▏       | 85256/400000 [00:09<00:34, 9146.77it/s] 22%|██▏       | 86179/400000 [00:09<00:34, 9170.82it/s] 22%|██▏       | 87097/400000 [00:09<00:34, 8954.33it/s] 22%|██▏       | 87994/400000 [00:10<00:35, 8902.21it/s] 22%|██▏       | 88929/400000 [00:10<00:34, 9030.21it/s] 22%|██▏       | 89837/400000 [00:10<00:34, 9043.55it/s] 23%|██▎       | 90745/400000 [00:10<00:34, 9053.51it/s] 23%|██▎       | 91651/400000 [00:10<00:34, 8971.88it/s] 23%|██▎       | 92561/400000 [00:10<00:34, 9008.11it/s] 23%|██▎       | 93467/400000 [00:10<00:34, 9013.28it/s] 24%|██▎       | 94369/400000 [00:10<00:33, 9002.01it/s] 24%|██▍       | 95299/400000 [00:10<00:33, 9088.92it/s] 24%|██▍       | 96221/400000 [00:10<00:33, 9125.74it/s] 24%|██▍       | 97134/400000 [00:11<00:33, 9118.51it/s] 25%|██▍       | 98058/400000 [00:11<00:32, 9154.26it/s] 25%|██▍       | 98974/400000 [00:11<00:33, 9072.76it/s] 25%|██▍       | 99882/400000 [00:11<00:33, 9019.41it/s] 25%|██▌       | 100785/400000 [00:11<00:33, 9017.93it/s] 25%|██▌       | 101714/400000 [00:11<00:32, 9095.90it/s] 26%|██▌       | 102626/400000 [00:11<00:32, 9100.23it/s] 26%|██▌       | 103537/400000 [00:11<00:32, 9063.67it/s] 26%|██▌       | 104444/400000 [00:11<00:32, 9045.02it/s] 26%|██▋       | 105349/400000 [00:11<00:33, 8908.84it/s] 27%|██▋       | 106241/400000 [00:12<00:33, 8789.37it/s] 27%|██▋       | 107129/400000 [00:12<00:33, 8814.83it/s] 27%|██▋       | 108012/400000 [00:12<00:34, 8560.21it/s] 27%|██▋       | 108908/400000 [00:12<00:33, 8675.29it/s] 27%|██▋       | 109802/400000 [00:12<00:33, 8752.54it/s] 28%|██▊       | 110725/400000 [00:12<00:32, 8889.08it/s] 28%|██▊       | 111629/400000 [00:12<00:32, 8931.06it/s] 28%|██▊       | 112524/400000 [00:12<00:32, 8915.81it/s] 28%|██▊       | 113417/400000 [00:12<00:32, 8850.50it/s] 29%|██▊       | 114303/400000 [00:12<00:32, 8819.61it/s] 29%|██▉       | 115186/400000 [00:13<00:32, 8769.08it/s] 29%|██▉       | 116075/400000 [00:13<00:32, 8803.28it/s] 29%|██▉       | 116979/400000 [00:13<00:31, 8872.58it/s] 29%|██▉       | 117909/400000 [00:13<00:31, 8996.23it/s] 30%|██▉       | 118810/400000 [00:13<00:31, 8994.66it/s] 30%|██▉       | 119723/400000 [00:13<00:31, 9033.55it/s] 30%|███       | 120627/400000 [00:13<00:30, 9027.37it/s] 30%|███       | 121533/400000 [00:13<00:30, 9036.88it/s] 31%|███       | 122450/400000 [00:13<00:30, 9073.53it/s] 31%|███       | 123358/400000 [00:13<00:30, 8951.23it/s] 31%|███       | 124304/400000 [00:14<00:30, 9095.38it/s] 31%|███▏      | 125215/400000 [00:14<00:30, 9087.54it/s] 32%|███▏      | 126125/400000 [00:14<00:30, 9060.21it/s] 32%|███▏      | 127062/400000 [00:14<00:29, 9149.64it/s] 32%|███▏      | 127978/400000 [00:14<00:29, 9134.11it/s] 32%|███▏      | 128899/400000 [00:14<00:29, 9156.10it/s] 32%|███▏      | 129826/400000 [00:14<00:29, 9188.23it/s] 33%|███▎      | 130748/400000 [00:14<00:29, 9197.00it/s] 33%|███▎      | 131668/400000 [00:14<00:29, 9132.58it/s] 33%|███▎      | 132582/400000 [00:15<00:29, 9022.36it/s] 33%|███▎      | 133486/400000 [00:15<00:29, 9026.99it/s] 34%|███▎      | 134390/400000 [00:15<00:29, 8980.02it/s] 34%|███▍      | 135308/400000 [00:15<00:29, 9036.33it/s] 34%|███▍      | 136212/400000 [00:15<00:29, 8897.98it/s] 34%|███▍      | 137103/400000 [00:15<00:29, 8766.31it/s] 34%|███▍      | 137991/400000 [00:15<00:29, 8798.58it/s] 35%|███▍      | 138917/400000 [00:15<00:29, 8931.10it/s] 35%|███▍      | 139849/400000 [00:15<00:28, 9043.68it/s] 35%|███▌      | 140804/400000 [00:15<00:28, 9187.67it/s] 35%|███▌      | 141725/400000 [00:16<00:28, 9044.31it/s] 36%|███▌      | 142655/400000 [00:16<00:28, 9116.40it/s] 36%|███▌      | 143568/400000 [00:16<00:28, 8989.43it/s] 36%|███▌      | 144487/400000 [00:16<00:28, 9048.53it/s] 36%|███▋      | 145393/400000 [00:16<00:28, 8999.75it/s] 37%|███▋      | 146294/400000 [00:16<00:28, 8985.05it/s] 37%|███▋      | 147258/400000 [00:16<00:27, 9170.38it/s] 37%|███▋      | 148177/400000 [00:16<00:27, 9025.88it/s] 37%|███▋      | 149081/400000 [00:16<00:27, 9012.49it/s] 38%|███▊      | 150029/400000 [00:16<00:27, 9145.33it/s] 38%|███▊      | 150945/400000 [00:17<00:27, 9041.87it/s] 38%|███▊      | 151851/400000 [00:17<00:27, 9045.42it/s] 38%|███▊      | 152757/400000 [00:17<00:28, 8744.32it/s] 38%|███▊      | 153679/400000 [00:17<00:27, 8880.41it/s] 39%|███▊      | 154570/400000 [00:17<00:27, 8804.19it/s] 39%|███▉      | 155453/400000 [00:17<00:28, 8726.21it/s] 39%|███▉      | 156328/400000 [00:17<00:28, 8577.85it/s] 39%|███▉      | 157196/400000 [00:17<00:28, 8605.74it/s] 40%|███▉      | 158058/400000 [00:17<00:28, 8595.49it/s] 40%|███▉      | 158919/400000 [00:17<00:28, 8550.68it/s] 40%|███▉      | 159793/400000 [00:18<00:27, 8605.18it/s] 40%|████      | 160672/400000 [00:18<00:27, 8658.44it/s] 40%|████      | 161587/400000 [00:18<00:27, 8799.08it/s] 41%|████      | 162499/400000 [00:18<00:26, 8892.36it/s] 41%|████      | 163390/400000 [00:18<00:27, 8748.64it/s] 41%|████      | 164266/400000 [00:18<00:27, 8695.41it/s] 41%|████▏     | 165176/400000 [00:18<00:26, 8812.72it/s] 42%|████▏     | 166059/400000 [00:18<00:26, 8766.39it/s] 42%|████▏     | 166937/400000 [00:18<00:26, 8736.09it/s] 42%|████▏     | 167848/400000 [00:18<00:26, 8842.59it/s] 42%|████▏     | 168763/400000 [00:19<00:25, 8930.66it/s] 42%|████▏     | 169657/400000 [00:19<00:25, 8894.31it/s] 43%|████▎     | 170552/400000 [00:19<00:25, 8908.41it/s] 43%|████▎     | 171480/400000 [00:19<00:25, 9016.35it/s] 43%|████▎     | 172441/400000 [00:19<00:24, 9185.78it/s] 43%|████▎     | 173361/400000 [00:19<00:25, 9048.79it/s] 44%|████▎     | 174286/400000 [00:19<00:24, 9105.53it/s] 44%|████▍     | 175241/400000 [00:19<00:24, 9233.67it/s] 44%|████▍     | 176166/400000 [00:19<00:24, 9069.34it/s] 44%|████▍     | 177121/400000 [00:19<00:24, 9207.16it/s] 45%|████▍     | 178044/400000 [00:20<00:24, 9206.63it/s] 45%|████▍     | 178974/400000 [00:20<00:23, 9233.65it/s] 45%|████▍     | 179905/400000 [00:20<00:23, 9254.11it/s] 45%|████▌     | 180831/400000 [00:20<00:23, 9215.24it/s] 45%|████▌     | 181753/400000 [00:20<00:24, 9065.78it/s] 46%|████▌     | 182661/400000 [00:20<00:24, 8934.25it/s] 46%|████▌     | 183613/400000 [00:20<00:23, 9101.95it/s] 46%|████▌     | 184542/400000 [00:20<00:23, 9154.85it/s] 46%|████▋     | 185459/400000 [00:20<00:23, 9128.24it/s] 47%|████▋     | 186424/400000 [00:21<00:23, 9277.06it/s] 47%|████▋     | 187353/400000 [00:21<00:23, 9097.73it/s] 47%|████▋     | 188265/400000 [00:21<00:23, 9045.16it/s] 47%|████▋     | 189171/400000 [00:21<00:23, 8852.58it/s] 48%|████▊     | 190085/400000 [00:21<00:23, 8934.41it/s] 48%|████▊     | 191006/400000 [00:21<00:23, 9014.33it/s] 48%|████▊     | 191909/400000 [00:21<00:23, 8959.50it/s] 48%|████▊     | 192831/400000 [00:21<00:22, 9034.95it/s] 48%|████▊     | 193742/400000 [00:21<00:22, 9056.72it/s] 49%|████▊     | 194686/400000 [00:21<00:22, 9166.77it/s] 49%|████▉     | 195604/400000 [00:22<00:22, 9109.23it/s] 49%|████▉     | 196516/400000 [00:22<00:22, 9082.23it/s] 49%|████▉     | 197425/400000 [00:22<00:22, 8868.96it/s] 50%|████▉     | 198314/400000 [00:22<00:23, 8751.04it/s] 50%|████▉     | 199212/400000 [00:22<00:22, 8816.61it/s] 50%|█████     | 200134/400000 [00:22<00:22, 8933.51it/s] 50%|█████     | 201029/400000 [00:22<00:22, 8894.60it/s] 50%|█████     | 201947/400000 [00:22<00:22, 8975.79it/s] 51%|█████     | 202856/400000 [00:22<00:21, 9007.63it/s] 51%|█████     | 203788/400000 [00:22<00:21, 9097.08it/s] 51%|█████     | 204723/400000 [00:23<00:21, 9171.37it/s] 51%|█████▏    | 205641/400000 [00:23<00:21, 9143.50it/s] 52%|█████▏    | 206564/400000 [00:23<00:21, 9169.26it/s] 52%|█████▏    | 207505/400000 [00:23<00:20, 9239.32it/s] 52%|█████▏    | 208430/400000 [00:23<00:20, 9224.25it/s] 52%|█████▏    | 209353/400000 [00:23<00:20, 9157.24it/s] 53%|█████▎    | 210270/400000 [00:23<00:21, 8887.40it/s] 53%|█████▎    | 211163/400000 [00:23<00:21, 8898.52it/s] 53%|█████▎    | 212055/400000 [00:23<00:21, 8555.98it/s] 53%|█████▎    | 212930/400000 [00:23<00:21, 8612.96it/s] 53%|█████▎    | 213854/400000 [00:24<00:21, 8790.47it/s] 54%|█████▎    | 214736/400000 [00:24<00:21, 8761.07it/s] 54%|█████▍    | 215615/400000 [00:24<00:21, 8756.26it/s] 54%|█████▍    | 216559/400000 [00:24<00:20, 8950.36it/s] 54%|█████▍    | 217501/400000 [00:24<00:20, 9084.26it/s] 55%|█████▍    | 218424/400000 [00:24<00:19, 9124.80it/s] 55%|█████▍    | 219340/400000 [00:24<00:19, 9133.56it/s] 55%|█████▌    | 220301/400000 [00:24<00:19, 9270.44it/s] 55%|█████▌    | 221230/400000 [00:24<00:19, 9160.58it/s] 56%|█████▌    | 222148/400000 [00:24<00:19, 9150.06it/s] 56%|█████▌    | 223064/400000 [00:25<00:19, 9147.17it/s] 56%|█████▌    | 223982/400000 [00:25<00:19, 9156.53it/s] 56%|█████▌    | 224903/400000 [00:25<00:19, 9171.29it/s] 56%|█████▋    | 225821/400000 [00:25<00:19, 8894.32it/s] 57%|█████▋    | 226764/400000 [00:25<00:19, 9047.06it/s] 57%|█████▋    | 227671/400000 [00:25<00:19, 8936.40it/s] 57%|█████▋    | 228567/400000 [00:25<00:19, 8804.74it/s] 57%|█████▋    | 229490/400000 [00:25<00:19, 8926.98it/s] 58%|█████▊    | 230408/400000 [00:25<00:18, 8999.43it/s] 58%|█████▊    | 231310/400000 [00:25<00:18, 8991.40it/s] 58%|█████▊    | 232247/400000 [00:26<00:18, 9099.30it/s] 58%|█████▊    | 233166/400000 [00:26<00:18, 9123.88it/s] 59%|█████▊    | 234086/400000 [00:26<00:18, 9146.47it/s] 59%|█████▉    | 235006/400000 [00:26<00:18, 9159.72it/s] 59%|█████▉    | 235923/400000 [00:26<00:18, 9060.71it/s] 59%|█████▉    | 236875/400000 [00:26<00:17, 9191.84it/s] 59%|█████▉    | 237803/400000 [00:26<00:17, 9217.89it/s] 60%|█████▉    | 238726/400000 [00:26<00:17, 9216.76it/s] 60%|█████▉    | 239649/400000 [00:26<00:17, 9212.93it/s] 60%|██████    | 240571/400000 [00:27<00:17, 9140.97it/s] 60%|██████    | 241486/400000 [00:27<00:17, 9029.76it/s] 61%|██████    | 242390/400000 [00:27<00:18, 8641.11it/s] 61%|██████    | 243259/400000 [00:27<00:18, 8655.54it/s] 61%|██████    | 244128/400000 [00:27<00:18, 8628.27it/s] 61%|██████▏   | 245036/400000 [00:27<00:17, 8757.31it/s] 61%|██████▏   | 245955/400000 [00:27<00:17, 8879.22it/s] 62%|██████▏   | 246845/400000 [00:27<00:17, 8663.99it/s] 62%|██████▏   | 247714/400000 [00:27<00:17, 8652.21it/s] 62%|██████▏   | 248581/400000 [00:27<00:17, 8616.74it/s] 62%|██████▏   | 249444/400000 [00:28<00:17, 8563.65it/s] 63%|██████▎   | 250306/400000 [00:28<00:17, 8579.60it/s] 63%|██████▎   | 251165/400000 [00:28<00:17, 8523.43it/s] 63%|██████▎   | 252025/400000 [00:28<00:17, 8544.87it/s] 63%|██████▎   | 252895/400000 [00:28<00:17, 8589.51it/s] 63%|██████▎   | 253755/400000 [00:28<00:17, 8591.23it/s] 64%|██████▎   | 254641/400000 [00:28<00:16, 8668.63it/s] 64%|██████▍   | 255526/400000 [00:28<00:16, 8720.13it/s] 64%|██████▍   | 256415/400000 [00:28<00:16, 8767.38it/s] 64%|██████▍   | 257323/400000 [00:28<00:16, 8857.97it/s] 65%|██████▍   | 258247/400000 [00:29<00:15, 8968.86it/s] 65%|██████▍   | 259172/400000 [00:29<00:15, 9050.46it/s] 65%|██████▌   | 260080/400000 [00:29<00:15, 9056.52it/s] 65%|██████▌   | 260987/400000 [00:29<00:15, 9057.26it/s] 65%|██████▌   | 261897/400000 [00:29<00:15, 9068.76it/s] 66%|██████▌   | 262805/400000 [00:29<00:15, 9061.72it/s] 66%|██████▌   | 263712/400000 [00:29<00:15, 9009.27it/s] 66%|██████▌   | 264644/400000 [00:29<00:14, 9098.21it/s] 66%|██████▋   | 265555/400000 [00:29<00:14, 9085.69it/s] 67%|██████▋   | 266464/400000 [00:29<00:14, 9086.39it/s] 67%|██████▋   | 267373/400000 [00:30<00:14, 9059.52it/s] 67%|██████▋   | 268280/400000 [00:30<00:14, 9031.10it/s] 67%|██████▋   | 269184/400000 [00:30<00:14, 8959.15it/s] 68%|██████▊   | 270081/400000 [00:30<00:14, 8940.35it/s] 68%|██████▊   | 271000/400000 [00:30<00:14, 9013.63it/s] 68%|██████▊   | 271902/400000 [00:30<00:14, 8983.32it/s] 68%|██████▊   | 272819/400000 [00:30<00:14, 9035.90it/s] 68%|██████▊   | 273723/400000 [00:30<00:14, 8897.74it/s] 69%|██████▊   | 274622/400000 [00:30<00:14, 8922.42it/s] 69%|██████▉   | 275545/400000 [00:30<00:13, 9011.18it/s] 69%|██████▉   | 276490/400000 [00:31<00:13, 9138.24it/s] 69%|██████▉   | 277426/400000 [00:31<00:13, 9201.24it/s] 70%|██████▉   | 278347/400000 [00:31<00:13, 8756.90it/s] 70%|██████▉   | 279229/400000 [00:31<00:13, 8773.31it/s] 70%|███████   | 280151/400000 [00:31<00:13, 8899.97it/s] 70%|███████   | 281103/400000 [00:31<00:13, 9075.79it/s] 71%|███████   | 282039/400000 [00:31<00:12, 9157.54it/s] 71%|███████   | 282957/400000 [00:31<00:13, 8978.56it/s] 71%|███████   | 283891/400000 [00:31<00:12, 9082.27it/s] 71%|███████   | 284802/400000 [00:31<00:12, 9039.75it/s] 71%|███████▏  | 285708/400000 [00:32<00:12, 9019.94it/s] 72%|███████▏  | 286644/400000 [00:32<00:12, 9118.09it/s] 72%|███████▏  | 287557/400000 [00:32<00:12, 9033.23it/s] 72%|███████▏  | 288466/400000 [00:32<00:12, 9049.09it/s] 72%|███████▏  | 289372/400000 [00:32<00:12, 9025.41it/s] 73%|███████▎  | 290351/400000 [00:32<00:11, 9240.99it/s] 73%|███████▎  | 291279/400000 [00:32<00:11, 9250.29it/s] 73%|███████▎  | 292208/400000 [00:32<00:11, 9259.83it/s] 73%|███████▎  | 293143/400000 [00:32<00:11, 9284.91it/s] 74%|███████▎  | 294073/400000 [00:32<00:11, 9227.46it/s] 74%|███████▎  | 294997/400000 [00:33<00:11, 9154.58it/s] 74%|███████▍  | 295926/400000 [00:33<00:11, 9193.49it/s] 74%|███████▍  | 296846/400000 [00:33<00:11, 9061.44it/s] 74%|███████▍  | 297753/400000 [00:33<00:11, 9030.97it/s] 75%|███████▍  | 298658/400000 [00:33<00:11, 9033.83it/s] 75%|███████▍  | 299562/400000 [00:33<00:11, 8957.09it/s] 75%|███████▌  | 300474/400000 [00:33<00:11, 9002.62it/s] 75%|███████▌  | 301375/400000 [00:33<00:10, 8975.74it/s] 76%|███████▌  | 302273/400000 [00:33<00:10, 8908.87it/s] 76%|███████▌  | 303165/400000 [00:34<00:11, 8732.30it/s] 76%|███████▌  | 304053/400000 [00:34<00:10, 8776.05it/s] 76%|███████▋  | 305013/400000 [00:34<00:10, 9006.58it/s] 76%|███████▋  | 305916/400000 [00:34<00:10, 8970.31it/s] 77%|███████▋  | 306820/400000 [00:34<00:10, 8989.12it/s] 77%|███████▋  | 307720/400000 [00:34<00:10, 8876.27it/s] 77%|███████▋  | 308609/400000 [00:34<00:10, 8833.60it/s] 77%|███████▋  | 309494/400000 [00:34<00:10, 8831.45it/s] 78%|███████▊  | 310396/400000 [00:34<00:10, 8886.37it/s] 78%|███████▊  | 311286/400000 [00:34<00:10, 8817.03it/s] 78%|███████▊  | 312191/400000 [00:35<00:09, 8885.05it/s] 78%|███████▊  | 313118/400000 [00:35<00:09, 8994.74it/s] 79%|███████▊  | 314058/400000 [00:35<00:09, 9110.78it/s] 79%|███████▊  | 314970/400000 [00:35<00:09, 9110.02it/s] 79%|███████▉  | 315914/400000 [00:35<00:09, 9205.95it/s] 79%|███████▉  | 316836/400000 [00:35<00:09, 9128.31it/s] 79%|███████▉  | 317750/400000 [00:35<00:09, 9083.46it/s] 80%|███████▉  | 318659/400000 [00:35<00:09, 9033.96it/s] 80%|███████▉  | 319567/400000 [00:35<00:08, 9045.79it/s] 80%|████████  | 320480/400000 [00:35<00:08, 9068.52it/s] 80%|████████  | 321426/400000 [00:36<00:08, 9181.81it/s] 81%|████████  | 322350/400000 [00:36<00:08, 9198.33it/s] 81%|████████  | 323284/400000 [00:36<00:08, 9239.81it/s] 81%|████████  | 324212/400000 [00:36<00:08, 9249.60it/s] 81%|████████▏ | 325138/400000 [00:36<00:08, 9102.73it/s] 82%|████████▏ | 326064/400000 [00:36<00:08, 9147.17it/s] 82%|████████▏ | 326980/400000 [00:36<00:07, 9133.97it/s] 82%|████████▏ | 327898/400000 [00:36<00:07, 9147.55it/s] 82%|████████▏ | 328814/400000 [00:36<00:07, 9114.58it/s] 82%|████████▏ | 329740/400000 [00:36<00:07, 9157.54it/s] 83%|████████▎ | 330682/400000 [00:37<00:07, 9234.45it/s] 83%|████████▎ | 331606/400000 [00:37<00:07, 8941.67it/s] 83%|████████▎ | 332503/400000 [00:37<00:07, 8935.34it/s] 83%|████████▎ | 333399/400000 [00:37<00:07, 8885.03it/s] 84%|████████▎ | 334294/400000 [00:37<00:07, 8901.98it/s] 84%|████████▍ | 335220/400000 [00:37<00:07, 9004.00it/s] 84%|████████▍ | 336137/400000 [00:37<00:07, 9052.60it/s] 84%|████████▍ | 337076/400000 [00:37<00:06, 9148.61it/s] 84%|████████▍ | 337992/400000 [00:37<00:06, 9094.59it/s] 85%|████████▍ | 338903/400000 [00:37<00:06, 8997.63it/s] 85%|████████▍ | 339816/400000 [00:38<00:06, 9034.55it/s] 85%|████████▌ | 340720/400000 [00:38<00:06, 9003.05it/s] 85%|████████▌ | 341621/400000 [00:38<00:06, 8912.85it/s] 86%|████████▌ | 342513/400000 [00:38<00:06, 8758.36it/s] 86%|████████▌ | 343390/400000 [00:38<00:06, 8716.12it/s] 86%|████████▌ | 344263/400000 [00:38<00:06, 8622.93it/s] 86%|████████▋ | 345128/400000 [00:38<00:06, 8628.72it/s] 87%|████████▋ | 346027/400000 [00:38<00:06, 8732.86it/s] 87%|████████▋ | 346901/400000 [00:38<00:06, 8692.71it/s] 87%|████████▋ | 347795/400000 [00:38<00:05, 8763.43it/s] 87%|████████▋ | 348708/400000 [00:39<00:05, 8868.93it/s] 87%|████████▋ | 349596/400000 [00:39<00:05, 8860.51it/s] 88%|████████▊ | 350516/400000 [00:39<00:05, 8958.91it/s] 88%|████████▊ | 351413/400000 [00:39<00:05, 8932.92it/s] 88%|████████▊ | 352307/400000 [00:39<00:05, 8837.82it/s] 88%|████████▊ | 353224/400000 [00:39<00:05, 8933.28it/s] 89%|████████▊ | 354138/400000 [00:39<00:05, 8991.84it/s] 89%|████████▉ | 355056/400000 [00:39<00:04, 9046.11it/s] 89%|████████▉ | 355962/400000 [00:39<00:04, 9034.63it/s] 89%|████████▉ | 356866/400000 [00:39<00:04, 8687.62it/s] 89%|████████▉ | 357790/400000 [00:40<00:04, 8845.84it/s] 90%|████████▉ | 358728/400000 [00:40<00:04, 8997.08it/s] 90%|████████▉ | 359671/400000 [00:40<00:04, 9120.68it/s] 90%|█████████ | 360602/400000 [00:40<00:04, 9174.15it/s] 90%|█████████ | 361522/400000 [00:40<00:04, 9113.16it/s] 91%|█████████ | 362471/400000 [00:40<00:04, 9221.15it/s] 91%|█████████ | 363395/400000 [00:40<00:03, 9206.63it/s] 91%|█████████ | 364317/400000 [00:40<00:03, 9184.39it/s] 91%|█████████▏| 365239/400000 [00:40<00:03, 9193.31it/s] 92%|█████████▏| 366159/400000 [00:41<00:03, 9120.06it/s] 92%|█████████▏| 367072/400000 [00:41<00:03, 9101.79it/s] 92%|█████████▏| 367985/400000 [00:41<00:03, 9109.44it/s] 92%|█████████▏| 368897/400000 [00:41<00:03, 9027.94it/s] 92%|█████████▏| 369801/400000 [00:41<00:03, 8910.40it/s] 93%|█████████▎| 370693/400000 [00:41<00:03, 8872.35it/s] 93%|█████████▎| 371603/400000 [00:41<00:03, 8939.22it/s] 93%|█████████▎| 372498/400000 [00:41<00:03, 8925.78it/s] 93%|█████████▎| 373391/400000 [00:41<00:02, 8889.80it/s] 94%|█████████▎| 374324/400000 [00:41<00:02, 9015.67it/s] 94%|█████████▍| 375258/400000 [00:42<00:02, 9109.43it/s] 94%|█████████▍| 376170/400000 [00:42<00:02, 8995.71it/s] 94%|█████████▍| 377071/400000 [00:42<00:02, 8988.07it/s] 94%|█████████▍| 377971/400000 [00:42<00:02, 8938.82it/s] 95%|█████████▍| 378866/400000 [00:42<00:02, 8881.55it/s] 95%|█████████▍| 379755/400000 [00:42<00:02, 8801.20it/s] 95%|█████████▌| 380661/400000 [00:42<00:02, 8874.58it/s] 95%|█████████▌| 381589/400000 [00:42<00:02, 8990.52it/s] 96%|█████████▌| 382489/400000 [00:42<00:01, 8947.10it/s] 96%|█████████▌| 383385/400000 [00:42<00:01, 8908.31it/s] 96%|█████████▌| 384279/400000 [00:43<00:01, 8915.77it/s] 96%|█████████▋| 385198/400000 [00:43<00:01, 8994.59it/s] 97%|█████████▋| 386100/400000 [00:43<00:01, 8999.91it/s] 97%|█████████▋| 387034/400000 [00:43<00:01, 9098.46it/s] 97%|█████████▋| 387945/400000 [00:43<00:01, 9084.10it/s] 97%|█████████▋| 388854/400000 [00:43<00:01, 8984.70it/s] 97%|█████████▋| 389753/400000 [00:43<00:01, 8946.58it/s] 98%|█████████▊| 390672/400000 [00:43<00:01, 9017.22it/s] 98%|█████████▊| 391633/400000 [00:43<00:00, 9185.27it/s] 98%|█████████▊| 392570/400000 [00:43<00:00, 9238.67it/s] 98%|█████████▊| 393495/400000 [00:44<00:00, 9151.14it/s] 99%|█████████▊| 394412/400000 [00:44<00:00, 9154.22it/s] 99%|█████████▉| 395333/400000 [00:44<00:00, 9170.60it/s] 99%|█████████▉| 396251/400000 [00:44<00:00, 9085.27it/s] 99%|█████████▉| 397160/400000 [00:44<00:00, 8959.06it/s]100%|█████████▉| 398057/400000 [00:44<00:00, 8952.91it/s]100%|█████████▉| 398970/400000 [00:44<00:00, 9004.77it/s]100%|█████████▉| 399906/400000 [00:44<00:00, 9107.31it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8936.37it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f59ad563d30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011307907633174302 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.011103412180042585 	 Accuracy: 54

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

  


### Running {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5} {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'} 

  #### Setup Model   ############################################## 

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
2020-05-13 19:25:38.833537: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 19:25:38.838005: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-13 19:25:38.838202: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55daeb1c8e90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 19:25:38.838216: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f5960aa1160> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.6666 - accuracy: 0.5000
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7203 - accuracy: 0.4965 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7011 - accuracy: 0.4978
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6298 - accuracy: 0.5024
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5900 - accuracy: 0.5050
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5516 - accuracy: 0.5075
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.5491 - accuracy: 0.5077
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5608 - accuracy: 0.5069
11000/25000 [============>.................] - ETA: 3s - loss: 7.6053 - accuracy: 0.5040
12000/25000 [=============>................] - ETA: 3s - loss: 7.6066 - accuracy: 0.5039
13000/25000 [==============>...............] - ETA: 2s - loss: 7.5770 - accuracy: 0.5058
14000/25000 [===============>..............] - ETA: 2s - loss: 7.5648 - accuracy: 0.5066
15000/25000 [=================>............] - ETA: 2s - loss: 7.6012 - accuracy: 0.5043
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5928 - accuracy: 0.5048
17000/25000 [===================>..........] - ETA: 1s - loss: 7.5936 - accuracy: 0.5048
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6070 - accuracy: 0.5039
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6174 - accuracy: 0.5032
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6498 - accuracy: 0.5011
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6418 - accuracy: 0.5016
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6624 - accuracy: 0.5003
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6713 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 7s 271us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f590e4396d8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f5952faee10> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 2.1898 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 2.0945 - val_crf_viterbi_accuracy: 0.0267

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


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f1b286d0f60> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 20:13:08.058716
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 20:13:08.065015
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 20:13:08.068205
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 20:13:08.071427
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f1b3449a400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 351527.5938
Epoch 2/10

1/1 [==============================] - 0s 97ms/step - loss: 209541.9688
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 98576.6250
Epoch 4/10

1/1 [==============================] - 0s 98ms/step - loss: 42820.8945
Epoch 5/10

1/1 [==============================] - 0s 98ms/step - loss: 21402.5586
Epoch 6/10

1/1 [==============================] - 0s 100ms/step - loss: 12346.2725
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 8029.9551
Epoch 8/10

1/1 [==============================] - 0s 100ms/step - loss: 5664.0283
Epoch 9/10

1/1 [==============================] - 0s 96ms/step - loss: 4307.8516
Epoch 10/10

1/1 [==============================] - 0s 102ms/step - loss: 3475.5825

  #### Inference Need return ypred, ytrue ######################### 
[[  0.36742768   0.610263    -0.5376874   -0.637426     1.2218804
    2.5774462    1.6530415    0.56060266  -1.2487202    0.03080022
    0.6976792    1.3201421   -1.7823346    1.868024     0.91437703
    0.30236715   2.4068222    0.5048129   -0.34091744  -0.9292885
   -1.3588475   -1.1097893   -1.393707     0.8661748   -1.1909077
   -0.05931416  -0.79134053  -1.8599677    0.4762271    0.8866743
   -0.21521354   0.04191503  -1.3771458   -0.6071481    0.9597601
   -1.1533492    1.55634      0.7933202    1.0347581   -0.8119372
   -0.30728525  -0.3247519    2.0716884   -0.4079506    0.90204054
    0.15930685   2.4681273   -0.49167758   0.7717961   -0.97229177
   -0.53080034   1.2968674    1.4543512   -2.322353     1.4081405
   -1.4540081   -0.31813356  -2.0162308   -0.26786217   0.04980344
   -0.7054089   10.382484    12.742186    12.432805    11.86312
   11.724561    13.543815    11.298591    12.015761    13.012712
   12.732975    11.956036    11.828472    11.487823    11.54399
   10.129693    10.744101    12.625575    13.869171    12.823336
    9.681293    12.008019    11.89932     12.114294    11.818497
   13.102916    11.720164    13.164968    10.227239    13.186837
   13.793834    11.346208    12.206213    12.585402    13.279226
   14.669511    12.499413    12.8195095   13.697496    10.187811
   11.460124    10.186183    13.232725    12.152882    12.430791
   12.802523    13.113497    13.68362     11.196016    13.0784445
   13.12726     12.345157    11.1785965   13.717592    11.631439
   14.292681    12.558078    12.113567    10.261562    11.289123
    3.4100237   -0.0486348   -0.39213783   0.86663437  -1.1791534
   -1.2400612    1.9557174    0.92904705   0.02526891   1.4309094
   -0.37942505   0.34876865  -0.20641994   0.57030475  -0.10292488
   -2.206607    -0.28176984  -0.15596692  -0.52099603   1.7252444
   -1.5470805   -3.8049426    1.0331116   -0.6196325    1.971361
   -0.73592234  -0.27131167  -0.7775692   -0.16362855   0.15005995
   -0.16926646  -0.5144759    1.5546893    0.3264943   -0.13342887
    0.82676935   0.27533165  -0.38224047  -0.08385831   0.8429788
    2.1763568   -0.11306153  -2.638029     2.8436759   -0.38641602
   -1.6884242   -2.1045532   -0.6625307   -1.7771227    0.4710286
    2.3646104    0.07481439  -2.3048687    0.8507801   -1.3902948
   -0.5640607   -1.2571013    0.25103587   3.24223     -0.7700931
    0.9342671    0.8543328    1.2352934    1.2607441    0.3731712
    0.13150752   3.6469254    1.6003159    0.21831143   0.20925748
    3.447444     0.5923093    0.07973921   2.3966236    1.3207273
    0.11049354   0.15424722   0.8256091    3.351956     0.6261325
    1.1780996    0.4673537    3.2130995    0.9740664    0.3691063
    2.3749008    0.45444798   3.3072186    4.0588355    1.1754203
    0.5135909    1.8694663    1.094661     2.457695     1.0758739
    1.6663938    0.05338103   1.6461802    1.8473629    1.920876
    2.4408517    0.3250389    1.6038649    0.35052836   1.735733
    0.27732623   0.29041547   0.7226423    1.5143311    2.5086021
    2.2274404    0.11232793   0.82447696   2.8163652    0.3410566
    0.27991635   2.478644     0.9987673    2.7706847    3.2848516
    0.55487883  11.321285    13.701499    11.805206    11.862226
   14.570775    10.545422    15.025102    11.723866    11.436089
   12.57657     12.353845     9.555221    11.276071     9.2169
   13.676097    11.533837    14.428918    13.646713    14.1508
    9.210117    12.341659    11.277858    11.496546    14.860399
   12.7425585   14.571798    13.819108    12.162451    10.543592
    8.920291    13.286335    13.540114    14.181715    14.336513
   12.839056    10.94305     10.169205    12.333892    10.973241
    9.2153015   13.599531    10.722119     9.70635     11.708668
   13.661541    12.094067    12.777395    11.490015    12.055497
    9.983111     9.978133    14.516817    11.785432    12.169282
   12.690415    10.127828    10.805528    14.8193035   10.365479
    1.8565891    2.1467786    0.4549806    0.41267198   0.80943143
    0.40616387   0.8787578    1.3517615    0.4026522    0.44697404
    0.23141444   0.59377974   0.95456314   1.5983684    1.4276509
    0.27775824   0.21531916   1.0705652    0.26434326   2.8150039
    1.4077195    0.21402156   2.8694472    1.7738001    3.2365942
    0.50520307   1.7472968    0.34359175   1.7254757    0.08101386
    0.6941608    0.6631314    1.7181666    0.59263384   0.20774299
    0.5774657    0.5834909    0.69164073   1.3688471    1.6110523
    3.143055     0.7834903    2.203526     0.969355     1.921941
    0.28610188   1.618099     1.2594495    0.26147676   1.2260184
    2.583148     0.55040514   3.3148556    0.36822325   1.6254914
    2.950602     2.118084     1.3166711    0.14055014   0.33173203
  -17.884293     1.6503994  -16.988375  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 20:13:16.543331
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   89.7741
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 20:13:16.547733
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8095.88
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 20:13:16.551447
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                     89.85
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 20:13:16.554847
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -724.043
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139754277159432
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139751747490368
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139751747490872
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139751747491376
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139751747491880
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139751747492384

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f1b28390b38> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.576637
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.544363
grad_step = 000002, loss = 0.523487
grad_step = 000003, loss = 0.504059
grad_step = 000004, loss = 0.484872
grad_step = 000005, loss = 0.465739
grad_step = 000006, loss = 0.453303
grad_step = 000007, loss = 0.442045
grad_step = 000008, loss = 0.428835
grad_step = 000009, loss = 0.417549
grad_step = 000010, loss = 0.407677
grad_step = 000011, loss = 0.398122
grad_step = 000012, loss = 0.387978
grad_step = 000013, loss = 0.377481
grad_step = 000014, loss = 0.367444
grad_step = 000015, loss = 0.357502
grad_step = 000016, loss = 0.346912
grad_step = 000017, loss = 0.336110
grad_step = 000018, loss = 0.325574
grad_step = 000019, loss = 0.315597
grad_step = 000020, loss = 0.306118
grad_step = 000021, loss = 0.296593
grad_step = 000022, loss = 0.286794
grad_step = 000023, loss = 0.277308
grad_step = 000024, loss = 0.268241
grad_step = 000025, loss = 0.259062
grad_step = 000026, loss = 0.250001
grad_step = 000027, loss = 0.241214
grad_step = 000028, loss = 0.232425
grad_step = 000029, loss = 0.223725
grad_step = 000030, loss = 0.215328
grad_step = 000031, loss = 0.207121
grad_step = 000032, loss = 0.198890
grad_step = 000033, loss = 0.190737
grad_step = 000034, loss = 0.182963
grad_step = 000035, loss = 0.175395
grad_step = 000036, loss = 0.167817
grad_step = 000037, loss = 0.160509
grad_step = 000038, loss = 0.153531
grad_step = 000039, loss = 0.146553
grad_step = 000040, loss = 0.139708
grad_step = 000041, loss = 0.133189
grad_step = 000042, loss = 0.126869
grad_step = 000043, loss = 0.120629
grad_step = 000044, loss = 0.114684
grad_step = 000045, loss = 0.108961
grad_step = 000046, loss = 0.103337
grad_step = 000047, loss = 0.097969
grad_step = 000048, loss = 0.092829
grad_step = 000049, loss = 0.087851
grad_step = 000050, loss = 0.083076
grad_step = 000051, loss = 0.078505
grad_step = 000052, loss = 0.074109
grad_step = 000053, loss = 0.069887
grad_step = 000054, loss = 0.065871
grad_step = 000055, loss = 0.062023
grad_step = 000056, loss = 0.058322
grad_step = 000057, loss = 0.054839
grad_step = 000058, loss = 0.051491
grad_step = 000059, loss = 0.048307
grad_step = 000060, loss = 0.045295
grad_step = 000061, loss = 0.042397
grad_step = 000062, loss = 0.039670
grad_step = 000063, loss = 0.037082
grad_step = 000064, loss = 0.034622
grad_step = 000065, loss = 0.032300
grad_step = 000066, loss = 0.030104
grad_step = 000067, loss = 0.028027
grad_step = 000068, loss = 0.026081
grad_step = 000069, loss = 0.024235
grad_step = 000070, loss = 0.022499
grad_step = 000071, loss = 0.020872
grad_step = 000072, loss = 0.019337
grad_step = 000073, loss = 0.017909
grad_step = 000074, loss = 0.016561
grad_step = 000075, loss = 0.015303
grad_step = 000076, loss = 0.014132
grad_step = 000077, loss = 0.013039
grad_step = 000078, loss = 0.012031
grad_step = 000079, loss = 0.011108
grad_step = 000080, loss = 0.010244
grad_step = 000081, loss = 0.009413
grad_step = 000082, loss = 0.008638
grad_step = 000083, loss = 0.007974
grad_step = 000084, loss = 0.007358
grad_step = 000085, loss = 0.006750
grad_step = 000086, loss = 0.006226
grad_step = 000087, loss = 0.005770
grad_step = 000088, loss = 0.005315
grad_step = 000089, loss = 0.004917
grad_step = 000090, loss = 0.004585
grad_step = 000091, loss = 0.004260
grad_step = 000092, loss = 0.003968
grad_step = 000093, loss = 0.003730
grad_step = 000094, loss = 0.003513
grad_step = 000095, loss = 0.003307
grad_step = 000096, loss = 0.003138
grad_step = 000097, loss = 0.002998
grad_step = 000098, loss = 0.002866
grad_step = 000099, loss = 0.002749
grad_step = 000100, loss = 0.002659
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002582
grad_step = 000102, loss = 0.002507
grad_step = 000103, loss = 0.002444
grad_step = 000104, loss = 0.002399
grad_step = 000105, loss = 0.002360
grad_step = 000106, loss = 0.002323
grad_step = 000107, loss = 0.002292
grad_step = 000108, loss = 0.002273
grad_step = 000109, loss = 0.002264
grad_step = 000110, loss = 0.002261
grad_step = 000111, loss = 0.002271
grad_step = 000112, loss = 0.002285
grad_step = 000113, loss = 0.002288
grad_step = 000114, loss = 0.002252
grad_step = 000115, loss = 0.002198
grad_step = 000116, loss = 0.002167
grad_step = 000117, loss = 0.002176
grad_step = 000118, loss = 0.002199
grad_step = 000119, loss = 0.002202
grad_step = 000120, loss = 0.002173
grad_step = 000121, loss = 0.002138
grad_step = 000122, loss = 0.002125
grad_step = 000123, loss = 0.002136
grad_step = 000124, loss = 0.002148
grad_step = 000125, loss = 0.002142
grad_step = 000126, loss = 0.002120
grad_step = 000127, loss = 0.002098
grad_step = 000128, loss = 0.002087
grad_step = 000129, loss = 0.002088
grad_step = 000130, loss = 0.002094
grad_step = 000131, loss = 0.002097
grad_step = 000132, loss = 0.002092
grad_step = 000133, loss = 0.002079
grad_step = 000134, loss = 0.002064
grad_step = 000135, loss = 0.002051
grad_step = 000136, loss = 0.002042
grad_step = 000137, loss = 0.002037
grad_step = 000138, loss = 0.002034
grad_step = 000139, loss = 0.002032
grad_step = 000140, loss = 0.002033
grad_step = 000141, loss = 0.002037
grad_step = 000142, loss = 0.002050
grad_step = 000143, loss = 0.002078
grad_step = 000144, loss = 0.002129
grad_step = 000145, loss = 0.002192
grad_step = 000146, loss = 0.002236
grad_step = 000147, loss = 0.002196
grad_step = 000148, loss = 0.002087
grad_step = 000149, loss = 0.002000
grad_step = 000150, loss = 0.002003
grad_step = 000151, loss = 0.002062
grad_step = 000152, loss = 0.002103
grad_step = 000153, loss = 0.002088
grad_step = 000154, loss = 0.002031
grad_step = 000155, loss = 0.001988
grad_step = 000156, loss = 0.001979
grad_step = 000157, loss = 0.002001
grad_step = 000158, loss = 0.002022
grad_step = 000159, loss = 0.002018
grad_step = 000160, loss = 0.001992
grad_step = 000161, loss = 0.001959
grad_step = 000162, loss = 0.001946
grad_step = 000163, loss = 0.001956
grad_step = 000164, loss = 0.001974
grad_step = 000165, loss = 0.001983
grad_step = 000166, loss = 0.001971
grad_step = 000167, loss = 0.001951
grad_step = 000168, loss = 0.001934
grad_step = 000169, loss = 0.001931
grad_step = 000170, loss = 0.001941
grad_step = 000171, loss = 0.001962
grad_step = 000172, loss = 0.001990
grad_step = 000173, loss = 0.002006
grad_step = 000174, loss = 0.002001
grad_step = 000175, loss = 0.001953
grad_step = 000176, loss = 0.001918
grad_step = 000177, loss = 0.001923
grad_step = 000178, loss = 0.001944
grad_step = 000179, loss = 0.001957
grad_step = 000180, loss = 0.001967
grad_step = 000181, loss = 0.002037
grad_step = 000182, loss = 0.002160
grad_step = 000183, loss = 0.002337
grad_step = 000184, loss = 0.002362
grad_step = 000185, loss = 0.002253
grad_step = 000186, loss = 0.001982
grad_step = 000187, loss = 0.001886
grad_step = 000188, loss = 0.002020
grad_step = 000189, loss = 0.002135
grad_step = 000190, loss = 0.002068
grad_step = 000191, loss = 0.001895
grad_step = 000192, loss = 0.001895
grad_step = 000193, loss = 0.002017
grad_step = 000194, loss = 0.002023
grad_step = 000195, loss = 0.001918
grad_step = 000196, loss = 0.001866
grad_step = 000197, loss = 0.001931
grad_step = 000198, loss = 0.001976
grad_step = 000199, loss = 0.001915
grad_step = 000200, loss = 0.001863
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001890
grad_step = 000202, loss = 0.001930
grad_step = 000203, loss = 0.001914
grad_step = 000204, loss = 0.001866
grad_step = 000205, loss = 0.001859
grad_step = 000206, loss = 0.001887
grad_step = 000207, loss = 0.001893
grad_step = 000208, loss = 0.001868
grad_step = 000209, loss = 0.001847
grad_step = 000210, loss = 0.001851
grad_step = 000211, loss = 0.001868
grad_step = 000212, loss = 0.001867
grad_step = 000213, loss = 0.001852
grad_step = 000214, loss = 0.001838
grad_step = 000215, loss = 0.001837
grad_step = 000216, loss = 0.001847
grad_step = 000217, loss = 0.001852
grad_step = 000218, loss = 0.001849
grad_step = 000219, loss = 0.001838
grad_step = 000220, loss = 0.001830
grad_step = 000221, loss = 0.001832
grad_step = 000222, loss = 0.001840
grad_step = 000223, loss = 0.001855
grad_step = 000224, loss = 0.001869
grad_step = 000225, loss = 0.001889
grad_step = 000226, loss = 0.001913
grad_step = 000227, loss = 0.001944
grad_step = 000228, loss = 0.001974
grad_step = 000229, loss = 0.001956
grad_step = 000230, loss = 0.001898
grad_step = 000231, loss = 0.001826
grad_step = 000232, loss = 0.001814
grad_step = 000233, loss = 0.001853
grad_step = 000234, loss = 0.001876
grad_step = 000235, loss = 0.001851
grad_step = 000236, loss = 0.001820
grad_step = 000237, loss = 0.001829
grad_step = 000238, loss = 0.001850
grad_step = 000239, loss = 0.001835
grad_step = 000240, loss = 0.001807
grad_step = 000241, loss = 0.001802
grad_step = 000242, loss = 0.001817
grad_step = 000243, loss = 0.001820
grad_step = 000244, loss = 0.001805
grad_step = 000245, loss = 0.001796
grad_step = 000246, loss = 0.001806
grad_step = 000247, loss = 0.001821
grad_step = 000248, loss = 0.001832
grad_step = 000249, loss = 0.001843
grad_step = 000250, loss = 0.001892
grad_step = 000251, loss = 0.001993
grad_step = 000252, loss = 0.002208
grad_step = 000253, loss = 0.002431
grad_step = 000254, loss = 0.002679
grad_step = 000255, loss = 0.002441
grad_step = 000256, loss = 0.002055
grad_step = 000257, loss = 0.001796
grad_step = 000258, loss = 0.001978
grad_step = 000259, loss = 0.002233
grad_step = 000260, loss = 0.002045
grad_step = 000261, loss = 0.001796
grad_step = 000262, loss = 0.001874
grad_step = 000263, loss = 0.002037
grad_step = 000264, loss = 0.001960
grad_step = 000265, loss = 0.001788
grad_step = 000266, loss = 0.001855
grad_step = 000267, loss = 0.001963
grad_step = 000268, loss = 0.001876
grad_step = 000269, loss = 0.001786
grad_step = 000270, loss = 0.001837
grad_step = 000271, loss = 0.001870
grad_step = 000272, loss = 0.001840
grad_step = 000273, loss = 0.001792
grad_step = 000274, loss = 0.001798
grad_step = 000275, loss = 0.001826
grad_step = 000276, loss = 0.001823
grad_step = 000277, loss = 0.001786
grad_step = 000278, loss = 0.001771
grad_step = 000279, loss = 0.001794
grad_step = 000280, loss = 0.001809
grad_step = 000281, loss = 0.001778
grad_step = 000282, loss = 0.001755
grad_step = 000283, loss = 0.001774
grad_step = 000284, loss = 0.001791
grad_step = 000285, loss = 0.001770
grad_step = 000286, loss = 0.001748
grad_step = 000287, loss = 0.001757
grad_step = 000288, loss = 0.001771
grad_step = 000289, loss = 0.001762
grad_step = 000290, loss = 0.001747
grad_step = 000291, loss = 0.001746
grad_step = 000292, loss = 0.001752
grad_step = 000293, loss = 0.001751
grad_step = 000294, loss = 0.001746
grad_step = 000295, loss = 0.001742
grad_step = 000296, loss = 0.001740
grad_step = 000297, loss = 0.001738
grad_step = 000298, loss = 0.001738
grad_step = 000299, loss = 0.001739
grad_step = 000300, loss = 0.001737
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001731
grad_step = 000302, loss = 0.001728
grad_step = 000303, loss = 0.001729
grad_step = 000304, loss = 0.001731
grad_step = 000305, loss = 0.001730
grad_step = 000306, loss = 0.001726
grad_step = 000307, loss = 0.001723
grad_step = 000308, loss = 0.001722
grad_step = 000309, loss = 0.001721
grad_step = 000310, loss = 0.001719
grad_step = 000311, loss = 0.001719
grad_step = 000312, loss = 0.001719
grad_step = 000313, loss = 0.001718
grad_step = 000314, loss = 0.001717
grad_step = 000315, loss = 0.001714
grad_step = 000316, loss = 0.001713
grad_step = 000317, loss = 0.001711
grad_step = 000318, loss = 0.001710
grad_step = 000319, loss = 0.001708
grad_step = 000320, loss = 0.001707
grad_step = 000321, loss = 0.001705
grad_step = 000322, loss = 0.001704
grad_step = 000323, loss = 0.001703
grad_step = 000324, loss = 0.001702
grad_step = 000325, loss = 0.001701
grad_step = 000326, loss = 0.001699
grad_step = 000327, loss = 0.001698
grad_step = 000328, loss = 0.001698
grad_step = 000329, loss = 0.001697
grad_step = 000330, loss = 0.001696
grad_step = 000331, loss = 0.001697
grad_step = 000332, loss = 0.001699
grad_step = 000333, loss = 0.001707
grad_step = 000334, loss = 0.001727
grad_step = 000335, loss = 0.001779
grad_step = 000336, loss = 0.001901
grad_step = 000337, loss = 0.002201
grad_step = 000338, loss = 0.002701
grad_step = 000339, loss = 0.003435
grad_step = 000340, loss = 0.003096
grad_step = 000341, loss = 0.002225
grad_step = 000342, loss = 0.001707
grad_step = 000343, loss = 0.002289
grad_step = 000344, loss = 0.002675
grad_step = 000345, loss = 0.001956
grad_step = 000346, loss = 0.001825
grad_step = 000347, loss = 0.002370
grad_step = 000348, loss = 0.002044
grad_step = 000349, loss = 0.001760
grad_step = 000350, loss = 0.002064
grad_step = 000351, loss = 0.001957
grad_step = 000352, loss = 0.001745
grad_step = 000353, loss = 0.001936
grad_step = 000354, loss = 0.001847
grad_step = 000355, loss = 0.001750
grad_step = 000356, loss = 0.001866
grad_step = 000357, loss = 0.001788
grad_step = 000358, loss = 0.001732
grad_step = 000359, loss = 0.001828
grad_step = 000360, loss = 0.001740
grad_step = 000361, loss = 0.001727
grad_step = 000362, loss = 0.001791
grad_step = 000363, loss = 0.001711
grad_step = 000364, loss = 0.001724
grad_step = 000365, loss = 0.001764
grad_step = 000366, loss = 0.001694
grad_step = 000367, loss = 0.001720
grad_step = 000368, loss = 0.001745
grad_step = 000369, loss = 0.001683
grad_step = 000370, loss = 0.001715
grad_step = 000371, loss = 0.001722
grad_step = 000372, loss = 0.001679
grad_step = 000373, loss = 0.001705
grad_step = 000374, loss = 0.001708
grad_step = 000375, loss = 0.001674
grad_step = 000376, loss = 0.001697
grad_step = 000377, loss = 0.001696
grad_step = 000378, loss = 0.001671
grad_step = 000379, loss = 0.001687
grad_step = 000380, loss = 0.001688
grad_step = 000381, loss = 0.001667
grad_step = 000382, loss = 0.001679
grad_step = 000383, loss = 0.001679
grad_step = 000384, loss = 0.001664
grad_step = 000385, loss = 0.001670
grad_step = 000386, loss = 0.001673
grad_step = 000387, loss = 0.001660
grad_step = 000388, loss = 0.001663
grad_step = 000389, loss = 0.001668
grad_step = 000390, loss = 0.001658
grad_step = 000391, loss = 0.001656
grad_step = 000392, loss = 0.001662
grad_step = 000393, loss = 0.001656
grad_step = 000394, loss = 0.001651
grad_step = 000395, loss = 0.001654
grad_step = 000396, loss = 0.001654
grad_step = 000397, loss = 0.001648
grad_step = 000398, loss = 0.001648
grad_step = 000399, loss = 0.001649
grad_step = 000400, loss = 0.001646
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001644
grad_step = 000402, loss = 0.001644
grad_step = 000403, loss = 0.001644
grad_step = 000404, loss = 0.001641
grad_step = 000405, loss = 0.001640
grad_step = 000406, loss = 0.001640
grad_step = 000407, loss = 0.001638
grad_step = 000408, loss = 0.001636
grad_step = 000409, loss = 0.001636
grad_step = 000410, loss = 0.001635
grad_step = 000411, loss = 0.001633
grad_step = 000412, loss = 0.001632
grad_step = 000413, loss = 0.001631
grad_step = 000414, loss = 0.001630
grad_step = 000415, loss = 0.001629
grad_step = 000416, loss = 0.001628
grad_step = 000417, loss = 0.001627
grad_step = 000418, loss = 0.001626
grad_step = 000419, loss = 0.001625
grad_step = 000420, loss = 0.001624
grad_step = 000421, loss = 0.001623
grad_step = 000422, loss = 0.001622
grad_step = 000423, loss = 0.001620
grad_step = 000424, loss = 0.001619
grad_step = 000425, loss = 0.001619
grad_step = 000426, loss = 0.001618
grad_step = 000427, loss = 0.001616
grad_step = 000428, loss = 0.001615
grad_step = 000429, loss = 0.001614
grad_step = 000430, loss = 0.001613
grad_step = 000431, loss = 0.001612
grad_step = 000432, loss = 0.001611
grad_step = 000433, loss = 0.001610
grad_step = 000434, loss = 0.001609
grad_step = 000435, loss = 0.001608
grad_step = 000436, loss = 0.001607
grad_step = 000437, loss = 0.001606
grad_step = 000438, loss = 0.001605
grad_step = 000439, loss = 0.001604
grad_step = 000440, loss = 0.001603
grad_step = 000441, loss = 0.001604
grad_step = 000442, loss = 0.001606
grad_step = 000443, loss = 0.001611
grad_step = 000444, loss = 0.001618
grad_step = 000445, loss = 0.001625
grad_step = 000446, loss = 0.001630
grad_step = 000447, loss = 0.001633
grad_step = 000448, loss = 0.001632
grad_step = 000449, loss = 0.001627
grad_step = 000450, loss = 0.001618
grad_step = 000451, loss = 0.001610
grad_step = 000452, loss = 0.001605
grad_step = 000453, loss = 0.001601
grad_step = 000454, loss = 0.001599
grad_step = 000455, loss = 0.001597
grad_step = 000456, loss = 0.001594
grad_step = 000457, loss = 0.001591
grad_step = 000458, loss = 0.001588
grad_step = 000459, loss = 0.001586
grad_step = 000460, loss = 0.001586
grad_step = 000461, loss = 0.001587
grad_step = 000462, loss = 0.001591
grad_step = 000463, loss = 0.001597
grad_step = 000464, loss = 0.001607
grad_step = 000465, loss = 0.001625
grad_step = 000466, loss = 0.001656
grad_step = 000467, loss = 0.001708
grad_step = 000468, loss = 0.001795
grad_step = 000469, loss = 0.001914
grad_step = 000470, loss = 0.002069
grad_step = 000471, loss = 0.002143
grad_step = 000472, loss = 0.002115
grad_step = 000473, loss = 0.001878
grad_step = 000474, loss = 0.001652
grad_step = 000475, loss = 0.001582
grad_step = 000476, loss = 0.001682
grad_step = 000477, loss = 0.001808
grad_step = 000478, loss = 0.001785
grad_step = 000479, loss = 0.001661
grad_step = 000480, loss = 0.001574
grad_step = 000481, loss = 0.001615
grad_step = 000482, loss = 0.001697
grad_step = 000483, loss = 0.001684
grad_step = 000484, loss = 0.001606
grad_step = 000485, loss = 0.001566
grad_step = 000486, loss = 0.001604
grad_step = 000487, loss = 0.001646
grad_step = 000488, loss = 0.001620
grad_step = 000489, loss = 0.001571
grad_step = 000490, loss = 0.001563
grad_step = 000491, loss = 0.001594
grad_step = 000492, loss = 0.001612
grad_step = 000493, loss = 0.001588
grad_step = 000494, loss = 0.001559
grad_step = 000495, loss = 0.001560
grad_step = 000496, loss = 0.001580
grad_step = 000497, loss = 0.001585
grad_step = 000498, loss = 0.001568
grad_step = 000499, loss = 0.001551
grad_step = 000500, loss = 0.001552
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001563
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

  date_run                              2020-05-13 20:13:39.442952
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.266181
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 20:13:39.449536
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.181198
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 20:13:39.457484
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.149885
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 20:13:39.462997
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.75337
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
0   2020-05-13 20:13:08.058716  ...    mean_absolute_error
1   2020-05-13 20:13:08.065015  ...     mean_squared_error
2   2020-05-13 20:13:08.068205  ...  median_absolute_error
3   2020-05-13 20:13:08.071427  ...               r2_score
4   2020-05-13 20:13:16.543331  ...    mean_absolute_error
5   2020-05-13 20:13:16.547733  ...     mean_squared_error
6   2020-05-13 20:13:16.551447  ...  median_absolute_error
7   2020-05-13 20:13:16.554847  ...               r2_score
8   2020-05-13 20:13:39.442952  ...    mean_absolute_error
9   2020-05-13 20:13:39.449536  ...     mean_squared_error
10  2020-05-13 20:13:39.457484  ...  median_absolute_error
11  2020-05-13 20:13:39.462997  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb050291c50> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 20%|█▉        | 1941504/9912422 [00:00<00:00, 18991294.07it/s] 87%|████████▋ | 8667136/9912422 [00:00<00:00, 24198518.36it/s]9920512it [00:00, 29764182.93it/s]                             
0it [00:00, ?it/s]32768it [00:00, 540534.30it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 461699.29it/s]1654784it [00:00, 11234725.16it/s]                         
0it [00:00, ?it/s]8192it [00:00, 205818.42it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb002c4deb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb00227c0f0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb002c4deb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb0021d2128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fafffa0d518> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fafff9f8c88> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb002c4deb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb00218f748> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fafffa0d518> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb00204a550> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f9fc1a9d208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=38f57e2854d95147429d48568b5cfc0f60294e0b1ea3c5356baf78620b31502f
  Stored in directory: /tmp/pip-ephem-wheel-cache-ta893jf_/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f9fb7c07080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2351104/17464789 [===>..........................] - ETA: 0s
 6283264/17464789 [=========>....................] - ETA: 0s
10846208/17464789 [=================>............] - ETA: 0s
15466496/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 20:15:04.819437: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 20:15:04.822854: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-13 20:15:04.823010: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d985387ff0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 20:15:04.823024: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.4366 - accuracy: 0.5150
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4060 - accuracy: 0.5170 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5542 - accuracy: 0.5073
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6705 - accuracy: 0.4997
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7065 - accuracy: 0.4974
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7280 - accuracy: 0.4960
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7849 - accuracy: 0.4923
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7605 - accuracy: 0.4939
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7654 - accuracy: 0.4936
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7203 - accuracy: 0.4965
11000/25000 [============>.................] - ETA: 4s - loss: 7.7210 - accuracy: 0.4965
12000/25000 [=============>................] - ETA: 4s - loss: 7.7190 - accuracy: 0.4966
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7126 - accuracy: 0.4970
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7345 - accuracy: 0.4956
15000/25000 [=================>............] - ETA: 3s - loss: 7.7136 - accuracy: 0.4969
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7117 - accuracy: 0.4971
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7198 - accuracy: 0.4965
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7211 - accuracy: 0.4964
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7199 - accuracy: 0.4965
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7165 - accuracy: 0.4967
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7170 - accuracy: 0.4967
22000/25000 [=========================>....] - ETA: 0s - loss: 7.7029 - accuracy: 0.4976
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6826 - accuracy: 0.4990
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 9s 370us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 20:15:20.962878
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 20:15:20.962878  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<130:06:33, 1.84kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:04<91:18:22, 2.62kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:04<63:57:54, 3.74kB/s] .vector_cache/glove.6B.zip:   0%|          | 885k/862M [00:04<44:45:13, 5.35kB/s].vector_cache/glove.6B.zip:   0%|          | 2.92M/862M [00:04<31:15:26, 7.64kB/s].vector_cache/glove.6B.zip:   1%|          | 6.39M/862M [00:05<21:47:37, 10.9kB/s].vector_cache/glove.6B.zip:   1%|          | 9.62M/862M [00:05<15:12:12, 15.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.3M/862M [00:05<10:34:22, 22.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.9M/862M [00:05<7:22:15, 31.8kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.0M/862M [00:05<5:08:12, 45.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.5M/862M [00:05<3:34:57, 64.8kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.5M/862M [00:05<2:29:51, 92.5kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.9M/862M [00:05<1:44:35, 132kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 37.2M/862M [00:05<1:13:02, 188kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.7M/862M [00:06<51:02, 268kB/s]  .vector_cache/glove.6B.zip:   5%|▌         | 44.7M/862M [00:06<35:39, 382kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.2M/862M [00:06<24:57, 543kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:06<17:55, 753kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:08<14:23, 933kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:08<11:58, 1.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:09<08:45, 1.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:09<06:20, 2.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:10<11:12, 1.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:10<10:03, 1.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:11<07:35, 1.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:12<07:37, 1.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:12<07:30, 1.77MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:13<05:47, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:13<04:10, 3.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:14<34:01, 389kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:14<26:00, 508kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:14<18:44, 705kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:16<15:22, 856kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:16<12:54, 1.02MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:16<09:29, 1.38MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.7M/862M [00:17<06:52, 1.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:18<09:36, 1.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:18<08:52, 1.47MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:18<06:39, 1.96MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:19<04:49, 2.70MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:20<12:54, 1.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:20<11:10, 1.16MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:20<08:21, 1.55MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:20<06:07, 2.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:22<07:53, 1.64MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:22<07:37, 1.70MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:22<05:50, 2.21MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.8M/862M [00:22<04:21, 2.96MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:24<07:09, 1.80MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:24<09:15, 1.39MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:24<07:34, 1.70MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.9M/862M [00:25<05:31, 2.32MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:26<07:32, 1.70MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:26<07:23, 1.73MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:26<05:42, 2.24MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.0M/862M [00:27<04:14, 3.01MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:28<06:47, 1.88MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:28<06:50, 1.86MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:28<05:16, 2.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<05:54, 2.15MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<06:11, 2.04MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:30<04:51, 2.60MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<05:36, 2.25MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<05:56, 2.12MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:32<04:39, 2.70MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<05:27, 2.30MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<05:58, 2.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:34<04:37, 2.70MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:34<03:24, 3.66MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<10:08, 1.23MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:37<18:21, 679kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<13:57, 888kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:38<10:46, 1.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:38<07:47, 1.59MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:40<09:10, 1.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<08:29, 1.45MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:40<06:26, 1.91MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:40<04:38, 2.65MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<28:59, 423kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<22:17, 550kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:42<16:02, 763kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:42<11:20, 1.08MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<19:25, 628kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<15:36, 781kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:44<11:24, 1.07MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<10:05, 1.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<09:02, 1.34MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:46<06:44, 1.79MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:46<04:53, 2.47MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<09:35, 1.26MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<08:22, 1.44MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:48<06:30, 1.85MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:48<04:45, 2.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<06:53, 1.74MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<06:46, 1.77MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:50<05:14, 2.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<05:44, 2.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<05:58, 1.99MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:52<04:40, 2.55MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:52<03:23, 3.49MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:54<23:26, 505kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:54<18:20, 645kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:54<13:18, 889kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<09:24, 1.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:56<1:09:17, 170kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:56<50:29, 233kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:56<35:46, 329kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:56<25:05, 467kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<30:15, 387kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<23:07, 506kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:58<16:35, 705kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:58<11:51, 984kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<11:44, 991kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<10:08, 1.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:00<07:31, 1.55MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:00<05:22, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:02<22:44, 509kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:02<17:42, 653kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:02<12:47, 903kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:02<09:09, 1.26MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:04<10:22, 1.11MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:04<09:08, 1.26MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:04<06:52, 1.67MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:06<06:47, 1.68MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:06<06:37, 1.72MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:06<05:06, 2.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<03:40, 3.09MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:08<5:35:37, 33.9kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:08<3:56:44, 48.0kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:08<2:45:56, 68.4kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:10<1:57:38, 96.0kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:10<1:24:09, 134kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:10<59:14, 190kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<41:30, 271kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<35:14, 319kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:12<26:31, 423kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:12<18:56, 592kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<13:23, 835kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<14:34, 765kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:14<12:07, 920kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:14<08:56, 1.25MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<08:09, 1.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:16<07:32, 1.47MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:16<05:42, 1.94MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:16<04:13, 2.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<06:14, 1.77MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:18<06:11, 1.78MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:18<04:45, 2.31MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<05:14, 2.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<05:30, 1.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:20<04:15, 2.57MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:20<03:12, 3.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<05:28, 1.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<05:19, 2.05MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:22<04:02, 2.69MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:22<03:00, 3.60MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<07:05, 1.52MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<08:47, 1.23MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:24<07:03, 1.53MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:24<05:06, 2.11MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<06:27, 1.66MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<06:22, 1.68MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:26<04:52, 2.20MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<05:15, 2.03MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<07:27, 1.43MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<06:00, 1.78MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:28<04:34, 2.32MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<05:05, 2.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<05:22, 1.98MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:29<04:11, 2.53MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<04:45, 2.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<05:04, 2.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:31<03:55, 2.68MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:32<02:55, 3.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<05:54, 1.77MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<05:50, 1.79MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:33<04:31, 2.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<04:58, 2.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<05:11, 2.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:35<03:59, 2.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:35<03:00, 3.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<05:11, 1.99MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<05:18, 1.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<04:03, 2.54MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:37<03:03, 3.36MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<05:27, 1.88MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<06:22, 1.61MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<05:06, 2.00MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:40<03:44, 2.73MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<05:38, 1.81MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<05:41, 1.79MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:41<04:25, 2.30MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<04:50, 2.09MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<05:02, 2.01MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:43<03:51, 2.62MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:43<02:53, 3.49MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<05:48, 1.73MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<05:42, 1.76MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:45<04:20, 2.31MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:45<03:09, 3.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:47<08:48, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:47<07:25, 1.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:47<05:30, 1.81MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<05:49, 1.70MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<07:20, 1.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:49<05:51, 1.69MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:49<04:18, 2.30MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<05:11, 1.89MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<04:59, 1.97MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:51<04:08, 2.37MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:51<03:03, 3.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<05:02, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<05:08, 1.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:53<03:59, 2.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<04:29, 2.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<04:43, 2.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:55<03:42, 2.61MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<04:16, 2.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<04:34, 2.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<03:35, 2.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<04:10, 2.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<04:33, 2.10MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:59<03:31, 2.71MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<02:33, 3.71MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<21:56, 433kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<16:56, 561kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:01<12:14, 775kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<10:10, 927kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<08:40, 1.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:03<06:27, 1.46MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<06:11, 1.51MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<07:25, 1.26MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<05:58, 1.56MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:05<04:20, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:07<05:36, 1.66MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<05:27, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:07<04:11, 2.21MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:09<04:32, 2.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<04:40, 1.97MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:09<03:35, 2.56MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:09<02:38, 3.47MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<06:47, 1.35MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<06:14, 1.47MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:11<04:41, 1.95MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:11<03:26, 2.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<06:00, 1.51MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<05:41, 1.60MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:13<04:20, 2.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:13<03:07, 2.89MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:15<23:24, 385kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:15<17:52, 504kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:15<12:52, 700kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:17<10:31, 851kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:17<08:50, 1.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:17<06:29, 1.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:17<04:39, 1.91MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:19<08:31, 1.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:19<07:28, 1.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:19<05:34, 1.59MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:19<04:00, 2.21MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:21<09:11, 958kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:21<07:54, 1.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:21<05:50, 1.51MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:21<04:12, 2.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:23<07:50, 1.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:23<08:34, 1.02MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:23<06:36, 1.32MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:23<04:54, 1.78MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:25<05:01, 1.73MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:25<04:55, 1.76MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:25<03:44, 2.31MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:25<02:46, 3.11MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:27<05:10, 1.66MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:27<05:02, 1.71MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:27<03:52, 2.22MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:29<04:11, 2.04MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:29<04:20, 1.97MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:29<03:22, 2.52MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:31<03:50, 2.21MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:31<04:04, 2.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:31<03:10, 2.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:33<03:41, 2.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:33<03:57, 2.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:33<03:06, 2.69MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:35<03:37, 2.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:35<03:53, 2.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:35<03:00, 2.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:35<02:13, 3.73MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:37<06:19, 1.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:37<05:47, 1.43MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:37<04:19, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:37<03:08, 2.61MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:39<05:47, 1.42MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:39<05:25, 1.51MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:39<04:07, 1.98MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:41<04:17, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:41<04:20, 1.87MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:41<03:19, 2.44MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:41<02:26, 3.30MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:43<05:44, 1.40MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:43<05:21, 1.51MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:43<04:01, 2.00MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:43<02:54, 2.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:45<07:49, 1.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:45<06:46, 1.18MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:45<05:04, 1.57MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<04:54, 1.61MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:47<04:45, 1.66MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:47<03:39, 2.16MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:49<04:07, 1.90MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:49<05:26, 1.44MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:49<04:28, 1.75MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:49<03:15, 2.40MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:51<04:23, 1.77MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:51<04:21, 1.79MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:51<03:20, 2.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:51<02:29, 3.10MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:53<04:05, 1.89MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:53<04:08, 1.86MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:53<03:10, 2.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:53<02:19, 3.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:55<05:48, 1.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:55<05:19, 1.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:55<04:00, 1.90MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:55<02:52, 2.63MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:57<15:59, 474kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:57<12:26, 609kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:57<09:00, 839kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:59<07:34, 991kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:59<06:34, 1.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:59<04:51, 1.54MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:59<03:29, 2.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:01<06:58, 1.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:01<06:05, 1.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:01<04:34, 1.62MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:01<03:16, 2.26MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:03<10:55, 675kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:03<08:50, 833kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:03<06:28, 1.13MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:04<05:47, 1.26MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:05<05:13, 1.40MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:05<03:57, 1.84MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:05<02:50, 2.55MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:06<10:38, 680kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:07<08:37, 838kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:07<06:19, 1.14MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:08<05:39, 1.27MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:09<05:07, 1.40MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:09<03:52, 1.84MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<02:46, 2.56MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:10<31:38, 224kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:11<23:17, 304kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:11<16:33, 427kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:12<12:44, 551kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:12<09:47, 717kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:13<07:02, 995kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<04:59, 1.40MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:14<20:04, 347kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:14<14:54, 466kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:15<10:35, 654kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<07:28, 923kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:16<14:23, 479kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:16<11:12, 614kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:17<08:04, 850kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<05:45, 1.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:18<06:30, 1.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:18<05:41, 1.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:19<04:15, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:19<03:04, 2.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:20<05:11, 1.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:20<04:43, 1.43MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:21<03:32, 1.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:21<02:35, 2.59MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:22<04:33, 1.47MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:22<04:17, 1.56MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:22<03:16, 2.03MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:23<02:21, 2.81MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:24<10:03, 657kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:24<08:07, 814kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:24<05:55, 1.11MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<04:12, 1.56MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:26<08:53, 736kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:26<07:16, 899kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:26<05:18, 1.23MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:27<03:49, 1.70MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:28<05:12, 1.24MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:28<05:54, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:29<04:35, 1.41MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:29<03:22, 1.91MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<02:25, 2.64MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:30<1:31:45, 69.8kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:30<1:05:14, 98.1kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:31<45:46, 139kB/s]   .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:31<31:57, 199kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:32<25:58, 244kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:32<19:14, 329kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:32<13:39, 462kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:33<09:36, 654kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:34<10:05, 621kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:34<08:05, 774kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:34<05:54, 1.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<04:11, 1.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:36<09:10, 676kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:36<07:25, 833kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:36<05:26, 1.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:38<04:51, 1.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:38<04:23, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:38<03:19, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:40<03:22, 1.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:40<03:11, 1.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:40<02:46, 2.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:40<02:02, 2.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:42<02:57, 2.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:42<03:02, 1.97MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:42<02:20, 2.55MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:42<01:45, 3.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:44<03:15, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:44<03:15, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:44<02:30, 2.35MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:46<02:46, 2.11MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:46<02:54, 2.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:46<02:15, 2.58MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:48<02:35, 2.24MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:48<02:47, 2.07MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:48<02:10, 2.64MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<01:35, 3.60MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:50<06:32, 873kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:50<06:34, 869kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:50<05:00, 1.14MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:51<03:39, 1.56MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:52<03:39, 1.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:52<03:30, 1.61MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:52<02:39, 2.12MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:53<01:58, 2.84MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:54<03:11, 1.75MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:54<03:09, 1.77MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:54<02:24, 2.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:54<01:45, 3.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:56<03:59, 1.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:56<03:41, 1.49MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:56<02:48, 1.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<02:01, 2.69MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:58<04:59, 1.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:58<04:23, 1.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:58<03:15, 1.66MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:58<02:21, 2.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:00<03:49, 1.40MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:00<03:33, 1.51MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:00<02:42, 1.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:00<01:56, 2.74MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:02<29:34, 179kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:02<21:34, 246kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:02<15:15, 346kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 547M/862M [04:02<10:40, 491kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:04<11:33, 453kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:04<08:56, 585kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:04<06:25, 811kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:04<04:33, 1.14MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:06<05:35, 923kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:06<04:46, 1.08MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:06<03:31, 1.46MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:06<02:32, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:08<03:41, 1.38MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:08<03:26, 1.48MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:08<02:36, 1.94MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:08<01:52, 2.69MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:10<10:30, 478kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:10<08:10, 614kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:10<05:54, 847kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:12<04:58, 998kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:12<04:17, 1.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:12<03:10, 1.56MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:12<02:17, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:14<03:49, 1.28MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:14<03:28, 1.41MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:14<02:37, 1.85MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:16<02:40, 1.81MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:16<02:32, 1.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:16<01:58, 2.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:16<01:28, 3.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:18<02:15, 2.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:18<03:08, 1.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:18<02:36, 1.82MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:18<01:54, 2.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:20<02:40, 1.75MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:20<02:38, 1.77MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:20<02:02, 2.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:22<02:13, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:22<02:18, 1.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:22<01:47, 2.55MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:24<02:02, 2.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:24<02:11, 2.08MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:24<01:41, 2.67MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:24<01:14, 3.61MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:26<03:16, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:26<03:01, 1.47MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:26<02:17, 1.94MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:28<02:21, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:28<02:22, 1.85MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:28<01:50, 2.39MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:30<02:01, 2.13MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:30<02:07, 2.03MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:30<01:37, 2.64MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:30<01:13, 3.52MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:32<02:23, 1.79MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:32<02:23, 1.78MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:32<01:49, 2.32MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<01:20, 3.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:34<02:52, 1.46MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:34<02:42, 1.55MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:34<02:01, 2.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<01:28, 2.80MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:36<02:50, 1.45MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:36<02:40, 1.54MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:36<02:01, 2.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:36<01:32, 2.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:36<01:07, 3.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:38<1:28:39, 45.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:38<1:03:06, 64.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:38<44:56, 90.1kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:38<31:33, 128kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:38<22:02, 182kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:40<16:31, 241kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:40<12:54, 308kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:40<09:21, 425kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:40<06:36, 599kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:41<04:38, 844kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:42<10:55, 358kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:42<08:17, 472kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:42<05:56, 656kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<04:09, 926kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:44<05:57, 646kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:44<04:47, 801kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:44<03:27, 1.11MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:44<02:30, 1.52MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:46<02:53, 1.31MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:46<02:38, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:46<02:00, 1.88MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<01:26, 2.59MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:48<03:58, 934kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:48<03:23, 1.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:48<02:30, 1.47MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<01:47, 2.04MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:50<05:25, 671kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:50<04:23, 828kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:50<03:12, 1.13MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:52<02:50, 1.26MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:52<02:34, 1.38MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:52<01:56, 1.82MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:52<01:23, 2.52MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:54<03:26, 1.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:54<03:00, 1.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:54<02:12, 1.57MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:54<01:36, 2.16MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:56<02:23, 1.43MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:56<02:15, 1.52MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:56<01:41, 2.02MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:56<01:13, 2.75MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:58<02:20, 1.43MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:58<02:11, 1.53MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:58<01:40, 2.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:58<01:11, 2.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:00<06:30, 507kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:00<05:05, 647kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:00<03:39, 894kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:00<02:35, 1.25MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [05:02<03:24, 946kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:02<02:55, 1.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:02<02:08, 1.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:02<01:32, 2.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:04<02:37, 1.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:04<02:20, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:04<01:45, 1.78MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:06<01:45, 1.76MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:06<01:44, 1.78MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:06<01:19, 2.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:06<00:57, 3.14MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:08<02:31, 1.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:08<02:15, 1.34MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:08<01:41, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:08<01:11, 2.47MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:10<58:33, 50.4kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:10<41:26, 71.2kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:10<28:57, 101kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:10<20:03, 144kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:12<15:55, 181kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:12<11:36, 248kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:12<08:11, 350kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:12<05:41, 496kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:14<05:41, 495kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:14<04:26, 632kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:14<03:12, 871kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:14<02:14, 1.23MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:16<05:41, 482kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:16<04:25, 619kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:16<03:11, 853kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:16<02:14, 1.20MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:18<03:30, 765kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:18<02:49, 945kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:18<02:04, 1.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:18<01:27, 1.79MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:19<03:33, 733kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:20<02:55, 893kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:20<02:08, 1.21MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:21<01:54, 1.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:22<01:45, 1.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:22<01:19, 1.90MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:23<01:20, 1.84MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:24<01:32, 1.60MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:24<01:13, 1.99MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:24<00:53, 2.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:25<01:58, 1.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:26<01:46, 1.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:26<01:20, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:27<01:19, 1.76MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:28<01:18, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:28<01:00, 2.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:29<01:05, 2.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:29<01:07, 2.00MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:30<00:52, 2.55MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:31<00:59, 2.22MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:31<01:02, 2.09MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:32<00:49, 2.65MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:32<00:35, 3.64MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:33<07:20, 289kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:33<05:29, 387kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:34<03:52, 543kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:34<02:42, 765kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:35<02:44, 751kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:35<02:15, 911kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:36<01:38, 1.24MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:36<01:09, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:37<02:17, 868kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:37<01:55, 1.03MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:38<01:24, 1.39MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:38<01:00, 1.93MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:39<01:41, 1.13MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:39<01:30, 1.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:39<01:07, 1.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:40<00:47, 2.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:41<02:48, 658kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:41<02:16, 810kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:41<01:39, 1.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:42<01:10, 1.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:43<01:21, 1.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:44<01:34, 1.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:44<01:13, 1.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:44<00:53, 1.96MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:45<00:57, 1.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:46<00:57, 1.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:46<00:44, 2.29MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:47<00:47, 2.08MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:48<00:49, 1.98MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:48<00:38, 2.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:48<00:27, 3.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:49<01:14, 1.26MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:49<01:07, 1.39MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:50<00:50, 1.85MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:50<00:35, 2.55MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:51<01:18, 1.14MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:51<01:09, 1.28MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:52<00:52, 1.70MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:52<00:36, 2.37MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:53<08:47, 163kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:53<06:22, 224kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:54<04:27, 315kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:55<03:15, 417kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:55<02:29, 542kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:56<01:46, 751kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:56<01:13, 1.06MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:57<02:02, 633kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:57<01:37, 787kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:58<01:10, 1.07MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:58<00:49, 1.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:59<01:17, 940kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:59<01:06, 1.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:59<00:48, 1.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:00<00:33, 2.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:01<02:47, 412kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:01<02:08, 537kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:01<01:30, 747kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:02<01:03, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:03<01:08, 954kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:03<00:58, 1.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:03<00:42, 1.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:04<00:29, 2.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:05<00:52, 1.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:05<00:47, 1.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:05<00:34, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:06<00:24, 2.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:07<00:51, 1.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:07<00:44, 1.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:07<00:33, 1.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<00:22, 2.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:09<01:46, 494kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:09<01:22, 632kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:09<00:58, 874kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:09<00:40, 1.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:11<00:49, 976kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:11<00:42, 1.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:11<00:31, 1.52MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:11<00:20, 2.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:13<02:39, 277kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:13<02:06, 347kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:13<01:31, 478kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:13<01:03, 668kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:15<00:49, 812kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:15<00:41, 969kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:15<00:29, 1.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:15<00:19, 1.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:17<00:48, 732kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:17<00:39, 892kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:17<00:28, 1.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:19<00:23, 1.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:19<00:21, 1.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:19<00:16, 1.90MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:10, 2.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:21<00:45, 603kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:21<00:36, 755kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:21<00:25, 1.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<00:16, 1.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:23<00:23, 989kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:23<00:20, 1.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:23<00:14, 1.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:23<00:09, 2.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:25<00:12, 1.57MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:25<00:10, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:25<00:07, 2.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:25<00:05, 3.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:27<00:07, 2.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:27<00:07, 2.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:27<00:05, 2.57MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:27<00:03, 3.50MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:29<00:10, 1.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:29<00:08, 1.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:29<00:05, 1.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:29<00:03, 2.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:31<00:04, 1.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:31<00:04, 1.54MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:31<00:02, 2.04MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:31<00:01, 2.75MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:33<00:01, 1.68MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:33<00:01, 1.83MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:33<00:00, 2.45MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:33<00:00, 3.11MB/s].vector_cache/glove.6B.zip: 862MB [06:33, 2.19MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 796/400000 [00:00<00:50, 7957.56it/s]  0%|          | 1666/400000 [00:00<00:48, 8164.66it/s]  1%|          | 2523/400000 [00:00<00:48, 8280.65it/s]  1%|          | 3387/400000 [00:00<00:47, 8384.10it/s]  1%|          | 4218/400000 [00:00<00:47, 8361.00it/s]  1%|▏         | 5007/400000 [00:00<00:48, 8212.49it/s]  1%|▏         | 5850/400000 [00:00<00:47, 8276.32it/s]  2%|▏         | 6745/400000 [00:00<00:46, 8466.45it/s]  2%|▏         | 7646/400000 [00:00<00:45, 8622.08it/s]  2%|▏         | 8555/400000 [00:01<00:44, 8755.88it/s]  2%|▏         | 9425/400000 [00:01<00:44, 8735.97it/s]  3%|▎         | 10308/400000 [00:01<00:44, 8758.15it/s]  3%|▎         | 11174/400000 [00:01<00:45, 8569.74it/s]  3%|▎         | 12048/400000 [00:01<00:45, 8617.72it/s]  3%|▎         | 12906/400000 [00:01<00:45, 8536.63it/s]  3%|▎         | 13757/400000 [00:01<00:45, 8449.30it/s]  4%|▎         | 14622/400000 [00:01<00:45, 8505.10it/s]  4%|▍         | 15472/400000 [00:01<00:45, 8464.80it/s]  4%|▍         | 16319/400000 [00:01<00:45, 8464.26it/s]  4%|▍         | 17225/400000 [00:02<00:44, 8633.52it/s]  5%|▍         | 18127/400000 [00:02<00:43, 8745.91it/s]  5%|▍         | 19023/400000 [00:02<00:43, 8806.91it/s]  5%|▍         | 19917/400000 [00:02<00:42, 8844.02it/s]  5%|▌         | 20842/400000 [00:02<00:42, 8959.78it/s]  5%|▌         | 21755/400000 [00:02<00:41, 9007.67it/s]  6%|▌         | 22657/400000 [00:02<00:42, 8894.96it/s]  6%|▌         | 23550/400000 [00:02<00:42, 8902.94it/s]  6%|▌         | 24441/400000 [00:02<00:42, 8895.60it/s]  6%|▋         | 25331/400000 [00:02<00:43, 8632.76it/s]  7%|▋         | 26197/400000 [00:03<00:45, 8263.00it/s]  7%|▋         | 27038/400000 [00:03<00:44, 8301.92it/s]  7%|▋         | 27913/400000 [00:03<00:44, 8431.26it/s]  7%|▋         | 28837/400000 [00:03<00:42, 8656.21it/s]  7%|▋         | 29738/400000 [00:03<00:42, 8757.75it/s]  8%|▊         | 30617/400000 [00:03<00:42, 8696.08it/s]  8%|▊         | 31489/400000 [00:03<00:42, 8693.36it/s]  8%|▊         | 32360/400000 [00:03<00:42, 8616.81it/s]  8%|▊         | 33300/400000 [00:03<00:41, 8835.32it/s]  9%|▊         | 34220/400000 [00:03<00:40, 8939.42it/s]  9%|▉         | 35173/400000 [00:04<00:40, 9106.22it/s]  9%|▉         | 36086/400000 [00:04<00:40, 8952.20it/s]  9%|▉         | 36999/400000 [00:04<00:40, 9002.22it/s]  9%|▉         | 37926/400000 [00:04<00:39, 9079.15it/s] 10%|▉         | 38836/400000 [00:04<00:40, 8871.02it/s] 10%|▉         | 39725/400000 [00:04<00:40, 8790.50it/s] 10%|█         | 40606/400000 [00:04<00:41, 8643.00it/s] 10%|█         | 41472/400000 [00:04<00:41, 8580.05it/s] 11%|█         | 42362/400000 [00:04<00:41, 8671.15it/s] 11%|█         | 43231/400000 [00:04<00:41, 8622.89it/s] 11%|█         | 44112/400000 [00:05<00:41, 8676.59it/s] 11%|█         | 44981/400000 [00:05<00:41, 8652.00it/s] 11%|█▏        | 45883/400000 [00:05<00:40, 8757.70it/s] 12%|█▏        | 46760/400000 [00:05<00:40, 8697.21it/s] 12%|█▏        | 47665/400000 [00:05<00:40, 8798.96it/s] 12%|█▏        | 48576/400000 [00:05<00:39, 8888.75it/s] 12%|█▏        | 49466/400000 [00:05<00:39, 8851.99it/s] 13%|█▎        | 50352/400000 [00:05<00:39, 8846.43it/s] 13%|█▎        | 51277/400000 [00:05<00:38, 8963.35it/s] 13%|█▎        | 52197/400000 [00:05<00:38, 9032.46it/s] 13%|█▎        | 53101/400000 [00:06<00:38, 8922.94it/s] 13%|█▎        | 53994/400000 [00:06<00:39, 8671.28it/s] 14%|█▎        | 54864/400000 [00:06<00:40, 8538.11it/s] 14%|█▍        | 55792/400000 [00:06<00:39, 8745.86it/s] 14%|█▍        | 56728/400000 [00:06<00:38, 8919.98it/s] 14%|█▍        | 57623/400000 [00:06<00:38, 8903.55it/s] 15%|█▍        | 58516/400000 [00:06<00:38, 8821.37it/s] 15%|█▍        | 59400/400000 [00:06<00:38, 8793.33it/s] 15%|█▌        | 60298/400000 [00:06<00:38, 8846.82it/s] 15%|█▌        | 61220/400000 [00:07<00:37, 8954.01it/s] 16%|█▌        | 62120/400000 [00:07<00:37, 8967.16it/s] 16%|█▌        | 63018/400000 [00:07<00:38, 8813.79it/s] 16%|█▌        | 63901/400000 [00:07<00:38, 8804.46it/s] 16%|█▌        | 64829/400000 [00:07<00:37, 8939.28it/s] 16%|█▋        | 65724/400000 [00:07<00:37, 8884.04it/s] 17%|█▋        | 66614/400000 [00:07<00:38, 8697.21it/s] 17%|█▋        | 67486/400000 [00:07<00:39, 8495.45it/s] 17%|█▋        | 68338/400000 [00:07<00:39, 8475.55it/s] 17%|█▋        | 69210/400000 [00:07<00:38, 8545.38it/s] 18%|█▊        | 70072/400000 [00:08<00:38, 8563.31it/s] 18%|█▊        | 70930/400000 [00:08<00:38, 8565.01it/s] 18%|█▊        | 71807/400000 [00:08<00:38, 8623.47it/s] 18%|█▊        | 72710/400000 [00:08<00:37, 8739.89it/s] 18%|█▊        | 73587/400000 [00:08<00:37, 8746.97it/s] 19%|█▊        | 74463/400000 [00:08<00:37, 8631.65it/s] 19%|█▉        | 75327/400000 [00:08<00:38, 8404.62it/s] 19%|█▉        | 76170/400000 [00:08<00:39, 8247.56it/s] 19%|█▉        | 77087/400000 [00:08<00:37, 8503.84it/s] 19%|█▉        | 77983/400000 [00:08<00:37, 8634.26it/s] 20%|█▉        | 78883/400000 [00:09<00:36, 8738.73it/s] 20%|█▉        | 79760/400000 [00:09<00:37, 8593.45it/s] 20%|██        | 80622/400000 [00:09<00:38, 8345.30it/s] 20%|██        | 81460/400000 [00:09<00:38, 8336.90it/s] 21%|██        | 82345/400000 [00:09<00:37, 8483.36it/s] 21%|██        | 83206/400000 [00:09<00:37, 8518.76it/s] 21%|██        | 84060/400000 [00:09<00:37, 8410.92it/s] 21%|██        | 84903/400000 [00:09<00:37, 8383.95it/s] 21%|██▏       | 85805/400000 [00:09<00:36, 8562.33it/s] 22%|██▏       | 86663/400000 [00:09<00:36, 8564.73it/s] 22%|██▏       | 87522/400000 [00:10<00:36, 8569.55it/s] 22%|██▏       | 88380/400000 [00:10<00:36, 8542.53it/s] 22%|██▏       | 89235/400000 [00:10<00:36, 8456.73it/s] 23%|██▎       | 90101/400000 [00:10<00:36, 8515.76it/s] 23%|██▎       | 91025/400000 [00:10<00:35, 8720.62it/s] 23%|██▎       | 91934/400000 [00:10<00:34, 8827.37it/s] 23%|██▎       | 92819/400000 [00:10<00:34, 8817.60it/s] 23%|██▎       | 93702/400000 [00:10<00:34, 8760.64it/s] 24%|██▎       | 94579/400000 [00:10<00:35, 8553.06it/s] 24%|██▍       | 95453/400000 [00:10<00:35, 8606.91it/s] 24%|██▍       | 96326/400000 [00:11<00:35, 8643.18it/s] 24%|██▍       | 97207/400000 [00:11<00:34, 8689.15it/s] 25%|██▍       | 98116/400000 [00:11<00:34, 8804.24it/s] 25%|██▍       | 99003/400000 [00:11<00:34, 8823.03it/s] 25%|██▍       | 99931/400000 [00:11<00:33, 8952.95it/s] 25%|██▌       | 100841/400000 [00:11<00:33, 8994.20it/s] 25%|██▌       | 101742/400000 [00:11<00:33, 8833.39it/s] 26%|██▌       | 102627/400000 [00:11<00:35, 8372.16it/s] 26%|██▌       | 103471/400000 [00:11<00:35, 8317.26it/s] 26%|██▌       | 104371/400000 [00:12<00:34, 8509.07it/s] 26%|██▋       | 105270/400000 [00:12<00:34, 8647.10it/s] 27%|██▋       | 106139/400000 [00:12<00:34, 8590.42it/s] 27%|██▋       | 107042/400000 [00:12<00:33, 8716.36it/s] 27%|██▋       | 107916/400000 [00:12<00:34, 8450.63it/s] 27%|██▋       | 108818/400000 [00:12<00:33, 8612.92it/s] 27%|██▋       | 109683/400000 [00:12<00:33, 8579.43it/s] 28%|██▊       | 110544/400000 [00:12<00:33, 8546.35it/s] 28%|██▊       | 111401/400000 [00:12<00:34, 8263.50it/s] 28%|██▊       | 112239/400000 [00:12<00:34, 8296.23it/s] 28%|██▊       | 113150/400000 [00:13<00:33, 8523.18it/s] 29%|██▊       | 114074/400000 [00:13<00:32, 8724.29it/s] 29%|██▊       | 114950/400000 [00:13<00:32, 8641.63it/s] 29%|██▉       | 115817/400000 [00:13<00:32, 8645.19it/s] 29%|██▉       | 116698/400000 [00:13<00:32, 8692.73it/s] 29%|██▉       | 117630/400000 [00:13<00:31, 8869.46it/s] 30%|██▉       | 118575/400000 [00:13<00:31, 9034.37it/s] 30%|██▉       | 119481/400000 [00:13<00:31, 8932.73it/s] 30%|███       | 120376/400000 [00:13<00:31, 8794.31it/s] 30%|███       | 121301/400000 [00:13<00:31, 8924.46it/s] 31%|███       | 122196/400000 [00:14<00:32, 8619.61it/s] 31%|███       | 123104/400000 [00:14<00:31, 8752.41it/s] 31%|███       | 123986/400000 [00:14<00:31, 8769.96it/s] 31%|███       | 124905/400000 [00:14<00:30, 8891.60it/s] 31%|███▏      | 125796/400000 [00:14<00:31, 8827.88it/s] 32%|███▏      | 126681/400000 [00:14<00:31, 8745.55it/s] 32%|███▏      | 127557/400000 [00:14<00:31, 8746.03it/s] 32%|███▏      | 128433/400000 [00:14<00:31, 8545.51it/s] 32%|███▏      | 129363/400000 [00:14<00:30, 8753.49it/s] 33%|███▎      | 130286/400000 [00:14<00:30, 8888.22it/s] 33%|███▎      | 131177/400000 [00:15<00:30, 8767.08it/s] 33%|███▎      | 132081/400000 [00:15<00:30, 8846.80it/s] 33%|███▎      | 132968/400000 [00:15<00:30, 8624.14it/s] 33%|███▎      | 133833/400000 [00:15<00:31, 8582.91it/s] 34%|███▎      | 134703/400000 [00:15<00:30, 8616.28it/s] 34%|███▍      | 135566/400000 [00:15<00:30, 8591.95it/s] 34%|███▍      | 136427/400000 [00:15<00:31, 8401.99it/s] 34%|███▍      | 137308/400000 [00:15<00:30, 8519.69it/s] 35%|███▍      | 138260/400000 [00:15<00:29, 8796.26it/s] 35%|███▍      | 139151/400000 [00:16<00:29, 8825.19it/s] 35%|███▌      | 140036/400000 [00:16<00:29, 8715.35it/s] 35%|███▌      | 140955/400000 [00:16<00:29, 8850.35it/s] 35%|███▌      | 141842/400000 [00:16<00:30, 8566.25it/s] 36%|███▌      | 142702/400000 [00:16<00:30, 8502.13it/s] 36%|███▌      | 143605/400000 [00:16<00:29, 8653.82it/s] 36%|███▌      | 144522/400000 [00:16<00:29, 8800.83it/s] 36%|███▋      | 145457/400000 [00:16<00:28, 8956.66it/s] 37%|███▋      | 146355/400000 [00:16<00:28, 8884.69it/s] 37%|███▋      | 147246/400000 [00:16<00:28, 8880.84it/s] 37%|███▋      | 148144/400000 [00:17<00:28, 8908.94it/s] 37%|███▋      | 149036/400000 [00:17<00:28, 8840.24it/s] 37%|███▋      | 149921/400000 [00:17<00:28, 8626.72it/s] 38%|███▊      | 150809/400000 [00:17<00:28, 8698.72it/s] 38%|███▊      | 151715/400000 [00:17<00:28, 8802.18it/s] 38%|███▊      | 152597/400000 [00:17<00:28, 8701.41it/s] 38%|███▊      | 153532/400000 [00:17<00:27, 8886.06it/s] 39%|███▊      | 154469/400000 [00:17<00:27, 9025.46it/s] 39%|███▉      | 155374/400000 [00:17<00:27, 8985.88it/s] 39%|███▉      | 156345/400000 [00:17<00:26, 9190.86it/s] 39%|███▉      | 157267/400000 [00:18<00:26, 9190.91it/s] 40%|███▉      | 158203/400000 [00:18<00:26, 9238.01it/s] 40%|███▉      | 159128/400000 [00:18<00:26, 9187.05it/s] 40%|████      | 160048/400000 [00:18<00:26, 8907.81it/s] 40%|████      | 160979/400000 [00:18<00:26, 9022.06it/s] 40%|████      | 161884/400000 [00:18<00:26, 8860.42it/s] 41%|████      | 162773/400000 [00:18<00:27, 8637.84it/s] 41%|████      | 163640/400000 [00:18<00:27, 8533.84it/s] 41%|████      | 164496/400000 [00:18<00:27, 8517.25it/s] 41%|████▏     | 165380/400000 [00:18<00:27, 8610.62it/s] 42%|████▏     | 166243/400000 [00:19<00:27, 8532.91it/s] 42%|████▏     | 167098/400000 [00:19<00:27, 8478.45it/s] 42%|████▏     | 168024/400000 [00:19<00:26, 8696.76it/s] 42%|████▏     | 168918/400000 [00:19<00:26, 8767.87it/s] 42%|████▏     | 169859/400000 [00:19<00:25, 8950.57it/s] 43%|████▎     | 170759/400000 [00:19<00:25, 8963.42it/s] 43%|████▎     | 171657/400000 [00:19<00:25, 8914.03it/s] 43%|████▎     | 172563/400000 [00:19<00:25, 8955.54it/s] 43%|████▎     | 173464/400000 [00:19<00:25, 8968.63it/s] 44%|████▎     | 174362/400000 [00:20<00:25, 8888.07it/s] 44%|████▍     | 175252/400000 [00:20<00:25, 8885.28it/s] 44%|████▍     | 176141/400000 [00:20<00:25, 8733.09it/s] 44%|████▍     | 177016/400000 [00:20<00:25, 8703.72it/s] 44%|████▍     | 177887/400000 [00:20<00:26, 8433.35it/s] 45%|████▍     | 178733/400000 [00:20<00:26, 8330.71it/s] 45%|████▍     | 179615/400000 [00:20<00:26, 8471.60it/s] 45%|████▌     | 180482/400000 [00:20<00:25, 8527.55it/s] 45%|████▌     | 181344/400000 [00:20<00:25, 8548.72it/s] 46%|████▌     | 182200/400000 [00:20<00:25, 8544.61it/s] 46%|████▌     | 183109/400000 [00:21<00:24, 8699.55it/s] 46%|████▌     | 183981/400000 [00:21<00:24, 8690.38it/s] 46%|████▌     | 184868/400000 [00:21<00:24, 8742.15it/s] 46%|████▋     | 185743/400000 [00:21<00:24, 8735.66it/s] 47%|████▋     | 186619/400000 [00:21<00:24, 8741.63it/s] 47%|████▋     | 187494/400000 [00:21<00:24, 8742.20it/s] 47%|████▋     | 188413/400000 [00:21<00:23, 8870.59it/s] 47%|████▋     | 189301/400000 [00:21<00:23, 8827.50it/s] 48%|████▊     | 190185/400000 [00:21<00:24, 8573.07it/s] 48%|████▊     | 191051/400000 [00:21<00:24, 8596.16it/s] 48%|████▊     | 191912/400000 [00:22<00:24, 8360.46it/s] 48%|████▊     | 192771/400000 [00:22<00:24, 8426.01it/s] 48%|████▊     | 193674/400000 [00:22<00:24, 8596.17it/s] 49%|████▊     | 194536/400000 [00:22<00:24, 8558.98it/s] 49%|████▉     | 195409/400000 [00:22<00:23, 8605.84it/s] 49%|████▉     | 196301/400000 [00:22<00:23, 8696.10it/s] 49%|████▉     | 197172/400000 [00:22<00:23, 8634.35it/s] 50%|████▉     | 198057/400000 [00:22<00:23, 8693.52it/s] 50%|████▉     | 198928/400000 [00:22<00:23, 8585.89it/s] 50%|████▉     | 199788/400000 [00:22<00:23, 8534.43it/s] 50%|█████     | 200643/400000 [00:23<00:23, 8501.04it/s] 50%|█████     | 201494/400000 [00:23<00:23, 8434.37it/s] 51%|█████     | 202338/400000 [00:23<00:23, 8373.10it/s] 51%|█████     | 203272/400000 [00:23<00:22, 8639.81it/s] 51%|█████     | 204139/400000 [00:23<00:22, 8572.05it/s] 51%|█████     | 204999/400000 [00:23<00:23, 8312.00it/s] 51%|█████▏    | 205834/400000 [00:23<00:23, 8272.27it/s] 52%|█████▏    | 206736/400000 [00:23<00:22, 8481.03it/s] 52%|█████▏    | 207587/400000 [00:23<00:22, 8382.65it/s] 52%|█████▏    | 208465/400000 [00:23<00:22, 8495.38it/s] 52%|█████▏    | 209355/400000 [00:24<00:22, 8612.22it/s] 53%|█████▎    | 210251/400000 [00:24<00:21, 8713.12it/s] 53%|█████▎    | 211124/400000 [00:24<00:21, 8604.68it/s] 53%|█████▎    | 212029/400000 [00:24<00:21, 8732.97it/s] 53%|█████▎    | 212904/400000 [00:24<00:21, 8649.40it/s] 53%|█████▎    | 213793/400000 [00:24<00:21, 8718.25it/s] 54%|█████▎    | 214717/400000 [00:24<00:20, 8865.88it/s] 54%|█████▍    | 215616/400000 [00:24<00:20, 8901.82it/s] 54%|█████▍    | 216508/400000 [00:24<00:21, 8356.04it/s] 54%|█████▍    | 217352/400000 [00:25<00:22, 8272.64it/s] 55%|█████▍    | 218236/400000 [00:25<00:21, 8434.59it/s] 55%|█████▍    | 219095/400000 [00:25<00:21, 8479.88it/s] 55%|█████▍    | 219997/400000 [00:25<00:20, 8633.67it/s] 55%|█████▌    | 220864/400000 [00:25<00:20, 8603.28it/s] 55%|█████▌    | 221727/400000 [00:25<00:21, 8122.73it/s] 56%|█████▌    | 222547/400000 [00:25<00:22, 8056.79it/s] 56%|█████▌    | 223358/400000 [00:25<00:22, 7986.02it/s] 56%|█████▌    | 224224/400000 [00:25<00:21, 8175.40it/s] 56%|█████▋    | 225073/400000 [00:25<00:21, 8265.40it/s] 56%|█████▋    | 225942/400000 [00:26<00:20, 8387.82it/s] 57%|█████▋    | 226856/400000 [00:26<00:20, 8597.99it/s] 57%|█████▋    | 227722/400000 [00:26<00:19, 8614.07it/s] 57%|█████▋    | 228586/400000 [00:26<00:20, 8467.65it/s] 57%|█████▋    | 229435/400000 [00:26<00:20, 8394.94it/s] 58%|█████▊    | 230353/400000 [00:26<00:19, 8614.52it/s] 58%|█████▊    | 231217/400000 [00:26<00:19, 8570.04it/s] 58%|█████▊    | 232076/400000 [00:26<00:19, 8463.71it/s] 58%|█████▊    | 232924/400000 [00:26<00:19, 8392.59it/s] 58%|█████▊    | 233776/400000 [00:26<00:19, 8428.53it/s] 59%|█████▊    | 234681/400000 [00:27<00:19, 8603.76it/s] 59%|█████▉    | 235595/400000 [00:27<00:18, 8756.86it/s] 59%|█████▉    | 236473/400000 [00:27<00:18, 8732.41it/s] 59%|█████▉    | 237383/400000 [00:27<00:18, 8837.52it/s] 60%|█████▉    | 238310/400000 [00:27<00:18, 8959.49it/s] 60%|█████▉    | 239208/400000 [00:27<00:17, 8933.83it/s] 60%|██████    | 240137/400000 [00:27<00:17, 9037.51it/s] 60%|██████    | 241051/400000 [00:27<00:17, 9066.73it/s] 60%|██████    | 242000/400000 [00:27<00:17, 9187.69it/s] 61%|██████    | 242920/400000 [00:27<00:17, 8928.48it/s] 61%|██████    | 243816/400000 [00:28<00:18, 8620.83it/s] 61%|██████    | 244682/400000 [00:28<00:18, 8603.11it/s] 61%|██████▏   | 245545/400000 [00:28<00:18, 8569.57it/s] 62%|██████▏   | 246404/400000 [00:28<00:18, 8481.82it/s] 62%|██████▏   | 247254/400000 [00:28<00:18, 8485.39it/s] 62%|██████▏   | 248131/400000 [00:28<00:17, 8567.37it/s] 62%|██████▏   | 249013/400000 [00:28<00:17, 8639.34it/s] 62%|██████▏   | 249878/400000 [00:28<00:17, 8622.68it/s] 63%|██████▎   | 250830/400000 [00:28<00:16, 8872.02it/s] 63%|██████▎   | 251742/400000 [00:29<00:16, 8943.59it/s] 63%|██████▎   | 252639/400000 [00:29<00:16, 8793.87it/s] 63%|██████▎   | 253570/400000 [00:29<00:16, 8942.13it/s] 64%|██████▎   | 254467/400000 [00:29<00:16, 8932.74it/s] 64%|██████▍   | 255362/400000 [00:29<00:16, 8855.02it/s] 64%|██████▍   | 256249/400000 [00:29<00:16, 8650.13it/s] 64%|██████▍   | 257116/400000 [00:29<00:16, 8570.30it/s] 64%|██████▍   | 257979/400000 [00:29<00:16, 8586.76it/s] 65%|██████▍   | 258839/400000 [00:29<00:16, 8520.77it/s] 65%|██████▍   | 259693/400000 [00:29<00:16, 8522.72it/s] 65%|██████▌   | 260546/400000 [00:30<00:16, 8430.94it/s] 65%|██████▌   | 261412/400000 [00:30<00:16, 8498.05it/s] 66%|██████▌   | 262305/400000 [00:30<00:15, 8620.62it/s] 66%|██████▌   | 263204/400000 [00:30<00:15, 8727.80it/s] 66%|██████▌   | 264134/400000 [00:30<00:15, 8890.66it/s] 66%|██████▋   | 265025/400000 [00:30<00:15, 8838.12it/s] 66%|██████▋   | 265941/400000 [00:30<00:15, 8931.10it/s] 67%|██████▋   | 266858/400000 [00:30<00:14, 8999.27it/s] 67%|██████▋   | 267759/400000 [00:30<00:14, 8879.90it/s] 67%|██████▋   | 268648/400000 [00:30<00:15, 8745.95it/s] 67%|██████▋   | 269524/400000 [00:31<00:14, 8709.15it/s] 68%|██████▊   | 270413/400000 [00:31<00:14, 8759.61it/s] 68%|██████▊   | 271290/400000 [00:31<00:14, 8733.22it/s] 68%|██████▊   | 272167/400000 [00:31<00:14, 8742.87it/s] 68%|██████▊   | 273082/400000 [00:31<00:14, 8860.87it/s] 68%|██████▊   | 273969/400000 [00:31<00:14, 8718.72it/s] 69%|██████▊   | 274853/400000 [00:31<00:14, 8752.75it/s] 69%|██████▉   | 275782/400000 [00:31<00:13, 8905.14it/s] 69%|██████▉   | 276674/400000 [00:31<00:13, 8825.66it/s] 69%|██████▉   | 277620/400000 [00:31<00:13, 9005.33it/s] 70%|██████▉   | 278523/400000 [00:32<00:13, 8724.14it/s] 70%|██████▉   | 279418/400000 [00:32<00:13, 8789.65it/s] 70%|███████   | 280367/400000 [00:32<00:13, 8987.44it/s] 70%|███████   | 281269/400000 [00:32<00:13, 8987.66it/s] 71%|███████   | 282212/400000 [00:32<00:12, 9112.22it/s] 71%|███████   | 283125/400000 [00:32<00:13, 8850.25it/s] 71%|███████   | 284058/400000 [00:32<00:12, 8986.88it/s] 71%|███████   | 284976/400000 [00:32<00:12, 9043.50it/s] 71%|███████▏  | 285883/400000 [00:32<00:13, 8719.70it/s] 72%|███████▏  | 286759/400000 [00:33<00:13, 8425.56it/s] 72%|███████▏  | 287607/400000 [00:33<00:13, 8281.43it/s] 72%|███████▏  | 288503/400000 [00:33<00:13, 8473.32it/s] 72%|███████▏  | 289403/400000 [00:33<00:12, 8623.02it/s] 73%|███████▎  | 290296/400000 [00:33<00:12, 8711.23it/s] 73%|███████▎  | 291194/400000 [00:33<00:12, 8789.76it/s] 73%|███████▎  | 292075/400000 [00:33<00:12, 8693.31it/s] 73%|███████▎  | 292989/400000 [00:33<00:12, 8820.96it/s] 73%|███████▎  | 293890/400000 [00:33<00:11, 8876.46it/s] 74%|███████▎  | 294779/400000 [00:33<00:11, 8859.15it/s] 74%|███████▍  | 295738/400000 [00:34<00:11, 9065.64it/s] 74%|███████▍  | 296647/400000 [00:34<00:11, 9050.49it/s] 74%|███████▍  | 297554/400000 [00:34<00:11, 8939.35it/s] 75%|███████▍  | 298487/400000 [00:34<00:11, 9049.97it/s] 75%|███████▍  | 299394/400000 [00:34<00:11, 8909.38it/s] 75%|███████▌  | 300287/400000 [00:34<00:11, 8831.52it/s] 75%|███████▌  | 301172/400000 [00:34<00:11, 8687.71it/s] 76%|███████▌  | 302043/400000 [00:34<00:11, 8589.57it/s] 76%|███████▌  | 302904/400000 [00:34<00:11, 8428.49it/s] 76%|███████▌  | 303773/400000 [00:34<00:11, 8503.63it/s] 76%|███████▌  | 304625/400000 [00:35<00:11, 8484.41it/s] 76%|███████▋  | 305528/400000 [00:35<00:10, 8640.51it/s] 77%|███████▋  | 306421/400000 [00:35<00:10, 8723.97it/s] 77%|███████▋  | 307295/400000 [00:35<00:10, 8603.92it/s] 77%|███████▋  | 308222/400000 [00:35<00:10, 8792.29it/s] 77%|███████▋  | 309104/400000 [00:35<00:10, 8735.40it/s] 77%|███████▋  | 309979/400000 [00:35<00:10, 8635.77it/s] 78%|███████▊  | 310896/400000 [00:35<00:10, 8788.44it/s] 78%|███████▊  | 311778/400000 [00:35<00:10, 8795.91it/s] 78%|███████▊  | 312701/400000 [00:35<00:09, 8919.50it/s] 78%|███████▊  | 313595/400000 [00:36<00:09, 8741.42it/s] 79%|███████▊  | 314471/400000 [00:36<00:09, 8570.62it/s] 79%|███████▉  | 315340/400000 [00:36<00:09, 8583.82it/s] 79%|███████▉  | 316200/400000 [00:36<00:09, 8587.36it/s] 79%|███████▉  | 317064/400000 [00:36<00:09, 8601.81it/s] 79%|███████▉  | 317931/400000 [00:36<00:09, 8620.77it/s] 80%|███████▉  | 318794/400000 [00:36<00:09, 8604.40it/s] 80%|███████▉  | 319725/400000 [00:36<00:09, 8802.33it/s] 80%|████████  | 320607/400000 [00:36<00:09, 8806.61it/s] 80%|████████  | 321510/400000 [00:36<00:08, 8869.76it/s] 81%|████████  | 322448/400000 [00:37<00:08, 9015.08it/s] 81%|████████  | 323351/400000 [00:37<00:08, 8904.80it/s] 81%|████████  | 324272/400000 [00:37<00:08, 8992.85it/s] 81%|████████▏ | 325188/400000 [00:37<00:08, 9041.68it/s] 82%|████████▏ | 326093/400000 [00:37<00:08, 9036.78it/s] 82%|████████▏ | 326998/400000 [00:37<00:08, 8897.01it/s] 82%|████████▏ | 327889/400000 [00:37<00:08, 8777.80it/s] 82%|████████▏ | 328772/400000 [00:37<00:08, 8791.77it/s] 82%|████████▏ | 329652/400000 [00:37<00:08, 8762.09it/s] 83%|████████▎ | 330529/400000 [00:37<00:07, 8717.52it/s] 83%|████████▎ | 331434/400000 [00:38<00:07, 8812.58it/s] 83%|████████▎ | 332316/400000 [00:38<00:07, 8645.62it/s] 83%|████████▎ | 333241/400000 [00:38<00:07, 8816.79it/s] 84%|████████▎ | 334167/400000 [00:38<00:07, 8943.78it/s] 84%|████████▍ | 335112/400000 [00:38<00:07, 9088.26it/s] 84%|████████▍ | 336023/400000 [00:38<00:07, 9049.34it/s] 84%|████████▍ | 336930/400000 [00:38<00:07, 8822.74it/s] 84%|████████▍ | 337832/400000 [00:38<00:07, 8879.95it/s] 85%|████████▍ | 338776/400000 [00:38<00:06, 9040.09it/s] 85%|████████▍ | 339707/400000 [00:39<00:06, 9118.09it/s] 85%|████████▌ | 340621/400000 [00:39<00:06, 9099.13it/s] 85%|████████▌ | 341532/400000 [00:39<00:06, 8889.91it/s] 86%|████████▌ | 342423/400000 [00:39<00:06, 8858.73it/s] 86%|████████▌ | 343311/400000 [00:39<00:06, 8819.69it/s] 86%|████████▌ | 344194/400000 [00:39<00:06, 8641.38it/s] 86%|████████▋ | 345084/400000 [00:39<00:06, 8715.66it/s] 86%|████████▋ | 345957/400000 [00:39<00:06, 8716.49it/s] 87%|████████▋ | 346832/400000 [00:39<00:06, 8722.70it/s] 87%|████████▋ | 347737/400000 [00:39<00:05, 8818.10it/s] 87%|████████▋ | 348657/400000 [00:40<00:05, 8926.81it/s] 87%|████████▋ | 349551/400000 [00:40<00:05, 8882.16it/s] 88%|████████▊ | 350440/400000 [00:40<00:05, 8770.01it/s] 88%|████████▊ | 351346/400000 [00:40<00:05, 8853.96it/s] 88%|████████▊ | 352276/400000 [00:40<00:05, 8980.13it/s] 88%|████████▊ | 353211/400000 [00:40<00:05, 9085.88it/s] 89%|████████▊ | 354121/400000 [00:40<00:05, 9029.66it/s] 89%|████████▉ | 355025/400000 [00:40<00:05, 8921.54it/s] 89%|████████▉ | 355925/400000 [00:40<00:04, 8944.20it/s] 89%|████████▉ | 356821/400000 [00:40<00:04, 8866.07it/s] 89%|████████▉ | 357709/400000 [00:41<00:04, 8865.20it/s] 90%|████████▉ | 358596/400000 [00:41<00:04, 8571.71it/s] 90%|████████▉ | 359456/400000 [00:41<00:04, 8520.31it/s] 90%|█████████ | 360313/400000 [00:41<00:04, 8535.04it/s] 90%|█████████ | 361193/400000 [00:41<00:04, 8611.92it/s] 91%|█████████ | 362061/400000 [00:41<00:04, 8627.98it/s] 91%|█████████ | 362940/400000 [00:41<00:04, 8675.16it/s] 91%|█████████ | 363809/400000 [00:41<00:04, 8531.20it/s] 91%|█████████ | 364667/400000 [00:41<00:04, 8545.26it/s] 91%|█████████▏| 365533/400000 [00:41<00:04, 8576.80it/s] 92%|█████████▏| 366460/400000 [00:42<00:03, 8772.80it/s] 92%|█████████▏| 367339/400000 [00:42<00:03, 8630.33it/s] 92%|█████████▏| 368204/400000 [00:42<00:03, 8613.55it/s] 92%|█████████▏| 369111/400000 [00:42<00:03, 8742.56it/s] 92%|█████████▏| 369987/400000 [00:42<00:03, 8747.46it/s] 93%|█████████▎| 370905/400000 [00:42<00:03, 8871.94it/s] 93%|█████████▎| 371845/400000 [00:42<00:03, 9021.88it/s] 93%|█████████▎| 372749/400000 [00:42<00:03, 8629.02it/s] 93%|█████████▎| 373658/400000 [00:42<00:03, 8760.20it/s] 94%|█████████▎| 374609/400000 [00:42<00:02, 8972.29it/s] 94%|█████████▍| 375511/400000 [00:43<00:02, 8983.00it/s] 94%|█████████▍| 376433/400000 [00:43<00:02, 9050.59it/s] 94%|█████████▍| 377341/400000 [00:43<00:02, 8866.79it/s] 95%|█████████▍| 378230/400000 [00:43<00:02, 8872.84it/s] 95%|█████████▍| 379186/400000 [00:43<00:02, 9068.29it/s] 95%|█████████▌| 380095/400000 [00:43<00:02, 9062.20it/s] 95%|█████████▌| 381003/400000 [00:43<00:02, 9055.12it/s] 95%|█████████▌| 381910/400000 [00:43<00:02, 8720.93it/s] 96%|█████████▌| 382797/400000 [00:43<00:01, 8764.64it/s] 96%|█████████▌| 383738/400000 [00:44<00:01, 8946.13it/s] 96%|█████████▌| 384636/400000 [00:44<00:01, 8792.36it/s] 96%|█████████▋| 385543/400000 [00:44<00:01, 8873.76it/s] 97%|█████████▋| 386433/400000 [00:44<00:01, 8700.51it/s] 97%|█████████▋| 387374/400000 [00:44<00:01, 8901.33it/s] 97%|█████████▋| 388318/400000 [00:44<00:01, 9054.88it/s] 97%|█████████▋| 389226/400000 [00:44<00:01, 9042.82it/s] 98%|█████████▊| 390150/400000 [00:44<00:01, 9099.06it/s] 98%|█████████▊| 391062/400000 [00:44<00:00, 8963.50it/s] 98%|█████████▊| 391998/400000 [00:44<00:00, 9077.15it/s] 98%|█████████▊| 392948/400000 [00:45<00:00, 9197.51it/s] 98%|█████████▊| 393870/400000 [00:45<00:00, 9091.72it/s] 99%|█████████▊| 394781/400000 [00:45<00:00, 9061.31it/s] 99%|█████████▉| 395688/400000 [00:45<00:00, 8960.68it/s] 99%|█████████▉| 396603/400000 [00:45<00:00, 9015.64it/s] 99%|█████████▉| 397506/400000 [00:45<00:00, 8801.59it/s]100%|█████████▉| 398388/400000 [00:45<00:00, 8784.67it/s]100%|█████████▉| 399268/400000 [00:45<00:00, 8759.63it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8727.55it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f6d0d158cc0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011499144488649818 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.011759248465598626 	 Accuracy: 53

  model saves at 53% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15975 out of table with 15699 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15975 out of table with 15699 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-13 20:24:30.766644: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 20:24:30.772113: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-13 20:24:30.772345: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d66fbb05d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 20:24:30.772362: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f6cb2bde2e8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7433 - accuracy: 0.4950
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6436 - accuracy: 0.5015 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7791 - accuracy: 0.4927
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7280 - accuracy: 0.4960
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6697 - accuracy: 0.4998
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7228 - accuracy: 0.4963
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6973 - accuracy: 0.4980
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6800 - accuracy: 0.4991
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6835 - accuracy: 0.4989
11000/25000 [============>.................] - ETA: 4s - loss: 7.6694 - accuracy: 0.4998
12000/25000 [=============>................] - ETA: 4s - loss: 7.6487 - accuracy: 0.5012
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6454 - accuracy: 0.5014
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6579 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 3s - loss: 7.6728 - accuracy: 0.4996
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6657 - accuracy: 0.5001
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6802 - accuracy: 0.4991
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6521 - accuracy: 0.5009
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6643 - accuracy: 0.5002
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6674 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6638 - accuracy: 0.5002
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6713 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
25000/25000 [==============================] - 9s 370us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f6c8908b160> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f6c8a2e80f0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 977ms/step - loss: 1.3462 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.2483 - val_crf_viterbi_accuracy: 0.0133

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

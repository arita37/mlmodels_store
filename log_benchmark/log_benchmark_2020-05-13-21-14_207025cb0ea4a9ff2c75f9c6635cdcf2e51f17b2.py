
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fea36550fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 21:14:42.091177
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 21:14:42.095029
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 21:14:42.098062
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 21:14:42.101148
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fea425684a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353009.1562
Epoch 2/10

1/1 [==============================] - 0s 96ms/step - loss: 232041.4844
Epoch 3/10

1/1 [==============================] - 0s 92ms/step - loss: 136498.4531
Epoch 4/10

1/1 [==============================] - 0s 92ms/step - loss: 68451.4297
Epoch 5/10

1/1 [==============================] - 0s 92ms/step - loss: 34938.6562
Epoch 6/10

1/1 [==============================] - 0s 93ms/step - loss: 19553.3047
Epoch 7/10

1/1 [==============================] - 0s 93ms/step - loss: 12081.5664
Epoch 8/10

1/1 [==============================] - 0s 91ms/step - loss: 8125.7173
Epoch 9/10

1/1 [==============================] - 0s 91ms/step - loss: 5909.1187
Epoch 10/10

1/1 [==============================] - 0s 91ms/step - loss: 4563.7095

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.37090969e+00  1.09047711e-01  1.99728930e+00  1.31985939e+00
  -1.27649164e+00  1.85751128e+00 -3.71184170e-01  1.55981708e+00
   9.31878150e-01  1.32686079e+00 -8.91452312e-01 -3.84909600e-01
   3.23933649e+00 -4.18535143e-01 -5.86881757e-01  1.23403788e+00
  -2.67963409e-01  4.20835912e-02  1.25384378e+00 -1.48094702e+00
  -2.75630593e-01 -3.09230387e-01  1.41814351e-02 -2.10849810e+00
  -6.70357198e-02  6.19587600e-01  6.09913886e-01 -2.45429826e+00
  -1.59531415e+00  8.95712733e-01 -2.51399040e-01 -1.07343459e+00
  -1.25634027e+00  6.14711642e-01 -4.76238430e-01  6.37695193e-03
   1.24294698e-01  2.09618139e+00  1.26974046e-01  7.91124582e-01
   1.79358363e+00  1.26883397e-02 -1.53079176e+00 -1.44526637e+00
  -1.50489497e+00 -7.86827326e-01  4.72278148e-03  2.18945503e+00
  -8.61944497e-01 -1.48953915e+00  2.69365728e-01 -1.40302205e+00
   4.89140600e-01  2.20414782e+00 -6.99571908e-01  1.38357997e+00
  -3.03873390e-01  2.34258962e+00  3.86650324e-01  4.41615522e-01
   1.12465358e+00  3.41668397e-01  2.35090542e+00  9.47661519e-01
   2.90892094e-01 -1.73957062e+00 -1.02046037e+00  4.62137461e-01
   1.57702661e+00  1.42018795e-02 -9.04693305e-01  1.17929518e+00
   4.66581017e-01 -3.10631901e-01  4.27655578e-01  1.27093434e+00
   2.28534603e+00  4.10016805e-01 -1.20946944e+00 -1.88687325e-01
  -1.02128458e+00 -1.54215455e+00 -5.38829267e-01  9.91478026e-01
  -3.86777431e-01  2.75891542e-01  4.69393462e-01 -4.36425149e-01
  -9.88404989e-01  6.64833248e-01  1.10382187e+00 -1.35282266e+00
   5.05863011e-01 -7.52498567e-01  2.09974313e+00  5.27888894e-01
   2.12551641e+00  4.14748192e-02 -2.02609682e+00  2.46637464e-02
  -1.94775426e+00  1.49140072e+00  3.29848945e-01 -1.15753925e+00
   3.51911306e-01  4.14269954e-01 -7.13905752e-01 -1.94018126e+00
  -2.21212721e+00  1.93892896e-01  1.78382242e+00 -9.86382604e-01
  -9.84274149e-02  2.17497635e+00 -7.15655804e-01 -1.65577412e-01
  -8.15792799e-01  9.48566198e-02  2.71954477e-01 -1.38749850e+00
  -3.77084672e-01  1.12584028e+01  9.57912540e+00  1.04510422e+01
   8.95957279e+00  1.28113794e+01  8.58273697e+00  7.83885288e+00
   8.16359997e+00  1.10998430e+01  1.03547010e+01  1.04259501e+01
   1.20286541e+01  1.08904552e+01  1.05858383e+01  9.73742580e+00
   1.14346313e+01  1.06809978e+01  1.04749222e+01  1.12894468e+01
   1.08700609e+01  9.31351376e+00  8.88485527e+00  1.07170382e+01
   1.02689905e+01  9.43728161e+00  9.04686260e+00  9.83314896e+00
   9.68365192e+00  1.08933630e+01  8.80071449e+00  9.54653549e+00
   8.80649376e+00  1.10917578e+01  1.06988602e+01  1.18087349e+01
   9.58101463e+00  9.44008541e+00  9.33613396e+00  1.01976471e+01
   9.92657661e+00  9.61267090e+00  9.54434109e+00  9.73323822e+00
   9.97715855e+00  1.06608963e+01  9.54254055e+00  8.89037323e+00
   9.76577187e+00  8.62302208e+00  1.01057577e+01  1.03745337e+01
   1.35594893e+01  1.04905500e+01  1.15903225e+01  8.60348225e+00
   1.09650812e+01  1.17796984e+01  1.07049208e+01  9.25334454e+00
   6.01034403e-01  2.56427646e-01  1.79144752e+00  6.51415229e-01
   1.87786460e+00  1.88366055e+00  2.19762945e+00  3.12468815e+00
   3.18294644e-01  3.76435280e-01  9.76898551e-01  2.94488001e+00
   1.22150707e+00  3.64426494e-01  7.94560909e-01  6.22757673e-01
   3.36184978e-01  5.42416096e-01  1.58573771e+00  9.15221334e-01
   4.95088816e-01  1.97758853e+00  2.87784815e-01  7.23348737e-01
   1.39579105e+00  1.16479492e+00  1.05907142e-01  1.04816961e+00
   9.62502241e-01  2.33700156e-01  1.74079394e+00  1.26979971e+00
   1.73432171e+00  4.17819858e-01  7.19855964e-01  3.14374781e+00
   2.14856720e+00  3.19631279e-01  1.67987406e-01  3.49900723e-01
   1.69017327e+00  3.88436198e-01  1.39289498e-01  2.24189520e-01
   1.33063161e+00  4.54511046e-01  1.28048158e+00  1.29367900e+00
   3.11465025e-01  8.51165414e-01  2.39391327e+00  7.11312056e-01
   1.68028212e+00  9.04485881e-01  1.21271670e-01  2.41940403e+00
   6.38083518e-01  5.40137231e-01  3.25072229e-01  5.50799489e-01
   8.40158582e-01  1.54768944e+00  2.98260546e+00  2.18686104e+00
   8.68650556e-01  1.54087961e-01  3.59293818e-02  1.62237346e+00
   2.17502952e+00  8.43674719e-01  2.18857336e+00  1.85342216e+00
   6.43037558e-02  3.20644665e+00  7.98897624e-01  1.57010019e-01
   1.89090276e+00  1.28826082e-01  9.68600750e-01  1.98124838e+00
   2.16378880e+00  2.56974030e+00  1.77097404e+00  1.78063273e-01
   1.45796561e+00  1.77786696e+00  1.47852778e+00  1.74054623e-01
   2.42287660e+00  1.01760972e+00  2.55755281e+00  1.55982614e-01
   3.00697470e+00  8.65790606e-01  9.24443185e-01  3.25068533e-01
   2.85023093e-01  1.59174514e+00  9.47713912e-01  1.84614182e+00
   2.36324251e-01  1.82156420e+00  7.12601960e-01  1.01445699e+00
   3.25846052e+00  9.44222212e-01  1.99733198e-01  4.18938756e-01
   1.00078046e+00  7.97590733e-01  2.52050400e+00  5.17811120e-01
   4.02579260e+00  1.78952587e+00  5.51617742e-02  5.84612072e-01
   3.19225788e-01  3.16597402e-01  2.79921889e-01  2.87783957e+00
   2.43001997e-01  1.11212797e+01  1.02228994e+01  1.03912134e+01
   9.97446537e+00  1.07697115e+01  1.03451824e+01  1.01164417e+01
   9.94583035e+00  1.11066208e+01  1.18616085e+01  1.07314167e+01
   1.15701027e+01  1.04793358e+01  8.42500305e+00  8.10027790e+00
   1.10509481e+01  9.04554749e+00  1.05965357e+01  1.16181135e+01
   1.06568813e+01  1.10973711e+01  8.51144314e+00  9.26905060e+00
   1.08812113e+01  1.16219931e+01  1.13008776e+01  1.20965061e+01
   1.22622004e+01  1.01794167e+01  8.13873959e+00  1.12158031e+01
   1.10374603e+01  1.03678713e+01  8.47383499e+00  1.01121998e+01
   1.07146549e+01  1.05734491e+01  8.81936073e+00  1.21408215e+01
   1.09493132e+01  1.01372414e+01  1.15393267e+01  1.17434683e+01
   1.15905590e+01  1.00762291e+01  9.72579193e+00  7.32820606e+00
   8.98454475e+00  9.87177086e+00  1.09576864e+01  1.06087484e+01
   1.09128466e+01  1.07518663e+01  1.00027914e+01  1.11707420e+01
   9.49204159e+00  1.02844543e+01  8.35967159e+00  1.14094191e+01
  -1.56539698e+01 -8.05565739e+00  7.31111908e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 21:14:50.528983
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.6771
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 21:14:50.532654
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8427.73
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 21:14:50.535709
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.2304
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 21:14:50.538687
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -753.763
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140643553859120
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140642612441720
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140642612442224
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140642612442728
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140642612443232
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140642612443736

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fea3844a048> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.498152
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.476206
grad_step = 000002, loss = 0.455423
grad_step = 000003, loss = 0.430901
grad_step = 000004, loss = 0.402571
grad_step = 000005, loss = 0.373942
grad_step = 000006, loss = 0.354713
grad_step = 000007, loss = 0.350889
grad_step = 000008, loss = 0.338159
grad_step = 000009, loss = 0.319342
grad_step = 000010, loss = 0.305059
grad_step = 000011, loss = 0.294950
grad_step = 000012, loss = 0.286530
grad_step = 000013, loss = 0.277853
grad_step = 000014, loss = 0.267926
grad_step = 000015, loss = 0.257114
grad_step = 000016, loss = 0.246544
grad_step = 000017, loss = 0.236992
grad_step = 000018, loss = 0.227921
grad_step = 000019, loss = 0.218683
grad_step = 000020, loss = 0.209074
grad_step = 000021, loss = 0.199576
grad_step = 000022, loss = 0.190785
grad_step = 000023, loss = 0.182485
grad_step = 000024, loss = 0.174209
grad_step = 000025, loss = 0.165813
grad_step = 000026, loss = 0.157431
grad_step = 000027, loss = 0.149336
grad_step = 000028, loss = 0.141736
grad_step = 000029, loss = 0.134497
grad_step = 000030, loss = 0.127295
grad_step = 000031, loss = 0.120025
grad_step = 000032, loss = 0.112944
grad_step = 000033, loss = 0.106262
grad_step = 000034, loss = 0.099884
grad_step = 000035, loss = 0.093696
grad_step = 000036, loss = 0.087689
grad_step = 000037, loss = 0.081954
grad_step = 000038, loss = 0.076513
grad_step = 000039, loss = 0.071249
grad_step = 000040, loss = 0.066150
grad_step = 000041, loss = 0.061295
grad_step = 000042, loss = 0.056695
grad_step = 000043, loss = 0.052311
grad_step = 000044, loss = 0.048167
grad_step = 000045, loss = 0.044305
grad_step = 000046, loss = 0.040712
grad_step = 000047, loss = 0.037346
grad_step = 000048, loss = 0.034177
grad_step = 000049, loss = 0.031182
grad_step = 000050, loss = 0.028382
grad_step = 000051, loss = 0.025808
grad_step = 000052, loss = 0.023452
grad_step = 000053, loss = 0.021283
grad_step = 000054, loss = 0.019292
grad_step = 000055, loss = 0.017475
grad_step = 000056, loss = 0.015825
grad_step = 000057, loss = 0.014324
grad_step = 000058, loss = 0.012945
grad_step = 000059, loss = 0.011687
grad_step = 000060, loss = 0.010567
grad_step = 000061, loss = 0.009576
grad_step = 000062, loss = 0.008682
grad_step = 000063, loss = 0.007875
grad_step = 000064, loss = 0.007169
grad_step = 000065, loss = 0.006548
grad_step = 000066, loss = 0.005989
grad_step = 000067, loss = 0.005487
grad_step = 000068, loss = 0.005050
grad_step = 000069, loss = 0.004677
grad_step = 000070, loss = 0.004349
grad_step = 000071, loss = 0.004059
grad_step = 000072, loss = 0.003809
grad_step = 000073, loss = 0.003597
grad_step = 000074, loss = 0.003411
grad_step = 000075, loss = 0.003248
grad_step = 000076, loss = 0.003109
grad_step = 000077, loss = 0.002992
grad_step = 000078, loss = 0.002893
grad_step = 000079, loss = 0.002809
grad_step = 000080, loss = 0.002737
grad_step = 000081, loss = 0.002678
grad_step = 000082, loss = 0.002627
grad_step = 000083, loss = 0.002585
grad_step = 000084, loss = 0.002548
grad_step = 000085, loss = 0.002518
grad_step = 000086, loss = 0.002494
grad_step = 000087, loss = 0.002475
grad_step = 000088, loss = 0.002457
grad_step = 000089, loss = 0.002442
grad_step = 000090, loss = 0.002430
grad_step = 000091, loss = 0.002419
grad_step = 000092, loss = 0.002409
grad_step = 000093, loss = 0.002400
grad_step = 000094, loss = 0.002393
grad_step = 000095, loss = 0.002386
grad_step = 000096, loss = 0.002379
grad_step = 000097, loss = 0.002373
grad_step = 000098, loss = 0.002366
grad_step = 000099, loss = 0.002360
grad_step = 000100, loss = 0.002353
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002347
grad_step = 000102, loss = 0.002341
grad_step = 000103, loss = 0.002334
grad_step = 000104, loss = 0.002328
grad_step = 000105, loss = 0.002321
grad_step = 000106, loss = 0.002315
grad_step = 000107, loss = 0.002308
grad_step = 000108, loss = 0.002302
grad_step = 000109, loss = 0.002296
grad_step = 000110, loss = 0.002290
grad_step = 000111, loss = 0.002285
grad_step = 000112, loss = 0.002279
grad_step = 000113, loss = 0.002274
grad_step = 000114, loss = 0.002269
grad_step = 000115, loss = 0.002264
grad_step = 000116, loss = 0.002259
grad_step = 000117, loss = 0.002255
grad_step = 000118, loss = 0.002250
grad_step = 000119, loss = 0.002246
grad_step = 000120, loss = 0.002242
grad_step = 000121, loss = 0.002239
grad_step = 000122, loss = 0.002235
grad_step = 000123, loss = 0.002232
grad_step = 000124, loss = 0.002228
grad_step = 000125, loss = 0.002226
grad_step = 000126, loss = 0.002223
grad_step = 000127, loss = 0.002223
grad_step = 000128, loss = 0.002228
grad_step = 000129, loss = 0.002246
grad_step = 000130, loss = 0.002286
grad_step = 000131, loss = 0.002375
grad_step = 000132, loss = 0.002445
grad_step = 000133, loss = 0.002461
grad_step = 000134, loss = 0.002282
grad_step = 000135, loss = 0.002196
grad_step = 000136, loss = 0.002284
grad_step = 000137, loss = 0.002336
grad_step = 000138, loss = 0.002261
grad_step = 000139, loss = 0.002184
grad_step = 000140, loss = 0.002243
grad_step = 000141, loss = 0.002288
grad_step = 000142, loss = 0.002209
grad_step = 000143, loss = 0.002176
grad_step = 000144, loss = 0.002230
grad_step = 000145, loss = 0.002226
grad_step = 000146, loss = 0.002172
grad_step = 000147, loss = 0.002171
grad_step = 000148, loss = 0.002203
grad_step = 000149, loss = 0.002187
grad_step = 000150, loss = 0.002152
grad_step = 000151, loss = 0.002164
grad_step = 000152, loss = 0.002180
grad_step = 000153, loss = 0.002157
grad_step = 000154, loss = 0.002139
grad_step = 000155, loss = 0.002151
grad_step = 000156, loss = 0.002155
grad_step = 000157, loss = 0.002137
grad_step = 000158, loss = 0.002126
grad_step = 000159, loss = 0.002133
grad_step = 000160, loss = 0.002134
grad_step = 000161, loss = 0.002122
grad_step = 000162, loss = 0.002112
grad_step = 000163, loss = 0.002113
grad_step = 000164, loss = 0.002114
grad_step = 000165, loss = 0.002108
grad_step = 000166, loss = 0.002100
grad_step = 000167, loss = 0.002094
grad_step = 000168, loss = 0.002091
grad_step = 000169, loss = 0.002088
grad_step = 000170, loss = 0.002084
grad_step = 000171, loss = 0.002084
grad_step = 000172, loss = 0.002096
grad_step = 000173, loss = 0.002089
grad_step = 000174, loss = 0.002074
grad_step = 000175, loss = 0.002060
grad_step = 000176, loss = 0.002053
grad_step = 000177, loss = 0.002055
grad_step = 000178, loss = 0.002049
grad_step = 000179, loss = 0.002062
grad_step = 000180, loss = 0.002092
grad_step = 000181, loss = 0.002082
grad_step = 000182, loss = 0.002099
grad_step = 000183, loss = 0.002075
grad_step = 000184, loss = 0.002049
grad_step = 000185, loss = 0.002054
grad_step = 000186, loss = 0.002035
grad_step = 000187, loss = 0.002002
grad_step = 000188, loss = 0.002008
grad_step = 000189, loss = 0.002048
grad_step = 000190, loss = 0.002084
grad_step = 000191, loss = 0.002157
grad_step = 000192, loss = 0.002221
grad_step = 000193, loss = 0.002284
grad_step = 000194, loss = 0.002033
grad_step = 000195, loss = 0.002026
grad_step = 000196, loss = 0.002097
grad_step = 000197, loss = 0.001972
grad_step = 000198, loss = 0.002090
grad_step = 000199, loss = 0.002117
grad_step = 000200, loss = 0.001960
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002058
grad_step = 000202, loss = 0.002017
grad_step = 000203, loss = 0.001947
grad_step = 000204, loss = 0.002032
grad_step = 000205, loss = 0.001984
grad_step = 000206, loss = 0.001961
grad_step = 000207, loss = 0.002030
grad_step = 000208, loss = 0.001933
grad_step = 000209, loss = 0.001917
grad_step = 000210, loss = 0.001969
grad_step = 000211, loss = 0.001912
grad_step = 000212, loss = 0.001900
grad_step = 000213, loss = 0.001963
grad_step = 000214, loss = 0.001964
grad_step = 000215, loss = 0.002066
grad_step = 000216, loss = 0.001933
grad_step = 000217, loss = 0.001967
grad_step = 000218, loss = 0.002055
grad_step = 000219, loss = 0.001906
grad_step = 000220, loss = 0.002096
grad_step = 000221, loss = 0.002232
grad_step = 000222, loss = 0.001996
grad_step = 000223, loss = 0.002196
grad_step = 000224, loss = 0.001966
grad_step = 000225, loss = 0.002118
grad_step = 000226, loss = 0.001957
grad_step = 000227, loss = 0.002096
grad_step = 000228, loss = 0.001965
grad_step = 000229, loss = 0.001955
grad_step = 000230, loss = 0.001937
grad_step = 000231, loss = 0.001978
grad_step = 000232, loss = 0.001934
grad_step = 000233, loss = 0.001967
grad_step = 000234, loss = 0.001946
grad_step = 000235, loss = 0.001952
grad_step = 000236, loss = 0.001939
grad_step = 000237, loss = 0.001967
grad_step = 000238, loss = 0.001956
grad_step = 000239, loss = 0.001988
grad_step = 000240, loss = 0.002023
grad_step = 000241, loss = 0.002103
grad_step = 000242, loss = 0.002156
grad_step = 000243, loss = 0.002193
grad_step = 000244, loss = 0.002129
grad_step = 000245, loss = 0.001999
grad_step = 000246, loss = 0.001897
grad_step = 000247, loss = 0.001929
grad_step = 000248, loss = 0.001998
grad_step = 000249, loss = 0.002038
grad_step = 000250, loss = 0.001967
grad_step = 000251, loss = 0.001895
grad_step = 000252, loss = 0.001877
grad_step = 000253, loss = 0.001929
grad_step = 000254, loss = 0.001955
grad_step = 000255, loss = 0.001939
grad_step = 000256, loss = 0.001893
grad_step = 000257, loss = 0.001863
grad_step = 000258, loss = 0.001868
grad_step = 000259, loss = 0.001889
grad_step = 000260, loss = 0.001909
grad_step = 000261, loss = 0.001891
grad_step = 000262, loss = 0.001863
grad_step = 000263, loss = 0.001841
grad_step = 000264, loss = 0.001846
grad_step = 000265, loss = 0.001858
grad_step = 000266, loss = 0.001863
grad_step = 000267, loss = 0.001859
grad_step = 000268, loss = 0.001843
grad_step = 000269, loss = 0.001830
grad_step = 000270, loss = 0.001825
grad_step = 000271, loss = 0.001828
grad_step = 000272, loss = 0.001835
grad_step = 000273, loss = 0.001836
grad_step = 000274, loss = 0.001833
grad_step = 000275, loss = 0.001826
grad_step = 000276, loss = 0.001819
grad_step = 000277, loss = 0.001810
grad_step = 000278, loss = 0.001807
grad_step = 000279, loss = 0.001806
grad_step = 000280, loss = 0.001807
grad_step = 000281, loss = 0.001811
grad_step = 000282, loss = 0.001815
grad_step = 000283, loss = 0.001822
grad_step = 000284, loss = 0.001827
grad_step = 000285, loss = 0.001837
grad_step = 000286, loss = 0.001837
grad_step = 000287, loss = 0.001838
grad_step = 000288, loss = 0.001829
grad_step = 000289, loss = 0.001819
grad_step = 000290, loss = 0.001806
grad_step = 000291, loss = 0.001794
grad_step = 000292, loss = 0.001786
grad_step = 000293, loss = 0.001785
grad_step = 000294, loss = 0.001788
grad_step = 000295, loss = 0.001791
grad_step = 000296, loss = 0.001795
grad_step = 000297, loss = 0.001800
grad_step = 000298, loss = 0.001814
grad_step = 000299, loss = 0.001830
grad_step = 000300, loss = 0.001863
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001863
grad_step = 000302, loss = 0.001863
grad_step = 000303, loss = 0.001855
grad_step = 000304, loss = 0.001850
grad_step = 000305, loss = 0.001828
grad_step = 000306, loss = 0.001789
grad_step = 000307, loss = 0.001776
grad_step = 000308, loss = 0.001781
grad_step = 000309, loss = 0.001782
grad_step = 000310, loss = 0.001790
grad_step = 000311, loss = 0.001811
grad_step = 000312, loss = 0.001828
grad_step = 000313, loss = 0.001845
grad_step = 000314, loss = 0.001834
grad_step = 000315, loss = 0.001825
grad_step = 000316, loss = 0.001809
grad_step = 000317, loss = 0.001786
grad_step = 000318, loss = 0.001764
grad_step = 000319, loss = 0.001755
grad_step = 000320, loss = 0.001760
grad_step = 000321, loss = 0.001768
grad_step = 000322, loss = 0.001775
grad_step = 000323, loss = 0.001779
grad_step = 000324, loss = 0.001781
grad_step = 000325, loss = 0.001775
grad_step = 000326, loss = 0.001766
grad_step = 000327, loss = 0.001756
grad_step = 000328, loss = 0.001749
grad_step = 000329, loss = 0.001744
grad_step = 000330, loss = 0.001741
grad_step = 000331, loss = 0.001741
grad_step = 000332, loss = 0.001743
grad_step = 000333, loss = 0.001745
grad_step = 000334, loss = 0.001748
grad_step = 000335, loss = 0.001751
grad_step = 000336, loss = 0.001755
grad_step = 000337, loss = 0.001761
grad_step = 000338, loss = 0.001765
grad_step = 000339, loss = 0.001773
grad_step = 000340, loss = 0.001778
grad_step = 000341, loss = 0.001787
grad_step = 000342, loss = 0.001792
grad_step = 000343, loss = 0.001799
grad_step = 000344, loss = 0.001798
grad_step = 000345, loss = 0.001799
grad_step = 000346, loss = 0.001788
grad_step = 000347, loss = 0.001778
grad_step = 000348, loss = 0.001759
grad_step = 000349, loss = 0.001742
grad_step = 000350, loss = 0.001727
grad_step = 000351, loss = 0.001717
grad_step = 000352, loss = 0.001711
grad_step = 000353, loss = 0.001710
grad_step = 000354, loss = 0.001713
grad_step = 000355, loss = 0.001717
grad_step = 000356, loss = 0.001723
grad_step = 000357, loss = 0.001729
grad_step = 000358, loss = 0.001740
grad_step = 000359, loss = 0.001750
grad_step = 000360, loss = 0.001770
grad_step = 000361, loss = 0.001779
grad_step = 000362, loss = 0.001797
grad_step = 000363, loss = 0.001795
grad_step = 000364, loss = 0.001793
grad_step = 000365, loss = 0.001776
grad_step = 000366, loss = 0.001751
grad_step = 000367, loss = 0.001722
grad_step = 000368, loss = 0.001699
grad_step = 000369, loss = 0.001689
grad_step = 000370, loss = 0.001691
grad_step = 000371, loss = 0.001702
grad_step = 000372, loss = 0.001724
grad_step = 000373, loss = 0.001760
grad_step = 000374, loss = 0.001775
grad_step = 000375, loss = 0.001811
grad_step = 000376, loss = 0.001788
grad_step = 000377, loss = 0.001776
grad_step = 000378, loss = 0.001759
grad_step = 000379, loss = 0.001754
grad_step = 000380, loss = 0.001740
grad_step = 000381, loss = 0.001704
grad_step = 000382, loss = 0.001678
grad_step = 000383, loss = 0.001675
grad_step = 000384, loss = 0.001680
grad_step = 000385, loss = 0.001687
grad_step = 000386, loss = 0.001692
grad_step = 000387, loss = 0.001700
grad_step = 000388, loss = 0.001710
grad_step = 000389, loss = 0.001704
grad_step = 000390, loss = 0.001708
grad_step = 000391, loss = 0.001705
grad_step = 000392, loss = 0.001725
grad_step = 000393, loss = 0.001736
grad_step = 000394, loss = 0.001742
grad_step = 000395, loss = 0.001711
grad_step = 000396, loss = 0.001674
grad_step = 000397, loss = 0.001646
grad_step = 000398, loss = 0.001643
grad_step = 000399, loss = 0.001655
grad_step = 000400, loss = 0.001662
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001658
grad_step = 000402, loss = 0.001644
grad_step = 000403, loss = 0.001642
grad_step = 000404, loss = 0.001653
grad_step = 000405, loss = 0.001675
grad_step = 000406, loss = 0.001690
grad_step = 000407, loss = 0.001724
grad_step = 000408, loss = 0.001755
grad_step = 000409, loss = 0.001823
grad_step = 000410, loss = 0.001816
grad_step = 000411, loss = 0.001814
grad_step = 000412, loss = 0.001758
grad_step = 000413, loss = 0.001694
grad_step = 000414, loss = 0.001641
grad_step = 000415, loss = 0.001624
grad_step = 000416, loss = 0.001656
grad_step = 000417, loss = 0.001699
grad_step = 000418, loss = 0.001695
grad_step = 000419, loss = 0.001654
grad_step = 000420, loss = 0.001613
grad_step = 000421, loss = 0.001607
grad_step = 000422, loss = 0.001628
grad_step = 000423, loss = 0.001633
grad_step = 000424, loss = 0.001619
grad_step = 000425, loss = 0.001598
grad_step = 000426, loss = 0.001594
grad_step = 000427, loss = 0.001609
grad_step = 000428, loss = 0.001615
grad_step = 000429, loss = 0.001604
grad_step = 000430, loss = 0.001584
grad_step = 000431, loss = 0.001580
grad_step = 000432, loss = 0.001589
grad_step = 000433, loss = 0.001591
grad_step = 000434, loss = 0.001583
grad_step = 000435, loss = 0.001573
grad_step = 000436, loss = 0.001574
grad_step = 000437, loss = 0.001580
grad_step = 000438, loss = 0.001578
grad_step = 000439, loss = 0.001571
grad_step = 000440, loss = 0.001564
grad_step = 000441, loss = 0.001565
grad_step = 000442, loss = 0.001569
grad_step = 000443, loss = 0.001568
grad_step = 000444, loss = 0.001566
grad_step = 000445, loss = 0.001567
grad_step = 000446, loss = 0.001575
grad_step = 000447, loss = 0.001590
grad_step = 000448, loss = 0.001614
grad_step = 000449, loss = 0.001637
grad_step = 000450, loss = 0.001675
grad_step = 000451, loss = 0.001669
grad_step = 000452, loss = 0.001657
grad_step = 000453, loss = 0.001599
grad_step = 000454, loss = 0.001569
grad_step = 000455, loss = 0.001567
grad_step = 000456, loss = 0.001567
grad_step = 000457, loss = 0.001572
grad_step = 000458, loss = 0.001579
grad_step = 000459, loss = 0.001579
grad_step = 000460, loss = 0.001565
grad_step = 000461, loss = 0.001543
grad_step = 000462, loss = 0.001536
grad_step = 000463, loss = 0.001545
grad_step = 000464, loss = 0.001556
grad_step = 000465, loss = 0.001557
grad_step = 000466, loss = 0.001551
grad_step = 000467, loss = 0.001549
grad_step = 000468, loss = 0.001570
grad_step = 000469, loss = 0.001595
grad_step = 000470, loss = 0.001642
grad_step = 000471, loss = 0.001632
grad_step = 000472, loss = 0.001614
grad_step = 000473, loss = 0.001573
grad_step = 000474, loss = 0.001567
grad_step = 000475, loss = 0.001568
grad_step = 000476, loss = 0.001537
grad_step = 000477, loss = 0.001522
grad_step = 000478, loss = 0.001544
grad_step = 000479, loss = 0.001561
grad_step = 000480, loss = 0.001548
grad_step = 000481, loss = 0.001518
grad_step = 000482, loss = 0.001512
grad_step = 000483, loss = 0.001529
grad_step = 000484, loss = 0.001535
grad_step = 000485, loss = 0.001527
grad_step = 000486, loss = 0.001510
grad_step = 000487, loss = 0.001512
grad_step = 000488, loss = 0.001528
grad_step = 000489, loss = 0.001541
grad_step = 000490, loss = 0.001544
grad_step = 000491, loss = 0.001552
grad_step = 000492, loss = 0.001560
grad_step = 000493, loss = 0.001573
grad_step = 000494, loss = 0.001549
grad_step = 000495, loss = 0.001526
grad_step = 000496, loss = 0.001505
grad_step = 000497, loss = 0.001501
grad_step = 000498, loss = 0.001503
grad_step = 000499, loss = 0.001500
grad_step = 000500, loss = 0.001503
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001507
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

  date_run                              2020-05-13 21:15:08.237164
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.260122
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 21:15:08.242791
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.164019
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 21:15:08.250153
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.163821
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 21:15:08.255128
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.49232
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
0   2020-05-13 21:14:42.091177  ...    mean_absolute_error
1   2020-05-13 21:14:42.095029  ...     mean_squared_error
2   2020-05-13 21:14:42.098062  ...  median_absolute_error
3   2020-05-13 21:14:42.101148  ...               r2_score
4   2020-05-13 21:14:50.528983  ...    mean_absolute_error
5   2020-05-13 21:14:50.532654  ...     mean_squared_error
6   2020-05-13 21:14:50.535709  ...  median_absolute_error
7   2020-05-13 21:14:50.538687  ...               r2_score
8   2020-05-13 21:15:08.237164  ...    mean_absolute_error
9   2020-05-13 21:15:08.242791  ...     mean_squared_error
10  2020-05-13 21:15:08.250153  ...  median_absolute_error
11  2020-05-13 21:15:08.255128  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa485a81d30> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:32, 307228.22it/s]  2%|▏         | 212992/9912422 [00:00<00:24, 401613.02it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 556807.69it/s] 30%|███       | 2998272/9912422 [00:00<00:08, 784178.85it/s] 57%|█████▋    | 5660672/9912422 [00:00<00:03, 1104539.61it/s] 87%|████████▋ | 8593408/9912422 [00:00<00:00, 1549533.75it/s]9920512it [00:00, 9945756.36it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 147398.52it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:04, 322966.30it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 418852.13it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 578917.41it/s]1654784it [00:00, 2892119.38it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 54174.24it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa43843de80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa437a6d0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa43843de80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4379c20f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4351fd4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4351e8c50> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa43843de80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa437980710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4351fd4e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa437a6d128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f41edd11208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=fcf13f099c25ed8a6ffecbf4a72fcd1d74fa2796b08bd8ef659dc3653f6417bf
  Stored in directory: /tmp/pip-ephem-wheel-cache-ke28b46q/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f4185b0d748> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   16384/17464789 [..............................] - ETA: 1:15
   49152/17464789 [..............................] - ETA: 49s 
   98304/17464789 [..............................] - ETA: 37s
  188416/17464789 [..............................] - ETA: 25s
  376832/17464789 [..............................] - ETA: 16s
  499712/17464789 [..............................] - ETA: 14s
  745472/17464789 [>.............................] - ETA: 11s
 1253376/17464789 [=>............................] - ETA: 7s 
 2228224/17464789 [==>...........................] - ETA: 4s
 4161536/17464789 [======>.......................] - ETA: 2s
 7192576/17464789 [===========>..................] - ETA: 1s
10190848/17464789 [================>.............] - ETA: 0s
13271040/17464789 [=====================>........] - ETA: 0s
16318464/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 21:16:38.511353: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 21:16:38.515292: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-13 21:16:38.515724: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563a39b7f950 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 21:16:38.515747: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 8.1113 - accuracy: 0.4710
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7663 - accuracy: 0.4935 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7280 - accuracy: 0.4960
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7203 - accuracy: 0.4965
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6605 - accuracy: 0.5004
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7177 - accuracy: 0.4967
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7104 - accuracy: 0.4971
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6734 - accuracy: 0.4996
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6298 - accuracy: 0.5024
11000/25000 [============>.................] - ETA: 3s - loss: 7.6318 - accuracy: 0.5023
12000/25000 [=============>................] - ETA: 3s - loss: 7.6513 - accuracy: 0.5010
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6395 - accuracy: 0.5018
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6261 - accuracy: 0.5026
15000/25000 [=================>............] - ETA: 2s - loss: 7.6278 - accuracy: 0.5025
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6091 - accuracy: 0.5038
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6423 - accuracy: 0.5016
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6436 - accuracy: 0.5015
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6368 - accuracy: 0.5019
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6344 - accuracy: 0.5021
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6433 - accuracy: 0.5015
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6360 - accuracy: 0.5020
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6540 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6558 - accuracy: 0.5007
25000/25000 [==============================] - 7s 276us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 21:16:52.002049
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 21:16:52.002049  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<11:52:07, 20.2kB/s].vector_cache/glove.6B.zip:   0%|          | 229k/862M [00:00<8:20:19, 28.7kB/s]  .vector_cache/glove.6B.zip:   0%|          | 3.72M/862M [00:00<5:48:56, 41.0kB/s].vector_cache/glove.6B.zip:   1%|          | 10.4M/862M [00:00<4:02:25, 58.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.1M/862M [00:00<2:49:04, 83.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.5M/862M [00:00<1:57:49, 119kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.3M/862M [00:01<1:22:13, 170kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.3M/862M [00:01<57:29, 243kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 27.9M/862M [00:01<40:16, 345kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.0M/862M [00:01<28:13, 491kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.5M/862M [00:01<19:47, 697kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.7M/862M [00:01<13:52, 989kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.8M/862M [00:01<09:44, 1.40MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.1M/862M [00:01<06:52, 1.97MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:02<05:24, 2.49MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.2M/862M [00:02<03:55, 3.42MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:04<07:35, 1.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:04<07:19, 1.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:04<05:12, 2.56MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<25:46, 518kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:06<19:33, 682kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<15:33, 854kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:08<12:33, 1.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:08<08:53, 1.49MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<14:11, 931kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:10<11:07, 1.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:10<07:52, 1.67MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<23:38, 556kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:12<17:34, 748kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:12<12:24, 1.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<14:38, 893kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:14<11:18, 1.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:14<07:59, 1.63MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<20:53, 623kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:16<15:28, 840kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.7M/862M [00:16<11:02, 1.17MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<11:08, 1.16MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:18<08:41, 1.49MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<08:02, 1.60MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:20<06:35, 1.95MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:20<04:43, 2.72MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<12:37, 1.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<10:05, 1.27MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:22<07:08, 1.78MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<24:16, 524kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:24<17:52, 712kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<14:23, 880kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<11:00, 1.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<09:34, 1.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<07:31, 1.67MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<05:21, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<21:48, 575kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<16:01, 782kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<11:17, 1.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:31<19:45, 631kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<15:13, 818kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<10:44, 1.16MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:33<14:55, 830kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<11:44, 1.05MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:34<08:16, 1.49MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<13:24:02, 15.3kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<9:23:15, 21.9kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<6:32:40, 31.2kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<4:51:26, 42.0kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<3:24:44, 59.8kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<2:24:28, 84.3kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<1:42:03, 119kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<1:12:52, 166kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:41<51:48, 234kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<37:54, 318kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<27:33, 437kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<20:57, 571kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:45<15:37, 766kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<12:39, 941kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<09:38, 1.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<08:32, 1.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<06:48, 1.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<06:33, 1.79MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<05:21, 2.19MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<05:32, 2.11MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<04:41, 2.49MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<05:03, 2.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<04:18, 2.70MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<04:46, 2.42MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<04:09, 2.78MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<04:44, 2.43MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<04:15, 2.69MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:01<04:41, 2.44MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<03:57, 2.89MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<04:30, 2.52MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<03:54, 2.90MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<04:27, 2.53MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<03:52, 2.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<04:26, 2.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<03:51, 2.90MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:24, 2.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<03:59, 2.79MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<04:36, 2.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<04:07, 2.69MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:31, 2.43MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<03:51, 2.85MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:23, 2.49MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:17, 2.54MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:34, 2.38MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:20, 2.51MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<03:08, 3.45MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<08:13, 1.31MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:36, 1.63MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:12, 1.73MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:08, 2.08MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:11, 2.06MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:20, 2.46MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<05:05, 2.08MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:26<05:04, 2.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<03:37, 2.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<18:22, 573kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<13:28, 780kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<11:01, 949kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<08:24, 1.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<07:26, 1.40MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<05:53, 1.76MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:41, 1.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:57, 2.08MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:57, 2.07MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:10, 2.45MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:28, 2.28MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:51, 2.63MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:13, 2.40MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<03:36, 2.80MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:03, 2.47MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:31, 2.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<03:59, 2.50MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<03:24, 2.93MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<03:53, 2.54MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:18, 2.99MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:47<03:49, 2.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<03:21, 2.93MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<03:50, 2.55MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<03:12, 3.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<03:46, 2.57MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<03:19, 2.91MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<03:47, 2.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<03:18, 2.90MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<03:46, 2.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<03:17, 2.90MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<03:45, 2.53MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:16, 2.89MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<03:42, 2.54MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<03:12, 2.93MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<03:45, 2.49MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<03:22, 2.78MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<03:44, 2.48MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<03:14, 2.86MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<03:45, 2.45MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<03:25, 2.70MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<02:28, 3.70MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<06:45, 1.36MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:20, 1.71MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:07, 1.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:11, 2.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<04:18, 2.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<03:39, 2.47MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<03:54, 2.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<03:19, 2.69MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<03:41, 2.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<03:37, 2.44MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<02:37, 3.37MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<06:41, 1.31MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<05:43, 1.54MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<04:02, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<1:13:27, 119kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<52:27, 167kB/s]  .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<36:35, 237kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<31:40, 274kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<23:50, 364kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<16:51, 513kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<13:25, 641kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<10:25, 825kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<08:28, 1.01MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<06:53, 1.24MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<04:52, 1.74MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<12:06, 699kB/s] .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<09:17, 910kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<06:31, 1.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<2:20:38, 59.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<1:38:55, 84.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<1:10:02, 119kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<50:25, 165kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<35:33, 234kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<25:11, 330kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<17:51, 464kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<12:41, 651kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<21:29, 384kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<16:12, 509kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<11:27, 718kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<09:54, 827kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<07:28, 1.09MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<06:26, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<05:04, 1.60MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:36, 2.23MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<07:57, 1.01MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<06:21, 1.27MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<04:31, 1.77MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<06:10, 1.29MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:41<04:49, 1.66MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:27, 2.30MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<05:40, 1.39MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<04:30, 1.75MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<04:20, 1.81MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<03:46, 2.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:41, 2.89MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<09:30, 818kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<07:14, 1.07MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<05:05, 1.51MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<23:32, 327kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<46:59, 164kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<33:34, 228kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<23:49, 320kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<17:47, 425kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<12:47, 591kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<10:05, 743kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<07:30, 997kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<06:21, 1.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:53, 1.52MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<04:33, 1.62MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:36, 2.04MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:38, 2.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:06, 2.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:14, 2.23MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<02:49, 2.56MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:01, 2.36MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<02:33, 2.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<02:57, 2.40MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<02:59, 2.37MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<02:10, 3.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:51, 1.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:18, 2.12MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<02:24, 2.91MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:47, 1.83MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:13, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:16, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<02:44, 2.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<01:58, 3.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<11:42, 582kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<10:11, 669kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<07:29, 909kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<06:13, 1.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<05:01, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<03:34, 1.87MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:33, 1.20MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:40, 1.43MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<03:21, 1.98MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<04:11, 1.58MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:49, 1.73MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:44, 2.39MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<04:15, 1.54MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<03:36, 1.81MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:34, 2.52MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<05:16, 1.23MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<04:06, 1.58MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:56, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<04:24, 1.45MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<04:09, 1.54MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:02, 2.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:29<03:56, 1.61MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<03:15, 1.94MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:18, 2.71MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<08:12, 763kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<05:48, 1.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<05:49, 1.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:44, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<03:21, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<06:33, 935kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<05:05, 1.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<03:35, 1.69MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<07:35, 797kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<05:46, 1.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<04:53, 1.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<03:52, 1.54MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<03:34, 1.66MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:50, 2.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<02:01, 2.89MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<06:49, 856kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<05:05, 1.15MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<04:26, 1.30MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:27, 1.67MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<03:17, 1.74MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<02:37, 2.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<02:41, 2.09MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<02:11, 2.57MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<02:23, 2.33MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<02:01, 2.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<02:15, 2.44MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<01:55, 2.87MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<02:10, 2.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<01:52, 2.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<02:07, 2.53MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<01:54, 2.82MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<01:21, 3.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<07:11, 736kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<05:18, 996kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<04:29, 1.16MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<03:28, 1.50MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:27, 2.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<43:09, 120kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<30:28, 169kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<21:53, 233kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<15:36, 326kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<11:34, 434kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<08:24, 596kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<06:35, 752kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<04:54, 1.01MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<04:09, 1.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<03:10, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:56, 1.63MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:21, 2.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:22, 2.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:04, 2.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:07, 2.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<01:53, 2.46MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<01:59, 2.31MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<01:44, 2.65MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<01:52, 2.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<01:50, 2.47MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<01:54, 2.33MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<01:38, 2.73MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<01:48, 2.44MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<01:34, 2.79MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<01:07, 3.85MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<07:49, 554kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<05:48, 746kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<04:38, 919kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:30, 1.21MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<02:28, 1.70MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<09:13, 455kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<06:49, 615kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<05:18, 777kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<04:09, 990kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<02:55, 1.39MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<04:24, 922kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:30, 1.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:29, 1.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:56, 1.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:32, 1.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:47, 2.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<04:49, 812kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<03:46, 1.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<02:38, 1.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<05:15, 732kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:54, 982kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<02:45, 1.38MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<03:27, 1.09MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<02:43, 1.38MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<02:25, 1.53MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:02, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:25, 2.55MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<08:09, 447kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<06:06, 596kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<04:15, 844kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<04:40, 767kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:28, 1.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:56, 1.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:19, 1.51MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<02:08, 1.60MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:49, 1.88MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:19, 2.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:47, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:41, 1.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:12, 2.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<01:58, 1.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<01:51, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:20, 2.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<01:58, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:57<01:46, 1.82MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:16, 2.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<01:47, 1.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:39, 1.90MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:11, 2.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<01:48, 1.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:01<01:41, 1.83MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:13, 2.51MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:34, 1.91MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:32, 1.96MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:07, 2.68MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:30, 1.96MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:53, 1.56MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:24, 2.09MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:00, 2.89MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<04:27, 649kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<03:49, 755kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:43, 1.05MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:56, 1.47MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:10<03:42, 761kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:10<02:49, 999kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:57, 1.41MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<08:46, 314kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:12<06:13, 440kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<04:42, 570kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:14<03:26, 775kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<02:46, 944kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:16<02:06, 1.24MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<01:50, 1.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:18<01:26, 1.77MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:21, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:06, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:07, 2.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<00:56, 2.55MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:23<01:00, 2.33MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<00:51, 2.74MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<00:55, 2.45MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<00:46, 2.89MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<00:52, 2.51MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<00:45, 2.88MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<00:50, 2.53MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<00:44, 2.86MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:31<00:49, 2.52MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<00:42, 2.87MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<00:48, 2.46MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<00:43, 2.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<00:46, 2.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<00:39, 2.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<00:44, 2.51MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<00:37, 2.95MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<00:42, 2.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<00:36, 2.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:26, 3.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:41<01:18, 1.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:41<01:02, 1.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:43, 2.29MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:43<02:05, 787kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:43<01:37, 1.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<01:07, 1.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:45<01:25, 1.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<01:09, 1.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<00:47, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:47<01:27, 1.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<01:06, 1.36MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:49<00:57, 1.50MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<00:45, 1.87MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:51<00:43, 1.89MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:51<00:35, 2.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<00:36, 2.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<00:32, 2.43MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:22, 3.38MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<03:35, 344kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<02:34, 477kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:57<01:53, 617kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<01:23, 834kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:59<01:05, 1.01MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:59<00:53, 1.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:36, 1.73MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:01<00:46, 1.32MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<00:38, 1.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:03<00:33, 1.72MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:03<00:30, 1.91MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:03<00:21, 2.59MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:15, 3.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:05<01:05, 822kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:05<00:49, 1.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:32, 1.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:07<02:25, 339kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:07<01:45, 465kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:09<01:14, 605kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:09<00:54, 821kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:36, 1.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:11<00:46, 886kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:11<00:36, 1.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:23, 1.59MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:13<08:16, 74.7kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:13<05:46, 106kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:15<03:42, 148kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:15<02:37, 207kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<01:38, 295kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:17<01:56, 248kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:17<01:22, 343kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:19<00:53, 457kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:19<00:38, 628kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:21<00:26, 787kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:21<00:19, 1.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:23<00:13, 1.21MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:23<00:10, 1.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:25<00:07, 1.67MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:25<00:06, 1.93MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.69MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:27<01:47, 75.7kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:27<01:10, 107kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:29<00:26, 150kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:29<00:17, 210kB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 854/400000 [00:00<00:46, 8533.82it/s]  0%|          | 1723/400000 [00:00<00:46, 8578.23it/s]  1%|          | 2590/400000 [00:00<00:46, 8603.98it/s]  1%|          | 3454/400000 [00:00<00:46, 8613.90it/s]  1%|          | 4322/400000 [00:00<00:45, 8631.64it/s]  1%|▏         | 5185/400000 [00:00<00:45, 8628.75it/s]  2%|▏         | 6050/400000 [00:00<00:45, 8634.80it/s]  2%|▏         | 6898/400000 [00:00<00:45, 8587.34it/s]  2%|▏         | 7717/400000 [00:00<00:46, 8462.83it/s]  2%|▏         | 8538/400000 [00:01<00:46, 8382.53it/s]  2%|▏         | 9374/400000 [00:01<00:46, 8373.55it/s]  3%|▎         | 10235/400000 [00:01<00:46, 8441.23it/s]  3%|▎         | 11113/400000 [00:01<00:45, 8537.59it/s]  3%|▎         | 11982/400000 [00:01<00:45, 8580.83it/s]  3%|▎         | 12850/400000 [00:01<00:44, 8610.19it/s]  3%|▎         | 13708/400000 [00:01<00:44, 8597.16it/s]  4%|▎         | 14570/400000 [00:01<00:44, 8603.34it/s]  4%|▍         | 15433/400000 [00:01<00:44, 8610.75it/s]  4%|▍         | 16293/400000 [00:01<00:46, 8316.26it/s]  4%|▍         | 17152/400000 [00:02<00:45, 8395.61it/s]  5%|▍         | 18012/400000 [00:02<00:45, 8454.25it/s]  5%|▍         | 18871/400000 [00:02<00:44, 8493.57it/s]  5%|▍         | 19737/400000 [00:02<00:44, 8542.06it/s]  5%|▌         | 20602/400000 [00:02<00:44, 8573.92it/s]  5%|▌         | 21466/400000 [00:02<00:44, 8593.04it/s]  6%|▌         | 22326/400000 [00:02<00:44, 8539.22it/s]  6%|▌         | 23181/400000 [00:02<00:44, 8396.83it/s]  6%|▌         | 24043/400000 [00:02<00:44, 8462.41it/s]  6%|▌         | 24908/400000 [00:02<00:44, 8516.25it/s]  6%|▋         | 25769/400000 [00:03<00:43, 8543.03it/s]  7%|▋         | 26624/400000 [00:03<00:43, 8535.55it/s]  7%|▋         | 27492/400000 [00:03<00:43, 8577.83it/s]  7%|▋         | 28373/400000 [00:03<00:42, 8643.54it/s]  7%|▋         | 29239/400000 [00:03<00:42, 8646.19it/s]  8%|▊         | 30113/400000 [00:03<00:42, 8671.52it/s]  8%|▊         | 30981/400000 [00:03<00:42, 8618.82it/s]  8%|▊         | 31846/400000 [00:03<00:42, 8627.17it/s]  8%|▊         | 32723/400000 [00:03<00:42, 8669.09it/s]  8%|▊         | 33591/400000 [00:03<00:43, 8500.91it/s]  9%|▊         | 34442/400000 [00:04<00:43, 8481.18it/s]  9%|▉         | 35291/400000 [00:04<00:43, 8446.40it/s]  9%|▉         | 36137/400000 [00:04<00:43, 8420.19it/s]  9%|▉         | 37002/400000 [00:04<00:42, 8486.32it/s]  9%|▉         | 37871/400000 [00:04<00:42, 8546.19it/s] 10%|▉         | 38742/400000 [00:04<00:42, 8593.91it/s] 10%|▉         | 39606/400000 [00:04<00:41, 8607.22it/s] 10%|█         | 40482/400000 [00:04<00:41, 8649.93it/s] 10%|█         | 41354/400000 [00:04<00:41, 8669.46it/s] 11%|█         | 42228/400000 [00:04<00:41, 8688.84it/s] 11%|█         | 43099/400000 [00:05<00:41, 8693.43it/s] 11%|█         | 43969/400000 [00:05<00:40, 8690.42it/s] 11%|█         | 44839/400000 [00:05<00:40, 8679.68it/s] 11%|█▏        | 45708/400000 [00:05<00:40, 8677.64it/s] 12%|█▏        | 46583/400000 [00:05<00:40, 8696.95it/s] 12%|█▏        | 47453/400000 [00:05<00:40, 8695.56it/s] 12%|█▏        | 48323/400000 [00:05<00:40, 8659.41it/s] 12%|█▏        | 49189/400000 [00:05<00:40, 8637.09it/s] 13%|█▎        | 50053/400000 [00:05<00:40, 8600.80it/s] 13%|█▎        | 50914/400000 [00:05<00:41, 8474.23it/s] 13%|█▎        | 51777/400000 [00:06<00:40, 8519.35it/s] 13%|█▎        | 52630/400000 [00:06<00:40, 8473.26it/s] 13%|█▎        | 53486/400000 [00:06<00:40, 8496.64it/s] 14%|█▎        | 54336/400000 [00:06<00:40, 8476.78it/s] 14%|█▍        | 55204/400000 [00:06<00:40, 8535.00it/s] 14%|█▍        | 56074/400000 [00:06<00:40, 8582.48it/s] 14%|█▍        | 56953/400000 [00:06<00:39, 8641.27it/s] 14%|█▍        | 57823/400000 [00:06<00:39, 8658.43it/s] 15%|█▍        | 58690/400000 [00:06<00:39, 8552.16it/s] 15%|█▍        | 59570/400000 [00:06<00:39, 8622.29it/s] 15%|█▌        | 60450/400000 [00:07<00:39, 8672.73it/s] 15%|█▌        | 61331/400000 [00:07<00:38, 8711.75it/s] 16%|█▌        | 62204/400000 [00:07<00:38, 8715.15it/s] 16%|█▌        | 63080/400000 [00:07<00:38, 8726.80it/s] 16%|█▌        | 63961/400000 [00:07<00:38, 8749.25it/s] 16%|█▌        | 64837/400000 [00:07<00:38, 8635.90it/s] 16%|█▋        | 65714/400000 [00:07<00:38, 8675.65it/s] 17%|█▋        | 66592/400000 [00:07<00:38, 8703.89it/s] 17%|█▋        | 67463/400000 [00:07<00:38, 8668.27it/s] 17%|█▋        | 68345/400000 [00:07<00:38, 8712.54it/s] 17%|█▋        | 69226/400000 [00:08<00:37, 8740.91it/s] 18%|█▊        | 70103/400000 [00:08<00:37, 8747.49it/s] 18%|█▊        | 70978/400000 [00:08<00:37, 8740.06it/s] 18%|█▊        | 71856/400000 [00:08<00:37, 8750.65it/s] 18%|█▊        | 72732/400000 [00:08<00:37, 8753.39it/s] 18%|█▊        | 73614/400000 [00:08<00:37, 8771.12it/s] 19%|█▊        | 74492/400000 [00:08<00:37, 8770.86it/s] 19%|█▉        | 75370/400000 [00:08<00:37, 8768.67it/s] 19%|█▉        | 76247/400000 [00:08<00:37, 8731.88it/s] 19%|█▉        | 77121/400000 [00:08<00:37, 8642.41it/s] 20%|█▉        | 78001/400000 [00:09<00:37, 8687.21it/s] 20%|█▉        | 78877/400000 [00:09<00:36, 8706.19it/s] 20%|█▉        | 79748/400000 [00:09<00:36, 8702.30it/s] 20%|██        | 80625/400000 [00:09<00:36, 8719.91it/s] 20%|██        | 81503/400000 [00:09<00:36, 8737.30it/s] 21%|██        | 82385/400000 [00:09<00:36, 8760.86it/s] 21%|██        | 83265/400000 [00:09<00:36, 8771.26it/s] 21%|██        | 84143/400000 [00:09<00:36, 8764.25it/s] 21%|██▏       | 85023/400000 [00:09<00:35, 8772.13it/s] 21%|██▏       | 85903/400000 [00:09<00:35, 8780.05it/s] 22%|██▏       | 86782/400000 [00:10<00:35, 8766.25it/s] 22%|██▏       | 87659/400000 [00:10<00:35, 8764.28it/s] 22%|██▏       | 88536/400000 [00:10<00:35, 8754.52it/s] 22%|██▏       | 89413/400000 [00:10<00:35, 8758.77it/s] 23%|██▎       | 90291/400000 [00:10<00:35, 8764.99it/s] 23%|██▎       | 91168/400000 [00:10<00:35, 8760.48it/s] 23%|██▎       | 92045/400000 [00:10<00:35, 8725.62it/s] 23%|██▎       | 92919/400000 [00:10<00:35, 8727.33it/s] 23%|██▎       | 93793/400000 [00:10<00:35, 8728.16it/s] 24%|██▎       | 94671/400000 [00:10<00:34, 8742.40it/s] 24%|██▍       | 95551/400000 [00:11<00:34, 8758.33it/s] 24%|██▍       | 96431/400000 [00:11<00:34, 8768.43it/s] 24%|██▍       | 97308/400000 [00:11<00:34, 8752.67it/s] 25%|██▍       | 98184/400000 [00:11<00:34, 8737.00it/s] 25%|██▍       | 99063/400000 [00:11<00:34, 8752.78it/s] 25%|██▍       | 99942/400000 [00:11<00:34, 8763.37it/s] 25%|██▌       | 100823/400000 [00:11<00:34, 8775.57it/s] 25%|██▌       | 101701/400000 [00:11<00:34, 8771.87it/s] 26%|██▌       | 102579/400000 [00:11<00:33, 8758.34it/s] 26%|██▌       | 103455/400000 [00:11<00:34, 8673.14it/s] 26%|██▌       | 104336/400000 [00:12<00:33, 8713.59it/s] 26%|██▋       | 105217/400000 [00:12<00:33, 8740.86it/s] 27%|██▋       | 106093/400000 [00:12<00:33, 8746.42it/s] 27%|██▋       | 106968/400000 [00:12<00:34, 8534.55it/s] 27%|██▋       | 107826/400000 [00:12<00:34, 8546.04it/s] 27%|██▋       | 108698/400000 [00:12<00:33, 8596.10it/s] 27%|██▋       | 109569/400000 [00:12<00:33, 8629.82it/s] 28%|██▊       | 110433/400000 [00:12<00:33, 8601.22it/s] 28%|██▊       | 111300/400000 [00:12<00:33, 8621.58it/s] 28%|██▊       | 112163/400000 [00:12<00:33, 8611.54it/s] 28%|██▊       | 113033/400000 [00:13<00:33, 8637.21it/s] 28%|██▊       | 113914/400000 [00:13<00:32, 8687.71it/s] 29%|██▊       | 114786/400000 [00:13<00:32, 8697.00it/s] 29%|██▉       | 115664/400000 [00:13<00:32, 8719.39it/s] 29%|██▉       | 116543/400000 [00:13<00:32, 8739.02it/s] 29%|██▉       | 117424/400000 [00:13<00:32, 8758.10it/s] 30%|██▉       | 118302/400000 [00:13<00:32, 8764.30it/s] 30%|██▉       | 119179/400000 [00:13<00:32, 8762.80it/s] 30%|███       | 120056/400000 [00:13<00:32, 8734.32it/s] 30%|███       | 120936/400000 [00:13<00:31, 8753.31it/s] 30%|███       | 121814/400000 [00:14<00:31, 8758.98it/s] 31%|███       | 122693/400000 [00:14<00:31, 8767.36it/s] 31%|███       | 123570/400000 [00:14<00:31, 8719.65it/s] 31%|███       | 124443/400000 [00:14<00:32, 8373.24it/s] 31%|███▏      | 125317/400000 [00:14<00:32, 8478.35it/s] 32%|███▏      | 126200/400000 [00:14<00:31, 8579.88it/s] 32%|███▏      | 127071/400000 [00:14<00:31, 8618.02it/s] 32%|███▏      | 127943/400000 [00:14<00:31, 8646.18it/s] 32%|███▏      | 128809/400000 [00:14<00:31, 8640.29it/s] 32%|███▏      | 129684/400000 [00:15<00:31, 8670.66it/s] 33%|███▎      | 130559/400000 [00:15<00:30, 8692.21it/s] 33%|███▎      | 131431/400000 [00:15<00:30, 8698.49it/s] 33%|███▎      | 132311/400000 [00:15<00:30, 8726.65it/s] 33%|███▎      | 133184/400000 [00:15<00:30, 8709.51it/s] 34%|███▎      | 134061/400000 [00:15<00:30, 8726.25it/s] 34%|███▎      | 134934/400000 [00:15<00:30, 8705.11it/s] 34%|███▍      | 135805/400000 [00:15<00:30, 8686.02it/s] 34%|███▍      | 136674/400000 [00:15<00:30, 8653.81it/s] 34%|███▍      | 137540/400000 [00:15<00:30, 8617.03it/s] 35%|███▍      | 138416/400000 [00:16<00:30, 8656.80it/s] 35%|███▍      | 139293/400000 [00:16<00:29, 8690.37it/s] 35%|███▌      | 140163/400000 [00:16<00:30, 8621.89it/s] 35%|███▌      | 141026/400000 [00:16<00:30, 8551.25it/s] 35%|███▌      | 141882/400000 [00:16<00:30, 8329.69it/s] 36%|███▌      | 142728/400000 [00:16<00:30, 8367.40it/s] 36%|███▌      | 143606/400000 [00:16<00:30, 8484.79it/s] 36%|███▌      | 144482/400000 [00:16<00:29, 8563.43it/s] 36%|███▋      | 145358/400000 [00:16<00:29, 8620.75it/s] 37%|███▋      | 146230/400000 [00:16<00:29, 8649.28it/s] 37%|███▋      | 147110/400000 [00:17<00:29, 8693.52it/s] 37%|███▋      | 147988/400000 [00:17<00:28, 8718.78it/s] 37%|███▋      | 148869/400000 [00:17<00:28, 8743.70it/s] 37%|███▋      | 149745/400000 [00:17<00:28, 8748.00it/s] 38%|███▊      | 150620/400000 [00:17<00:28, 8730.31it/s] 38%|███▊      | 151494/400000 [00:17<00:28, 8592.30it/s] 38%|███▊      | 152354/400000 [00:17<00:29, 8403.58it/s] 38%|███▊      | 153233/400000 [00:17<00:28, 8514.26it/s] 39%|███▊      | 154111/400000 [00:17<00:28, 8590.83it/s] 39%|███▊      | 154983/400000 [00:17<00:28, 8626.53it/s] 39%|███▉      | 155863/400000 [00:18<00:28, 8676.33it/s] 39%|███▉      | 156740/400000 [00:18<00:27, 8703.09it/s] 39%|███▉      | 157619/400000 [00:18<00:27, 8726.13it/s] 40%|███▉      | 158492/400000 [00:18<00:27, 8725.35it/s] 40%|███▉      | 159365/400000 [00:18<00:27, 8723.44it/s] 40%|████      | 160242/400000 [00:18<00:27, 8735.82it/s] 40%|████      | 161122/400000 [00:18<00:27, 8753.70it/s] 40%|████      | 161998/400000 [00:18<00:27, 8748.08it/s] 41%|████      | 162877/400000 [00:18<00:27, 8759.55it/s] 41%|████      | 163754/400000 [00:18<00:26, 8757.07it/s] 41%|████      | 164634/400000 [00:19<00:26, 8769.02it/s] 41%|████▏     | 165511/400000 [00:19<00:27, 8650.05it/s] 42%|████▏     | 166388/400000 [00:19<00:26, 8684.26it/s] 42%|████▏     | 167257/400000 [00:19<00:26, 8657.28it/s] 42%|████▏     | 168123/400000 [00:19<00:27, 8431.56it/s] 42%|████▏     | 169001/400000 [00:19<00:27, 8531.07it/s] 42%|████▏     | 169879/400000 [00:19<00:26, 8603.32it/s] 43%|████▎     | 170751/400000 [00:19<00:26, 8635.85it/s] 43%|████▎     | 171626/400000 [00:19<00:26, 8668.29it/s] 43%|████▎     | 172494/400000 [00:19<00:26, 8655.85it/s] 43%|████▎     | 173365/400000 [00:20<00:26, 8670.10it/s] 44%|████▎     | 174241/400000 [00:20<00:25, 8696.76it/s] 44%|████▍     | 175111/400000 [00:20<00:25, 8678.30it/s] 44%|████▍     | 175979/400000 [00:20<00:26, 8589.47it/s] 44%|████▍     | 176852/400000 [00:20<00:25, 8629.83it/s] 44%|████▍     | 177727/400000 [00:20<00:25, 8663.89it/s] 45%|████▍     | 178602/400000 [00:20<00:25, 8687.87it/s] 45%|████▍     | 179479/400000 [00:20<00:25, 8711.45it/s] 45%|████▌     | 180351/400000 [00:20<00:25, 8674.45it/s] 45%|████▌     | 181219/400000 [00:20<00:25, 8415.87it/s] 46%|████▌     | 182063/400000 [00:21<00:26, 8257.05it/s] 46%|████▌     | 182915/400000 [00:21<00:26, 8332.60it/s] 46%|████▌     | 183796/400000 [00:21<00:25, 8467.72it/s] 46%|████▌     | 184675/400000 [00:21<00:25, 8560.88it/s] 46%|████▋     | 185545/400000 [00:21<00:24, 8601.39it/s] 47%|████▋     | 186428/400000 [00:21<00:24, 8667.52it/s] 47%|████▋     | 187309/400000 [00:21<00:24, 8709.41it/s] 47%|████▋     | 188183/400000 [00:21<00:24, 8717.78it/s] 47%|████▋     | 189056/400000 [00:21<00:24, 8605.09it/s] 47%|████▋     | 189918/400000 [00:21<00:24, 8607.31it/s] 48%|████▊     | 190796/400000 [00:22<00:24, 8655.88it/s] 48%|████▊     | 191663/400000 [00:22<00:24, 8659.86it/s] 48%|████▊     | 192530/400000 [00:22<00:23, 8654.90it/s] 48%|████▊     | 193400/400000 [00:22<00:23, 8668.27it/s] 49%|████▊     | 194267/400000 [00:22<00:23, 8665.70it/s] 49%|████▉     | 195142/400000 [00:22<00:23, 8689.18it/s] 49%|████▉     | 196011/400000 [00:22<00:24, 8343.13it/s] 49%|████▉     | 196849/400000 [00:22<00:24, 8245.10it/s] 49%|████▉     | 197723/400000 [00:22<00:24, 8387.53it/s] 50%|████▉     | 198598/400000 [00:23<00:23, 8492.08it/s] 50%|████▉     | 199477/400000 [00:23<00:23, 8577.22it/s] 50%|█████     | 200340/400000 [00:23<00:23, 8592.66it/s] 50%|█████     | 201222/400000 [00:23<00:22, 8658.51it/s] 51%|█████     | 202089/400000 [00:23<00:23, 8566.35it/s] 51%|█████     | 202960/400000 [00:23<00:22, 8608.24it/s] 51%|█████     | 203835/400000 [00:23<00:22, 8649.93it/s] 51%|█████     | 204714/400000 [00:23<00:22, 8689.30it/s] 51%|█████▏    | 205592/400000 [00:23<00:22, 8716.01it/s] 52%|█████▏    | 206472/400000 [00:23<00:22, 8739.64it/s] 52%|█████▏    | 207347/400000 [00:24<00:22, 8727.79it/s] 52%|█████▏    | 208227/400000 [00:24<00:21, 8747.78it/s] 52%|█████▏    | 209102/400000 [00:24<00:22, 8581.97it/s] 52%|█████▏    | 209984/400000 [00:24<00:21, 8649.54it/s] 53%|█████▎    | 210854/400000 [00:24<00:21, 8664.30it/s] 53%|█████▎    | 211726/400000 [00:24<00:21, 8679.99it/s] 53%|█████▎    | 212605/400000 [00:24<00:21, 8712.48it/s] 53%|█████▎    | 213484/400000 [00:24<00:21, 8735.55it/s] 54%|█████▎    | 214358/400000 [00:24<00:21, 8604.76it/s] 54%|█████▍    | 215233/400000 [00:24<00:21, 8645.71it/s] 54%|█████▍    | 216099/400000 [00:25<00:21, 8613.28it/s] 54%|█████▍    | 216971/400000 [00:25<00:21, 8643.78it/s] 54%|█████▍    | 217846/400000 [00:25<00:21, 8673.42it/s] 55%|█████▍    | 218722/400000 [00:25<00:20, 8699.15it/s] 55%|█████▍    | 219597/400000 [00:25<00:20, 8712.74it/s] 55%|█████▌    | 220469/400000 [00:25<00:20, 8618.02it/s] 55%|█████▌    | 221344/400000 [00:25<00:20, 8655.75it/s] 56%|█████▌    | 222217/400000 [00:25<00:20, 8675.37it/s] 56%|█████▌    | 223085/400000 [00:25<00:20, 8631.58it/s] 56%|█████▌    | 223949/400000 [00:25<00:20, 8591.84it/s] 56%|█████▌    | 224813/400000 [00:26<00:20, 8605.43it/s] 56%|█████▋    | 225679/400000 [00:26<00:20, 8620.76it/s] 57%|█████▋    | 226542/400000 [00:26<00:20, 8486.98it/s] 57%|█████▋    | 227392/400000 [00:26<00:20, 8439.37it/s] 57%|█████▋    | 228259/400000 [00:26<00:20, 8506.81it/s] 57%|█████▋    | 229114/400000 [00:26<00:20, 8519.49it/s] 57%|█████▋    | 229987/400000 [00:26<00:19, 8580.22it/s] 58%|█████▊    | 230862/400000 [00:26<00:19, 8628.03it/s] 58%|█████▊    | 231726/400000 [00:26<00:19, 8626.95it/s] 58%|█████▊    | 232589/400000 [00:26<00:19, 8574.11it/s] 58%|█████▊    | 233462/400000 [00:27<00:19, 8618.84it/s] 59%|█████▊    | 234344/400000 [00:27<00:19, 8677.03it/s] 59%|█████▉    | 235212/400000 [00:27<00:19, 8662.99it/s] 59%|█████▉    | 236092/400000 [00:27<00:18, 8703.09it/s] 59%|█████▉    | 236963/400000 [00:27<00:18, 8697.90it/s] 59%|█████▉    | 237838/400000 [00:27<00:18, 8710.50it/s] 60%|█████▉    | 238713/400000 [00:27<00:18, 8720.95it/s] 60%|█████▉    | 239591/400000 [00:27<00:18, 8735.67it/s] 60%|██████    | 240465/400000 [00:27<00:18, 8735.46it/s] 60%|██████    | 241346/400000 [00:27<00:18, 8756.97it/s] 61%|██████    | 242222/400000 [00:28<00:18, 8751.44it/s] 61%|██████    | 243098/400000 [00:28<00:17, 8743.01it/s] 61%|██████    | 243980/400000 [00:28<00:17, 8763.82it/s] 61%|██████    | 244858/400000 [00:28<00:17, 8766.26it/s] 61%|██████▏   | 245735/400000 [00:28<00:17, 8744.73it/s] 62%|██████▏   | 246610/400000 [00:28<00:17, 8717.84it/s] 62%|██████▏   | 247482/400000 [00:28<00:17, 8671.87it/s] 62%|██████▏   | 248355/400000 [00:28<00:17, 8688.77it/s] 62%|██████▏   | 249224/400000 [00:28<00:17, 8649.07it/s] 63%|██████▎   | 250095/400000 [00:28<00:17, 8666.34it/s] 63%|██████▎   | 250965/400000 [00:29<00:17, 8673.75it/s] 63%|██████▎   | 251833/400000 [00:29<00:17, 8531.95it/s] 63%|██████▎   | 252693/400000 [00:29<00:17, 8551.98it/s] 63%|██████▎   | 253556/400000 [00:29<00:17, 8574.44it/s] 64%|██████▎   | 254416/400000 [00:29<00:16, 8581.82it/s] 64%|██████▍   | 255275/400000 [00:29<00:16, 8556.46it/s] 64%|██████▍   | 256131/400000 [00:29<00:16, 8523.25it/s] 64%|██████▍   | 256999/400000 [00:29<00:16, 8567.56it/s] 64%|██████▍   | 257866/400000 [00:29<00:16, 8597.46it/s] 65%|██████▍   | 258727/400000 [00:29<00:16, 8599.87it/s] 65%|██████▍   | 259588/400000 [00:30<00:17, 8256.28it/s] 65%|██████▌   | 260454/400000 [00:30<00:16, 8371.81it/s] 65%|██████▌   | 261324/400000 [00:30<00:16, 8465.00it/s] 66%|██████▌   | 262192/400000 [00:30<00:16, 8526.34it/s] 66%|██████▌   | 263061/400000 [00:30<00:15, 8572.44it/s] 66%|██████▌   | 263934/400000 [00:30<00:15, 8617.46it/s] 66%|██████▌   | 264797/400000 [00:30<00:15, 8571.84it/s] 66%|██████▋   | 265672/400000 [00:30<00:15, 8623.14it/s] 67%|██████▋   | 266544/400000 [00:30<00:15, 8651.81it/s] 67%|██████▋   | 267414/400000 [00:30<00:15, 8665.83it/s] 67%|██████▋   | 268287/400000 [00:31<00:15, 8684.94it/s] 67%|██████▋   | 269156/400000 [00:31<00:15, 8679.01it/s] 68%|██████▊   | 270028/400000 [00:31<00:14, 8690.02it/s] 68%|██████▊   | 270901/400000 [00:31<00:14, 8700.97it/s] 68%|██████▊   | 271772/400000 [00:31<00:14, 8678.81it/s] 68%|██████▊   | 272645/400000 [00:31<00:14, 8691.21it/s] 68%|██████▊   | 273515/400000 [00:31<00:14, 8651.96it/s] 69%|██████▊   | 274387/400000 [00:31<00:14, 8670.89it/s] 69%|██████▉   | 275255/400000 [00:31<00:14, 8558.06it/s] 69%|██████▉   | 276115/400000 [00:31<00:14, 8567.42it/s] 69%|██████▉   | 276990/400000 [00:32<00:14, 8618.57it/s] 69%|██████▉   | 277853/400000 [00:32<00:14, 8576.97it/s] 70%|██████▉   | 278718/400000 [00:32<00:14, 8596.13it/s] 70%|██████▉   | 279578/400000 [00:32<00:14, 8427.97it/s] 70%|███████   | 280448/400000 [00:32<00:14, 8507.22it/s] 70%|███████   | 281310/400000 [00:32<00:13, 8539.52it/s] 71%|███████   | 282166/400000 [00:32<00:13, 8545.23it/s] 71%|███████   | 283021/400000 [00:32<00:14, 8354.06it/s] 71%|███████   | 283880/400000 [00:32<00:13, 8422.41it/s] 71%|███████   | 284724/400000 [00:32<00:13, 8347.50it/s] 71%|███████▏  | 285568/400000 [00:33<00:13, 8374.33it/s] 72%|███████▏  | 286407/400000 [00:33<00:13, 8331.36it/s] 72%|███████▏  | 287273/400000 [00:33<00:13, 8425.89it/s] 72%|███████▏  | 288125/400000 [00:33<00:13, 8452.67it/s] 72%|███████▏  | 288989/400000 [00:33<00:13, 8507.24it/s] 72%|███████▏  | 289847/400000 [00:33<00:12, 8527.22it/s] 73%|███████▎  | 290701/400000 [00:33<00:12, 8490.32it/s] 73%|███████▎  | 291568/400000 [00:33<00:12, 8542.69it/s] 73%|███████▎  | 292425/400000 [00:33<00:12, 8550.86it/s] 73%|███████▎  | 293299/400000 [00:33<00:12, 8604.82it/s] 74%|███████▎  | 294169/400000 [00:34<00:12, 8631.80it/s] 74%|███████▍  | 295033/400000 [00:34<00:12, 8631.17it/s] 74%|███████▍  | 295904/400000 [00:34<00:12, 8652.04it/s] 74%|███████▍  | 296779/400000 [00:34<00:11, 8680.91it/s] 74%|███████▍  | 297652/400000 [00:34<00:11, 8693.33it/s] 75%|███████▍  | 298526/400000 [00:34<00:11, 8707.21it/s] 75%|███████▍  | 299397/400000 [00:34<00:11, 8640.33it/s] 75%|███████▌  | 300267/400000 [00:34<00:11, 8657.54it/s] 75%|███████▌  | 301133/400000 [00:34<00:11, 8607.64it/s] 76%|███████▌  | 302005/400000 [00:35<00:11, 8639.24it/s] 76%|███████▌  | 302875/400000 [00:35<00:11, 8654.54it/s] 76%|███████▌  | 303744/400000 [00:35<00:11, 8662.71it/s] 76%|███████▌  | 304618/400000 [00:35<00:10, 8683.71it/s] 76%|███████▋  | 305493/400000 [00:35<00:10, 8702.91it/s] 77%|███████▋  | 306364/400000 [00:35<00:10, 8702.00it/s] 77%|███████▋  | 307235/400000 [00:35<00:10, 8665.76it/s] 77%|███████▋  | 308102/400000 [00:35<00:10, 8663.34it/s] 77%|███████▋  | 308969/400000 [00:35<00:10, 8650.91it/s] 77%|███████▋  | 309835/400000 [00:35<00:10, 8521.38it/s] 78%|███████▊  | 310688/400000 [00:36<00:10, 8512.16it/s] 78%|███████▊  | 311555/400000 [00:36<00:10, 8556.55it/s] 78%|███████▊  | 312418/400000 [00:36<00:10, 8578.10it/s] 78%|███████▊  | 313291/400000 [00:36<00:10, 8620.44it/s] 79%|███████▊  | 314154/400000 [00:36<00:10, 8485.09it/s] 79%|███████▉  | 315004/400000 [00:36<00:10, 8484.93it/s] 79%|███████▉  | 315856/400000 [00:36<00:09, 8494.25it/s] 79%|███████▉  | 316706/400000 [00:36<00:09, 8330.52it/s] 79%|███████▉  | 317541/400000 [00:36<00:09, 8334.28it/s] 80%|███████▉  | 318401/400000 [00:36<00:09, 8409.50it/s] 80%|███████▉  | 319275/400000 [00:37<00:09, 8505.27it/s] 80%|████████  | 320140/400000 [00:37<00:09, 8546.42it/s] 80%|████████  | 320996/400000 [00:37<00:09, 8549.26it/s] 80%|████████  | 321862/400000 [00:37<00:09, 8580.87it/s] 81%|████████  | 322736/400000 [00:37<00:08, 8627.05it/s] 81%|████████  | 323612/400000 [00:37<00:08, 8664.47it/s] 81%|████████  | 324479/400000 [00:37<00:08, 8516.11it/s] 81%|████████▏ | 325332/400000 [00:37<00:08, 8458.04it/s] 82%|████████▏ | 326195/400000 [00:37<00:08, 8506.13it/s] 82%|████████▏ | 327058/400000 [00:37<00:08, 8540.62it/s] 82%|████████▏ | 327921/400000 [00:38<00:08, 8564.90it/s] 82%|████████▏ | 328778/400000 [00:38<00:08, 8528.93it/s] 82%|████████▏ | 329635/400000 [00:38<00:08, 8539.96it/s] 83%|████████▎ | 330509/400000 [00:38<00:08, 8598.27it/s] 83%|████████▎ | 331380/400000 [00:38<00:07, 8629.45it/s] 83%|████████▎ | 332250/400000 [00:38<00:07, 8649.99it/s] 83%|████████▎ | 333120/400000 [00:38<00:07, 8662.50it/s] 83%|████████▎ | 333987/400000 [00:38<00:07, 8529.49it/s] 84%|████████▎ | 334861/400000 [00:38<00:07, 8590.10it/s] 84%|████████▍ | 335734/400000 [00:38<00:07, 8629.45it/s] 84%|████████▍ | 336606/400000 [00:39<00:07, 8653.73it/s] 84%|████████▍ | 337480/400000 [00:39<00:07, 8678.89it/s] 85%|████████▍ | 338349/400000 [00:39<00:07, 8603.75it/s] 85%|████████▍ | 339210/400000 [00:39<00:07, 8424.55it/s] 85%|████████▌ | 340062/400000 [00:39<00:07, 8451.76it/s] 85%|████████▌ | 340934/400000 [00:39<00:06, 8528.78it/s] 85%|████████▌ | 341803/400000 [00:39<00:06, 8575.16it/s] 86%|████████▌ | 342672/400000 [00:39<00:06, 8607.48it/s] 86%|████████▌ | 343546/400000 [00:39<00:06, 8646.25it/s] 86%|████████▌ | 344421/400000 [00:39<00:06, 8675.13it/s] 86%|████████▋ | 345297/400000 [00:40<00:06, 8697.74it/s] 87%|████████▋ | 346167/400000 [00:40<00:06, 8695.82it/s] 87%|████████▋ | 347039/400000 [00:40<00:06, 8702.38it/s] 87%|████████▋ | 347910/400000 [00:40<00:05, 8702.44it/s] 87%|████████▋ | 348781/400000 [00:40<00:05, 8699.11it/s] 87%|████████▋ | 349657/400000 [00:40<00:05, 8716.16it/s] 88%|████████▊ | 350529/400000 [00:40<00:05, 8714.01it/s] 88%|████████▊ | 351401/400000 [00:40<00:05, 8695.92it/s] 88%|████████▊ | 352271/400000 [00:40<00:05, 8679.12it/s] 88%|████████▊ | 353139/400000 [00:40<00:05, 8573.39it/s] 89%|████████▊ | 354001/400000 [00:41<00:05, 8585.22it/s] 89%|████████▊ | 354870/400000 [00:41<00:05, 8615.71it/s] 89%|████████▉ | 355738/400000 [00:41<00:05, 8632.63it/s] 89%|████████▉ | 356602/400000 [00:41<00:05, 8620.11it/s] 89%|████████▉ | 357465/400000 [00:41<00:05, 8420.52it/s] 90%|████████▉ | 358324/400000 [00:41<00:04, 8469.41it/s] 90%|████████▉ | 359191/400000 [00:41<00:04, 8525.82it/s] 90%|█████████ | 360055/400000 [00:41<00:04, 8558.38it/s] 90%|█████████ | 360928/400000 [00:41<00:04, 8607.74it/s] 90%|█████████ | 361790/400000 [00:41<00:04, 8544.27it/s] 91%|█████████ | 362657/400000 [00:42<00:04, 8580.34it/s] 91%|█████████ | 363516/400000 [00:42<00:04, 8543.90it/s] 91%|█████████ | 364371/400000 [00:42<00:04, 8350.10it/s] 91%|█████████▏| 365208/400000 [00:42<00:04, 8200.21it/s] 92%|█████████▏| 366030/400000 [00:42<00:04, 8108.25it/s] 92%|█████████▏| 366883/400000 [00:42<00:04, 8230.10it/s] 92%|█████████▏| 367744/400000 [00:42<00:03, 8338.99it/s] 92%|█████████▏| 368588/400000 [00:42<00:03, 8366.32it/s] 92%|█████████▏| 369455/400000 [00:42<00:03, 8454.48it/s] 93%|█████████▎| 370324/400000 [00:42<00:03, 8523.48it/s] 93%|█████████▎| 371186/400000 [00:43<00:03, 8551.33it/s] 93%|█████████▎| 372043/400000 [00:43<00:03, 8554.74it/s] 93%|█████████▎| 372899/400000 [00:43<00:03, 8509.18it/s] 93%|█████████▎| 373768/400000 [00:43<00:03, 8561.21it/s] 94%|█████████▎| 374643/400000 [00:43<00:02, 8616.33it/s] 94%|█████████▍| 375516/400000 [00:43<00:02, 8648.15it/s] 94%|█████████▍| 376392/400000 [00:43<00:02, 8679.96it/s] 94%|█████████▍| 377262/400000 [00:43<00:02, 8685.00it/s] 95%|█████████▍| 378131/400000 [00:43<00:02, 8655.13it/s] 95%|█████████▍| 379005/400000 [00:43<00:02, 8678.97it/s] 95%|█████████▍| 379876/400000 [00:44<00:02, 8686.37it/s] 95%|█████████▌| 380745/400000 [00:44<00:02, 8685.59it/s] 95%|█████████▌| 381615/400000 [00:44<00:02, 8687.09it/s] 96%|█████████▌| 382485/400000 [00:44<00:02, 8689.82it/s] 96%|█████████▌| 383357/400000 [00:44<00:01, 8697.79it/s] 96%|█████████▌| 384229/400000 [00:44<00:01, 8701.87it/s] 96%|█████████▋| 385100/400000 [00:44<00:01, 8701.00it/s] 96%|█████████▋| 385971/400000 [00:44<00:01, 8691.30it/s] 97%|█████████▋| 386842/400000 [00:44<00:01, 8696.49it/s] 97%|█████████▋| 387716/400000 [00:44<00:01, 8706.54it/s] 97%|█████████▋| 388588/400000 [00:45<00:01, 8708.66it/s] 97%|█████████▋| 389459/400000 [00:45<00:01, 8690.93it/s] 98%|█████████▊| 390329/400000 [00:45<00:01, 8683.26it/s] 98%|█████████▊| 391200/400000 [00:45<00:01, 8690.88it/s] 98%|█████████▊| 392070/400000 [00:45<00:00, 8502.89it/s] 98%|█████████▊| 392942/400000 [00:45<00:00, 8565.32it/s] 98%|█████████▊| 393802/400000 [00:45<00:00, 8574.30it/s] 99%|█████████▊| 394669/400000 [00:45<00:00, 8601.17it/s] 99%|█████████▉| 395539/400000 [00:45<00:00, 8629.96it/s] 99%|█████████▉| 396411/400000 [00:46<00:00, 8655.65it/s] 99%|█████████▉| 397280/400000 [00:46<00:00, 8664.28it/s]100%|█████████▉| 398147/400000 [00:46<00:00, 8575.41it/s]100%|█████████▉| 399017/400000 [00:46<00:00, 8610.88it/s]100%|█████████▉| 399879/400000 [00:46<00:00, 8524.73it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8615.55it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fd766b10d30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010844756775830095 	 Accuracy: 55
Train Epoch: 1 	 Loss: 0.011012541609862975 	 Accuracy: 53

  model saves at 53% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15612 out of table with 15565 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15612 out of table with 15565 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-13 21:26:00.963404: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 21:26:00.967377: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-13 21:26:00.967584: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556b1fdbe4a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 21:26:00.967597: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd70c60af98> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.2526 - accuracy: 0.5270
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4366 - accuracy: 0.5150 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5440 - accuracy: 0.5080
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5133 - accuracy: 0.5100
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6298 - accuracy: 0.5024
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6155 - accuracy: 0.5033
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6338 - accuracy: 0.5021
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6494 - accuracy: 0.5011
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6308 - accuracy: 0.5023
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6344 - accuracy: 0.5021
11000/25000 [============>.................] - ETA: 3s - loss: 7.6694 - accuracy: 0.4998
12000/25000 [=============>................] - ETA: 3s - loss: 7.6334 - accuracy: 0.5022
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6301 - accuracy: 0.5024
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6458 - accuracy: 0.5014
15000/25000 [=================>............] - ETA: 2s - loss: 7.6482 - accuracy: 0.5012
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6561 - accuracy: 0.5007
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6486 - accuracy: 0.5012
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6615 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6577 - accuracy: 0.5006
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6643 - accuracy: 0.5002
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6703 - accuracy: 0.4998
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6736 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6720 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6583 - accuracy: 0.5005
25000/25000 [==============================] - 7s 294us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fd6c713c198> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fd6d474e128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 2.0289 - crf_viterbi_accuracy: 0.1067 - val_loss: 1.9515 - val_crf_viterbi_accuracy: 0.1467

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

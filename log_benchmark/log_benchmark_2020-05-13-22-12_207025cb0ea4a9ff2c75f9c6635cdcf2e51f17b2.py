
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f850ddbaf60> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 22:12:42.234001
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 22:12:42.237982
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 22:12:42.241205
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 22:12:42.244569
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f8519b843c8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355044.5625
Epoch 2/10

1/1 [==============================] - 0s 110ms/step - loss: 243563.4375
Epoch 3/10

1/1 [==============================] - 0s 115ms/step - loss: 129335.5312
Epoch 4/10

1/1 [==============================] - 0s 105ms/step - loss: 64699.2617
Epoch 5/10

1/1 [==============================] - 0s 98ms/step - loss: 33814.8945
Epoch 6/10

1/1 [==============================] - 0s 99ms/step - loss: 19366.7363
Epoch 7/10

1/1 [==============================] - 0s 105ms/step - loss: 12177.1230
Epoch 8/10

1/1 [==============================] - 0s 102ms/step - loss: 8235.1250
Epoch 9/10

1/1 [==============================] - 0s 109ms/step - loss: 5914.3833
Epoch 10/10

1/1 [==============================] - 0s 104ms/step - loss: 4480.3076

  #### Inference Need return ypred, ytrue ######################### 
[[  0.5700476    9.876222    11.582583    11.766815    10.409483
    9.165274    10.009336    11.707261     9.610434     9.287843
   11.204359    10.690839    11.208465     7.9819694    9.703917
    9.692098    11.353089     9.17264     11.013124     9.842344
   10.755537     9.449827    11.071393     8.276637    10.211464
   11.071896    11.149316    11.40175     10.226782     8.223754
   10.115726    11.883422     9.705885    11.373449    10.679007
    9.310447    12.072767     9.564819     9.229407     9.959108
    9.193046     9.46194     10.875245    11.07789     10.270862
    8.599404    11.523207    11.411521    10.035272    11.134177
   10.50942      9.660892    10.8821745    9.756344    11.724738
   10.869899     8.438604    11.973658    12.409986    11.627336
   -0.81744254   0.08406249  -0.02975595   1.5135406   -0.06463063
   -0.14588688  -1.5497373   -0.12059698  -1.4225142    1.9499886
   -1.4257529    0.16372198   0.84226084  -0.69025844   0.6012704
   -1.0643823    0.20107773   1.0805879   -0.6479342    0.08341175
   -0.08051726   0.9039597   -0.49543023   0.28396922   0.103552
   -1.7580311   -0.77333826   0.6506902    1.0118954    1.4813213
    0.842484     1.3851237    1.3343831    0.53824836   0.80837154
    0.22693938  -0.8217055    0.9072778   -0.19992031   1.7272424
   -0.66828406  -0.97875524  -0.3735888    1.35674      0.31319356
    1.1185415    0.4981708    0.3929016    0.9540129   -0.73983437
    0.585957     1.0462905    1.6135654    0.7637234   -0.27268016
   -1.4883001    0.29808903   0.4358601   -0.44374937  -0.4822695
   -0.24811621  -0.7757808    0.4135618    1.404851    -0.727402
   -0.04841547   0.6067465   -0.5463182   -0.68415713   0.19437523
    0.95734125  -0.1273337    0.06206504   0.9541523    1.061214
   -1.2140403    1.4424635    0.119368    -2.026997     0.6164639
    1.5780506    1.0638595   -1.7920581    1.2693224   -0.1852327
    0.7486308   -0.07220687   0.54698575  -0.5571078   -0.17573905
    0.7897763    0.42178622  -1.2528896    0.7345389    0.7226038
    0.18323746   0.75590336   0.02095738   0.25304684  -0.3481531
    0.7649284   -0.63184667  -0.29959333   0.18541539  -0.2264719
   -0.16207847   0.02030173   0.6618272    0.6803348   -0.09985389
   -0.55314434   1.0913754    1.7861178    0.5038054   -1.9344052
   -1.5618155    1.1514314   -0.4480115    0.27071598  -1.6639184
    0.42244673  10.65053      9.530668    10.992487    10.128582
   10.859508    10.048441    11.3262205   10.424651    10.922679
   12.315556    10.924985    11.127506    10.284866    10.996571
   10.170817     9.724857     9.888312    10.720015    11.356251
   10.534331    11.397136     9.739374    10.048832    11.435025
    9.984613    10.93395     11.34364     10.471945    10.065679
    9.943042    11.012095     9.6230345    9.763318    10.121659
   10.444012    10.027572    11.050725    10.950721     8.58178
    9.360861    10.380747    10.287001     9.832318    10.519157
   10.615416     9.839685    10.925915    10.2914295   11.69237
    8.888025    10.320772    10.188476    10.648059    10.965112
   10.020976    10.704235    10.601533     8.979131    10.206259
    0.15271741   0.20359623   0.54276454   0.27957714   1.9411557
    1.2409209    0.86625916   1.6505092    0.41753072   0.77855504
    0.17091608   1.864263     0.6729454    2.626967     1.1365107
    2.1326344    1.2364031    0.44223076   2.3968668    1.9686291
    0.34357786   0.35019565   1.7013113    0.2131691    1.3798362
    1.9557043    1.1658783    0.8409672    0.7206371    1.4864843
    1.3830899    0.5275344    1.4952996    0.5897327    0.21904099
    0.6110941    0.6571907    1.8096161    0.1889506    0.8580349
    0.46850812   1.9862995    2.9152203    0.50917727   1.0381203
    1.5147445    0.74763966   1.1409748    0.19755727   0.10772192
    1.6856962    0.51767695   2.10256      0.6539668    0.08496386
    1.3746996    0.2633595    1.1327323    1.9182798    1.5903451
    0.68830955   1.8641704    0.38742387   0.09362596   0.6711589
    1.1300352    0.43916178   1.2543464    1.2937105    2.4447165
    0.23931384   1.5913197    1.6302059    1.5105945    0.29884946
    1.3950852    0.6975661    0.20817542   0.45057034   0.3607831
    1.0535542    0.47648787   0.67005754   0.8218646    0.11483955
    1.5486928    2.5529528    0.10227495   1.311491     0.80705273
    0.18508768   1.5010633    1.5125151    0.33131927   0.40705717
    1.836264     1.183448     2.8992143    0.45393324   2.0291042
    2.0682385    1.6055914    0.39943093   1.4159838    1.5469968
    1.2060351    0.79027784   2.1607995    1.0149243    1.0478659
    0.10042077   2.8846412    0.7741823    3.477189     0.3355736
    1.2739565    2.2491245    0.63094574   0.74751246   0.23806608
    7.501333   -10.593832   -11.506302  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 22:12:51.344001
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.5459
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 22:12:51.348393
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8411.82
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 22:12:51.352250
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    91.601
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 22:12:51.356180
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -752.338
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140209097958848
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140208139174408
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140208139174912
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140208139175416
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140208139175920
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140208139176424

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f8507503860> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.456384
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.425228
grad_step = 000002, loss = 0.407457
grad_step = 000003, loss = 0.391143
grad_step = 000004, loss = 0.374403
grad_step = 000005, loss = 0.359222
grad_step = 000006, loss = 0.347467
grad_step = 000007, loss = 0.341538
grad_step = 000008, loss = 0.335194
grad_step = 000009, loss = 0.323893
grad_step = 000010, loss = 0.313742
grad_step = 000011, loss = 0.306332
grad_step = 000012, loss = 0.300522
grad_step = 000013, loss = 0.294535
grad_step = 000014, loss = 0.287598
grad_step = 000015, loss = 0.279813
grad_step = 000016, loss = 0.271798
grad_step = 000017, loss = 0.264315
grad_step = 000018, loss = 0.257886
grad_step = 000019, loss = 0.252028
grad_step = 000020, loss = 0.245624
grad_step = 000021, loss = 0.238545
grad_step = 000022, loss = 0.231621
grad_step = 000023, loss = 0.225386
grad_step = 000024, loss = 0.219635
grad_step = 000025, loss = 0.213888
grad_step = 000026, loss = 0.207906
grad_step = 000027, loss = 0.201777
grad_step = 000028, loss = 0.195755
grad_step = 000029, loss = 0.190042
grad_step = 000030, loss = 0.184630
grad_step = 000031, loss = 0.179312
grad_step = 000032, loss = 0.173932
grad_step = 000033, loss = 0.168567
grad_step = 000034, loss = 0.163365
grad_step = 000035, loss = 0.158370
grad_step = 000036, loss = 0.153487
grad_step = 000037, loss = 0.148620
grad_step = 000038, loss = 0.143814
grad_step = 000039, loss = 0.139166
grad_step = 000040, loss = 0.134686
grad_step = 000041, loss = 0.130306
grad_step = 000042, loss = 0.125947
grad_step = 000043, loss = 0.121652
grad_step = 000044, loss = 0.117508
grad_step = 000045, loss = 0.113526
grad_step = 000046, loss = 0.109632
grad_step = 000047, loss = 0.105770
grad_step = 000048, loss = 0.101988
grad_step = 000049, loss = 0.098341
grad_step = 000050, loss = 0.094820
grad_step = 000051, loss = 0.091371
grad_step = 000052, loss = 0.087992
grad_step = 000053, loss = 0.084701
grad_step = 000054, loss = 0.081517
grad_step = 000055, loss = 0.078428
grad_step = 000056, loss = 0.075418
grad_step = 000057, loss = 0.072485
grad_step = 000058, loss = 0.069646
grad_step = 000059, loss = 0.066897
grad_step = 000060, loss = 0.064216
grad_step = 000061, loss = 0.061613
grad_step = 000062, loss = 0.059101
grad_step = 000063, loss = 0.056483
grad_step = 000064, loss = 0.053831
grad_step = 000065, loss = 0.051350
grad_step = 000066, loss = 0.049128
grad_step = 000067, loss = 0.047050
grad_step = 000068, loss = 0.044982
grad_step = 000069, loss = 0.042858
grad_step = 000070, loss = 0.040750
grad_step = 000071, loss = 0.038754
grad_step = 000072, loss = 0.036867
grad_step = 000073, loss = 0.035092
grad_step = 000074, loss = 0.033387
grad_step = 000075, loss = 0.031467
grad_step = 000076, loss = 0.029434
grad_step = 000077, loss = 0.027528
grad_step = 000078, loss = 0.025815
grad_step = 000079, loss = 0.024378
grad_step = 000080, loss = 0.023013
grad_step = 000081, loss = 0.021634
grad_step = 000082, loss = 0.020239
grad_step = 000083, loss = 0.018876
grad_step = 000084, loss = 0.017546
grad_step = 000085, loss = 0.016238
grad_step = 000086, loss = 0.015219
grad_step = 000087, loss = 0.014274
grad_step = 000088, loss = 0.013218
grad_step = 000089, loss = 0.012296
grad_step = 000090, loss = 0.011439
grad_step = 000091, loss = 0.010502
grad_step = 000092, loss = 0.009797
grad_step = 000093, loss = 0.009167
grad_step = 000094, loss = 0.008462
grad_step = 000095, loss = 0.007886
grad_step = 000096, loss = 0.007330
grad_step = 000097, loss = 0.006738
grad_step = 000098, loss = 0.006303
grad_step = 000099, loss = 0.005907
grad_step = 000100, loss = 0.005464
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.005124
grad_step = 000102, loss = 0.004804
grad_step = 000103, loss = 0.004450
grad_step = 000104, loss = 0.004186
grad_step = 000105, loss = 0.003962
grad_step = 000106, loss = 0.003703
grad_step = 000107, loss = 0.003490
grad_step = 000108, loss = 0.003325
grad_step = 000109, loss = 0.003135
grad_step = 000110, loss = 0.002956
grad_step = 000111, loss = 0.002826
grad_step = 000112, loss = 0.002713
grad_step = 000113, loss = 0.002589
grad_step = 000114, loss = 0.002475
grad_step = 000115, loss = 0.002391
grad_step = 000116, loss = 0.002317
grad_step = 000117, loss = 0.002235
grad_step = 000118, loss = 0.002162
grad_step = 000119, loss = 0.002108
grad_step = 000120, loss = 0.002067
grad_step = 000121, loss = 0.002027
grad_step = 000122, loss = 0.001983
grad_step = 000123, loss = 0.001941
grad_step = 000124, loss = 0.001908
grad_step = 000125, loss = 0.001884
grad_step = 000126, loss = 0.001868
grad_step = 000127, loss = 0.001854
grad_step = 000128, loss = 0.001840
grad_step = 000129, loss = 0.001826
grad_step = 000130, loss = 0.001815
grad_step = 000131, loss = 0.001802
grad_step = 000132, loss = 0.001791
grad_step = 000133, loss = 0.001777
grad_step = 000134, loss = 0.001768
grad_step = 000135, loss = 0.001766
grad_step = 000136, loss = 0.001782
grad_step = 000137, loss = 0.001808
grad_step = 000138, loss = 0.001808
grad_step = 000139, loss = 0.001786
grad_step = 000140, loss = 0.001830
grad_step = 000141, loss = 0.001860
grad_step = 000142, loss = 0.001832
grad_step = 000143, loss = 0.001720
grad_step = 000144, loss = 0.001719
grad_step = 000145, loss = 0.001775
grad_step = 000146, loss = 0.001729
grad_step = 000147, loss = 0.001680
grad_step = 000148, loss = 0.001705
grad_step = 000149, loss = 0.001723
grad_step = 000150, loss = 0.001673
grad_step = 000151, loss = 0.001645
grad_step = 000152, loss = 0.001674
grad_step = 000153, loss = 0.001679
grad_step = 000154, loss = 0.001638
grad_step = 000155, loss = 0.001616
grad_step = 000156, loss = 0.001641
grad_step = 000157, loss = 0.001637
grad_step = 000158, loss = 0.001606
grad_step = 000159, loss = 0.001595
grad_step = 000160, loss = 0.001616
grad_step = 000161, loss = 0.001627
grad_step = 000162, loss = 0.001632
grad_step = 000163, loss = 0.001688
grad_step = 000164, loss = 0.001844
grad_step = 000165, loss = 0.001883
grad_step = 000166, loss = 0.001863
grad_step = 000167, loss = 0.001669
grad_step = 000168, loss = 0.001556
grad_step = 000169, loss = 0.001641
grad_step = 000170, loss = 0.001725
grad_step = 000171, loss = 0.001629
grad_step = 000172, loss = 0.001526
grad_step = 000173, loss = 0.001609
grad_step = 000174, loss = 0.001620
grad_step = 000175, loss = 0.001588
grad_step = 000176, loss = 0.001544
grad_step = 000177, loss = 0.001509
grad_step = 000178, loss = 0.001587
grad_step = 000179, loss = 0.001540
grad_step = 000180, loss = 0.001499
grad_step = 000181, loss = 0.001518
grad_step = 000182, loss = 0.001504
grad_step = 000183, loss = 0.001532
grad_step = 000184, loss = 0.001507
grad_step = 000185, loss = 0.001465
grad_step = 000186, loss = 0.001493
grad_step = 000187, loss = 0.001495
grad_step = 000188, loss = 0.001498
grad_step = 000189, loss = 0.001491
grad_step = 000190, loss = 0.001452
grad_step = 000191, loss = 0.001454
grad_step = 000192, loss = 0.001458
grad_step = 000193, loss = 0.001463
grad_step = 000194, loss = 0.001465
grad_step = 000195, loss = 0.001451
grad_step = 000196, loss = 0.001436
grad_step = 000197, loss = 0.001425
grad_step = 000198, loss = 0.001428
grad_step = 000199, loss = 0.001425
grad_step = 000200, loss = 0.001434
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001436
grad_step = 000202, loss = 0.001428
grad_step = 000203, loss = 0.001429
grad_step = 000204, loss = 0.001412
grad_step = 000205, loss = 0.001406
grad_step = 000206, loss = 0.001399
grad_step = 000207, loss = 0.001391
grad_step = 000208, loss = 0.001392
grad_step = 000209, loss = 0.001388
grad_step = 000210, loss = 0.001389
grad_step = 000211, loss = 0.001391
grad_step = 000212, loss = 0.001397
grad_step = 000213, loss = 0.001411
grad_step = 000214, loss = 0.001446
grad_step = 000215, loss = 0.001496
grad_step = 000216, loss = 0.001602
grad_step = 000217, loss = 0.001605
grad_step = 000218, loss = 0.001592
grad_step = 000219, loss = 0.001425
grad_step = 000220, loss = 0.001369
grad_step = 000221, loss = 0.001447
grad_step = 000222, loss = 0.001487
grad_step = 000223, loss = 0.001430
grad_step = 000224, loss = 0.001356
grad_step = 000225, loss = 0.001385
grad_step = 000226, loss = 0.001448
grad_step = 000227, loss = 0.001422
grad_step = 000228, loss = 0.001364
grad_step = 000229, loss = 0.001337
grad_step = 000230, loss = 0.001364
grad_step = 000231, loss = 0.001402
grad_step = 000232, loss = 0.001389
grad_step = 000233, loss = 0.001353
grad_step = 000234, loss = 0.001325
grad_step = 000235, loss = 0.001329
grad_step = 000236, loss = 0.001354
grad_step = 000237, loss = 0.001366
grad_step = 000238, loss = 0.001365
grad_step = 000239, loss = 0.001340
grad_step = 000240, loss = 0.001319
grad_step = 000241, loss = 0.001308
grad_step = 000242, loss = 0.001311
grad_step = 000243, loss = 0.001322
grad_step = 000244, loss = 0.001328
grad_step = 000245, loss = 0.001330
grad_step = 000246, loss = 0.001318
grad_step = 000247, loss = 0.001304
grad_step = 000248, loss = 0.001294
grad_step = 000249, loss = 0.001292
grad_step = 000250, loss = 0.001295
grad_step = 000251, loss = 0.001299
grad_step = 000252, loss = 0.001303
grad_step = 000253, loss = 0.001300
grad_step = 000254, loss = 0.001296
grad_step = 000255, loss = 0.001289
grad_step = 000256, loss = 0.001283
grad_step = 000257, loss = 0.001277
grad_step = 000258, loss = 0.001273
grad_step = 000259, loss = 0.001271
grad_step = 000260, loss = 0.001270
grad_step = 000261, loss = 0.001271
grad_step = 000262, loss = 0.001272
grad_step = 000263, loss = 0.001277
grad_step = 000264, loss = 0.001283
grad_step = 000265, loss = 0.001299
grad_step = 000266, loss = 0.001318
grad_step = 000267, loss = 0.001364
grad_step = 000268, loss = 0.001393
grad_step = 000269, loss = 0.001460
grad_step = 000270, loss = 0.001411
grad_step = 000271, loss = 0.001350
grad_step = 000272, loss = 0.001268
grad_step = 000273, loss = 0.001260
grad_step = 000274, loss = 0.001309
grad_step = 000275, loss = 0.001333
grad_step = 000276, loss = 0.001316
grad_step = 000277, loss = 0.001263
grad_step = 000278, loss = 0.001241
grad_step = 000279, loss = 0.001256
grad_step = 000280, loss = 0.001284
grad_step = 000281, loss = 0.001306
grad_step = 000282, loss = 0.001290
grad_step = 000283, loss = 0.001269
grad_step = 000284, loss = 0.001242
grad_step = 000285, loss = 0.001227
grad_step = 000286, loss = 0.001226
grad_step = 000287, loss = 0.001236
grad_step = 000288, loss = 0.001253
grad_step = 000289, loss = 0.001271
grad_step = 000290, loss = 0.001297
grad_step = 000291, loss = 0.001307
grad_step = 000292, loss = 0.001326
grad_step = 000293, loss = 0.001313
grad_step = 000294, loss = 0.001306
grad_step = 000295, loss = 0.001267
grad_step = 000296, loss = 0.001231
grad_step = 000297, loss = 0.001208
grad_step = 000298, loss = 0.001209
grad_step = 000299, loss = 0.001225
grad_step = 000300, loss = 0.001240
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001246
grad_step = 000302, loss = 0.001233
grad_step = 000303, loss = 0.001216
grad_step = 000304, loss = 0.001199
grad_step = 000305, loss = 0.001191
grad_step = 000306, loss = 0.001192
grad_step = 000307, loss = 0.001198
grad_step = 000308, loss = 0.001208
grad_step = 000309, loss = 0.001215
grad_step = 000310, loss = 0.001224
grad_step = 000311, loss = 0.001224
grad_step = 000312, loss = 0.001227
grad_step = 000313, loss = 0.001220
grad_step = 000314, loss = 0.001213
grad_step = 000315, loss = 0.001199
grad_step = 000316, loss = 0.001187
grad_step = 000317, loss = 0.001177
grad_step = 000318, loss = 0.001170
grad_step = 000319, loss = 0.001167
grad_step = 000320, loss = 0.001167
grad_step = 000321, loss = 0.001170
grad_step = 000322, loss = 0.001174
grad_step = 000323, loss = 0.001179
grad_step = 000324, loss = 0.001184
grad_step = 000325, loss = 0.001194
grad_step = 000326, loss = 0.001202
grad_step = 000327, loss = 0.001220
grad_step = 000328, loss = 0.001234
grad_step = 000329, loss = 0.001264
grad_step = 000330, loss = 0.001272
grad_step = 000331, loss = 0.001294
grad_step = 000332, loss = 0.001266
grad_step = 000333, loss = 0.001232
grad_step = 000334, loss = 0.001178
grad_step = 000335, loss = 0.001145
grad_step = 000336, loss = 0.001143
grad_step = 000337, loss = 0.001163
grad_step = 000338, loss = 0.001189
grad_step = 000339, loss = 0.001195
grad_step = 000340, loss = 0.001191
grad_step = 000341, loss = 0.001167
grad_step = 000342, loss = 0.001145
grad_step = 000343, loss = 0.001129
grad_step = 000344, loss = 0.001124
grad_step = 000345, loss = 0.001128
grad_step = 000346, loss = 0.001137
grad_step = 000347, loss = 0.001148
grad_step = 000348, loss = 0.001155
grad_step = 000349, loss = 0.001163
grad_step = 000350, loss = 0.001161
grad_step = 000351, loss = 0.001160
grad_step = 000352, loss = 0.001150
grad_step = 000353, loss = 0.001141
grad_step = 000354, loss = 0.001127
grad_step = 000355, loss = 0.001116
grad_step = 000356, loss = 0.001107
grad_step = 000357, loss = 0.001102
grad_step = 000358, loss = 0.001100
grad_step = 000359, loss = 0.001100
grad_step = 000360, loss = 0.001103
grad_step = 000361, loss = 0.001106
grad_step = 000362, loss = 0.001112
grad_step = 000363, loss = 0.001119
grad_step = 000364, loss = 0.001132
grad_step = 000365, loss = 0.001147
grad_step = 000366, loss = 0.001181
grad_step = 000367, loss = 0.001209
grad_step = 000368, loss = 0.001270
grad_step = 000369, loss = 0.001281
grad_step = 000370, loss = 0.001299
grad_step = 000371, loss = 0.001220
grad_step = 000372, loss = 0.001139
grad_step = 000373, loss = 0.001082
grad_step = 000374, loss = 0.001090
grad_step = 000375, loss = 0.001137
grad_step = 000376, loss = 0.001162
grad_step = 000377, loss = 0.001155
grad_step = 000378, loss = 0.001107
grad_step = 000379, loss = 0.001071
grad_step = 000380, loss = 0.001067
grad_step = 000381, loss = 0.001087
grad_step = 000382, loss = 0.001115
grad_step = 000383, loss = 0.001121
grad_step = 000384, loss = 0.001116
grad_step = 000385, loss = 0.001088
grad_step = 000386, loss = 0.001065
grad_step = 000387, loss = 0.001052
grad_step = 000388, loss = 0.001053
grad_step = 000389, loss = 0.001063
grad_step = 000390, loss = 0.001072
grad_step = 000391, loss = 0.001079
grad_step = 000392, loss = 0.001075
grad_step = 000393, loss = 0.001068
grad_step = 000394, loss = 0.001055
grad_step = 000395, loss = 0.001045
grad_step = 000396, loss = 0.001037
grad_step = 000397, loss = 0.001035
grad_step = 000398, loss = 0.001036
grad_step = 000399, loss = 0.001039
grad_step = 000400, loss = 0.001043
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001045
grad_step = 000402, loss = 0.001048
grad_step = 000403, loss = 0.001047
grad_step = 000404, loss = 0.001049
grad_step = 000405, loss = 0.001048
grad_step = 000406, loss = 0.001051
grad_step = 000407, loss = 0.001052
grad_step = 000408, loss = 0.001057
grad_step = 000409, loss = 0.001059
grad_step = 000410, loss = 0.001067
grad_step = 000411, loss = 0.001068
grad_step = 000412, loss = 0.001074
grad_step = 000413, loss = 0.001069
grad_step = 000414, loss = 0.001066
grad_step = 000415, loss = 0.001052
grad_step = 000416, loss = 0.001040
grad_step = 000417, loss = 0.001024
grad_step = 000418, loss = 0.001011
grad_step = 000419, loss = 0.001002
grad_step = 000420, loss = 0.000997
grad_step = 000421, loss = 0.000995
grad_step = 000422, loss = 0.000997
grad_step = 000423, loss = 0.001000
grad_step = 000424, loss = 0.001006
grad_step = 000425, loss = 0.001016
grad_step = 000426, loss = 0.001031
grad_step = 000427, loss = 0.001064
grad_step = 000428, loss = 0.001103
grad_step = 000429, loss = 0.001184
grad_step = 000430, loss = 0.001229
grad_step = 000431, loss = 0.001306
grad_step = 000432, loss = 0.001219
grad_step = 000433, loss = 0.001112
grad_step = 000434, loss = 0.000996
grad_step = 000435, loss = 0.000989
grad_step = 000436, loss = 0.001061
grad_step = 000437, loss = 0.001090
grad_step = 000438, loss = 0.001059
grad_step = 000439, loss = 0.000988
grad_step = 000440, loss = 0.000975
grad_step = 000441, loss = 0.001016
grad_step = 000442, loss = 0.001039
grad_step = 000443, loss = 0.001028
grad_step = 000444, loss = 0.000981
grad_step = 000445, loss = 0.000959
grad_step = 000446, loss = 0.000971
grad_step = 000447, loss = 0.000996
grad_step = 000448, loss = 0.001017
grad_step = 000449, loss = 0.001007
grad_step = 000450, loss = 0.000989
grad_step = 000451, loss = 0.000964
grad_step = 000452, loss = 0.000949
grad_step = 000453, loss = 0.000948
grad_step = 000454, loss = 0.000956
grad_step = 000455, loss = 0.000968
grad_step = 000456, loss = 0.000973
grad_step = 000457, loss = 0.000973
grad_step = 000458, loss = 0.000962
grad_step = 000459, loss = 0.000950
grad_step = 000460, loss = 0.000939
grad_step = 000461, loss = 0.000934
grad_step = 000462, loss = 0.000933
grad_step = 000463, loss = 0.000937
grad_step = 000464, loss = 0.000941
grad_step = 000465, loss = 0.000944
grad_step = 000466, loss = 0.000946
grad_step = 000467, loss = 0.000945
grad_step = 000468, loss = 0.000944
grad_step = 000469, loss = 0.000941
grad_step = 000470, loss = 0.000939
grad_step = 000471, loss = 0.000935
grad_step = 000472, loss = 0.000932
grad_step = 000473, loss = 0.000929
grad_step = 000474, loss = 0.000926
grad_step = 000475, loss = 0.000923
grad_step = 000476, loss = 0.000920
grad_step = 000477, loss = 0.000918
grad_step = 000478, loss = 0.000918
grad_step = 000479, loss = 0.000917
grad_step = 000480, loss = 0.000918
grad_step = 000481, loss = 0.000921
grad_step = 000482, loss = 0.000929
grad_step = 000483, loss = 0.000939
grad_step = 000484, loss = 0.000962
grad_step = 000485, loss = 0.000988
grad_step = 000486, loss = 0.001043
grad_step = 000487, loss = 0.001084
grad_step = 000488, loss = 0.001156
grad_step = 000489, loss = 0.001142
grad_step = 000490, loss = 0.001107
grad_step = 000491, loss = 0.000992
grad_step = 000492, loss = 0.000907
grad_step = 000493, loss = 0.000895
grad_step = 000494, loss = 0.000945
grad_step = 000495, loss = 0.001001
grad_step = 000496, loss = 0.000997
grad_step = 000497, loss = 0.000959
grad_step = 000498, loss = 0.000902
grad_step = 000499, loss = 0.000881
grad_step = 000500, loss = 0.000900
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000930
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

  date_run                              2020-05-13 22:13:14.035101
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.250923
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 22:13:14.042419
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.175573
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 22:13:14.050274
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.12599
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 22:13:14.055289
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.66789
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
0   2020-05-13 22:12:42.234001  ...    mean_absolute_error
1   2020-05-13 22:12:42.237982  ...     mean_squared_error
2   2020-05-13 22:12:42.241205  ...  median_absolute_error
3   2020-05-13 22:12:42.244569  ...               r2_score
4   2020-05-13 22:12:51.344001  ...    mean_absolute_error
5   2020-05-13 22:12:51.348393  ...     mean_squared_error
6   2020-05-13 22:12:51.352250  ...  median_absolute_error
7   2020-05-13 22:12:51.356180  ...               r2_score
8   2020-05-13 22:13:14.035101  ...    mean_absolute_error
9   2020-05-13 22:13:14.042419  ...     mean_squared_error
10  2020-05-13 22:13:14.050274  ...  median_absolute_error
11  2020-05-13 22:13:14.055289  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b8f2b6cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 314407.77it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 406322.07it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 562228.63it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 794437.28it/s] 77%|███████▋  | 7675904/9912422 [00:00<00:01, 1123722.47it/s]9920512it [00:00, 10708136.11it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 147763.80it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 313975.38it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 406177.45it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 561885.18it/s]1654784it [00:00, 2823793.56it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 51445.67it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b41c6fe80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b412a00b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b41c6fe80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b411f60f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b3ea314e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b3ea1c748> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b41c6fe80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b411b4710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b3ea314e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8b8f278f28> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f5a64ed8240> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=8413beeb021aecfe118b84a7409e56f1d6cb0dfde8bd4b43ebe687d2d8717481
  Stored in directory: /tmp/pip-ephem-wheel-cache-_5d69uer/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f5a5b25e080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 2:17
   40960/17464789 [..............................] - ETA: 54s 
   90112/17464789 [..............................] - ETA: 37s
  163840/17464789 [..............................] - ETA: 27s
  278528/17464789 [..............................] - ETA: 19s
  557056/17464789 [..............................] - ETA: 11s
 1097728/17464789 [>.............................] - ETA: 6s 
 2170880/17464789 [==>...........................] - ETA: 3s
 4300800/17464789 [======>.......................] - ETA: 1s
 7315456/17464789 [===========>..................] - ETA: 0s
10313728/17464789 [================>.............] - ETA: 0s
13213696/17464789 [=====================>........] - ETA: 0s
16277504/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 22:14:43.409165: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 22:14:43.412858: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-13 22:14:43.413545: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562d8ad82000 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 22:14:43.413560: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8660 - accuracy: 0.4870
 2000/25000 [=>............................] - ETA: 10s - loss: 8.0576 - accuracy: 0.4745
 3000/25000 [==>...........................] - ETA: 8s - loss: 8.0857 - accuracy: 0.4727 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.9388 - accuracy: 0.4823
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.8568 - accuracy: 0.4876
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8481 - accuracy: 0.4882
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.8353 - accuracy: 0.4890
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.8238 - accuracy: 0.4897
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7740 - accuracy: 0.4930
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7602 - accuracy: 0.4939
11000/25000 [============>.................] - ETA: 4s - loss: 7.7126 - accuracy: 0.4970
12000/25000 [=============>................] - ETA: 4s - loss: 7.7113 - accuracy: 0.4971
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7079 - accuracy: 0.4973
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6896 - accuracy: 0.4985
15000/25000 [=================>............] - ETA: 3s - loss: 7.6922 - accuracy: 0.4983
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6676 - accuracy: 0.4999
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6829 - accuracy: 0.4989
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6794 - accuracy: 0.4992
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6707 - accuracy: 0.4997
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6728 - accuracy: 0.4996
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6871 - accuracy: 0.4987
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6903 - accuracy: 0.4985
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6766 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6711 - accuracy: 0.4997
25000/25000 [==============================] - 10s 385us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 22:15:00.407091
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 22:15:00.407091  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<9:38:25, 24.8kB/s].vector_cache/glove.6B.zip:   0%|          | 385k/862M [00:00<6:45:51, 35.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 4.01M/862M [00:00<4:43:01, 50.5kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.8M/862M [00:00<3:16:22, 72.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.8M/862M [00:00<2:16:33, 103kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.7M/862M [00:00<1:35:07, 147kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.3M/862M [00:00<1:06:24, 210kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.5M/862M [00:01<46:21, 299kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 34.4M/862M [00:01<32:24, 426kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.8M/862M [00:01<22:39, 606kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.6M/862M [00:01<15:53, 859kB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.5M/862M [00:01<11:08, 1.22MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.3M/862M [00:01<07:55, 1.71MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:01<06:16, 2.15MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<04:45, 2.83MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<10:51:27, 20.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:03<7:36:16, 29.4kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<5:20:10, 41.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:05<3:45:32, 59.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.8M/862M [00:05<2:37:25, 84.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:07<1:57:53, 113kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:07<1:24:28, 158kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:07<59:03, 224kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:09<47:43, 277kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<35:16, 375kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:09<24:43, 533kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:11<24:38, 535kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:11<19:09, 687kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:11<13:30, 972kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:13<15:20, 854kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:13<12:37, 1.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.8M/862M [00:13<08:58, 1.46MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:15<11:20, 1.15MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:15<09:56, 1.31MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:15<07:08, 1.82MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:17<08:31, 1.52MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:17<07:52, 1.64MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:17<05:39, 2.28MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:19<08:21, 1.54MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:19<07:40, 1.68MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:19<05:30, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:21<08:19, 1.54MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:21<07:39, 1.67MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:21<05:29, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:23<09:17, 1.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:23<08:23, 1.52MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:23<06:05, 2.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<07:17, 1.74MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<06:11, 2.05MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:25<04:29, 2.81MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<07:37, 1.66MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<06:55, 1.82MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:27<05:01, 2.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<06:57, 1.80MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<06:38, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:29<04:50, 2.58MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:29<04:42, 2.65MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<7:00:27, 29.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<4:53:34, 42.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:30<3:28:36, 59.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:31<5:34:27, 37.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:31<3:54:28, 53.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<9:09:42, 22.6kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:32<6:23:37, 32.3kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<4:32:18, 45.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<3:12:10, 64.3kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:34<2:14:06, 91.8kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<1:40:56, 122kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:36<1:10:58, 173kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<51:32, 237kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<37:52, 323kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:38<26:31, 459kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<25:22, 479kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<19:34, 621kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:40<13:47, 878kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<14:22, 841kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<11:45, 1.03MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:42<08:19, 1.45MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<12:53, 933kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<10:45, 1.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:44<07:37, 1.57MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<11:22, 1.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<09:31, 1.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:46<06:44, 1.76MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<13:39, 870kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<11:17, 1.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:48<07:59, 1.48MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<14:06, 837kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<10:44, 1.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<07:37, 1.54MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<11:23, 1.03MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<09:03, 1.30MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:52<06:26, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:54<10:32, 1.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<08:21, 1.40MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:54<05:55, 1.96MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<12:59, 894kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<10:25, 1.11MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:56<07:22, 1.57MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<13:56, 828kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<10:44, 1.07MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [00:58<07:34, 1.51MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<23:22, 491kB/s] .vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<17:24, 658kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:00<12:14, 933kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<19:05, 597kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<14:17, 798kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:02<10:03, 1.13MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<17:49, 636kB/s] .vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<13:46, 823kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:04<09:42, 1.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<13:55, 808kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<11:04, 1.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:06<07:50, 1.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<11:53, 941kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<09:16, 1.21MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:08<06:34, 1.69MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<11:47, 943kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<09:50, 1.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:10<06:57, 1.59MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<11:15, 982kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<08:50, 1.25MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<07:47, 1.41MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<06:31, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<04:40, 2.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<08:46, 1.24MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<07:23, 1.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:16<05:15, 2.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<09:53, 1.10MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<07:48, 1.39MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<07:03, 1.53MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<05:45, 1.87MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:20<04:06, 2.61MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<12:38, 847kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<09:34, 1.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:22<06:46, 1.57MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<16:24, 648kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<12:27, 854kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:24<08:46, 1.21MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<16:51, 628kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<12:57, 815kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:26<09:07, 1.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<15:35, 674kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<12:06, 867kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<08:32, 1.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<12:55, 807kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<10:14, 1.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<07:14, 1.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:30<06:15, 1.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<9:20:22, 18.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<6:32:54, 26.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<4:34:54, 37.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<3:13:39, 53.2kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<2:14:58, 75.9kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<1:41:24, 101kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<1:11:44, 143kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:36<50:04, 203kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<42:29, 239kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<30:29, 333kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:38<21:21, 473kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<20:53, 483kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<15:28, 652kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<12:15, 818kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<09:37, 1.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<06:50, 1.46MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<08:42, 1.14MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<06:54, 1.44MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<06:17, 1.57MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<05:48, 1.70MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<04:08, 2.37MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<08:14, 1.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<06:35, 1.49MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<04:41, 2.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<09:01, 1.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<07:30, 1.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:50<05:18, 1.83MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<13:12, 733kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<09:55, 976kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:51<06:59, 1.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<16:11, 594kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<12:25, 773kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:53<08:44, 1.09MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<13:28, 709kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<10:30, 907kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:55<07:25, 1.28MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<10:42, 884kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<08:52, 1.07MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<06:16, 1.50MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<09:51, 955kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<07:40, 1.23MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<06:44, 1.39MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<05:35, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:01<03:58, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<13:29, 687kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<10:07, 915kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:03<07:07, 1.29MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<59:06, 156kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<42:05, 218kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<30:38, 298kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<22:11, 411kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:07<15:31, 584kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<21:09, 428kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<15:32, 582kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<12:09, 740kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<09:10, 979kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<06:28, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<13:01, 685kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<09:48, 909kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:13<06:54, 1.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<11:50, 747kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<09:00, 981kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<07:35, 1.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:55, 1.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:17<04:11, 2.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<21:11, 411kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<16:04, 542kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<11:16, 768kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<12:48, 675kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<09:37, 898kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:21<06:46, 1.27MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<11:06, 773kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<08:32, 1.00MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<06:01, 1.41MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<08:48, 967kB/s] .vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<07:06, 1.20MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<05:01, 1.68MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<09:50, 857kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<07:59, 1.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<05:39, 1.48MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<07:22, 1.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<06:25, 1.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<04:32, 1.83MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<08:46, 947kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<07:18, 1.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<05:10, 1.60MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<08:49, 933kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<07:15, 1.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<05:07, 1.60MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<09:19, 876kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<07:45, 1.05MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:35<05:28, 1.48MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<07:43, 1.05MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<06:13, 1.30MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<04:23, 1.83MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<21:37, 371kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<16:11, 496kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<11:26, 699kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<08:06, 982kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<14:02, 567kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<10:57, 726kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<07:47, 1.02MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:41<05:33, 1.42MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<13:42, 576kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<10:42, 736kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<07:37, 1.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:43<05:26, 1.44MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<15:01, 521kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<11:17, 692kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<08:00, 972kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<05:42, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<26:29, 293kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<19:33, 396kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<13:46, 561kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<09:42, 792kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<58:52, 131kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<41:47, 184kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<29:18, 261kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<20:31, 371kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<1:47:04, 71.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<1:15:52, 100kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<53:03, 143kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<37:04, 204kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<51:30, 147kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 410M/862M [02:53<37:06, 203kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<26:01, 289kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<19:54, 376kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<14:41, 509kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<10:21, 718kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:55<07:19, 1.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<1:58:20, 62.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<1:23:21, 88.8kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<58:11, 127kB/s]   .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<43:00, 171kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<30:45, 238kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<21:34, 339kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<16:59, 428kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<12:55, 562kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<09:08, 792kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<08:08, 885kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<06:16, 1.15MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:29, 1.60MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<05:25, 1.31MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<04:49, 1.48MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:29, 2.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<02:32, 2.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<2:24:36, 48.9kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<1:42:07, 69.2kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<1:11:19, 98.6kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<49:50, 141kB/s]   .vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<42:36, 164kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<30:51, 227kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<21:37, 322kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<16:46, 413kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<12:45, 543kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<09:00, 765kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<06:22, 1.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<3:33:39, 32.1kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<2:30:28, 45.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:13<1:45:00, 65.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<1:13:12, 92.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<7:46:00, 14.6kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<5:26:42, 20.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<3:47:51, 29.7kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<2:40:19, 41.9kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<1:52:42, 59.6kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<1:18:40, 85.0kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<56:33, 118kB/s]   .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<40:18, 165kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<28:11, 235kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<21:23, 308kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<15:30, 424kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:20<10:53, 601kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<09:49, 663kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<07:19, 888kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<05:13, 1.24MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<05:19, 1.21MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:37, 1.39MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<03:21, 1.92MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:27, 2.61MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<06:27, 988kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<05:25, 1.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<03:55, 1.62MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:26<02:51, 2.21MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<05:15, 1.20MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:17, 1.47MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<03:06, 2.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:28<02:16, 2.74MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<09:46, 639kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<07:25, 839kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<05:17, 1.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<03:48, 1.62MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<07:37, 810kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<05:57, 1.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<04:16, 1.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<03:05, 1.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<06:45, 903kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<05:33, 1.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<03:59, 1.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<02:53, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<06:43, 896kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<05:26, 1.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<03:54, 1.54MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<02:49, 2.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<08:02, 743kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<06:30, 917kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<04:38, 1.28MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<03:20, 1.77MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<06:28, 910kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<05:22, 1.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:52, 1.52MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<02:47, 2.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<06:44, 864kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<05:34, 1.04MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<04:00, 1.45MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<02:53, 2.00MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<06:12, 927kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<05:10, 1.11MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:43, 1.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<02:40, 2.13MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<07:07, 798kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<05:48, 980kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<04:09, 1.36MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<02:59, 1.88MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<06:25, 875kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<05:18, 1.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<03:48, 1.47MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:48<02:44, 2.02MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<07:07, 780kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<05:25, 1.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<03:53, 1.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<02:47, 1.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<09:04, 604kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<06:45, 811kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<04:49, 1.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<03:26, 1.58MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<09:26, 573kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<07:02, 769kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<05:00, 1.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<03:34, 1.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<10:27, 512kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<07:54, 676kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<05:37, 946kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<03:59, 1.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<08:26, 625kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<06:20, 831kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<04:29, 1.17MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:00<04:43, 1.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:01, 1.29MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:53, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<02:05, 2.45MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<08:55, 576kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<06:37, 775kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<04:42, 1.08MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<03:21, 1.51MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<13:23, 379kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<09:50, 515kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<06:56, 726kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<04:56, 1.01MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<06:42, 746kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<05:16, 947kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:47, 1.31MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<02:44, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<04:07, 1.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:35, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:36, 1.88MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:54, 2.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<03:47, 1.29MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<03:02, 1.59MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:12, 2.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<01:37, 2.95MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<06:17, 762kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<04:59, 961kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<03:33, 1.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<02:34, 1.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<05:02, 938kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<04:04, 1.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:56, 1.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<02:07, 2.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<04:52, 956kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:56, 1.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:50, 1.63MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<02:03, 2.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<04:58, 922kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<04:00, 1.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:52, 1.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<02:04, 2.19MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<07:17, 620kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<05:44, 786kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<04:05, 1.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<02:54, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<07:39, 582kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<05:44, 774kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<04:04, 1.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<02:54, 1.51MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<11:15, 389kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<08:12, 534kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<05:48, 750kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<04:06, 1.05MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<07:02, 612kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<05:32, 778kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<03:56, 1.09MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<02:48, 1.52MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<06:25, 661kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<05:06, 831kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:37, 1.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<02:35, 1.62MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<06:21, 657kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<05:04, 823kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:36, 1.15MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<02:34, 1.60MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<07:29, 549kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<05:35, 734kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<03:58, 1.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<02:49, 1.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<07:56, 509kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<06:01, 669kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<04:16, 938kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<03:01, 1.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<07:44, 513kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<05:57, 666kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<04:12, 936kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<02:58, 1.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<08:11, 476kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<06:00, 648kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<04:15, 909kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<03:00, 1.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<13:14, 290kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<09:30, 402kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<06:41, 568kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<04:41, 802kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<12:52, 293kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<09:22, 401kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<06:35, 566kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<04:37, 800kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<07:58, 463kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<06:02, 610kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<04:16, 858kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<03:00, 1.20MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<08:59, 403kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<06:34, 549kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<04:36, 776kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<04:20, 820kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:24, 1.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<02:26, 1.45MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:44, 2.00MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<08:18, 420kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<06:06, 570kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<04:17, 803kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<03:02, 1.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<22:58, 149kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<16:33, 206kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<11:33, 293kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<08:04, 416kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<10:23, 322kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<07:36, 440kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<05:20, 621kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<03:44, 875kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<12:19, 266kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<08:56, 366kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<06:15, 519kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<04:23, 733kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<15:18, 210kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<11:10, 287kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<07:48, 407kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:57<05:26, 577kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<15:46, 199kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<11:15, 279kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<07:51, 395kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<05:29, 560kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<1:37:35, 31.5kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<1:08:41, 44.7kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<47:41, 63.8kB/s]  .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<33:02, 91.0kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<33:52, 88.7kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<24:06, 124kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<16:46, 177kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<11:38, 252kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<20:04, 146kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<14:17, 205kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<09:57, 292kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<07:37, 376kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<05:36, 510kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<03:56, 718kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:07<02:46, 1.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<20:54, 134kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<14:49, 188kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<10:19, 268kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<07:47, 350kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<05:41, 479kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<03:59, 675kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<03:24, 782kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:34, 1.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:49, 1.44MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:56, 1.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:33, 1.65MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:07, 2.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:28, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:23, 1.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:00, 2.47MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:17, 1.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:08, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<00:49, 2.93MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:12, 1.97MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:10, 2.02MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:51, 2.76MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:13, 1.90MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:10, 1.98MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:50, 2.71MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:12, 1.86MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:08, 1.97MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:48, 2.71MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:12, 1.79MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:05, 1.99MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:47, 2.73MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:08, 1.85MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:00, 2.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:43, 2.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:07, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:04, 1.89MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:47, 2.54MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:35, 3.36MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:12, 1.64MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:07, 1.75MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:49, 2.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:36, 3.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:13, 1.56MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:07, 1.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:49, 2.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:36, 3.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:09, 1.57MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:00, 1.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:44, 2.45MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:32, 3.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<01:14, 1.41MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:01, 1.72MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:45, 2.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:32, 3.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<00:24, 4.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:20, 1.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:10, 1.41MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:51, 1.92MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:36, 2.62MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:19, 1.20MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:08, 1.40MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:49, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:35, 2.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:13, 1.24MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:04, 1.41MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:46, 1.93MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:33, 2.62MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:09, 1.26MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:01, 1.43MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:44, 1.96MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:31, 2.64MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:09, 1.20MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:00, 1.37MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:43, 1.88MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:31, 2.56MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:06, 1.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:52, 1.50MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:37, 2.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<00:27, 2.78MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:11, 1.05MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:01, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<00:43, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:31, 2.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:03, 1.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:51, 1.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:36, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<00:26, 2.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:55<00:19, 3.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:20, 811kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:02, 1.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:44, 1.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:31, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:12, 849kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:57, 1.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:40, 1.47MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:28, 2.03MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<01:12, 789kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:55, 1.03MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:38, 1.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:26, 1.98MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<04:24, 200kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<03:10, 277kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<02:10, 393kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<01:28, 556kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<01:53, 429kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:23, 583kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:57, 818kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:39, 1.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<01:27, 513kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<01:04, 691kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:43, 971kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:40, 1.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:31, 1.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:22, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:15, 2.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:57, 629kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:43, 837kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:28, 1.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:29, 1.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:24, 1.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:16, 1.81MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:11, 2.49MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:52, 543kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:38, 727kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:25, 1.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:17, 1.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:50, 477kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:36, 645kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:24, 906kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:15, 1.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<01:18, 254kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:55, 351kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:35, 497kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:26, 591kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:19, 785kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:12, 1.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:10, 1.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:08, 1.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:05, 1.86MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:05, 1.51MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 1.84MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.50MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 3.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:16, 206kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:11, 281kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:03, 399kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 747/400000 [00:00<00:53, 7468.46it/s]  0%|          | 1525/400000 [00:00<00:52, 7557.14it/s]  1%|          | 2273/400000 [00:00<00:52, 7531.68it/s]  1%|          | 3024/400000 [00:00<00:52, 7520.48it/s]  1%|          | 3796/400000 [00:00<00:52, 7577.95it/s]  1%|          | 4538/400000 [00:00<00:52, 7528.86it/s]  1%|▏         | 5295/400000 [00:00<00:52, 7539.03it/s]  2%|▏         | 6026/400000 [00:00<00:52, 7468.28it/s]  2%|▏         | 6769/400000 [00:00<00:52, 7455.68it/s]  2%|▏         | 7523/400000 [00:01<00:52, 7478.92it/s]  2%|▏         | 8284/400000 [00:01<00:52, 7516.61it/s]  2%|▏         | 9058/400000 [00:01<00:51, 7581.87it/s]  2%|▏         | 9836/400000 [00:01<00:51, 7639.39it/s]  3%|▎         | 10636/400000 [00:01<00:50, 7743.35it/s]  3%|▎         | 11452/400000 [00:01<00:49, 7861.72it/s]  3%|▎         | 12236/400000 [00:01<00:50, 7728.83it/s]  3%|▎         | 13053/400000 [00:01<00:49, 7855.24it/s]  3%|▎         | 13839/400000 [00:01<00:49, 7771.44it/s]  4%|▎         | 14617/400000 [00:01<00:50, 7676.32it/s]  4%|▍         | 15385/400000 [00:02<00:50, 7591.17it/s]  4%|▍         | 16185/400000 [00:02<00:49, 7708.53it/s]  4%|▍         | 16990/400000 [00:02<00:49, 7805.34it/s]  4%|▍         | 17773/400000 [00:02<00:48, 7810.65it/s]  5%|▍         | 18555/400000 [00:02<00:49, 7761.33it/s]  5%|▍         | 19332/400000 [00:02<00:49, 7704.08it/s]  5%|▌         | 20103/400000 [00:02<00:50, 7597.07it/s]  5%|▌         | 20864/400000 [00:02<00:50, 7579.13it/s]  5%|▌         | 21639/400000 [00:02<00:49, 7627.00it/s]  6%|▌         | 22410/400000 [00:02<00:49, 7650.29it/s]  6%|▌         | 23176/400000 [00:03<00:49, 7550.10it/s]  6%|▌         | 23934/400000 [00:03<00:49, 7556.94it/s]  6%|▌         | 24691/400000 [00:03<00:50, 7493.34it/s]  6%|▋         | 25441/400000 [00:03<00:50, 7490.84it/s]  7%|▋         | 26223/400000 [00:03<00:49, 7583.73it/s]  7%|▋         | 26982/400000 [00:03<00:49, 7510.81it/s]  7%|▋         | 27734/400000 [00:03<00:50, 7398.87it/s]  7%|▋         | 28477/400000 [00:03<00:50, 7405.32it/s]  7%|▋         | 29239/400000 [00:03<00:49, 7466.94it/s]  7%|▋         | 29987/400000 [00:03<00:49, 7448.90it/s]  8%|▊         | 30735/400000 [00:04<00:49, 7456.20it/s]  8%|▊         | 31484/400000 [00:04<00:49, 7466.09it/s]  8%|▊         | 32232/400000 [00:04<00:49, 7468.66it/s]  8%|▊         | 32979/400000 [00:04<00:50, 7266.16it/s]  8%|▊         | 33750/400000 [00:04<00:49, 7392.29it/s]  9%|▊         | 34505/400000 [00:04<00:49, 7435.94it/s]  9%|▉         | 35273/400000 [00:04<00:48, 7506.65it/s]  9%|▉         | 36025/400000 [00:04<00:48, 7495.09it/s]  9%|▉         | 36804/400000 [00:04<00:47, 7581.19it/s]  9%|▉         | 37563/400000 [00:04<00:47, 7581.24it/s] 10%|▉         | 38338/400000 [00:05<00:47, 7630.95it/s] 10%|▉         | 39102/400000 [00:05<00:47, 7629.15it/s] 10%|▉         | 39901/400000 [00:05<00:46, 7733.33it/s] 10%|█         | 40675/400000 [00:05<00:46, 7676.96it/s] 10%|█         | 41444/400000 [00:05<00:46, 7670.23it/s] 11%|█         | 42212/400000 [00:05<00:46, 7620.02it/s] 11%|█         | 42975/400000 [00:05<00:47, 7596.06it/s] 11%|█         | 43768/400000 [00:05<00:46, 7692.32it/s] 11%|█         | 44560/400000 [00:05<00:45, 7757.14it/s] 11%|█▏        | 45337/400000 [00:05<00:45, 7742.50it/s] 12%|█▏        | 46117/400000 [00:06<00:45, 7757.80it/s] 12%|█▏        | 46897/400000 [00:06<00:45, 7768.17it/s] 12%|█▏        | 47685/400000 [00:06<00:45, 7797.36it/s] 12%|█▏        | 48465/400000 [00:06<00:45, 7797.87it/s] 12%|█▏        | 49245/400000 [00:06<00:45, 7794.26it/s] 13%|█▎        | 50025/400000 [00:06<00:45, 7754.52it/s] 13%|█▎        | 50801/400000 [00:06<00:45, 7658.17it/s] 13%|█▎        | 51568/400000 [00:06<00:45, 7650.42it/s] 13%|█▎        | 52334/400000 [00:06<00:45, 7609.28it/s] 13%|█▎        | 53122/400000 [00:06<00:45, 7688.24it/s] 13%|█▎        | 53892/400000 [00:07<00:45, 7609.71it/s] 14%|█▎        | 54654/400000 [00:07<00:45, 7528.48it/s] 14%|█▍        | 55408/400000 [00:07<00:45, 7525.48it/s] 14%|█▍        | 56174/400000 [00:07<00:45, 7563.58it/s] 14%|█▍        | 56961/400000 [00:07<00:44, 7652.96it/s] 14%|█▍        | 57727/400000 [00:07<00:45, 7490.20it/s] 15%|█▍        | 58486/400000 [00:07<00:45, 7514.40it/s] 15%|█▍        | 59239/400000 [00:07<00:45, 7509.77it/s] 15%|█▌        | 60010/400000 [00:07<00:44, 7566.04it/s] 15%|█▌        | 60768/400000 [00:07<00:44, 7557.76it/s] 15%|█▌        | 61553/400000 [00:08<00:44, 7640.45it/s] 16%|█▌        | 62352/400000 [00:08<00:43, 7742.01it/s] 16%|█▌        | 63127/400000 [00:08<00:43, 7731.49it/s] 16%|█▌        | 63914/400000 [00:08<00:43, 7769.88it/s] 16%|█▌        | 64692/400000 [00:08<00:43, 7739.63it/s] 16%|█▋        | 65467/400000 [00:08<00:43, 7727.25it/s] 17%|█▋        | 66240/400000 [00:08<00:43, 7703.11it/s] 17%|█▋        | 67019/400000 [00:08<00:43, 7727.84it/s] 17%|█▋        | 67809/400000 [00:08<00:42, 7778.50it/s] 17%|█▋        | 68615/400000 [00:08<00:42, 7859.59it/s] 17%|█▋        | 69406/400000 [00:09<00:41, 7872.73it/s] 18%|█▊        | 70194/400000 [00:09<00:42, 7803.86it/s] 18%|█▊        | 70982/400000 [00:09<00:42, 7824.92it/s] 18%|█▊        | 71771/400000 [00:09<00:41, 7841.83it/s] 18%|█▊        | 72556/400000 [00:09<00:41, 7815.07it/s] 18%|█▊        | 73338/400000 [00:09<00:42, 7647.97it/s] 19%|█▊        | 74104/400000 [00:09<00:42, 7645.75it/s] 19%|█▊        | 74870/400000 [00:09<00:42, 7624.53it/s] 19%|█▉        | 75633/400000 [00:09<00:42, 7603.06it/s] 19%|█▉        | 76394/400000 [00:10<00:42, 7530.96it/s] 19%|█▉        | 77148/400000 [00:10<00:42, 7533.64it/s] 19%|█▉        | 77918/400000 [00:10<00:42, 7582.65it/s] 20%|█▉        | 78677/400000 [00:10<00:43, 7390.43it/s] 20%|█▉        | 79418/400000 [00:10<00:43, 7391.16it/s] 20%|██        | 80196/400000 [00:10<00:42, 7503.53it/s] 20%|██        | 80948/400000 [00:10<00:42, 7498.98it/s] 20%|██        | 81699/400000 [00:10<00:42, 7464.53it/s] 21%|██        | 82446/400000 [00:10<00:43, 7300.22it/s] 21%|██        | 83218/400000 [00:10<00:42, 7420.20it/s] 21%|██        | 83966/400000 [00:11<00:42, 7437.72it/s] 21%|██        | 84743/400000 [00:11<00:41, 7532.01it/s] 21%|██▏       | 85498/400000 [00:11<00:41, 7493.46it/s] 22%|██▏       | 86285/400000 [00:11<00:41, 7601.26it/s] 22%|██▏       | 87049/400000 [00:11<00:41, 7611.65it/s] 22%|██▏       | 87817/400000 [00:11<00:40, 7631.46it/s] 22%|██▏       | 88581/400000 [00:11<00:41, 7582.66it/s] 22%|██▏       | 89340/400000 [00:11<00:41, 7549.32it/s] 23%|██▎       | 90139/400000 [00:11<00:40, 7675.15it/s] 23%|██▎       | 90914/400000 [00:11<00:40, 7694.42it/s] 23%|██▎       | 91684/400000 [00:12<00:40, 7693.07it/s] 23%|██▎       | 92454/400000 [00:12<00:40, 7680.03it/s] 23%|██▎       | 93229/400000 [00:12<00:39, 7699.84it/s] 24%|██▎       | 94006/400000 [00:12<00:39, 7718.22it/s] 24%|██▎       | 94812/400000 [00:12<00:39, 7817.56it/s] 24%|██▍       | 95595/400000 [00:12<00:39, 7730.15it/s] 24%|██▍       | 96369/400000 [00:12<00:39, 7674.63it/s] 24%|██▍       | 97150/400000 [00:12<00:39, 7714.43it/s] 24%|██▍       | 97922/400000 [00:12<00:39, 7653.63it/s] 25%|██▍       | 98722/400000 [00:12<00:38, 7753.21it/s] 25%|██▍       | 99498/400000 [00:13<00:39, 7550.26it/s] 25%|██▌       | 100255/400000 [00:13<00:41, 7137.69it/s] 25%|██▌       | 100975/400000 [00:13<00:42, 7026.45it/s] 25%|██▌       | 101721/400000 [00:13<00:41, 7150.69it/s] 26%|██▌       | 102477/400000 [00:13<00:40, 7268.07it/s] 26%|██▌       | 103221/400000 [00:13<00:40, 7316.19it/s] 26%|██▌       | 103955/400000 [00:13<00:40, 7320.00it/s] 26%|██▌       | 104720/400000 [00:13<00:39, 7414.13it/s] 26%|██▋       | 105465/400000 [00:13<00:39, 7422.52it/s] 27%|██▋       | 106227/400000 [00:13<00:39, 7480.08it/s] 27%|██▋       | 107009/400000 [00:14<00:38, 7576.04it/s] 27%|██▋       | 107787/400000 [00:14<00:38, 7634.14it/s] 27%|██▋       | 108589/400000 [00:14<00:37, 7745.55it/s] 27%|██▋       | 109365/400000 [00:14<00:37, 7701.03it/s] 28%|██▊       | 110143/400000 [00:14<00:37, 7724.18it/s] 28%|██▊       | 110916/400000 [00:14<00:37, 7682.49it/s] 28%|██▊       | 111685/400000 [00:14<00:37, 7606.02it/s] 28%|██▊       | 112447/400000 [00:14<00:37, 7598.90it/s] 28%|██▊       | 113208/400000 [00:14<00:37, 7574.12it/s] 28%|██▊       | 113971/400000 [00:14<00:37, 7589.80it/s] 29%|██▊       | 114745/400000 [00:15<00:37, 7633.21it/s] 29%|██▉       | 115511/400000 [00:15<00:37, 7639.77it/s] 29%|██▉       | 116276/400000 [00:15<00:37, 7522.61it/s] 29%|██▉       | 117029/400000 [00:15<00:37, 7464.14it/s] 29%|██▉       | 117795/400000 [00:15<00:37, 7519.56it/s] 30%|██▉       | 118559/400000 [00:15<00:37, 7553.04it/s] 30%|██▉       | 119331/400000 [00:15<00:36, 7602.34it/s] 30%|███       | 120092/400000 [00:15<00:36, 7578.00it/s] 30%|███       | 120851/400000 [00:15<00:37, 7531.81it/s] 30%|███       | 121630/400000 [00:16<00:36, 7607.45it/s] 31%|███       | 122470/400000 [00:16<00:35, 7828.67it/s] 31%|███       | 123255/400000 [00:16<00:35, 7823.56it/s] 31%|███       | 124039/400000 [00:16<00:35, 7789.21it/s] 31%|███       | 124880/400000 [00:16<00:34, 7964.08it/s] 31%|███▏      | 125679/400000 [00:16<00:34, 7886.02it/s] 32%|███▏      | 126527/400000 [00:16<00:33, 8054.85it/s] 32%|███▏      | 127335/400000 [00:16<00:34, 7975.08it/s] 32%|███▏      | 128134/400000 [00:16<00:34, 7972.21it/s] 32%|███▏      | 128933/400000 [00:16<00:34, 7957.17it/s] 32%|███▏      | 129730/400000 [00:17<00:34, 7879.86it/s] 33%|███▎      | 130521/400000 [00:17<00:34, 7887.06it/s] 33%|███▎      | 131339/400000 [00:17<00:33, 7970.26it/s] 33%|███▎      | 132137/400000 [00:17<00:33, 7938.82it/s] 33%|███▎      | 132978/400000 [00:17<00:33, 8072.79it/s] 33%|███▎      | 133810/400000 [00:17<00:32, 8143.07it/s] 34%|███▎      | 134626/400000 [00:17<00:32, 8052.11it/s] 34%|███▍      | 135432/400000 [00:17<00:33, 7982.68it/s] 34%|███▍      | 136231/400000 [00:17<00:33, 7763.34it/s] 34%|███▍      | 137010/400000 [00:17<00:33, 7764.42it/s] 34%|███▍      | 137788/400000 [00:18<00:33, 7765.05it/s] 35%|███▍      | 138566/400000 [00:18<00:33, 7694.53it/s] 35%|███▍      | 139352/400000 [00:18<00:33, 7741.49it/s] 35%|███▌      | 140127/400000 [00:18<00:34, 7597.90it/s] 35%|███▌      | 140888/400000 [00:18<00:34, 7562.82it/s] 35%|███▌      | 141646/400000 [00:18<00:34, 7474.78it/s] 36%|███▌      | 142395/400000 [00:18<00:34, 7439.67it/s] 36%|███▌      | 143144/400000 [00:18<00:34, 7453.53it/s] 36%|███▌      | 143890/400000 [00:18<00:34, 7442.12it/s] 36%|███▌      | 144659/400000 [00:18<00:33, 7513.88it/s] 36%|███▋      | 145429/400000 [00:19<00:33, 7567.86it/s] 37%|███▋      | 146224/400000 [00:19<00:33, 7676.01it/s] 37%|███▋      | 147013/400000 [00:19<00:32, 7738.43it/s] 37%|███▋      | 147790/400000 [00:19<00:32, 7747.38it/s] 37%|███▋      | 148570/400000 [00:19<00:32, 7761.09it/s] 37%|███▋      | 149349/400000 [00:19<00:32, 7767.73it/s] 38%|███▊      | 150128/400000 [00:19<00:32, 7769.22it/s] 38%|███▊      | 150906/400000 [00:19<00:32, 7687.29it/s] 38%|███▊      | 151676/400000 [00:19<00:33, 7459.91it/s] 38%|███▊      | 152444/400000 [00:19<00:32, 7522.39it/s] 38%|███▊      | 153202/400000 [00:20<00:32, 7537.08it/s] 38%|███▊      | 153961/400000 [00:20<00:32, 7551.39it/s] 39%|███▊      | 154717/400000 [00:20<00:32, 7528.18it/s] 39%|███▉      | 155471/400000 [00:20<00:33, 7367.87it/s] 39%|███▉      | 156209/400000 [00:20<00:33, 7345.40it/s] 39%|███▉      | 156968/400000 [00:20<00:32, 7414.27it/s] 39%|███▉      | 157747/400000 [00:20<00:32, 7522.76it/s] 40%|███▉      | 158501/400000 [00:20<00:32, 7473.94it/s] 40%|███▉      | 159250/400000 [00:20<00:32, 7423.26it/s] 40%|████      | 160023/400000 [00:20<00:31, 7510.83it/s] 40%|████      | 160809/400000 [00:21<00:31, 7610.20it/s] 40%|████      | 161571/400000 [00:21<00:31, 7561.41it/s] 41%|████      | 162344/400000 [00:21<00:31, 7610.50it/s] 41%|████      | 163106/400000 [00:21<00:31, 7551.86it/s] 41%|████      | 163899/400000 [00:21<00:30, 7660.61it/s] 41%|████      | 164713/400000 [00:21<00:30, 7796.15it/s] 41%|████▏     | 165494/400000 [00:21<00:30, 7778.07it/s] 42%|████▏     | 166273/400000 [00:21<00:30, 7716.80it/s] 42%|████▏     | 167046/400000 [00:21<00:30, 7718.11it/s] 42%|████▏     | 167823/400000 [00:21<00:30, 7732.27it/s] 42%|████▏     | 168597/400000 [00:22<00:30, 7700.60it/s] 42%|████▏     | 169389/400000 [00:22<00:29, 7762.64it/s] 43%|████▎     | 170185/400000 [00:22<00:29, 7819.30it/s] 43%|████▎     | 170991/400000 [00:22<00:29, 7889.57it/s] 43%|████▎     | 171793/400000 [00:22<00:28, 7927.97it/s] 43%|████▎     | 172598/400000 [00:22<00:28, 7962.52it/s] 43%|████▎     | 173395/400000 [00:22<00:28, 7849.02it/s] 44%|████▎     | 174181/400000 [00:22<00:29, 7707.90it/s] 44%|████▎     | 174953/400000 [00:22<00:29, 7695.80it/s] 44%|████▍     | 175724/400000 [00:23<00:29, 7699.55it/s] 44%|████▍     | 176591/400000 [00:23<00:28, 7960.73it/s] 44%|████▍     | 177467/400000 [00:23<00:27, 8184.56it/s] 45%|████▍     | 178289/400000 [00:23<00:27, 8042.53it/s] 45%|████▍     | 179097/400000 [00:23<00:27, 8020.14it/s] 45%|████▍     | 179902/400000 [00:23<00:27, 7940.32it/s] 45%|████▌     | 180700/400000 [00:23<00:27, 7949.85it/s] 45%|████▌     | 181497/400000 [00:23<00:27, 7924.12it/s] 46%|████▌     | 182291/400000 [00:23<00:28, 7726.54it/s] 46%|████▌     | 183066/400000 [00:23<00:28, 7724.86it/s] 46%|████▌     | 183929/400000 [00:24<00:27, 7974.92it/s] 46%|████▌     | 184789/400000 [00:24<00:26, 8152.44it/s] 46%|████▋     | 185625/400000 [00:24<00:26, 8212.56it/s] 47%|████▋     | 186449/400000 [00:24<00:26, 8062.85it/s] 47%|████▋     | 187258/400000 [00:24<00:26, 8003.65it/s] 47%|████▋     | 188060/400000 [00:24<00:26, 7974.48it/s] 47%|████▋     | 188859/400000 [00:24<00:26, 7951.95it/s] 47%|████▋     | 189665/400000 [00:24<00:26, 7983.46it/s] 48%|████▊     | 190464/400000 [00:24<00:26, 7822.14it/s] 48%|████▊     | 191248/400000 [00:24<00:27, 7727.80it/s] 48%|████▊     | 192042/400000 [00:25<00:26, 7787.79it/s] 48%|████▊     | 192824/400000 [00:25<00:26, 7795.95it/s] 48%|████▊     | 193608/400000 [00:25<00:26, 7806.77it/s] 49%|████▊     | 194390/400000 [00:25<00:26, 7725.12it/s] 49%|████▉     | 195164/400000 [00:25<00:27, 7548.68it/s] 49%|████▉     | 195953/400000 [00:25<00:26, 7647.02it/s] 49%|████▉     | 196737/400000 [00:25<00:26, 7703.57it/s] 49%|████▉     | 197542/400000 [00:25<00:25, 7802.13it/s] 50%|████▉     | 198324/400000 [00:25<00:25, 7803.12it/s] 50%|████▉     | 199105/400000 [00:25<00:26, 7566.32it/s] 50%|████▉     | 199892/400000 [00:26<00:26, 7653.37it/s] 50%|█████     | 200698/400000 [00:26<00:25, 7770.43it/s] 50%|█████     | 201520/400000 [00:26<00:25, 7898.06it/s] 51%|█████     | 202312/400000 [00:26<00:25, 7827.25it/s] 51%|█████     | 203097/400000 [00:26<00:25, 7791.88it/s] 51%|█████     | 203903/400000 [00:26<00:24, 7868.33it/s] 51%|█████     | 204697/400000 [00:26<00:24, 7887.47it/s] 51%|█████▏    | 205508/400000 [00:26<00:24, 7952.58it/s] 52%|█████▏    | 206304/400000 [00:26<00:24, 7819.32it/s] 52%|█████▏    | 207088/400000 [00:26<00:24, 7823.74it/s] 52%|█████▏    | 207871/400000 [00:27<00:25, 7608.46it/s] 52%|█████▏    | 208634/400000 [00:27<00:25, 7487.87it/s] 52%|█████▏    | 209441/400000 [00:27<00:24, 7652.72it/s] 53%|█████▎    | 210252/400000 [00:27<00:24, 7781.84it/s] 53%|█████▎    | 211041/400000 [00:27<00:24, 7812.41it/s] 53%|█████▎    | 211824/400000 [00:27<00:24, 7807.77it/s] 53%|█████▎    | 212606/400000 [00:27<00:24, 7777.46it/s] 53%|█████▎    | 213385/400000 [00:27<00:24, 7701.18it/s] 54%|█████▎    | 214168/400000 [00:27<00:24, 7738.17it/s] 54%|█████▎    | 214982/400000 [00:28<00:23, 7853.80it/s] 54%|█████▍    | 215769/400000 [00:28<00:23, 7839.49it/s] 54%|█████▍    | 216554/400000 [00:28<00:23, 7756.68it/s] 54%|█████▍    | 217331/400000 [00:28<00:23, 7697.08it/s] 55%|█████▍    | 218102/400000 [00:28<00:23, 7698.39it/s] 55%|█████▍    | 218887/400000 [00:28<00:23, 7741.58it/s] 55%|█████▍    | 219662/400000 [00:28<00:23, 7689.28it/s] 55%|█████▌    | 220432/400000 [00:28<00:23, 7533.80it/s] 55%|█████▌    | 221187/400000 [00:28<00:24, 7259.46it/s] 55%|█████▌    | 221931/400000 [00:28<00:24, 7312.35it/s] 56%|█████▌    | 222684/400000 [00:29<00:24, 7373.84it/s] 56%|█████▌    | 223423/400000 [00:29<00:23, 7367.95it/s] 56%|█████▌    | 224194/400000 [00:29<00:23, 7467.31it/s] 56%|█████▌    | 224951/400000 [00:29<00:23, 7496.24it/s] 56%|█████▋    | 225718/400000 [00:29<00:23, 7544.41it/s] 57%|█████▋    | 226500/400000 [00:29<00:22, 7624.65it/s] 57%|█████▋    | 227277/400000 [00:29<00:22, 7665.98it/s] 57%|█████▋    | 228074/400000 [00:29<00:22, 7752.55it/s] 57%|█████▋    | 228867/400000 [00:29<00:21, 7803.43it/s] 57%|█████▋    | 229660/400000 [00:29<00:21, 7840.78it/s] 58%|█████▊    | 230445/400000 [00:30<00:21, 7761.53it/s] 58%|█████▊    | 231244/400000 [00:30<00:21, 7828.12it/s] 58%|█████▊    | 232080/400000 [00:30<00:21, 7979.03it/s] 58%|█████▊    | 232879/400000 [00:30<00:21, 7897.58it/s] 58%|█████▊    | 233670/400000 [00:30<00:21, 7822.99it/s] 59%|█████▊    | 234454/400000 [00:30<00:21, 7762.55it/s] 59%|█████▉    | 235231/400000 [00:30<00:22, 7416.38it/s] 59%|█████▉    | 235977/400000 [00:30<00:22, 7359.01it/s] 59%|█████▉    | 236716/400000 [00:30<00:22, 7363.73it/s] 59%|█████▉    | 237480/400000 [00:30<00:21, 7443.79it/s] 60%|█████▉    | 238262/400000 [00:31<00:21, 7550.32it/s] 60%|█████▉    | 239019/400000 [00:31<00:21, 7421.39it/s] 60%|█████▉    | 239791/400000 [00:31<00:21, 7507.43it/s] 60%|██████    | 240585/400000 [00:31<00:20, 7631.50it/s] 60%|██████    | 241414/400000 [00:31<00:20, 7817.75it/s] 61%|██████    | 242200/400000 [00:31<00:20, 7828.63it/s] 61%|██████    | 242996/400000 [00:31<00:19, 7867.44it/s] 61%|██████    | 243784/400000 [00:31<00:19, 7865.79it/s] 61%|██████    | 244572/400000 [00:31<00:19, 7868.58it/s] 61%|██████▏   | 245361/400000 [00:31<00:19, 7872.62it/s] 62%|██████▏   | 246149/400000 [00:32<00:20, 7684.03it/s] 62%|██████▏   | 246919/400000 [00:32<00:20, 7644.57it/s] 62%|██████▏   | 247690/400000 [00:32<00:19, 7663.24it/s] 62%|██████▏   | 248523/400000 [00:32<00:19, 7849.45it/s] 62%|██████▏   | 249313/400000 [00:32<00:19, 7863.27it/s] 63%|██████▎   | 250101/400000 [00:32<00:19, 7852.84it/s] 63%|██████▎   | 250907/400000 [00:32<00:18, 7912.64it/s] 63%|██████▎   | 251699/400000 [00:32<00:18, 7849.18it/s] 63%|██████▎   | 252485/400000 [00:32<00:18, 7781.39it/s] 63%|██████▎   | 253293/400000 [00:32<00:18, 7866.91it/s] 64%|██████▎   | 254110/400000 [00:33<00:18, 7954.39it/s] 64%|██████▎   | 254907/400000 [00:33<00:18, 7889.99it/s] 64%|██████▍   | 255697/400000 [00:33<00:18, 7883.15it/s] 64%|██████▍   | 256514/400000 [00:33<00:18, 7965.26it/s] 64%|██████▍   | 257343/400000 [00:33<00:17, 8055.55it/s] 65%|██████▍   | 258150/400000 [00:33<00:17, 7941.50it/s] 65%|██████▍   | 258945/400000 [00:33<00:17, 7871.51it/s] 65%|██████▍   | 259733/400000 [00:33<00:18, 7720.87it/s] 65%|██████▌   | 260507/400000 [00:33<00:18, 7585.47it/s] 65%|██████▌   | 261267/400000 [00:34<00:18, 7562.91it/s] 66%|██████▌   | 262062/400000 [00:34<00:17, 7673.10it/s] 66%|██████▌   | 262831/400000 [00:34<00:17, 7661.78it/s] 66%|██████▌   | 263598/400000 [00:34<00:18, 7494.03it/s] 66%|██████▌   | 264374/400000 [00:34<00:17, 7569.84it/s] 66%|██████▋   | 265133/400000 [00:34<00:17, 7552.23it/s] 66%|██████▋   | 265963/400000 [00:34<00:17, 7760.23it/s] 67%|██████▋   | 266743/400000 [00:34<00:17, 7770.53it/s] 67%|██████▋   | 267533/400000 [00:34<00:16, 7807.33it/s] 67%|██████▋   | 268335/400000 [00:34<00:16, 7869.15it/s] 67%|██████▋   | 269168/400000 [00:35<00:16, 8000.84it/s] 67%|██████▋   | 269979/400000 [00:35<00:16, 8032.39it/s] 68%|██████▊   | 270831/400000 [00:35<00:15, 8171.06it/s] 68%|██████▊   | 271650/400000 [00:35<00:15, 8127.98it/s] 68%|██████▊   | 272464/400000 [00:35<00:16, 7888.09it/s] 68%|██████▊   | 273255/400000 [00:35<00:16, 7879.52it/s] 69%|██████▊   | 274053/400000 [00:35<00:15, 7908.44it/s] 69%|██████▊   | 274861/400000 [00:35<00:15, 7958.34it/s] 69%|██████▉   | 275658/400000 [00:35<00:15, 7914.63it/s] 69%|██████▉   | 276451/400000 [00:35<00:15, 7892.87it/s] 69%|██████▉   | 277258/400000 [00:36<00:15, 7943.64it/s] 70%|██████▉   | 278053/400000 [00:36<00:15, 7846.87it/s] 70%|██████▉   | 278839/400000 [00:36<00:15, 7652.12it/s] 70%|██████▉   | 279661/400000 [00:36<00:15, 7812.20it/s] 70%|███████   | 280457/400000 [00:36<00:15, 7854.31it/s] 70%|███████   | 281266/400000 [00:36<00:14, 7921.31it/s] 71%|███████   | 282081/400000 [00:36<00:14, 7988.29it/s] 71%|███████   | 282887/400000 [00:36<00:14, 8008.90it/s] 71%|███████   | 283689/400000 [00:36<00:14, 7940.91it/s] 71%|███████   | 284484/400000 [00:36<00:14, 7930.36it/s] 71%|███████▏  | 285278/400000 [00:37<00:14, 7830.31it/s] 72%|███████▏  | 286065/400000 [00:37<00:14, 7840.25it/s] 72%|███████▏  | 286883/400000 [00:37<00:14, 7936.80it/s] 72%|███████▏  | 287729/400000 [00:37<00:13, 8085.44it/s] 72%|███████▏  | 288539/400000 [00:37<00:13, 7992.50it/s] 72%|███████▏  | 289346/400000 [00:37<00:13, 8014.35it/s] 73%|███████▎  | 290149/400000 [00:37<00:14, 7749.21it/s] 73%|███████▎  | 290928/400000 [00:37<00:14, 7759.31it/s] 73%|███████▎  | 291711/400000 [00:37<00:13, 7778.02it/s] 73%|███████▎  | 292490/400000 [00:37<00:13, 7775.25it/s] 73%|███████▎  | 293269/400000 [00:38<00:13, 7756.49it/s] 74%|███████▎  | 294083/400000 [00:38<00:13, 7864.80it/s] 74%|███████▎  | 294939/400000 [00:38<00:13, 8058.58it/s] 74%|███████▍  | 295762/400000 [00:38<00:12, 8108.03it/s] 74%|███████▍  | 296629/400000 [00:38<00:12, 8268.59it/s] 74%|███████▍  | 297485/400000 [00:38<00:12, 8351.72it/s] 75%|███████▍  | 298322/400000 [00:38<00:12, 8162.51it/s] 75%|███████▍  | 299141/400000 [00:38<00:12, 7957.94it/s] 75%|███████▍  | 299940/400000 [00:38<00:12, 7946.11it/s] 75%|███████▌  | 300812/400000 [00:39<00:12, 8161.53it/s] 75%|███████▌  | 301631/400000 [00:39<00:12, 8163.20it/s] 76%|███████▌  | 302505/400000 [00:39<00:11, 8327.51it/s] 76%|███████▌  | 303346/400000 [00:39<00:11, 8350.16it/s] 76%|███████▌  | 304192/400000 [00:39<00:11, 8382.12it/s] 76%|███████▋  | 305049/400000 [00:39<00:11, 8435.90it/s] 76%|███████▋  | 305894/400000 [00:39<00:11, 8345.55it/s] 77%|███████▋  | 306730/400000 [00:39<00:11, 8208.43it/s] 77%|███████▋  | 307552/400000 [00:39<00:11, 8162.11it/s] 77%|███████▋  | 308370/400000 [00:39<00:11, 8156.83it/s] 77%|███████▋  | 309198/400000 [00:40<00:11, 8192.63it/s] 78%|███████▊  | 310018/400000 [00:40<00:11, 7945.29it/s] 78%|███████▊  | 310826/400000 [00:40<00:11, 7983.55it/s] 78%|███████▊  | 311682/400000 [00:40<00:10, 8145.48it/s] 78%|███████▊  | 312499/400000 [00:40<00:11, 7874.96it/s] 78%|███████▊  | 313316/400000 [00:40<00:10, 7960.14it/s] 79%|███████▊  | 314115/400000 [00:40<00:10, 7942.95it/s] 79%|███████▊  | 314922/400000 [00:40<00:10, 7979.50it/s] 79%|███████▉  | 315722/400000 [00:40<00:10, 7905.50it/s] 79%|███████▉  | 316574/400000 [00:40<00:10, 8079.62it/s] 79%|███████▉  | 317384/400000 [00:41<00:10, 8066.67it/s] 80%|███████▉  | 318192/400000 [00:41<00:10, 8032.50it/s] 80%|███████▉  | 318997/400000 [00:41<00:10, 7958.18it/s] 80%|███████▉  | 319794/400000 [00:41<00:10, 7925.32it/s] 80%|████████  | 320588/400000 [00:41<00:10, 7875.04it/s] 80%|████████  | 321376/400000 [00:41<00:10, 7819.49it/s] 81%|████████  | 322171/400000 [00:41<00:09, 7854.66it/s] 81%|████████  | 322957/400000 [00:41<00:09, 7833.84it/s] 81%|████████  | 323741/400000 [00:41<00:09, 7794.37it/s] 81%|████████  | 324521/400000 [00:41<00:09, 7791.58it/s] 81%|████████▏ | 325301/400000 [00:42<00:09, 7768.05it/s] 82%|████████▏ | 326078/400000 [00:42<00:09, 7716.83it/s] 82%|████████▏ | 326916/400000 [00:42<00:09, 7902.38it/s] 82%|████████▏ | 327723/400000 [00:42<00:09, 7951.14it/s] 82%|████████▏ | 328520/400000 [00:42<00:09, 7869.52it/s] 82%|████████▏ | 329308/400000 [00:42<00:08, 7869.15it/s] 83%|████████▎ | 330098/400000 [00:42<00:08, 7877.54it/s] 83%|████████▎ | 330896/400000 [00:42<00:08, 7906.14it/s] 83%|████████▎ | 331687/400000 [00:42<00:08, 7794.09it/s] 83%|████████▎ | 332483/400000 [00:42<00:08, 7842.90it/s] 83%|████████▎ | 333287/400000 [00:43<00:08, 7900.55it/s] 84%|████████▎ | 334078/400000 [00:43<00:08, 7825.67it/s] 84%|████████▎ | 334862/400000 [00:43<00:08, 7671.66it/s] 84%|████████▍ | 335631/400000 [00:43<00:08, 7651.36it/s] 84%|████████▍ | 336397/400000 [00:43<00:08, 7640.05it/s] 84%|████████▍ | 337219/400000 [00:43<00:08, 7802.62it/s] 85%|████████▍ | 338008/400000 [00:43<00:07, 7827.87it/s] 85%|████████▍ | 338792/400000 [00:43<00:07, 7811.54it/s] 85%|████████▍ | 339574/400000 [00:43<00:07, 7781.30it/s] 85%|████████▌ | 340353/400000 [00:43<00:07, 7782.66it/s] 85%|████████▌ | 341155/400000 [00:44<00:07, 7850.62it/s] 85%|████████▌ | 341941/400000 [00:44<00:07, 7594.21it/s] 86%|████████▌ | 342703/400000 [00:44<00:07, 7516.93it/s] 86%|████████▌ | 343457/400000 [00:44<00:07, 7370.41it/s] 86%|████████▌ | 344216/400000 [00:44<00:07, 7434.75it/s] 86%|████████▌ | 344965/400000 [00:44<00:07, 7449.97it/s] 86%|████████▋ | 345711/400000 [00:44<00:07, 7410.94it/s] 87%|████████▋ | 346453/400000 [00:44<00:07, 7389.16it/s] 87%|████████▋ | 347223/400000 [00:44<00:07, 7479.62it/s] 87%|████████▋ | 347972/400000 [00:45<00:07, 7396.31it/s] 87%|████████▋ | 348770/400000 [00:45<00:06, 7560.81it/s] 87%|████████▋ | 349528/400000 [00:45<00:06, 7558.62it/s] 88%|████████▊ | 350285/400000 [00:45<00:06, 7555.57it/s] 88%|████████▊ | 351042/400000 [00:45<00:06, 7456.23it/s] 88%|████████▊ | 351791/400000 [00:45<00:06, 7465.37it/s] 88%|████████▊ | 352541/400000 [00:45<00:06, 7473.60it/s] 88%|████████▊ | 353327/400000 [00:45<00:06, 7583.76it/s] 89%|████████▊ | 354090/400000 [00:45<00:06, 7596.26it/s] 89%|████████▊ | 354869/400000 [00:45<00:05, 7650.33it/s] 89%|████████▉ | 355635/400000 [00:46<00:05, 7599.65it/s] 89%|████████▉ | 356422/400000 [00:46<00:05, 7678.08it/s] 89%|████████▉ | 357191/400000 [00:46<00:05, 7654.01it/s] 89%|████████▉ | 357957/400000 [00:46<00:05, 7612.87it/s] 90%|████████▉ | 358719/400000 [00:46<00:05, 7611.61it/s] 90%|████████▉ | 359505/400000 [00:46<00:05, 7684.15it/s] 90%|█████████ | 360280/400000 [00:46<00:05, 7701.22it/s] 90%|█████████ | 361061/400000 [00:46<00:05, 7731.05it/s] 90%|█████████ | 361854/400000 [00:46<00:04, 7786.99it/s] 91%|█████████ | 362664/400000 [00:46<00:04, 7875.58it/s] 91%|█████████ | 363452/400000 [00:47<00:04, 7774.83it/s] 91%|█████████ | 364232/400000 [00:47<00:04, 7780.48it/s] 91%|█████████▏| 365055/400000 [00:47<00:04, 7905.06it/s] 91%|█████████▏| 365847/400000 [00:47<00:04, 7826.17it/s] 92%|█████████▏| 366631/400000 [00:47<00:04, 7830.00it/s] 92%|█████████▏| 367469/400000 [00:47<00:04, 7986.44it/s] 92%|█████████▏| 368269/400000 [00:47<00:03, 7985.09it/s] 92%|█████████▏| 369078/400000 [00:47<00:03, 8015.97it/s] 92%|█████████▏| 369881/400000 [00:47<00:03, 7946.56it/s] 93%|█████████▎| 370677/400000 [00:47<00:03, 7908.78it/s] 93%|█████████▎| 371469/400000 [00:48<00:03, 7882.13it/s] 93%|█████████▎| 372319/400000 [00:48<00:03, 8056.08it/s] 93%|█████████▎| 373126/400000 [00:48<00:03, 8015.01it/s] 93%|█████████▎| 373929/400000 [00:48<00:03, 7982.05it/s] 94%|█████████▎| 374754/400000 [00:48<00:03, 8059.10it/s] 94%|█████████▍| 375582/400000 [00:48<00:03, 8124.09it/s] 94%|█████████▍| 376395/400000 [00:48<00:02, 8079.48it/s] 94%|█████████▍| 377204/400000 [00:48<00:02, 7967.18it/s] 95%|█████████▍| 378002/400000 [00:48<00:02, 7840.93it/s] 95%|█████████▍| 378788/400000 [00:48<00:02, 7831.87it/s] 95%|█████████▍| 379586/400000 [00:49<00:02, 7874.19it/s] 95%|█████████▌| 380374/400000 [00:49<00:02, 7845.85it/s] 95%|█████████▌| 381159/400000 [00:49<00:02, 7826.01it/s] 95%|█████████▌| 381942/400000 [00:49<00:02, 7793.93it/s] 96%|█████████▌| 382766/400000 [00:49<00:02, 7920.25it/s] 96%|█████████▌| 383559/400000 [00:49<00:02, 7825.65it/s] 96%|█████████▌| 384374/400000 [00:49<00:01, 7920.06it/s] 96%|█████████▋| 385178/400000 [00:49<00:01, 7955.15it/s] 96%|█████████▋| 385975/400000 [00:49<00:01, 7871.56it/s] 97%|█████████▋| 386786/400000 [00:49<00:01, 7941.19it/s] 97%|█████████▋| 387581/400000 [00:50<00:01, 7861.30it/s] 97%|█████████▋| 388368/400000 [00:50<00:01, 7824.76it/s] 97%|█████████▋| 389162/400000 [00:50<00:01, 7855.62it/s] 97%|█████████▋| 389948/400000 [00:50<00:01, 7795.53it/s] 98%|█████████▊| 390728/400000 [00:50<00:01, 7781.85it/s] 98%|█████████▊| 391507/400000 [00:50<00:01, 7783.30it/s] 98%|█████████▊| 392305/400000 [00:50<00:00, 7836.72it/s] 98%|█████████▊| 393089/400000 [00:50<00:00, 7804.98it/s] 98%|█████████▊| 393870/400000 [00:50<00:00, 7740.63it/s] 99%|█████████▊| 394645/400000 [00:50<00:00, 7678.51it/s] 99%|█████████▉| 395414/400000 [00:51<00:00, 7673.63it/s] 99%|█████████▉| 396192/400000 [00:51<00:00, 7703.13it/s] 99%|█████████▉| 396963/400000 [00:51<00:00, 7651.13it/s] 99%|█████████▉| 397729/400000 [00:51<00:00, 7259.32it/s]100%|█████████▉| 398498/400000 [00:51<00:00, 7382.31it/s]100%|█████████▉| 399285/400000 [00:51<00:00, 7520.45it/s]100%|█████████▉| 399999/400000 [00:51<00:00, 7737.59it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fcdc16aed68> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011434619859905543 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.011403175102029756 	 Accuracy: 50

  model saves at 50% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15903 out of table with 15851 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15903 out of table with 15851 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-13 22:24:14.435967: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 22:24:14.440195: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-13 22:24:14.440356: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5654face0b60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 22:24:14.440370: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fcdc45ae5c0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.7586 - accuracy: 0.4940
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6053 - accuracy: 0.5040 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7024 - accuracy: 0.4977
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7356 - accuracy: 0.4955
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6850 - accuracy: 0.4988
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6896 - accuracy: 0.4985
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6907 - accuracy: 0.4984
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6551 - accuracy: 0.5008
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6632 - accuracy: 0.5002
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6758 - accuracy: 0.4994
11000/25000 [============>.................] - ETA: 4s - loss: 7.6917 - accuracy: 0.4984
12000/25000 [=============>................] - ETA: 4s - loss: 7.6820 - accuracy: 0.4990
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6678 - accuracy: 0.4999
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6962 - accuracy: 0.4981
15000/25000 [=================>............] - ETA: 3s - loss: 7.6830 - accuracy: 0.4989
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7097 - accuracy: 0.4972
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6910 - accuracy: 0.4984
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7067 - accuracy: 0.4974
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6884 - accuracy: 0.4986
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7034 - accuracy: 0.4976
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6980 - accuracy: 0.4980
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6813 - accuracy: 0.4990
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6653 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 10s 395us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fcd320cbb38> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fcdc8e0c160> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.5461 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.4960 - val_crf_viterbi_accuracy: 0.0000e+00

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

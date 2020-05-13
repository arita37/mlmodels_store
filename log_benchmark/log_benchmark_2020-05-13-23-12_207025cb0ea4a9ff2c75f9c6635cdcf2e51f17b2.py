
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7efdc24a0fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 23:13:07.749224
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 23:13:07.753827
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 23:13:07.757894
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 23:13:07.763485
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7efdce26a438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 357099.5000
Epoch 2/10

1/1 [==============================] - 0s 103ms/step - loss: 277487.5312
Epoch 3/10

1/1 [==============================] - 0s 92ms/step - loss: 196571.2188
Epoch 4/10

1/1 [==============================] - 0s 100ms/step - loss: 124336.2109
Epoch 5/10

1/1 [==============================] - 0s 97ms/step - loss: 75680.8438
Epoch 6/10

1/1 [==============================] - 0s 96ms/step - loss: 46639.2617
Epoch 7/10

1/1 [==============================] - 0s 96ms/step - loss: 30132.1680
Epoch 8/10

1/1 [==============================] - 0s 94ms/step - loss: 20428.6836
Epoch 9/10

1/1 [==============================] - 0s 94ms/step - loss: 14483.8408
Epoch 10/10

1/1 [==============================] - 0s 100ms/step - loss: 10700.9180

  #### Inference Need return ypred, ytrue ######################### 
[[ 9.5999815e-02  6.0674725e+00  7.3020945e+00  5.8793855e+00
   5.8561726e+00  6.8074579e+00  7.1102366e+00  7.2318511e+00
   6.9542856e+00  5.0994749e+00  6.8129125e+00  4.9611959e+00
   5.4076166e+00  5.6987243e+00  5.4161477e+00  4.9895134e+00
   5.3162036e+00  6.6786423e+00  5.5706873e+00  6.9953408e+00
   6.6082692e+00  7.2252049e+00  5.9822927e+00  6.6032739e+00
   4.5051661e+00  5.7048450e+00  5.8015442e+00  7.3275380e+00
   6.1711307e+00  6.7017956e+00  7.4057460e+00  6.8415303e+00
   5.0964465e+00  6.0492597e+00  4.2304001e+00  5.6242781e+00
   5.6749258e+00  7.1606431e+00  7.7283287e+00  4.9509172e+00
   7.3796763e+00  6.0183821e+00  4.9472370e+00  6.1859865e+00
   6.9627810e+00  5.4057941e+00  4.4639030e+00  6.6890993e+00
   5.0263805e+00  5.8497963e+00  6.4036217e+00  6.4081984e+00
   6.3113160e+00  4.4137907e+00  5.6007104e+00  5.1192088e+00
   6.1033697e+00  5.7563853e+00  5.0851412e+00  5.1382103e+00
   7.0280030e-02  1.2855198e+00  3.0641031e-01 -1.9756318e+00
  -4.4401923e-01  1.3315190e+00  3.3505335e-01 -1.6535147e+00
   1.2041087e+00 -6.3243222e-01  9.0107012e-01 -7.1004117e-01
   9.8427504e-01 -4.1394639e-01 -8.7630230e-01  3.0781475e-01
  -4.2074415e-01 -5.3294992e-01 -5.4336333e-01  9.1032952e-01
  -1.1356033e+00 -4.8324126e-01 -1.1718647e+00  7.4186486e-01
  -6.6817844e-01  2.7366823e-01  3.7336595e-02  1.7349133e+00
  -2.8399494e-01 -5.3058076e-01  9.6549517e-01  1.2089847e+00
  -1.0697145e+00  1.3908272e+00  1.2553195e+00  1.6571027e-01
   8.8374026e-02 -5.2964056e-01 -5.6672074e-02  7.4248016e-01
  -1.1053350e+00 -1.5257040e+00  1.0745490e+00 -1.5489854e+00
   8.2089365e-01  3.4190607e-01  1.7252061e+00  4.3975535e-01
   4.4131365e-01  1.4800464e+00 -5.9921206e-03 -1.5425617e-01
   3.5114086e-01  8.2779229e-01  3.6261916e-01  1.0678651e+00
   1.5360454e+00  1.9613995e-01 -1.0397474e+00  1.5157743e-01
  -7.1290874e-01  6.8220752e-01  6.9187117e-01 -1.2597500e+00
   5.2973652e-01  7.4843216e-01  1.5537196e+00 -1.8722123e-01
  -9.2854905e-01  4.6011192e-01 -9.4438577e-01 -2.3860762e-01
  -1.2930968e-01 -6.8203568e-01  1.5380731e-01 -1.6863148e-01
  -1.5865321e+00  3.3840847e-01  1.3393303e+00  1.3845456e+00
   4.8256162e-01 -8.4884650e-01  1.0325918e+00 -5.6160104e-01
  -3.1872502e-01  2.4237144e-01  3.8871345e-01  8.2719922e-01
   5.9627330e-01 -4.6945024e-01 -1.1830199e+00  5.8471549e-01
   1.3211242e+00 -2.6701176e-01  2.8311542e-01  3.9725184e-02
  -1.5059392e+00  8.1209260e-01  1.3146030e+00  5.1780701e-02
  -1.1209606e+00  1.3862122e+00 -1.1078093e+00  3.8940233e-01
  -4.2341137e-01  2.2643107e-01 -8.7660939e-01  5.2018100e-01
   4.2087126e-01  1.0799518e+00  6.3199449e-01 -4.5377031e-01
  -7.2861660e-01 -2.1231070e-01 -3.3167201e-01 -6.9631571e-01
   1.3501461e+00 -1.7574799e-01  1.3534226e+00  3.3545721e-01
   8.4172904e-02  6.7713871e+00  6.6922188e+00  7.9894862e+00
   8.1979513e+00  6.2986536e+00  6.7083020e+00  7.6151423e+00
   6.6460333e+00  7.3576193e+00  5.0038886e+00  7.0636153e+00
   6.5239153e+00  5.9207959e+00  8.3184776e+00  7.2088785e+00
   7.0792255e+00  6.5664620e+00  7.8089247e+00  7.5627947e+00
   6.6315370e+00  6.2429218e+00  6.5380640e+00  7.7847142e+00
   7.2027464e+00  7.3836107e+00  5.9950237e+00  6.6469545e+00
   6.2077889e+00  6.7069902e+00  6.5905323e+00  7.5604014e+00
   7.4965596e+00  7.7121344e+00  6.2597504e+00  7.8453965e+00
   7.1072297e+00  6.0978122e+00  6.5825233e+00  5.4842606e+00
   7.2017407e+00  6.6289358e+00  4.9056997e+00  6.6089931e+00
   6.3736362e+00  7.4501543e+00  6.4064932e+00  6.2997746e+00
   4.7684999e+00  6.5111585e+00  5.8679676e+00  7.8233352e+00
   5.7256575e+00  6.4239721e+00  5.6633463e+00  5.6123090e+00
   6.7516184e+00  5.0146632e+00  7.8703032e+00  6.9353857e+00
   2.1149960e+00  1.0850418e+00  6.3859349e-01  7.2696924e-01
   1.2697606e+00  2.6401252e-01  8.4323889e-01  3.9895034e-01
   1.3114851e+00  3.8557905e-01  3.8790917e-01  8.3217645e-01
   8.2553816e-01  8.1476510e-01  2.3650813e-01  1.3264298e+00
   1.2530200e+00  1.8621576e+00  1.0685232e+00  1.5855167e+00
   9.5833898e-01  1.3281333e+00  4.5287681e-01  5.9868371e-01
   6.9362861e-01  6.1366034e-01  9.7037441e-01  4.7414106e-01
   2.1276772e+00  1.2127250e+00  1.7896965e+00  8.7529802e-01
   6.6004109e-01  1.2603086e+00  2.7811983e+00  4.8385918e-01
   1.4162736e+00  9.6366501e-01  8.4031916e-01  1.5155246e+00
   2.4887185e+00  1.2316819e+00  3.9815557e-01  5.8578271e-01
   2.0266776e+00  1.1813512e+00  2.3793271e+00  1.8261323e+00
   2.9680419e-01  2.4425857e+00  1.9456118e-01  8.5438883e-01
   4.8688197e-01  2.7389090e+00  4.8560798e-01  1.4993204e+00
   6.5502298e-01  1.0450817e+00  5.0912666e-01  3.2070434e-01
   1.5242724e+00  1.8092960e-01  4.0967059e-01  1.7222539e+00
   1.0875022e+00  1.5790834e+00  9.3461090e-01  1.9852523e+00
   1.8869774e+00  1.3940403e+00  7.1954072e-01  3.0623752e-01
   2.0105605e+00  2.2146654e-01  2.2510157e+00  9.6486479e-01
   1.3161172e+00  6.9562018e-01  4.8375887e-01  2.9927101e+00
   4.6483117e-01  3.5826075e-01  2.9986691e-01  1.8472879e+00
   1.3010027e+00  5.7772863e-01  2.3871148e-01  2.1814361e+00
   9.6114695e-01  2.1881711e+00  3.6582994e-01  5.1482558e-01
   8.8205087e-01  1.7633784e-01  1.7603561e+00  7.4958289e-01
   1.8056893e-01  4.1385704e-01  1.1925325e+00  6.5042734e-01
   3.8989151e-01  4.5466584e-01  2.5688963e+00  1.9912670e+00
   6.5943730e-01  1.2091856e+00  7.2361553e-01  1.1353583e+00
   1.0819000e+00  5.3229374e-01  1.0867682e+00  5.8964986e-01
   2.3533955e+00  2.4622245e+00  1.4704928e+00  1.7119280e+00
   5.9737176e-01  1.7583007e+00  2.0535984e+00  1.2421439e+00
   2.8331349e+00 -8.7669343e-02 -1.8118279e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 23:13:16.971004
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.9516
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 23:13:16.974666
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9228.88
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 23:13:16.977874
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.5669
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 23:13:16.981308
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -825.512
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139627992307248
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139627050844792
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139627050845296
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139627050845800
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139627050846304
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139627050846808

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7efdbbc355f8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.548294
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.507192
grad_step = 000002, loss = 0.477377
grad_step = 000003, loss = 0.446389
grad_step = 000004, loss = 0.414287
grad_step = 000005, loss = 0.386831
grad_step = 000006, loss = 0.376219
grad_step = 000007, loss = 0.371293
grad_step = 000008, loss = 0.349824
grad_step = 000009, loss = 0.328547
grad_step = 000010, loss = 0.314761
grad_step = 000011, loss = 0.307168
grad_step = 000012, loss = 0.301784
grad_step = 000013, loss = 0.294051
grad_step = 000014, loss = 0.282915
grad_step = 000015, loss = 0.271014
grad_step = 000016, loss = 0.260032
grad_step = 000017, loss = 0.250554
grad_step = 000018, loss = 0.242584
grad_step = 000019, loss = 0.234312
grad_step = 000020, loss = 0.224440
grad_step = 000021, loss = 0.214402
grad_step = 000022, loss = 0.205526
grad_step = 000023, loss = 0.197697
grad_step = 000024, loss = 0.190484
grad_step = 000025, loss = 0.183117
grad_step = 000026, loss = 0.175198
grad_step = 000027, loss = 0.167136
grad_step = 000028, loss = 0.159621
grad_step = 000029, loss = 0.152795
grad_step = 000030, loss = 0.146257
grad_step = 000031, loss = 0.139583
grad_step = 000032, loss = 0.132765
grad_step = 000033, loss = 0.126170
grad_step = 000034, loss = 0.119944
grad_step = 000035, loss = 0.114069
grad_step = 000036, loss = 0.108533
grad_step = 000037, loss = 0.103125
grad_step = 000038, loss = 0.097754
grad_step = 000039, loss = 0.092583
grad_step = 000040, loss = 0.087761
grad_step = 000041, loss = 0.083204
grad_step = 000042, loss = 0.078662
grad_step = 000043, loss = 0.074167
grad_step = 000044, loss = 0.069880
grad_step = 000045, loss = 0.065915
grad_step = 000046, loss = 0.062185
grad_step = 000047, loss = 0.058571
grad_step = 000048, loss = 0.055099
grad_step = 000049, loss = 0.051793
grad_step = 000050, loss = 0.048678
grad_step = 000051, loss = 0.045725
grad_step = 000052, loss = 0.042922
grad_step = 000053, loss = 0.040209
grad_step = 000054, loss = 0.037630
grad_step = 000055, loss = 0.035226
grad_step = 000056, loss = 0.032971
grad_step = 000057, loss = 0.030801
grad_step = 000058, loss = 0.028750
grad_step = 000059, loss = 0.026824
grad_step = 000060, loss = 0.025016
grad_step = 000061, loss = 0.023314
grad_step = 000062, loss = 0.021701
grad_step = 000063, loss = 0.020169
grad_step = 000064, loss = 0.018740
grad_step = 000065, loss = 0.017424
grad_step = 000066, loss = 0.016189
grad_step = 000067, loss = 0.015020
grad_step = 000068, loss = 0.013923
grad_step = 000069, loss = 0.012906
grad_step = 000070, loss = 0.011966
grad_step = 000071, loss = 0.011093
grad_step = 000072, loss = 0.010281
grad_step = 000073, loss = 0.009528
grad_step = 000074, loss = 0.008841
grad_step = 000075, loss = 0.008208
grad_step = 000076, loss = 0.007618
grad_step = 000077, loss = 0.007079
grad_step = 000078, loss = 0.006588
grad_step = 000079, loss = 0.006140
grad_step = 000080, loss = 0.005732
grad_step = 000081, loss = 0.005355
grad_step = 000082, loss = 0.005015
grad_step = 000083, loss = 0.004710
grad_step = 000084, loss = 0.004434
grad_step = 000085, loss = 0.004182
grad_step = 000086, loss = 0.003956
grad_step = 000087, loss = 0.003752
grad_step = 000088, loss = 0.003571
grad_step = 000089, loss = 0.003408
grad_step = 000090, loss = 0.003261
grad_step = 000091, loss = 0.003131
grad_step = 000092, loss = 0.003016
grad_step = 000093, loss = 0.002913
grad_step = 000094, loss = 0.002821
grad_step = 000095, loss = 0.002740
grad_step = 000096, loss = 0.002670
grad_step = 000097, loss = 0.002607
grad_step = 000098, loss = 0.002550
grad_step = 000099, loss = 0.002501
grad_step = 000100, loss = 0.002457
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002522
grad_step = 000102, loss = 0.002412
grad_step = 000103, loss = 0.002387
grad_step = 000104, loss = 0.002349
grad_step = 000105, loss = 0.002310
grad_step = 000106, loss = 0.002293
grad_step = 000107, loss = 0.002286
grad_step = 000108, loss = 0.002261
grad_step = 000109, loss = 0.002237
grad_step = 000110, loss = 0.002225
grad_step = 000111, loss = 0.002217
grad_step = 000112, loss = 0.002208
grad_step = 000113, loss = 0.002191
grad_step = 000114, loss = 0.002178
grad_step = 000115, loss = 0.002171
grad_step = 000116, loss = 0.002165
grad_step = 000117, loss = 0.002160
grad_step = 000118, loss = 0.002149
grad_step = 000119, loss = 0.002140
grad_step = 000120, loss = 0.002132
grad_step = 000121, loss = 0.002128
grad_step = 000122, loss = 0.002124
grad_step = 000123, loss = 0.002120
grad_step = 000124, loss = 0.002114
grad_step = 000125, loss = 0.002106
grad_step = 000126, loss = 0.002100
grad_step = 000127, loss = 0.002094
grad_step = 000128, loss = 0.002090
grad_step = 000129, loss = 0.002086
grad_step = 000130, loss = 0.002083
grad_step = 000131, loss = 0.002080
grad_step = 000132, loss = 0.002078
grad_step = 000133, loss = 0.002076
grad_step = 000134, loss = 0.002075
grad_step = 000135, loss = 0.002074
grad_step = 000136, loss = 0.002074
grad_step = 000137, loss = 0.002077
grad_step = 000138, loss = 0.002079
grad_step = 000139, loss = 0.002080
grad_step = 000140, loss = 0.002076
grad_step = 000141, loss = 0.002068
grad_step = 000142, loss = 0.002054
grad_step = 000143, loss = 0.002038
grad_step = 000144, loss = 0.002026
grad_step = 000145, loss = 0.002019
grad_step = 000146, loss = 0.002016
grad_step = 000147, loss = 0.002018
grad_step = 000148, loss = 0.002021
grad_step = 000149, loss = 0.002026
grad_step = 000150, loss = 0.002032
grad_step = 000151, loss = 0.002036
grad_step = 000152, loss = 0.002035
grad_step = 000153, loss = 0.002025
grad_step = 000154, loss = 0.002010
grad_step = 000155, loss = 0.001991
grad_step = 000156, loss = 0.001976
grad_step = 000157, loss = 0.001968
grad_step = 000158, loss = 0.001967
grad_step = 000159, loss = 0.001970
grad_step = 000160, loss = 0.001978
grad_step = 000161, loss = 0.001993
grad_step = 000162, loss = 0.002016
grad_step = 000163, loss = 0.002044
grad_step = 000164, loss = 0.002048
grad_step = 000165, loss = 0.002028
grad_step = 000166, loss = 0.001978
grad_step = 000167, loss = 0.001936
grad_step = 000168, loss = 0.001928
grad_step = 000169, loss = 0.001951
grad_step = 000170, loss = 0.001975
grad_step = 000171, loss = 0.001970
grad_step = 000172, loss = 0.001954
grad_step = 000173, loss = 0.001935
grad_step = 000174, loss = 0.001918
grad_step = 000175, loss = 0.001902
grad_step = 000176, loss = 0.001902
grad_step = 000177, loss = 0.001918
grad_step = 000178, loss = 0.001933
grad_step = 000179, loss = 0.001942
grad_step = 000180, loss = 0.001926
grad_step = 000181, loss = 0.001915
grad_step = 000182, loss = 0.001907
grad_step = 000183, loss = 0.001892
grad_step = 000184, loss = 0.001875
grad_step = 000185, loss = 0.001871
grad_step = 000186, loss = 0.001880
grad_step = 000187, loss = 0.001890
grad_step = 000188, loss = 0.001894
grad_step = 000189, loss = 0.001896
grad_step = 000190, loss = 0.001913
grad_step = 000191, loss = 0.001946
grad_step = 000192, loss = 0.001991
grad_step = 000193, loss = 0.001945
grad_step = 000194, loss = 0.001895
grad_step = 000195, loss = 0.001884
grad_step = 000196, loss = 0.001886
grad_step = 000197, loss = 0.001863
grad_step = 000198, loss = 0.001847
grad_step = 000199, loss = 0.001876
grad_step = 000200, loss = 0.001895
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001870
grad_step = 000202, loss = 0.001849
grad_step = 000203, loss = 0.001854
grad_step = 000204, loss = 0.001845
grad_step = 000205, loss = 0.001829
grad_step = 000206, loss = 0.001828
grad_step = 000207, loss = 0.001844
grad_step = 000208, loss = 0.001851
grad_step = 000209, loss = 0.001836
grad_step = 000210, loss = 0.001828
grad_step = 000211, loss = 0.001830
grad_step = 000212, loss = 0.001826
grad_step = 000213, loss = 0.001812
grad_step = 000214, loss = 0.001804
grad_step = 000215, loss = 0.001805
grad_step = 000216, loss = 0.001806
grad_step = 000217, loss = 0.001801
grad_step = 000218, loss = 0.001797
grad_step = 000219, loss = 0.001798
grad_step = 000220, loss = 0.001802
grad_step = 000221, loss = 0.001805
grad_step = 000222, loss = 0.001808
grad_step = 000223, loss = 0.001814
grad_step = 000224, loss = 0.001829
grad_step = 000225, loss = 0.001853
grad_step = 000226, loss = 0.001879
grad_step = 000227, loss = 0.001898
grad_step = 000228, loss = 0.001894
grad_step = 000229, loss = 0.001870
grad_step = 000230, loss = 0.001825
grad_step = 000231, loss = 0.001783
grad_step = 000232, loss = 0.001767
grad_step = 000233, loss = 0.001787
grad_step = 000234, loss = 0.001811
grad_step = 000235, loss = 0.001812
grad_step = 000236, loss = 0.001797
grad_step = 000237, loss = 0.001780
grad_step = 000238, loss = 0.001763
grad_step = 000239, loss = 0.001753
grad_step = 000240, loss = 0.001759
grad_step = 000241, loss = 0.001772
grad_step = 000242, loss = 0.001780
grad_step = 000243, loss = 0.001771
grad_step = 000244, loss = 0.001756
grad_step = 000245, loss = 0.001745
grad_step = 000246, loss = 0.001742
grad_step = 000247, loss = 0.001742
grad_step = 000248, loss = 0.001740
grad_step = 000249, loss = 0.001735
grad_step = 000250, loss = 0.001729
grad_step = 000251, loss = 0.001725
grad_step = 000252, loss = 0.001726
grad_step = 000253, loss = 0.001728
grad_step = 000254, loss = 0.001730
grad_step = 000255, loss = 0.001729
grad_step = 000256, loss = 0.001728
grad_step = 000257, loss = 0.001721
grad_step = 000258, loss = 0.001714
grad_step = 000259, loss = 0.001711
grad_step = 000260, loss = 0.001712
grad_step = 000261, loss = 0.001722
grad_step = 000262, loss = 0.001737
grad_step = 000263, loss = 0.001760
grad_step = 000264, loss = 0.001775
grad_step = 000265, loss = 0.001786
grad_step = 000266, loss = 0.001789
grad_step = 000267, loss = 0.001782
grad_step = 000268, loss = 0.001791
grad_step = 000269, loss = 0.001763
grad_step = 000270, loss = 0.001726
grad_step = 000271, loss = 0.001680
grad_step = 000272, loss = 0.001689
grad_step = 000273, loss = 0.001727
grad_step = 000274, loss = 0.001713
grad_step = 000275, loss = 0.001683
grad_step = 000276, loss = 0.001683
grad_step = 000277, loss = 0.001689
grad_step = 000278, loss = 0.001667
grad_step = 000279, loss = 0.001646
grad_step = 000280, loss = 0.001653
grad_step = 000281, loss = 0.001669
grad_step = 000282, loss = 0.001666
grad_step = 000283, loss = 0.001654
grad_step = 000284, loss = 0.001642
grad_step = 000285, loss = 0.001644
grad_step = 000286, loss = 0.001655
grad_step = 000287, loss = 0.001660
grad_step = 000288, loss = 0.001658
grad_step = 000289, loss = 0.001644
grad_step = 000290, loss = 0.001634
grad_step = 000291, loss = 0.001635
grad_step = 000292, loss = 0.001651
grad_step = 000293, loss = 0.001679
grad_step = 000294, loss = 0.001698
grad_step = 000295, loss = 0.001704
grad_step = 000296, loss = 0.001686
grad_step = 000297, loss = 0.001667
grad_step = 000298, loss = 0.001661
grad_step = 000299, loss = 0.001665
grad_step = 000300, loss = 0.001666
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001629
grad_step = 000302, loss = 0.001600
grad_step = 000303, loss = 0.001598
grad_step = 000304, loss = 0.001620
grad_step = 000305, loss = 0.001641
grad_step = 000306, loss = 0.001627
grad_step = 000307, loss = 0.001603
grad_step = 000308, loss = 0.001583
grad_step = 000309, loss = 0.001585
grad_step = 000310, loss = 0.001600
grad_step = 000311, loss = 0.001606
grad_step = 000312, loss = 0.001604
grad_step = 000313, loss = 0.001587
grad_step = 000314, loss = 0.001573
grad_step = 000315, loss = 0.001569
grad_step = 000316, loss = 0.001574
grad_step = 000317, loss = 0.001582
grad_step = 000318, loss = 0.001584
grad_step = 000319, loss = 0.001586
grad_step = 000320, loss = 0.001587
grad_step = 000321, loss = 0.001593
grad_step = 000322, loss = 0.001601
grad_step = 000323, loss = 0.001623
grad_step = 000324, loss = 0.001647
grad_step = 000325, loss = 0.001678
grad_step = 000326, loss = 0.001695
grad_step = 000327, loss = 0.001687
grad_step = 000328, loss = 0.001654
grad_step = 000329, loss = 0.001600
grad_step = 000330, loss = 0.001566
grad_step = 000331, loss = 0.001566
grad_step = 000332, loss = 0.001590
grad_step = 000333, loss = 0.001607
grad_step = 000334, loss = 0.001598
grad_step = 000335, loss = 0.001570
grad_step = 000336, loss = 0.001548
grad_step = 000337, loss = 0.001549
grad_step = 000338, loss = 0.001566
grad_step = 000339, loss = 0.001579
grad_step = 000340, loss = 0.001575
grad_step = 000341, loss = 0.001558
grad_step = 000342, loss = 0.001542
grad_step = 000343, loss = 0.001537
grad_step = 000344, loss = 0.001543
grad_step = 000345, loss = 0.001551
grad_step = 000346, loss = 0.001555
grad_step = 000347, loss = 0.001550
grad_step = 000348, loss = 0.001541
grad_step = 000349, loss = 0.001532
grad_step = 000350, loss = 0.001527
grad_step = 000351, loss = 0.001527
grad_step = 000352, loss = 0.001531
grad_step = 000353, loss = 0.001534
grad_step = 000354, loss = 0.001536
grad_step = 000355, loss = 0.001534
grad_step = 000356, loss = 0.001531
grad_step = 000357, loss = 0.001528
grad_step = 000358, loss = 0.001529
grad_step = 000359, loss = 0.001534
grad_step = 000360, loss = 0.001548
grad_step = 000361, loss = 0.001566
grad_step = 000362, loss = 0.001599
grad_step = 000363, loss = 0.001614
grad_step = 000364, loss = 0.001627
grad_step = 000365, loss = 0.001582
grad_step = 000366, loss = 0.001535
grad_step = 000367, loss = 0.001509
grad_step = 000368, loss = 0.001525
grad_step = 000369, loss = 0.001553
grad_step = 000370, loss = 0.001548
grad_step = 000371, loss = 0.001522
grad_step = 000372, loss = 0.001504
grad_step = 000373, loss = 0.001512
grad_step = 000374, loss = 0.001530
grad_step = 000375, loss = 0.001529
grad_step = 000376, loss = 0.001517
grad_step = 000377, loss = 0.001502
grad_step = 000378, loss = 0.001497
grad_step = 000379, loss = 0.001502
grad_step = 000380, loss = 0.001509
grad_step = 000381, loss = 0.001513
grad_step = 000382, loss = 0.001509
grad_step = 000383, loss = 0.001501
grad_step = 000384, loss = 0.001492
grad_step = 000385, loss = 0.001489
grad_step = 000386, loss = 0.001491
grad_step = 000387, loss = 0.001496
grad_step = 000388, loss = 0.001502
grad_step = 000389, loss = 0.001507
grad_step = 000390, loss = 0.001513
grad_step = 000391, loss = 0.001520
grad_step = 000392, loss = 0.001538
grad_step = 000393, loss = 0.001569
grad_step = 000394, loss = 0.001628
grad_step = 000395, loss = 0.001685
grad_step = 000396, loss = 0.001717
grad_step = 000397, loss = 0.001662
grad_step = 000398, loss = 0.001553
grad_step = 000399, loss = 0.001479
grad_step = 000400, loss = 0.001495
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001559
grad_step = 000402, loss = 0.001576
grad_step = 000403, loss = 0.001526
grad_step = 000404, loss = 0.001473
grad_step = 000405, loss = 0.001478
grad_step = 000406, loss = 0.001519
grad_step = 000407, loss = 0.001528
grad_step = 000408, loss = 0.001496
grad_step = 000409, loss = 0.001464
grad_step = 000410, loss = 0.001470
grad_step = 000411, loss = 0.001494
grad_step = 000412, loss = 0.001498
grad_step = 000413, loss = 0.001477
grad_step = 000414, loss = 0.001456
grad_step = 000415, loss = 0.001459
grad_step = 000416, loss = 0.001473
grad_step = 000417, loss = 0.001475
grad_step = 000418, loss = 0.001462
grad_step = 000419, loss = 0.001450
grad_step = 000420, loss = 0.001450
grad_step = 000421, loss = 0.001459
grad_step = 000422, loss = 0.001461
grad_step = 000423, loss = 0.001454
grad_step = 000424, loss = 0.001446
grad_step = 000425, loss = 0.001443
grad_step = 000426, loss = 0.001448
grad_step = 000427, loss = 0.001454
grad_step = 000428, loss = 0.001459
grad_step = 000429, loss = 0.001463
grad_step = 000430, loss = 0.001477
grad_step = 000431, loss = 0.001501
grad_step = 000432, loss = 0.001540
grad_step = 000433, loss = 0.001552
grad_step = 000434, loss = 0.001529
grad_step = 000435, loss = 0.001457
grad_step = 000436, loss = 0.001441
grad_step = 000437, loss = 0.001479
grad_step = 000438, loss = 0.001485
grad_step = 000439, loss = 0.001454
grad_step = 000440, loss = 0.001429
grad_step = 000441, loss = 0.001448
grad_step = 000442, loss = 0.001471
grad_step = 000443, loss = 0.001451
grad_step = 000444, loss = 0.001429
grad_step = 000445, loss = 0.001429
grad_step = 000446, loss = 0.001443
grad_step = 000447, loss = 0.001448
grad_step = 000448, loss = 0.001434
grad_step = 000449, loss = 0.001420
grad_step = 000450, loss = 0.001418
grad_step = 000451, loss = 0.001427
grad_step = 000452, loss = 0.001433
grad_step = 000453, loss = 0.001428
grad_step = 000454, loss = 0.001419
grad_step = 000455, loss = 0.001412
grad_step = 000456, loss = 0.001414
grad_step = 000457, loss = 0.001419
grad_step = 000458, loss = 0.001421
grad_step = 000459, loss = 0.001419
grad_step = 000460, loss = 0.001414
grad_step = 000461, loss = 0.001411
grad_step = 000462, loss = 0.001414
grad_step = 000463, loss = 0.001420
grad_step = 000464, loss = 0.001429
grad_step = 000465, loss = 0.001437
grad_step = 000466, loss = 0.001447
grad_step = 000467, loss = 0.001461
grad_step = 000468, loss = 0.001478
grad_step = 000469, loss = 0.001505
grad_step = 000470, loss = 0.001520
grad_step = 000471, loss = 0.001533
grad_step = 000472, loss = 0.001506
grad_step = 000473, loss = 0.001466
grad_step = 000474, loss = 0.001417
grad_step = 000475, loss = 0.001392
grad_step = 000476, loss = 0.001401
grad_step = 000477, loss = 0.001425
grad_step = 000478, loss = 0.001444
grad_step = 000479, loss = 0.001439
grad_step = 000480, loss = 0.001420
grad_step = 000481, loss = 0.001398
grad_step = 000482, loss = 0.001387
grad_step = 000483, loss = 0.001390
grad_step = 000484, loss = 0.001400
grad_step = 000485, loss = 0.001410
grad_step = 000486, loss = 0.001411
grad_step = 000487, loss = 0.001407
grad_step = 000488, loss = 0.001397
grad_step = 000489, loss = 0.001387
grad_step = 000490, loss = 0.001379
grad_step = 000491, loss = 0.001375
grad_step = 000492, loss = 0.001375
grad_step = 000493, loss = 0.001378
grad_step = 000494, loss = 0.001382
grad_step = 000495, loss = 0.001386
grad_step = 000496, loss = 0.001390
grad_step = 000497, loss = 0.001392
grad_step = 000498, loss = 0.001393
grad_step = 000499, loss = 0.001392
grad_step = 000500, loss = 0.001390
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001386
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

  date_run                              2020-05-13 23:13:37.326319
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.257914
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 23:13:37.332685
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.180954
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 23:13:37.340843
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.15212
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 23:13:37.346465
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.74966
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
0   2020-05-13 23:13:07.749224  ...    mean_absolute_error
1   2020-05-13 23:13:07.753827  ...     mean_squared_error
2   2020-05-13 23:13:07.757894  ...  median_absolute_error
3   2020-05-13 23:13:07.763485  ...               r2_score
4   2020-05-13 23:13:16.971004  ...    mean_absolute_error
5   2020-05-13 23:13:16.974666  ...     mean_squared_error
6   2020-05-13 23:13:16.977874  ...  median_absolute_error
7   2020-05-13 23:13:16.981308  ...               r2_score
8   2020-05-13 23:13:37.326319  ...    mean_absolute_error
9   2020-05-13 23:13:37.332685  ...     mean_squared_error
10  2020-05-13 23:13:37.340843  ...  median_absolute_error
11  2020-05-13 23:13:37.346465  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f765bdfffd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:30, 319424.41it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 411996.50it/s]  9%|▉         | 876544/9912422 [00:00<00:15, 570130.33it/s] 31%|███       | 3047424/9912422 [00:00<00:08, 803484.36it/s] 59%|█████▉    | 5824512/9912422 [00:00<00:03, 1130661.61it/s] 89%|████████▉ | 8855552/9912422 [00:01<00:00, 1586512.25it/s]9920512it [00:01, 9617826.94it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 152554.34it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 316022.33it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 408335.50it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 565306.86it/s]1654784it [00:00, 2816400.57it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 51887.49it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f760e802e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f760de300b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f760e802e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f760dd870f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f760b5c34e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f760b5adc50> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f760e802e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f760dd45710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f760b5c34e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f760de30128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7ff385894240> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=a9cc294f59941700d5301820f0558a2c5bf86be6463f14aeba9d65a873b87d43
  Stored in directory: /tmp/pip-ephem-wheel-cache-l408l2_w/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ff31d68e7f0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 47s
   57344/17464789 [..............................] - ETA: 40s
   90112/17464789 [..............................] - ETA: 38s
  180224/17464789 [..............................] - ETA: 25s
  352256/17464789 [..............................] - ETA: 16s
  679936/17464789 [>.............................] - ETA: 9s 
 1359872/17464789 [=>............................] - ETA: 5s
 2695168/17464789 [===>..........................] - ETA: 2s
 5218304/17464789 [=======>......................] - ETA: 1s
 7741440/17464789 [============>.................] - ETA: 0s
10248192/17464789 [================>.............] - ETA: 0s
12754944/17464789 [====================>.........] - ETA: 0s
15294464/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 23:15:08.054726: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 23:15:08.058715: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095070000 Hz
2020-05-13 23:15:08.058865: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5634775e5500 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 23:15:08.058880: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6513 - accuracy: 0.5010
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6053 - accuracy: 0.5040 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6411 - accuracy: 0.5017
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6605 - accuracy: 0.5004
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7050 - accuracy: 0.4975
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7148 - accuracy: 0.4969
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7337 - accuracy: 0.4956
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7245 - accuracy: 0.4962
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7326 - accuracy: 0.4957
11000/25000 [============>.................] - ETA: 3s - loss: 7.7224 - accuracy: 0.4964
12000/25000 [=============>................] - ETA: 3s - loss: 7.7126 - accuracy: 0.4970
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7103 - accuracy: 0.4972
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6688 - accuracy: 0.4999
15000/25000 [=================>............] - ETA: 2s - loss: 7.6697 - accuracy: 0.4998
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6522 - accuracy: 0.5009
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6513 - accuracy: 0.5010
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6743 - accuracy: 0.4995
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6771 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6689 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6466 - accuracy: 0.5013
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
25000/25000 [==============================] - 7s 297us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 23:15:22.388945
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 23:15:22.388945  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<11:54:01, 20.1kB/s].vector_cache/glove.6B.zip:   0%|          | 270k/862M [00:00<8:21:19, 28.7kB/s]  .vector_cache/glove.6B.zip:   0%|          | 3.38M/862M [00:00<5:49:47, 40.9kB/s].vector_cache/glove.6B.zip:   1%|          | 7.80M/862M [00:00<4:03:41, 58.4kB/s].vector_cache/glove.6B.zip:   1%|▏         | 10.9M/862M [00:00<2:50:05, 83.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.9M/862M [00:00<1:58:36, 119kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.1M/862M [00:01<1:22:36, 170kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.4M/862M [00:01<57:32, 242kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.5M/862M [00:01<40:03, 346kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.9M/862M [00:01<27:58, 492kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.5M/862M [00:01<19:31, 700kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.6M/862M [00:01<13:42, 993kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.1M/862M [00:01<09:36, 1.41MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:01<08:13, 1.64MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:03<07:38, 1.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:03<06:22, 2.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:04<04:34, 2.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:05<15:44, 849kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:05<12:20, 1.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:06<08:42, 1.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:07<23:43, 561kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:07<17:52, 744kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:08<12:35, 1.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:09<20:03, 659kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:09<15:09, 872kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.7M/862M [00:09<10:42, 1.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:11<15:50, 831kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:11<12:09, 1.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:13<10:26, 1.25MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:13<08:23, 1.56MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:13<05:58, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:14<07:31, 1.73MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:15<10:44:10, 20.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:15<7:31:04, 28.9kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:17<5:16:28, 40.9kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:17<3:42:20, 58.2kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.1M/862M [00:17<2:35:10, 83.1kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:19<1:57:41, 110kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:19<1:23:24, 154kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:21<1:00:01, 214kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:21<43:16, 296kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:21<30:15, 421kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:23<47:45, 267kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:23<35:20, 361kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:23<24:46, 513kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<24:19, 521kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<18:53, 671kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:25<13:17, 950kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<17:16, 730kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<13:52, 908kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:27<09:48, 1.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<14:43, 852kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<12:15, 1.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:29<08:40, 1.44MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<12:28, 1.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<10:44, 1.16MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:31<07:36, 1.63MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<12:08, 1.02MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<10:15, 1.21MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:33<07:17, 1.70MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<11:22, 1.08MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<09:19, 1.32MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:35<06:41, 1.84MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<08:16, 1.48MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<07:20, 1.67MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:37<05:20, 2.29MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<06:47, 1.80MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<06:28, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:39<04:40, 2.60MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<07:25, 1.64MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<06:14, 1.94MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<04:32, 2.66MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<06:50, 1.76MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<05:42, 2.11MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<04:07, 2.92MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<08:48, 1.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<07:52, 1.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:45<05:36, 2.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<10:27, 1.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<08:17, 1.44MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<07:33, 1.57MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<06:18, 1.88MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:09, 1.92MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:58, 1.97MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<04:16, 2.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<11:19, 1.03MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<09:38, 1.21MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<06:50, 1.71MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<11:37, 1.00MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<09:50, 1.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:55<06:58, 1.66MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<11:10, 1.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<09:28, 1.22MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:57<06:42, 1.72MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<12:17, 937kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<10:13, 1.13MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<07:14, 1.58MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<12:27, 918kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<10:13, 1.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<07:14, 1.57MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<11:45, 968kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<09:46, 1.16MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:03<06:55, 1.64MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<12:37, 895kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<10:25, 1.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:04<07:22, 1.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<12:16, 915kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<10:10, 1.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:06<07:12, 1.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<11:40, 957kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<09:49, 1.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:08<07:03, 1.58MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<05:31, 2.01MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<8:56:14, 20.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<6:15:32, 29.5kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:10<4:22:26, 42.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:10<3:03:37, 60.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<2:17:20, 80.3kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<1:37:43, 113kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<1:08:26, 161kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<50:17, 218kB/s]  .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<36:48, 298kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<25:52, 423kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<20:51, 523kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<15:33, 700kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:16<11:05, 979kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<10:11, 1.06MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<08:40, 1.25MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:18<06:17, 1.72MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:18<04:35, 2.35MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<10:06, 1.06MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<08:03, 1.33MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:20<05:49, 1.84MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<06:39, 1.61MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<06:14, 1.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:22<04:31, 2.35MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<06:33, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<05:50, 1.82MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:24<04:10, 2.53MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<11:01, 957kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<09:14, 1.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:26<06:33, 1.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<09:59, 1.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<08:24, 1.25MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:28<05:57, 1.75MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<10:26, 998kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<08:49, 1.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:30<06:15, 1.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<10:47, 959kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<09:02, 1.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:32<06:23, 1.61MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<11:07, 924kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:34<09:11, 1.12MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:34<06:31, 1.57MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<08:25, 1.21MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<07:21, 1.39MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:36<05:15, 1.93MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<07:27, 1.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<06:39, 1.52MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:38<04:44, 2.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<08:29, 1.19MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<07:26, 1.35MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:40<05:17, 1.90MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<08:48, 1.14MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<07:34, 1.32MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<05:23, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<08:06, 1.22MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<07:06, 1.40MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<05:08, 1.92MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:52, 1.68MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:18, 1.86MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<03:48, 2.58MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<07:43, 1.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<06:47, 1.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:48<04:51, 2.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<06:43, 1.45MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:36, 1.73MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:50<04:04, 2.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:34, 1.73MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:02, 1.91MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<03:37, 2.65MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<06:27, 1.48MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:57, 1.61MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<04:14, 2.25MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<08:33, 1.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<07:22, 1.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<05:14, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<08:29, 1.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:44, 1.40MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:58<04:48, 1.96MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<08:15, 1.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<07:02, 1.33MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:00<05:00, 1.86MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<08:35, 1.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<06:57, 1.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<04:55, 1.88MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<15:15, 606kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<12:03, 767kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<08:29, 1.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<10:56, 839kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<09:01, 1.02MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<06:24, 1.42MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<07:20, 1.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<06:26, 1.41MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<04:35, 1.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<06:58, 1.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:11, 1.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<04:24, 2.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<06:36, 1.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:54, 1.52MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<04:12, 2.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<07:59, 1.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<07:04, 1.26MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<05:06, 1.74MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<05:32, 1.59MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<05:13, 1.69MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<03:44, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<05:32, 1.58MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<05:04, 1.73MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<03:38, 2.39MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<05:47, 1.50MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<05:18, 1.64MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:19<03:47, 2.28MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<07:42, 1.12MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<06:37, 1.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:21<04:41, 1.83MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<08:27, 1.01MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<07:13, 1.19MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:23<05:06, 1.66MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<08:01, 1.06MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<06:52, 1.24MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<04:53, 1.73MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<06:29, 1.30MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<05:44, 1.47MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:27<04:04, 2.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<06:54, 1.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<05:25, 1.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:29<03:52, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<08:23, 988kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<07:04, 1.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:31<05:02, 1.63MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<06:02, 1.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<05:23, 1.52MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:33<03:49, 2.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<07:38, 1.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<06:27, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<04:37, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<05:31, 1.46MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<04:42, 1.71MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:37<03:21, 2.39MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<06:43, 1.19MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<05:51, 1.37MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:39<04:12, 1.89MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<04:55, 1.61MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<04:34, 1.74MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<03:15, 2.42MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<06:04, 1.30MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<05:25, 1.45MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:43<03:53, 2.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<05:05, 1.53MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:41, 1.66MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<03:22, 2.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<05:03, 1.53MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:39, 1.66MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<03:19, 2.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<06:30, 1.18MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:42, 1.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:49<04:02, 1.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<08:02, 944kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<06:42, 1.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:51<04:44, 1.59MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<07:53, 953kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<06:36, 1.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:53<04:40, 1.60MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<07:22, 1.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<06:17, 1.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<04:27, 1.66MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:53, 1.25MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:15, 1.40MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<03:46, 1.95MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:41, 1.56MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:23, 1.66MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<03:07, 2.32MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<06:13, 1.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<05:25, 1.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<03:52, 1.86MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<05:13, 1.37MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<04:42, 1.52MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<03:21, 2.12MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:59, 1.43MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<04:23, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<03:11, 2.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:56, 1.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:45, 1.87MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:46, 2.52MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<02:02, 3.41MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<07:14, 963kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<05:37, 1.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<04:02, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<04:42, 1.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:16, 1.62MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:06, 2.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<02:20, 2.93MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<04:43, 1.45MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:59, 1.71MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:13<02:49, 2.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<36:12, 187kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<26:20, 257kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<18:24, 365kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<15:11, 441kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<11:38, 575kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<08:10, 814kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<08:38, 767kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<07:03, 940kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<04:58, 1.32MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<06:27, 1.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<05:28, 1.20MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<03:52, 1.68MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<05:53, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<05:02, 1.29MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:22<03:35, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<04:32, 1.42MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<04:05, 1.57MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:24<02:54, 2.19MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<05:06, 1.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<04:06, 1.55MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:26<02:57, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<03:48, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<03:33, 1.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:28<02:32, 2.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<04:13, 1.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:30, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<02:33, 2.42MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:13, 1.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:07, 1.96MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:32<02:14, 2.73MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:54, 1.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:22, 1.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:34<03:07, 1.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<04:13, 1.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<03:46, 1.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<02:44, 2.19MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<03:20, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<03:14, 1.83MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<02:20, 2.52MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:21, 1.75MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:06, 1.89MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:40<02:12, 2.63MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<04:57, 1.17MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<04:17, 1.35MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:42<03:02, 1.89MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<04:46, 1.20MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<04:10, 1.38MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:44<02:57, 1.93MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<04:56, 1.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<04:10, 1.35MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:46<02:58, 1.89MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:22, 1.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:51, 1.45MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:48<02:43, 2.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<06:29, 853kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<04:57, 1.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:50<03:29, 1.57MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<06:30, 840kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<05:02, 1.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:52<03:32, 1.53MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<09:54, 544kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<07:41, 701kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<05:25, 987kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<05:20, 996kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<04:19, 1.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<03:05, 1.71MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:35, 1.46MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:00, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [03:58<02:09, 2.42MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:26, 1.51MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:47, 1.85MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<02:00, 2.56MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:12, 1.59MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:57, 1.73MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<02:06, 2.41MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:27, 1.46MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:07, 1.61MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<02:14, 2.23MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:08, 1.59MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:41, 1.84MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<01:54, 2.57MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<04:56, 993kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:56, 1.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:08<02:46, 1.75MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<05:23, 899kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<04:28, 1.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<03:09, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<05:26, 876kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:26, 1.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:12<03:06, 1.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<08:38, 544kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<06:38, 708kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<04:38, 1.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<07:31, 616kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<05:50, 792kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:16<04:05, 1.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<07:04, 645kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<05:25, 841kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:18<03:47, 1.19MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<07:56, 566kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<05:56, 757kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<04:09, 1.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<06:28, 683kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<05:09, 857kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<03:37, 1.21MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<04:11, 1.04MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:30, 1.24MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:24<02:27, 1.75MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<06:29, 660kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<04:51, 880kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<03:59, 1.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:13, 1.31MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<02:16, 1.83MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<03:40, 1.13MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<02:56, 1.41MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<02:05, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:55, 1.40MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:27, 1.66MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:45, 2.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:57, 1.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:33, 1.56MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:48, 2.19MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<03:46, 1.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<03:09, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:35<02:13, 1.75MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<04:00, 969kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<03:05, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:37<02:10, 1.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:39<04:22, 869kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<03:19, 1.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:39<02:19, 1.61MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<06:11, 604kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<04:48, 778kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:41<03:20, 1.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<07:23, 497kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<05:38, 650kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:43<03:55, 920kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<06:03, 594kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<04:37, 779kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:45<03:12, 1.10MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<08:06, 436kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<06:05, 580kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:47<04:12, 823kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<16:58, 204kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<12:21, 280kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:49<08:32, 399kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<09:06, 373kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<06:50, 495kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<04:46, 702kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<04:34, 727kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<03:34, 931kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:53<02:29, 1.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<03:42, 879kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<03:00, 1.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:55<02:05, 1.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<03:42, 861kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<02:49, 1.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:57<01:58, 1.59MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<04:10, 746kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<03:10, 983kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<02:13, 1.38MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<03:20, 915kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<02:41, 1.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:52, 1.59MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:52, 1.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:05, 1.42MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:58, 1.48MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:49, 1.60MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:05<01:16, 2.24MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<03:33, 801kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<02:53, 985kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<02:00, 1.39MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<05:08, 540kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<04:00, 692kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<02:46, 980kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<04:01, 674kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<03:10, 851kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:11<02:12, 1.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<04:03, 652kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<03:12, 821kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<02:13, 1.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<03:10, 809kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:35, 991kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:15<01:47, 1.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<04:41, 534kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<03:29, 714kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<02:25, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<04:33, 534kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<03:31, 689kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<02:26, 974kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<03:11, 740kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<02:34, 917kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<01:48, 1.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:03, 1.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:45, 1.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:23<01:13, 1.83MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 729M/862M [05:25<03:04, 725kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<02:22, 937kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<01:39, 1.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:53, 1.14MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:38, 1.31MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<01:09, 1.83MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:38, 1.27MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:26, 1.44MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<01:00, 2.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<02:23, 843kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:56, 1.04MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<01:20, 1.47MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<04:10, 468kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<03:06, 626kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<02:22, 791kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:48, 1.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<01:15, 1.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:39, 1.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:21, 1.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:37<00:56, 1.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<02:10, 800kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:36, 1.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<01:06, 1.52MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<02:15, 741kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<01:45, 952kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:24, 1.14MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:07, 1.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:46, 1.99MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:36, 961kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:19, 1.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:44<00:54, 1.64MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<02:08, 687kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:41, 869kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:46<01:08, 1.23MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<02:04, 675kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:36, 868kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:48<01:06, 1.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:29, 893kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<01:09, 1.14MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:50<00:47, 1.60MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:46, 713kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:52<01:21, 927kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:52<00:55, 1.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<02:01, 591kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<01:33, 762kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:54<01:03, 1.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<01:36, 700kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<01:14, 900kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:56<00:50, 1.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<03:26, 308kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<02:30, 419kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:58<01:39, 596kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<03:13, 306kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<02:22, 414kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:00<01:35, 588kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<01:34, 584kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<01:10, 779kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:53, 955kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:40, 1.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:33, 1.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:26, 1.76MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:23, 1.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:21, 1.96MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:19, 2.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:16, 2.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:15, 2.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:13, 2.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:12<00:08, 3.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<01:09, 437kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:50, 592kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:34, 752kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:27, 951kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:19, 1.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:16, 1.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:18<00:09, 1.88MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<01:25, 211kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<01:00, 294kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:20<00:34, 418kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:38, 363kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:27, 495kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:15, 701kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:13, 719kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:10, 910kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:06, 1.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:24<00:03, 1.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:04, 1.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.41MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.55MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.63MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 792/400000 [00:00<00:50, 7917.84it/s]  0%|          | 1647/400000 [00:00<00:49, 8097.10it/s]  1%|          | 2497/400000 [00:00<00:48, 8212.15it/s]  1%|          | 3326/400000 [00:00<00:48, 8235.11it/s]  1%|          | 4194/400000 [00:00<00:47, 8361.93it/s]  1%|▏         | 5049/400000 [00:00<00:46, 8415.64it/s]  1%|▏         | 5895/400000 [00:00<00:46, 8427.66it/s]  2%|▏         | 6744/400000 [00:00<00:46, 8443.42it/s]  2%|▏         | 7600/400000 [00:00<00:46, 8476.67it/s]  2%|▏         | 8457/400000 [00:01<00:46, 8502.61it/s]  2%|▏         | 9286/400000 [00:01<00:46, 8434.88it/s]  3%|▎         | 10132/400000 [00:01<00:46, 8440.39it/s]  3%|▎         | 10976/400000 [00:01<00:46, 8439.11it/s]  3%|▎         | 11823/400000 [00:01<00:45, 8448.13it/s]  3%|▎         | 12663/400000 [00:01<00:46, 8385.52it/s]  3%|▎         | 13532/400000 [00:01<00:45, 8474.13it/s]  4%|▎         | 14395/400000 [00:01<00:45, 8517.95it/s]  4%|▍         | 15258/400000 [00:01<00:45, 8549.65it/s]  4%|▍         | 16132/400000 [00:01<00:44, 8598.95it/s]  4%|▍         | 16992/400000 [00:02<00:45, 8383.70it/s]  4%|▍         | 17850/400000 [00:02<00:45, 8441.59it/s]  5%|▍         | 18724/400000 [00:02<00:44, 8528.71it/s]  5%|▍         | 19601/400000 [00:02<00:44, 8597.86it/s]  5%|▌         | 20480/400000 [00:02<00:43, 8654.03it/s]  5%|▌         | 21346/400000 [00:02<00:44, 8575.05it/s]  6%|▌         | 22205/400000 [00:02<00:44, 8529.87it/s]  6%|▌         | 23062/400000 [00:02<00:44, 8541.25it/s]  6%|▌         | 23939/400000 [00:02<00:43, 8607.05it/s]  6%|▌         | 24801/400000 [00:02<00:44, 8377.23it/s]  6%|▋         | 25641/400000 [00:03<00:45, 8295.91it/s]  7%|▋         | 26501/400000 [00:03<00:44, 8383.68it/s]  7%|▋         | 27342/400000 [00:03<00:44, 8390.30it/s]  7%|▋         | 28201/400000 [00:03<00:44, 8448.10it/s]  7%|▋         | 29069/400000 [00:03<00:43, 8514.62it/s]  7%|▋         | 29922/400000 [00:03<00:44, 8297.02it/s]  8%|▊         | 30754/400000 [00:03<00:45, 8095.39it/s]  8%|▊         | 31579/400000 [00:03<00:45, 8139.22it/s]  8%|▊         | 32445/400000 [00:03<00:44, 8287.87it/s]  8%|▊         | 33296/400000 [00:03<00:43, 8351.65it/s]  9%|▊         | 34158/400000 [00:04<00:43, 8430.18it/s]  9%|▉         | 35006/400000 [00:04<00:43, 8442.73it/s]  9%|▉         | 35872/400000 [00:04<00:42, 8504.74it/s]  9%|▉         | 36743/400000 [00:04<00:42, 8563.21it/s]  9%|▉         | 37608/400000 [00:04<00:42, 8587.08it/s] 10%|▉         | 38468/400000 [00:04<00:42, 8566.13it/s] 10%|▉         | 39327/400000 [00:04<00:42, 8573.12it/s] 10%|█         | 40185/400000 [00:04<00:42, 8400.83it/s] 10%|█         | 41056/400000 [00:04<00:42, 8488.57it/s] 10%|█         | 41917/400000 [00:04<00:42, 8524.26it/s] 11%|█         | 42774/400000 [00:05<00:41, 8536.80it/s] 11%|█         | 43629/400000 [00:05<00:42, 8479.49it/s] 11%|█         | 44478/400000 [00:05<00:42, 8399.85it/s] 11%|█▏        | 45336/400000 [00:05<00:41, 8450.00it/s] 12%|█▏        | 46195/400000 [00:05<00:41, 8489.11it/s] 12%|█▏        | 47047/400000 [00:05<00:41, 8497.61it/s] 12%|█▏        | 47897/400000 [00:05<00:41, 8472.47it/s] 12%|█▏        | 48745/400000 [00:05<00:41, 8461.72it/s] 12%|█▏        | 49600/400000 [00:05<00:41, 8487.64it/s] 13%|█▎        | 50451/400000 [00:05<00:41, 8492.37it/s] 13%|█▎        | 51314/400000 [00:06<00:40, 8531.15it/s] 13%|█▎        | 52168/400000 [00:06<00:40, 8495.90it/s] 13%|█▎        | 53025/400000 [00:06<00:40, 8516.33it/s] 13%|█▎        | 53877/400000 [00:06<00:40, 8503.94it/s] 14%|█▎        | 54728/400000 [00:06<00:40, 8499.93it/s] 14%|█▍        | 55592/400000 [00:06<00:40, 8540.40it/s] 14%|█▍        | 56464/400000 [00:06<00:39, 8592.87it/s] 14%|█▍        | 57335/400000 [00:06<00:39, 8626.90it/s] 15%|█▍        | 58208/400000 [00:06<00:39, 8656.71it/s] 15%|█▍        | 59086/400000 [00:06<00:39, 8692.64it/s] 15%|█▍        | 59963/400000 [00:07<00:39, 8712.78it/s] 15%|█▌        | 60837/400000 [00:07<00:38, 8718.80it/s] 15%|█▌        | 61709/400000 [00:07<00:38, 8713.22it/s] 16%|█▌        | 62584/400000 [00:07<00:38, 8724.10it/s] 16%|█▌        | 63463/400000 [00:07<00:38, 8740.94it/s] 16%|█▌        | 64342/400000 [00:07<00:38, 8753.93it/s] 16%|█▋        | 65218/400000 [00:07<00:38, 8727.82it/s] 17%|█▋        | 66091/400000 [00:07<00:38, 8717.58it/s] 17%|█▋        | 66966/400000 [00:07<00:38, 8724.45it/s] 17%|█▋        | 67844/400000 [00:07<00:38, 8738.92it/s] 17%|█▋        | 68723/400000 [00:08<00:37, 8753.21it/s] 17%|█▋        | 69600/400000 [00:08<00:37, 8755.54it/s] 18%|█▊        | 70476/400000 [00:08<00:37, 8749.58it/s] 18%|█▊        | 71351/400000 [00:08<00:37, 8744.08it/s] 18%|█▊        | 72226/400000 [00:08<00:37, 8719.13it/s] 18%|█▊        | 73100/400000 [00:08<00:37, 8723.25it/s] 18%|█▊        | 73973/400000 [00:08<00:37, 8708.62it/s] 19%|█▊        | 74844/400000 [00:08<00:38, 8386.85it/s] 19%|█▉        | 75706/400000 [00:08<00:38, 8451.44it/s] 19%|█▉        | 76583/400000 [00:08<00:37, 8542.55it/s] 19%|█▉        | 77462/400000 [00:09<00:37, 8611.36it/s] 20%|█▉        | 78338/400000 [00:09<00:37, 8653.54it/s] 20%|█▉        | 79215/400000 [00:09<00:36, 8688.14it/s] 20%|██        | 80085/400000 [00:09<00:37, 8565.27it/s] 20%|██        | 80943/400000 [00:09<00:37, 8476.28it/s] 20%|██        | 81798/400000 [00:09<00:37, 8496.10it/s] 21%|██        | 82669/400000 [00:09<00:37, 8557.35it/s] 21%|██        | 83527/400000 [00:09<00:36, 8561.41it/s] 21%|██        | 84384/400000 [00:09<00:37, 8494.30it/s] 21%|██▏       | 85241/400000 [00:09<00:36, 8516.36it/s] 22%|██▏       | 86102/400000 [00:10<00:36, 8543.05it/s] 22%|██▏       | 86957/400000 [00:10<00:36, 8497.23it/s] 22%|██▏       | 87820/400000 [00:10<00:36, 8534.51it/s] 22%|██▏       | 88680/400000 [00:10<00:36, 8552.11it/s] 22%|██▏       | 89542/400000 [00:10<00:36, 8569.55it/s] 23%|██▎       | 90412/400000 [00:10<00:35, 8606.89it/s] 23%|██▎       | 91273/400000 [00:10<00:36, 8543.80it/s] 23%|██▎       | 92128/400000 [00:10<00:36, 8515.46it/s] 23%|██▎       | 92980/400000 [00:10<00:36, 8512.58it/s] 23%|██▎       | 93832/400000 [00:11<00:36, 8442.66it/s] 24%|██▎       | 94688/400000 [00:11<00:36, 8476.12it/s] 24%|██▍       | 95543/400000 [00:11<00:35, 8496.62it/s] 24%|██▍       | 96403/400000 [00:11<00:35, 8525.40it/s] 24%|██▍       | 97264/400000 [00:11<00:35, 8548.95it/s] 25%|██▍       | 98135/400000 [00:11<00:35, 8596.54it/s] 25%|██▍       | 99013/400000 [00:11<00:34, 8649.65it/s] 25%|██▍       | 99883/400000 [00:11<00:34, 8663.20it/s] 25%|██▌       | 100755/400000 [00:11<00:34, 8679.08it/s] 25%|██▌       | 101629/400000 [00:11<00:34, 8694.57it/s] 26%|██▌       | 102499/400000 [00:12<00:34, 8639.55it/s] 26%|██▌       | 103364/400000 [00:12<00:34, 8516.34it/s] 26%|██▌       | 104217/400000 [00:12<00:34, 8517.72it/s] 26%|██▋       | 105073/400000 [00:12<00:34, 8529.73it/s] 26%|██▋       | 105933/400000 [00:12<00:34, 8549.29it/s] 27%|██▋       | 106789/400000 [00:12<00:34, 8494.90it/s] 27%|██▋       | 107639/400000 [00:12<00:34, 8382.26it/s] 27%|██▋       | 108484/400000 [00:12<00:34, 8402.45it/s] 27%|██▋       | 109331/400000 [00:12<00:34, 8421.20it/s] 28%|██▊       | 110174/400000 [00:12<00:34, 8378.73it/s] 28%|██▊       | 111019/400000 [00:13<00:34, 8399.18it/s] 28%|██▊       | 111860/400000 [00:13<00:35, 8230.50it/s] 28%|██▊       | 112684/400000 [00:13<00:35, 8111.67it/s] 28%|██▊       | 113497/400000 [00:13<00:36, 7806.03it/s] 29%|██▊       | 114281/400000 [00:13<00:37, 7661.16it/s] 29%|██▉       | 115121/400000 [00:13<00:36, 7868.79it/s] 29%|██▉       | 115984/400000 [00:13<00:35, 8080.73it/s] 29%|██▉       | 116847/400000 [00:13<00:34, 8235.83it/s] 29%|██▉       | 117688/400000 [00:13<00:34, 8286.13it/s] 30%|██▉       | 118547/400000 [00:13<00:33, 8372.81it/s] 30%|██▉       | 119420/400000 [00:14<00:33, 8476.20it/s] 30%|███       | 120270/400000 [00:14<00:33, 8431.28it/s] 30%|███       | 121115/400000 [00:14<00:34, 8085.02it/s] 30%|███       | 121928/400000 [00:14<00:35, 7841.77it/s] 31%|███       | 122762/400000 [00:14<00:34, 7984.67it/s] 31%|███       | 123577/400000 [00:14<00:34, 8032.13it/s] 31%|███       | 124448/400000 [00:14<00:33, 8222.46it/s] 31%|███▏      | 125313/400000 [00:14<00:32, 8343.66it/s] 32%|███▏      | 126190/400000 [00:14<00:32, 8466.52it/s] 32%|███▏      | 127059/400000 [00:14<00:31, 8530.47it/s] 32%|███▏      | 127934/400000 [00:15<00:31, 8594.49it/s] 32%|███▏      | 128795/400000 [00:15<00:32, 8311.29it/s] 32%|███▏      | 129630/400000 [00:15<00:33, 8170.50it/s] 33%|███▎      | 130450/400000 [00:15<00:33, 8081.61it/s] 33%|███▎      | 131261/400000 [00:15<00:33, 8054.66it/s] 33%|███▎      | 132068/400000 [00:15<00:33, 7965.44it/s] 33%|███▎      | 132866/400000 [00:15<00:34, 7828.55it/s] 33%|███▎      | 133651/400000 [00:15<00:34, 7813.00it/s] 34%|███▎      | 134434/400000 [00:15<00:34, 7758.82it/s] 34%|███▍      | 135258/400000 [00:16<00:33, 7896.88it/s] 34%|███▍      | 136138/400000 [00:16<00:32, 8147.04it/s] 34%|███▍      | 137019/400000 [00:16<00:31, 8333.05it/s] 34%|███▍      | 137895/400000 [00:16<00:31, 8454.63it/s] 35%|███▍      | 138770/400000 [00:16<00:30, 8540.61it/s] 35%|███▍      | 139627/400000 [00:16<00:30, 8540.03it/s] 35%|███▌      | 140504/400000 [00:16<00:30, 8607.49it/s] 35%|███▌      | 141382/400000 [00:16<00:29, 8656.76it/s] 36%|███▌      | 142258/400000 [00:16<00:29, 8684.65it/s] 36%|███▌      | 143128/400000 [00:16<00:29, 8674.59it/s] 36%|███▌      | 144006/400000 [00:17<00:29, 8704.11it/s] 36%|███▌      | 144885/400000 [00:17<00:29, 8729.41it/s] 36%|███▋      | 145759/400000 [00:17<00:29, 8716.02it/s] 37%|███▋      | 146631/400000 [00:17<00:29, 8705.22it/s] 37%|███▋      | 147502/400000 [00:17<00:30, 8282.43it/s] 37%|███▋      | 148335/400000 [00:17<00:30, 8294.77it/s] 37%|███▋      | 149196/400000 [00:17<00:29, 8385.15it/s] 38%|███▊      | 150060/400000 [00:17<00:29, 8458.48it/s] 38%|███▊      | 150908/400000 [00:17<00:29, 8461.92it/s] 38%|███▊      | 151764/400000 [00:17<00:29, 8489.97it/s] 38%|███▊      | 152614/400000 [00:18<00:29, 8466.91it/s] 38%|███▊      | 153462/400000 [00:18<00:29, 8459.34it/s] 39%|███▊      | 154315/400000 [00:18<00:28, 8478.39it/s] 39%|███▉      | 155179/400000 [00:18<00:28, 8526.08it/s] 39%|███▉      | 156035/400000 [00:18<00:28, 8534.08it/s] 39%|███▉      | 156915/400000 [00:18<00:28, 8609.52it/s] 39%|███▉      | 157786/400000 [00:18<00:28, 8636.83it/s] 40%|███▉      | 158650/400000 [00:18<00:28, 8463.80it/s] 40%|███▉      | 159511/400000 [00:18<00:28, 8504.61it/s] 40%|████      | 160363/400000 [00:18<00:28, 8344.67it/s] 40%|████      | 161199/400000 [00:19<00:29, 8106.39it/s] 41%|████      | 162012/400000 [00:19<00:29, 8012.67it/s] 41%|████      | 162816/400000 [00:19<00:29, 7989.48it/s] 41%|████      | 163683/400000 [00:19<00:28, 8180.47it/s] 41%|████      | 164504/400000 [00:19<00:28, 8151.03it/s] 41%|████▏     | 165321/400000 [00:19<00:29, 8041.46it/s] 42%|████▏     | 166127/400000 [00:19<00:29, 7810.36it/s] 42%|████▏     | 166917/400000 [00:19<00:29, 7835.91it/s] 42%|████▏     | 167767/400000 [00:19<00:28, 8023.59it/s] 42%|████▏     | 168619/400000 [00:19<00:28, 8164.63it/s] 42%|████▏     | 169474/400000 [00:20<00:27, 8274.95it/s] 43%|████▎     | 170309/400000 [00:20<00:27, 8296.72it/s] 43%|████▎     | 171141/400000 [00:20<00:28, 8001.38it/s] 43%|████▎     | 171971/400000 [00:20<00:28, 8087.67it/s] 43%|████▎     | 172805/400000 [00:20<00:27, 8161.52it/s] 43%|████▎     | 173641/400000 [00:20<00:27, 8217.41it/s] 44%|████▎     | 174492/400000 [00:20<00:27, 8302.58it/s] 44%|████▍     | 175370/400000 [00:20<00:26, 8439.45it/s] 44%|████▍     | 176245/400000 [00:20<00:26, 8525.95it/s] 44%|████▍     | 177099/400000 [00:20<00:26, 8524.02it/s] 44%|████▍     | 177980/400000 [00:21<00:25, 8606.03it/s] 45%|████▍     | 178860/400000 [00:21<00:25, 8661.61it/s] 45%|████▍     | 179742/400000 [00:21<00:25, 8706.35it/s] 45%|████▌     | 180621/400000 [00:21<00:25, 8730.53it/s] 45%|████▌     | 181495/400000 [00:21<00:25, 8607.68it/s] 46%|████▌     | 182357/400000 [00:21<00:25, 8536.69it/s] 46%|████▌     | 183212/400000 [00:21<00:25, 8527.38it/s] 46%|████▌     | 184075/400000 [00:21<00:25, 8555.15it/s] 46%|████▌     | 184953/400000 [00:21<00:24, 8619.75it/s] 46%|████▋     | 185816/400000 [00:22<00:24, 8610.93it/s] 47%|████▋     | 186693/400000 [00:22<00:24, 8657.41it/s] 47%|████▋     | 187571/400000 [00:22<00:24, 8692.52it/s] 47%|████▋     | 188450/400000 [00:22<00:24, 8719.32it/s] 47%|████▋     | 189323/400000 [00:22<00:24, 8691.96it/s] 48%|████▊     | 190193/400000 [00:22<00:24, 8436.88it/s] 48%|████▊     | 191039/400000 [00:22<00:25, 8247.76it/s] 48%|████▊     | 191866/400000 [00:22<00:25, 8041.86it/s] 48%|████▊     | 192685/400000 [00:22<00:25, 8084.02it/s] 48%|████▊     | 193556/400000 [00:22<00:25, 8257.59it/s] 49%|████▊     | 194421/400000 [00:23<00:24, 8368.85it/s] 49%|████▉     | 195295/400000 [00:23<00:24, 8475.91it/s] 49%|████▉     | 196165/400000 [00:23<00:23, 8540.48it/s] 49%|████▉     | 197038/400000 [00:23<00:23, 8594.61it/s] 49%|████▉     | 197913/400000 [00:23<00:23, 8638.48it/s] 50%|████▉     | 198778/400000 [00:23<00:23, 8582.15it/s] 50%|████▉     | 199654/400000 [00:23<00:23, 8632.82it/s] 50%|█████     | 200527/400000 [00:23<00:23, 8660.19it/s] 50%|█████     | 201394/400000 [00:23<00:24, 8147.91it/s] 51%|█████     | 202232/400000 [00:23<00:24, 8214.94it/s] 51%|█████     | 203059/400000 [00:24<00:24, 8139.35it/s] 51%|█████     | 203877/400000 [00:24<00:24, 8141.21it/s] 51%|█████     | 204752/400000 [00:24<00:23, 8312.30it/s] 51%|█████▏    | 205633/400000 [00:24<00:22, 8452.14it/s] 52%|█████▏    | 206503/400000 [00:24<00:22, 8523.68it/s] 52%|█████▏    | 207365/400000 [00:24<00:22, 8550.17it/s] 52%|█████▏    | 208224/400000 [00:24<00:22, 8561.20it/s] 52%|█████▏    | 209094/400000 [00:24<00:22, 8600.45it/s] 52%|█████▏    | 209955/400000 [00:24<00:22, 8593.21it/s] 53%|█████▎    | 210815/400000 [00:24<00:22, 8573.71it/s] 53%|█████▎    | 211673/400000 [00:25<00:22, 8296.37it/s] 53%|█████▎    | 212538/400000 [00:25<00:22, 8398.99it/s] 53%|█████▎    | 213408/400000 [00:25<00:21, 8486.32it/s] 54%|█████▎    | 214288/400000 [00:25<00:21, 8576.06it/s] 54%|█████▍    | 215147/400000 [00:25<00:21, 8418.27it/s] 54%|█████▍    | 216009/400000 [00:25<00:21, 8476.34it/s] 54%|█████▍    | 216877/400000 [00:25<00:21, 8533.67it/s] 54%|█████▍    | 217743/400000 [00:25<00:21, 8569.54it/s] 55%|█████▍    | 218608/400000 [00:25<00:21, 8590.67it/s] 55%|█████▍    | 219473/400000 [00:25<00:20, 8606.22it/s] 55%|█████▌    | 220338/400000 [00:26<00:20, 8619.23it/s] 55%|█████▌    | 221206/400000 [00:26<00:20, 8636.29it/s] 56%|█████▌    | 222070/400000 [00:26<00:20, 8590.63it/s] 56%|█████▌    | 222937/400000 [00:26<00:20, 8614.11it/s] 56%|█████▌    | 223799/400000 [00:26<00:20, 8591.04it/s] 56%|█████▌    | 224659/400000 [00:26<00:20, 8574.74it/s] 56%|█████▋    | 225519/400000 [00:26<00:20, 8580.42it/s] 57%|█████▋    | 226389/400000 [00:26<00:20, 8614.06it/s] 57%|█████▋    | 227251/400000 [00:26<00:20, 8574.89it/s] 57%|█████▋    | 228109/400000 [00:27<00:21, 8149.36it/s] 57%|█████▋    | 228929/400000 [00:27<00:21, 8055.81it/s] 57%|█████▋    | 229738/400000 [00:27<00:21, 7934.76it/s] 58%|█████▊    | 230535/400000 [00:27<00:21, 7916.31it/s] 58%|█████▊    | 231340/400000 [00:27<00:21, 7954.23it/s] 58%|█████▊    | 232197/400000 [00:27<00:20, 8127.24it/s] 58%|█████▊    | 233058/400000 [00:27<00:20, 8265.75it/s] 58%|█████▊    | 233920/400000 [00:27<00:19, 8367.42it/s] 59%|█████▊    | 234759/400000 [00:27<00:19, 8263.33it/s] 59%|█████▉    | 235587/400000 [00:27<00:20, 8148.27it/s] 59%|█████▉    | 236430/400000 [00:28<00:19, 8230.59it/s] 59%|█████▉    | 237300/400000 [00:28<00:19, 8365.61it/s] 60%|█████▉    | 238159/400000 [00:28<00:19, 8430.83it/s] 60%|█████▉    | 239004/400000 [00:28<00:19, 8306.89it/s] 60%|█████▉    | 239862/400000 [00:28<00:19, 8384.17it/s] 60%|██████    | 240709/400000 [00:28<00:18, 8407.25it/s] 60%|██████    | 241572/400000 [00:28<00:18, 8470.82it/s] 61%|██████    | 242442/400000 [00:28<00:18, 8537.14it/s] 61%|██████    | 243308/400000 [00:28<00:18, 8571.46it/s] 61%|██████    | 244167/400000 [00:28<00:18, 8576.14it/s] 61%|██████▏   | 245037/400000 [00:29<00:17, 8611.06it/s] 61%|██████▏   | 245903/400000 [00:29<00:17, 8625.68it/s] 62%|██████▏   | 246766/400000 [00:29<00:17, 8594.63it/s] 62%|██████▏   | 247626/400000 [00:29<00:17, 8588.30it/s] 62%|██████▏   | 248485/400000 [00:29<00:18, 8407.06it/s] 62%|██████▏   | 249327/400000 [00:29<00:17, 8404.49it/s] 63%|██████▎   | 250190/400000 [00:29<00:17, 8468.77it/s] 63%|██████▎   | 251047/400000 [00:29<00:17, 8498.01it/s] 63%|██████▎   | 251918/400000 [00:29<00:17, 8558.48it/s] 63%|██████▎   | 252790/400000 [00:29<00:17, 8603.77it/s] 63%|██████▎   | 253663/400000 [00:30<00:16, 8638.80it/s] 64%|██████▎   | 254533/400000 [00:30<00:16, 8656.27it/s] 64%|██████▍   | 255399/400000 [00:30<00:16, 8643.73it/s] 64%|██████▍   | 256264/400000 [00:30<00:16, 8615.52it/s] 64%|██████▍   | 257126/400000 [00:30<00:16, 8570.10it/s] 64%|██████▍   | 257991/400000 [00:30<00:16, 8592.14it/s] 65%|██████▍   | 258858/400000 [00:30<00:16, 8612.62it/s] 65%|██████▍   | 259723/400000 [00:30<00:16, 8622.29it/s] 65%|██████▌   | 260586/400000 [00:30<00:16, 8578.11it/s] 65%|██████▌   | 261450/400000 [00:30<00:16, 8595.71it/s] 66%|██████▌   | 262311/400000 [00:31<00:16, 8598.58it/s] 66%|██████▌   | 263180/400000 [00:31<00:15, 8622.88it/s] 66%|██████▌   | 264043/400000 [00:31<00:15, 8623.17it/s] 66%|██████▌   | 264906/400000 [00:31<00:15, 8583.76it/s] 66%|██████▋   | 265769/400000 [00:31<00:15, 8596.99it/s] 67%|██████▋   | 266629/400000 [00:31<00:15, 8546.18it/s] 67%|██████▋   | 267484/400000 [00:31<00:15, 8536.41it/s] 67%|██████▋   | 268338/400000 [00:31<00:15, 8500.92it/s] 67%|██████▋   | 269199/400000 [00:31<00:15, 8532.29it/s] 68%|██████▊   | 270053/400000 [00:31<00:15, 8478.79it/s] 68%|██████▊   | 270902/400000 [00:32<00:15, 8325.69it/s] 68%|██████▊   | 271736/400000 [00:32<00:15, 8132.57it/s] 68%|██████▊   | 272557/400000 [00:32<00:15, 8154.46it/s] 68%|██████▊   | 273374/400000 [00:32<00:15, 8136.19it/s] 69%|██████▊   | 274230/400000 [00:32<00:15, 8256.57it/s] 69%|██████▉   | 275082/400000 [00:32<00:14, 8331.64it/s] 69%|██████▉   | 275916/400000 [00:32<00:14, 8333.87it/s] 69%|██████▉   | 276781/400000 [00:32<00:14, 8424.66it/s] 69%|██████▉   | 277649/400000 [00:32<00:14, 8498.76it/s] 70%|██████▉   | 278515/400000 [00:32<00:14, 8544.57it/s] 70%|██████▉   | 279370/400000 [00:33<00:14, 8528.29it/s] 70%|███████   | 280224/400000 [00:33<00:14, 8427.97it/s] 70%|███████   | 281077/400000 [00:33<00:14, 8458.10it/s] 70%|███████   | 281924/400000 [00:33<00:13, 8434.17it/s] 71%|███████   | 282778/400000 [00:33<00:13, 8465.47it/s] 71%|███████   | 283629/400000 [00:33<00:13, 8478.29it/s] 71%|███████   | 284477/400000 [00:33<00:13, 8464.03it/s] 71%|███████▏  | 285324/400000 [00:33<00:13, 8382.00it/s] 72%|███████▏  | 286175/400000 [00:33<00:13, 8419.32it/s] 72%|███████▏  | 287027/400000 [00:33<00:13, 8448.24it/s] 72%|███████▏  | 287873/400000 [00:34<00:13, 8449.83it/s] 72%|███████▏  | 288719/400000 [00:34<00:13, 8443.69it/s] 72%|███████▏  | 289576/400000 [00:34<00:13, 8480.98it/s] 73%|███████▎  | 290437/400000 [00:34<00:12, 8517.59it/s] 73%|███████▎  | 291299/400000 [00:34<00:12, 8545.47it/s] 73%|███████▎  | 292154/400000 [00:34<00:12, 8501.97it/s] 73%|███████▎  | 293005/400000 [00:34<00:12, 8477.75it/s] 73%|███████▎  | 293855/400000 [00:34<00:12, 8482.03it/s] 74%|███████▎  | 294708/400000 [00:34<00:12, 8494.52it/s] 74%|███████▍  | 295564/400000 [00:34<00:12, 8511.91it/s] 74%|███████▍  | 296416/400000 [00:35<00:12, 8487.10it/s] 74%|███████▍  | 297274/400000 [00:35<00:12, 8513.85it/s] 75%|███████▍  | 298138/400000 [00:35<00:11, 8549.29it/s] 75%|███████▍  | 298994/400000 [00:35<00:11, 8543.85it/s] 75%|███████▍  | 299849/400000 [00:35<00:11, 8497.73it/s] 75%|███████▌  | 300699/400000 [00:35<00:11, 8307.41it/s] 75%|███████▌  | 301531/400000 [00:35<00:11, 8244.33it/s] 76%|███████▌  | 302385/400000 [00:35<00:11, 8329.84it/s] 76%|███████▌  | 303232/400000 [00:35<00:11, 8369.29it/s] 76%|███████▌  | 304085/400000 [00:36<00:11, 8416.40it/s] 76%|███████▌  | 304932/400000 [00:36<00:11, 8430.27it/s] 76%|███████▋  | 305776/400000 [00:36<00:11, 8409.32it/s] 77%|███████▋  | 306629/400000 [00:36<00:11, 8442.64it/s] 77%|███████▋  | 307480/400000 [00:36<00:10, 8461.78it/s] 77%|███████▋  | 308330/400000 [00:36<00:10, 8473.06it/s] 77%|███████▋  | 309178/400000 [00:36<00:10, 8465.74it/s] 78%|███████▊  | 310025/400000 [00:36<00:10, 8324.42it/s] 78%|███████▊  | 310862/400000 [00:36<00:10, 8336.02it/s] 78%|███████▊  | 311719/400000 [00:36<00:10, 8402.43it/s] 78%|███████▊  | 312571/400000 [00:37<00:10, 8437.29it/s] 78%|███████▊  | 313416/400000 [00:37<00:10, 8438.48it/s] 79%|███████▊  | 314271/400000 [00:37<00:10, 8469.47it/s] 79%|███████▉  | 315131/400000 [00:37<00:09, 8505.75it/s] 79%|███████▉  | 315982/400000 [00:37<00:09, 8492.99it/s] 79%|███████▉  | 316832/400000 [00:37<00:10, 8264.16it/s] 79%|███████▉  | 317664/400000 [00:37<00:09, 8279.48it/s] 80%|███████▉  | 318493/400000 [00:37<00:09, 8263.20it/s] 80%|███████▉  | 319335/400000 [00:37<00:09, 8306.86it/s] 80%|████████  | 320180/400000 [00:37<00:09, 8346.98it/s] 80%|████████  | 321028/400000 [00:38<00:09, 8383.72it/s] 80%|████████  | 321875/400000 [00:38<00:09, 8405.49it/s] 81%|████████  | 322716/400000 [00:38<00:09, 8394.22it/s] 81%|████████  | 323572/400000 [00:38<00:09, 8441.00it/s] 81%|████████  | 324417/400000 [00:38<00:09, 8389.40it/s] 81%|████████▏ | 325257/400000 [00:38<00:08, 8389.93it/s] 82%|████████▏ | 326097/400000 [00:38<00:08, 8391.71it/s] 82%|████████▏ | 326942/400000 [00:38<00:08, 8406.85it/s] 82%|████████▏ | 327783/400000 [00:38<00:08, 8324.67it/s] 82%|████████▏ | 328616/400000 [00:38<00:08, 8259.21it/s] 82%|████████▏ | 329443/400000 [00:39<00:08, 8186.22it/s] 83%|████████▎ | 330263/400000 [00:39<00:08, 8188.72it/s] 83%|████████▎ | 331133/400000 [00:39<00:08, 8333.43it/s] 83%|████████▎ | 331978/400000 [00:39<00:08, 8367.98it/s] 83%|████████▎ | 332825/400000 [00:39<00:08, 8395.75it/s] 83%|████████▎ | 333681/400000 [00:39<00:07, 8443.71it/s] 84%|████████▎ | 334543/400000 [00:39<00:07, 8493.12it/s] 84%|████████▍ | 335399/400000 [00:39<00:07, 8509.91it/s] 84%|████████▍ | 336251/400000 [00:39<00:07, 8274.60it/s] 84%|████████▍ | 337111/400000 [00:39<00:07, 8367.26it/s] 84%|████████▍ | 337950/400000 [00:40<00:07, 8301.86it/s] 85%|████████▍ | 338800/400000 [00:40<00:07, 8360.02it/s] 85%|████████▍ | 339648/400000 [00:40<00:07, 8394.65it/s] 85%|████████▌ | 340489/400000 [00:40<00:07, 8305.23it/s] 85%|████████▌ | 341321/400000 [00:40<00:07, 8300.67it/s] 86%|████████▌ | 342161/400000 [00:40<00:06, 8329.49it/s] 86%|████████▌ | 343024/400000 [00:40<00:06, 8414.66it/s] 86%|████████▌ | 343867/400000 [00:40<00:06, 8417.41it/s] 86%|████████▌ | 344710/400000 [00:40<00:06, 8318.56it/s] 86%|████████▋ | 345563/400000 [00:40<00:06, 8380.18it/s] 87%|████████▋ | 346416/400000 [00:41<00:06, 8397.37it/s] 87%|████████▋ | 347277/400000 [00:41<00:06, 8459.71it/s] 87%|████████▋ | 348133/400000 [00:41<00:06, 8486.38it/s] 87%|████████▋ | 348982/400000 [00:41<00:06, 8431.29it/s] 87%|████████▋ | 349826/400000 [00:41<00:05, 8381.26it/s] 88%|████████▊ | 350665/400000 [00:41<00:06, 8221.77it/s] 88%|████████▊ | 351515/400000 [00:41<00:05, 8302.98it/s] 88%|████████▊ | 352373/400000 [00:41<00:05, 8382.29it/s] 88%|████████▊ | 353237/400000 [00:41<00:05, 8457.68it/s] 89%|████████▊ | 354097/400000 [00:41<00:05, 8497.05it/s] 89%|████████▊ | 354960/400000 [00:42<00:05, 8534.11it/s] 89%|████████▉ | 355814/400000 [00:42<00:05, 8464.83it/s] 89%|████████▉ | 356677/400000 [00:42<00:05, 8512.97it/s] 89%|████████▉ | 357536/400000 [00:42<00:04, 8535.61it/s] 90%|████████▉ | 358396/400000 [00:42<00:04, 8554.79it/s] 90%|████████▉ | 359264/400000 [00:42<00:04, 8584.67it/s] 90%|█████████ | 360143/400000 [00:42<00:04, 8642.91it/s] 90%|█████████ | 361021/400000 [00:42<00:04, 8681.45it/s] 90%|█████████ | 361890/400000 [00:42<00:04, 8595.97it/s] 91%|█████████ | 362750/400000 [00:42<00:04, 8580.95it/s] 91%|█████████ | 363609/400000 [00:43<00:04, 8525.33it/s] 91%|█████████ | 364462/400000 [00:43<00:04, 8497.42it/s] 91%|█████████▏| 365312/400000 [00:43<00:04, 8290.31it/s] 92%|█████████▏| 366161/400000 [00:43<00:04, 8347.86it/s] 92%|█████████▏| 367009/400000 [00:43<00:03, 8386.06it/s] 92%|█████████▏| 367868/400000 [00:43<00:03, 8445.74it/s] 92%|█████████▏| 368733/400000 [00:43<00:03, 8506.00it/s] 92%|█████████▏| 369585/400000 [00:43<00:03, 8414.41it/s] 93%|█████████▎| 370430/400000 [00:43<00:03, 8422.80it/s] 93%|█████████▎| 371273/400000 [00:43<00:03, 8421.55it/s] 93%|█████████▎| 372116/400000 [00:44<00:03, 8396.09it/s] 93%|█████████▎| 372969/400000 [00:44<00:03, 8433.39it/s] 93%|█████████▎| 373813/400000 [00:44<00:03, 8241.50it/s] 94%|█████████▎| 374648/400000 [00:44<00:03, 8271.07it/s] 94%|█████████▍| 375491/400000 [00:44<00:02, 8317.29it/s] 94%|█████████▍| 376328/400000 [00:44<00:02, 8331.94it/s] 94%|█████████▍| 377170/400000 [00:44<00:02, 8356.80it/s] 95%|█████████▍| 378022/400000 [00:44<00:02, 8404.91it/s] 95%|█████████▍| 378864/400000 [00:44<00:02, 8409.23it/s] 95%|█████████▍| 379717/400000 [00:44<00:02, 8444.58it/s] 95%|█████████▌| 380562/400000 [00:45<00:02, 8441.24it/s] 95%|█████████▌| 381411/400000 [00:45<00:02, 8455.33it/s] 96%|█████████▌| 382257/400000 [00:45<00:02, 8275.30it/s] 96%|█████████▌| 383122/400000 [00:45<00:02, 8383.93it/s] 96%|█████████▌| 383962/400000 [00:45<00:01, 8304.72it/s] 96%|█████████▌| 384795/400000 [00:45<00:01, 8310.52it/s] 96%|█████████▋| 385645/400000 [00:45<00:01, 8350.36it/s] 97%|█████████▋| 386481/400000 [00:45<00:01, 8293.67it/s] 97%|█████████▋| 387325/400000 [00:45<00:01, 8336.37it/s] 97%|█████████▋| 388166/400000 [00:46<00:01, 8357.63it/s] 97%|█████████▋| 389003/400000 [00:46<00:01, 8297.67it/s] 97%|█████████▋| 389847/400000 [00:46<00:01, 8337.74it/s] 98%|█████████▊| 390711/400000 [00:46<00:01, 8426.13it/s] 98%|█████████▊| 391555/400000 [00:46<00:01, 8382.60it/s] 98%|█████████▊| 392394/400000 [00:46<00:00, 8359.82it/s] 98%|█████████▊| 393239/400000 [00:46<00:00, 8385.03it/s] 99%|█████████▊| 394096/400000 [00:46<00:00, 8437.05it/s] 99%|█████████▊| 394940/400000 [00:46<00:00, 8314.39it/s] 99%|█████████▉| 395795/400000 [00:46<00:00, 8381.52it/s] 99%|█████████▉| 396642/400000 [00:47<00:00, 8406.80it/s] 99%|█████████▉| 397493/400000 [00:47<00:00, 8435.11it/s]100%|█████████▉| 398349/400000 [00:47<00:00, 8469.36it/s]100%|█████████▉| 399197/400000 [00:47<00:00, 8413.66it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8434.70it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f421dd69d30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011654398813097603 	 Accuracy: 50
Train Epoch: 1 	 Loss: 0.010996436594321975 	 Accuracy: 68

  model saves at 68% accuracy 

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
2020-05-13 23:24:24.403423: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 23:24:24.407769: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095070000 Hz
2020-05-13 23:24:24.407928: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b2aa632560 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 23:24:24.407942: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f41ca74aba8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.4826 - accuracy: 0.5120
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5670 - accuracy: 0.5065 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7944 - accuracy: 0.4917
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6015 - accuracy: 0.5042
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6421 - accuracy: 0.5016
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6896 - accuracy: 0.4985
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6820 - accuracy: 0.4990
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6302 - accuracy: 0.5024
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6206 - accuracy: 0.5030
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6084 - accuracy: 0.5038
11000/25000 [============>.................] - ETA: 3s - loss: 7.6276 - accuracy: 0.5025
12000/25000 [=============>................] - ETA: 3s - loss: 7.6538 - accuracy: 0.5008
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6690 - accuracy: 0.4998
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6776 - accuracy: 0.4993
15000/25000 [=================>............] - ETA: 2s - loss: 7.6564 - accuracy: 0.5007
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6360 - accuracy: 0.5020
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6495 - accuracy: 0.5011
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6240 - accuracy: 0.5028
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6222 - accuracy: 0.5029
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6321 - accuracy: 0.5023
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6345 - accuracy: 0.5021
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6290 - accuracy: 0.5025
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6533 - accuracy: 0.5009
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6717 - accuracy: 0.4997
25000/25000 [==============================] - 7s 291us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f417eb89518> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f41c7359518> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.2165 - crf_viterbi_accuracy: 0.6800 - val_loss: 1.1560 - val_crf_viterbi_accuracy: 0.6533

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

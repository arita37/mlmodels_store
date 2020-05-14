
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f5148d67fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 00:17:13.175600
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 00:17:13.178643
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 00:17:13.181498
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 00:17:13.185092
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f5154b314a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354005.3125
Epoch 2/10

1/1 [==============================] - 0s 88ms/step - loss: 287237.7812
Epoch 3/10

1/1 [==============================] - 0s 91ms/step - loss: 186025.2656
Epoch 4/10

1/1 [==============================] - 0s 88ms/step - loss: 115189.5078
Epoch 5/10

1/1 [==============================] - 0s 87ms/step - loss: 69508.1719
Epoch 6/10

1/1 [==============================] - 0s 88ms/step - loss: 42925.9102
Epoch 7/10

1/1 [==============================] - 0s 83ms/step - loss: 27878.7402
Epoch 8/10

1/1 [==============================] - 0s 83ms/step - loss: 19155.3770
Epoch 9/10

1/1 [==============================] - 0s 89ms/step - loss: 13882.2197
Epoch 10/10

1/1 [==============================] - 0s 98ms/step - loss: 10556.8545

  #### Inference Need return ypred, ytrue ######################### 
[[-1.79955870e-01  6.25010252e+00  5.86932087e+00  6.96612692e+00
   6.97652578e+00  4.80746078e+00  4.97458982e+00  5.40928793e+00
   4.70197248e+00  5.91602087e+00  6.55093861e+00  6.52595615e+00
   4.49503708e+00  5.77715874e+00  6.32480240e+00  6.94525623e+00
   7.05565548e+00  6.45504189e+00  5.33908033e+00  5.82341957e+00
   5.48740959e+00  6.17557430e+00  4.94754648e+00  5.78886652e+00
   6.66124916e+00  7.22774601e+00  6.55192137e+00  5.53553534e+00
   5.01513386e+00  6.06568241e+00  5.53001738e+00  7.29856348e+00
   4.63021946e+00  5.95798635e+00  5.53534603e+00  5.06061935e+00
   6.49187613e+00  5.24765730e+00  5.88382864e+00  5.97880936e+00
   6.77679253e+00  5.52007914e+00  5.62523794e+00  5.96788216e+00
   6.53610277e+00  6.55476522e+00  6.32897377e+00  7.02588558e+00
   6.77786827e+00  5.00155830e+00  6.07562542e+00  5.44895554e+00
   5.87773848e+00  6.08351803e+00  6.30176592e+00  5.53419685e+00
   5.26372385e+00  5.68299484e+00  6.78334284e+00  6.33279085e+00
   1.18068409e+00 -1.34351110e+00 -2.67666280e-01  8.86518776e-01
   5.50517380e-01 -8.65785241e-01  7.74299383e-01  7.11917996e-01
   3.14353824e-01  1.16526258e+00  2.17712164e-01 -9.83711481e-02
  -4.25841630e-01  1.13794351e+00 -8.62579644e-01 -5.20301878e-01
   3.84011865e-02 -2.98993230e-01 -6.92248344e-04 -1.56338751e-01
  -6.63501024e-03 -8.49083841e-01  1.06533825e+00  5.49930334e-03
   5.52733123e-01  5.99863350e-01 -4.94313061e-01 -3.91747952e-02
   2.59930372e-01 -8.99812818e-01  1.79841661e+00 -4.76418138e-02
  -1.45195615e+00  9.37132180e-01  5.56978405e-01 -1.58426166e-02
  -9.20653343e-02  4.83502269e-01 -1.21752828e-01  2.75091827e-02
   1.45310163e+00  8.72063637e-02 -4.71291661e-01 -4.93292421e-01
  -1.74358308e-01  2.89875954e-01 -4.10161614e-01  2.78380483e-01
  -1.41403759e+00 -1.93785250e-01 -1.32226378e-01  1.01470327e+00
  -4.74026442e-01  5.13523102e-01 -9.85155702e-01 -2.88333595e-01
  -4.69016135e-01 -4.30248737e-01 -2.30041742e-01  6.52637780e-01
  -4.86432850e-01  7.38940477e-01  5.52155852e-01  8.55131805e-01
  -4.91172552e-01 -9.61725116e-02  5.62943459e-01 -1.00796282e+00
   1.27287269e+00  9.01010931e-02 -2.08299965e-01  1.48679399e+00
  -4.47919041e-01  2.94299126e-01 -6.76382363e-01  3.67911577e-01
   5.50228238e-01  1.96247160e-01  3.54762554e-01 -6.68976977e-02
   1.89035714e-01 -5.44969618e-01  3.78011465e-02 -2.51779199e-01
   4.81699407e-01  5.61480761e-01  2.91056722e-01  1.10832620e+00
   3.92393947e-01 -2.82323629e-01 -6.03912354e-01  5.69570303e-01
  -2.45887637e-01  8.57767165e-01 -7.06640601e-01  1.35925186e+00
  -2.09839284e-01 -8.43388677e-01  5.89586318e-01 -4.58199739e-01
  -1.49981117e+00 -5.00724137e-01  6.45424187e-01  3.28141689e-01
  -1.30167976e-02  3.68721485e-02  2.92360008e-01 -2.25206733e-01
   2.63805389e-01 -2.76611447e-01  3.98464084e-01 -1.45244122e-01
  -1.88149825e-01  1.12007126e-01  7.78974712e-01 -3.44016850e-01
   4.75581527e-01  2.35102147e-01  5.38443565e-01 -6.65804505e-01
   3.42667699e-02  6.35844421e+00  5.94354582e+00  6.66929197e+00
   7.03152561e+00  6.47322798e+00  6.70282650e+00  6.36577415e+00
   7.77545834e+00  6.95321226e+00  7.35092640e+00  6.87716770e+00
   6.46065092e+00  6.59973478e+00  5.96853399e+00  6.51267290e+00
   5.84145975e+00  6.27023602e+00  7.35026026e+00  6.73846865e+00
   5.46101093e+00  6.31557941e+00  6.59973145e+00  6.20379925e+00
   6.10475588e+00  6.23372746e+00  6.85437346e+00  6.76006603e+00
   7.24833822e+00  5.53622007e+00  6.35949612e+00  5.89836550e+00
   6.68008137e+00  7.47980309e+00  7.46995831e+00  5.90279961e+00
   6.76703358e+00  7.10813570e+00  7.74317122e+00  6.27201176e+00
   6.02854347e+00  6.21577358e+00  6.08830738e+00  6.65816116e+00
   6.22253799e+00  7.12545061e+00  6.82638550e+00  6.56164980e+00
   5.85182095e+00  6.33686113e+00  6.22118759e+00  6.82906628e+00
   5.69756842e+00  7.21143055e+00  5.92820740e+00  7.39533091e+00
   6.41638374e+00  6.95482588e+00  6.95738935e+00  7.36606836e+00
   1.44626451e+00  9.98316526e-01  1.98363221e+00  7.13818848e-01
   8.81495297e-01  2.17216849e+00  1.59146941e+00  1.22238326e+00
   6.86824679e-01  1.41141319e+00  8.21113586e-01  5.38961828e-01
   8.17976475e-01  9.23113346e-01  1.00762355e+00  2.95497775e-01
   1.27497125e+00  1.54448080e+00  9.43639278e-01  1.78064847e+00
   7.66495705e-01  1.25837040e+00  1.40355134e+00  1.84608924e+00
   2.21913910e+00  1.85451639e+00  4.27458823e-01  1.41517341e+00
   7.28069186e-01  1.14112926e+00  1.54858756e+00  2.69054365e+00
   1.31620204e+00  1.07805693e+00  3.39518666e-01  2.01963663e+00
   1.58703554e+00  5.04893064e-01  1.78935122e+00  1.68264115e+00
   1.18688536e+00  4.81009066e-01  9.60023940e-01  1.29930913e+00
   1.60581315e+00  7.23207712e-01  4.53793705e-01  4.22594666e-01
   1.45902848e+00  8.71281505e-01  8.47690523e-01  2.58007193e+00
   1.62871802e+00  2.12083793e+00  2.24858880e+00  3.30276370e-01
   4.44851220e-01  8.98307681e-01  3.81354868e-01  1.91278946e+00
   1.53761148e+00  6.55109942e-01  5.63919663e-01  1.34243655e+00
   7.04221189e-01  1.24057353e+00  1.17020774e+00  1.44525099e+00
   1.17499423e+00  6.27564609e-01  3.74268532e-01  5.19827902e-01
   6.31425381e-01  1.19974101e+00  2.20617676e+00  1.40174580e+00
   2.41230130e-01  2.54110909e+00  6.34260535e-01  1.12231648e+00
   1.14432502e+00  1.33991742e+00  1.54058373e+00  1.37695384e+00
   1.53329599e+00  5.92889190e-01  1.11864507e+00  5.82005978e-01
   8.59082401e-01  1.65598035e+00  1.03381622e+00  8.99400353e-01
   4.58454609e-01  1.41114140e+00  1.74534595e+00  5.66648841e-01
   1.83645439e+00  2.14437819e+00  8.92381907e-01  5.49427152e-01
   1.53728426e+00  1.03971028e+00  1.38396966e+00  1.60738564e+00
   1.24396825e+00  7.70686567e-01  7.46773720e-01  4.34760749e-01
   1.64168310e+00  1.35320747e+00  9.64314640e-01  7.83667564e-01
   1.83491707e+00  2.47363472e+00  3.10466409e-01  8.70191038e-01
   1.17197263e+00  2.93259668e+00  1.53207123e+00  5.81321418e-01
   7.26755190e+00 -8.80951786e+00 -1.00655298e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 00:17:20.828153
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.7844
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 00:17:20.831441
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9196.65
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 00:17:20.833991
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.4055
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 00:17:20.836497
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -822.625
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139986731942352
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139985522189032
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139985522189536
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139985522190040
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139985522190544
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139985522191048

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f51424fb668> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.438994
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.408489
grad_step = 000002, loss = 0.391644
grad_step = 000003, loss = 0.371881
grad_step = 000004, loss = 0.349168
grad_step = 000005, loss = 0.325265
grad_step = 000006, loss = 0.304528
grad_step = 000007, loss = 0.292375
grad_step = 000008, loss = 0.285175
grad_step = 000009, loss = 0.275765
grad_step = 000010, loss = 0.261313
grad_step = 000011, loss = 0.246395
grad_step = 000012, loss = 0.235655
grad_step = 000013, loss = 0.228375
grad_step = 000014, loss = 0.221657
grad_step = 000015, loss = 0.214236
grad_step = 000016, loss = 0.206085
grad_step = 000017, loss = 0.197685
grad_step = 000018, loss = 0.189461
grad_step = 000019, loss = 0.181981
grad_step = 000020, loss = 0.175584
grad_step = 000021, loss = 0.169638
grad_step = 000022, loss = 0.163310
grad_step = 000023, loss = 0.156597
grad_step = 000024, loss = 0.149993
grad_step = 000025, loss = 0.144027
grad_step = 000026, loss = 0.138432
grad_step = 000027, loss = 0.132832
grad_step = 000028, loss = 0.127286
grad_step = 000029, loss = 0.121957
grad_step = 000030, loss = 0.116931
grad_step = 000031, loss = 0.112126
grad_step = 000032, loss = 0.107279
grad_step = 000033, loss = 0.102378
grad_step = 000034, loss = 0.097679
grad_step = 000035, loss = 0.093345
grad_step = 000036, loss = 0.089378
grad_step = 000037, loss = 0.085546
grad_step = 000038, loss = 0.081674
grad_step = 000039, loss = 0.077902
grad_step = 000040, loss = 0.074347
grad_step = 000041, loss = 0.070978
grad_step = 000042, loss = 0.067695
grad_step = 000043, loss = 0.064485
grad_step = 000044, loss = 0.061407
grad_step = 000045, loss = 0.058487
grad_step = 000046, loss = 0.055729
grad_step = 000047, loss = 0.053054
grad_step = 000048, loss = 0.050482
grad_step = 000049, loss = 0.048071
grad_step = 000050, loss = 0.045810
grad_step = 000051, loss = 0.043588
grad_step = 000052, loss = 0.041420
grad_step = 000053, loss = 0.039348
grad_step = 000054, loss = 0.037399
grad_step = 000055, loss = 0.035552
grad_step = 000056, loss = 0.033776
grad_step = 000057, loss = 0.032064
grad_step = 000058, loss = 0.030443
grad_step = 000059, loss = 0.028895
grad_step = 000060, loss = 0.027398
grad_step = 000061, loss = 0.025979
grad_step = 000062, loss = 0.024642
grad_step = 000063, loss = 0.023372
grad_step = 000064, loss = 0.022147
grad_step = 000065, loss = 0.020961
grad_step = 000066, loss = 0.019839
grad_step = 000067, loss = 0.018780
grad_step = 000068, loss = 0.017766
grad_step = 000069, loss = 0.016800
grad_step = 000070, loss = 0.015890
grad_step = 000071, loss = 0.015027
grad_step = 000072, loss = 0.014207
grad_step = 000073, loss = 0.013428
grad_step = 000074, loss = 0.012694
grad_step = 000075, loss = 0.011999
grad_step = 000076, loss = 0.011335
grad_step = 000077, loss = 0.010710
grad_step = 000078, loss = 0.010126
grad_step = 000079, loss = 0.009576
grad_step = 000080, loss = 0.009055
grad_step = 000081, loss = 0.008566
grad_step = 000082, loss = 0.008109
grad_step = 000083, loss = 0.007679
grad_step = 000084, loss = 0.007275
grad_step = 000085, loss = 0.006900
grad_step = 000086, loss = 0.006550
grad_step = 000087, loss = 0.006223
grad_step = 000088, loss = 0.005924
grad_step = 000089, loss = 0.005657
grad_step = 000090, loss = 0.005415
grad_step = 000091, loss = 0.005195
grad_step = 000092, loss = 0.004946
grad_step = 000093, loss = 0.004706
grad_step = 000094, loss = 0.004482
grad_step = 000095, loss = 0.004296
grad_step = 000096, loss = 0.004141
grad_step = 000097, loss = 0.003992
grad_step = 000098, loss = 0.003840
grad_step = 000099, loss = 0.003695
grad_step = 000100, loss = 0.003568
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003461
grad_step = 000102, loss = 0.003360
grad_step = 000103, loss = 0.003263
grad_step = 000104, loss = 0.003167
grad_step = 000105, loss = 0.003078
grad_step = 000106, loss = 0.003001
grad_step = 000107, loss = 0.002933
grad_step = 000108, loss = 0.002872
grad_step = 000109, loss = 0.002813
grad_step = 000110, loss = 0.002756
grad_step = 000111, loss = 0.002701
grad_step = 000112, loss = 0.002651
grad_step = 000113, loss = 0.002606
grad_step = 000114, loss = 0.002565
grad_step = 000115, loss = 0.002529
grad_step = 000116, loss = 0.002497
grad_step = 000117, loss = 0.002468
grad_step = 000118, loss = 0.002442
grad_step = 000119, loss = 0.002420
grad_step = 000120, loss = 0.002399
grad_step = 000121, loss = 0.002383
grad_step = 000122, loss = 0.002369
grad_step = 000123, loss = 0.002359
grad_step = 000124, loss = 0.002347
grad_step = 000125, loss = 0.002337
grad_step = 000126, loss = 0.002319
grad_step = 000127, loss = 0.002300
grad_step = 000128, loss = 0.002276
grad_step = 000129, loss = 0.002255
grad_step = 000130, loss = 0.002237
grad_step = 000131, loss = 0.002227
grad_step = 000132, loss = 0.002223
grad_step = 000133, loss = 0.002222
grad_step = 000134, loss = 0.002222
grad_step = 000135, loss = 0.002221
grad_step = 000136, loss = 0.002220
grad_step = 000137, loss = 0.002215
grad_step = 000138, loss = 0.002208
grad_step = 000139, loss = 0.002198
grad_step = 000140, loss = 0.002188
grad_step = 000141, loss = 0.002178
grad_step = 000142, loss = 0.002171
grad_step = 000143, loss = 0.002164
grad_step = 000144, loss = 0.002160
grad_step = 000145, loss = 0.002157
grad_step = 000146, loss = 0.002155
grad_step = 000147, loss = 0.002154
grad_step = 000148, loss = 0.002154
grad_step = 000149, loss = 0.002157
grad_step = 000150, loss = 0.002164
grad_step = 000151, loss = 0.002178
grad_step = 000152, loss = 0.002199
grad_step = 000153, loss = 0.002236
grad_step = 000154, loss = 0.002260
grad_step = 000155, loss = 0.002275
grad_step = 000156, loss = 0.002226
grad_step = 000157, loss = 0.002160
grad_step = 000158, loss = 0.002117
grad_step = 000159, loss = 0.002125
grad_step = 000160, loss = 0.002163
grad_step = 000161, loss = 0.002179
grad_step = 000162, loss = 0.002161
grad_step = 000163, loss = 0.002117
grad_step = 000164, loss = 0.002093
grad_step = 000165, loss = 0.002101
grad_step = 000166, loss = 0.002120
grad_step = 000167, loss = 0.002124
grad_step = 000168, loss = 0.002107
grad_step = 000169, loss = 0.002087
grad_step = 000170, loss = 0.002069
grad_step = 000171, loss = 0.002070
grad_step = 000172, loss = 0.002073
grad_step = 000173, loss = 0.002072
grad_step = 000174, loss = 0.002070
grad_step = 000175, loss = 0.002068
grad_step = 000176, loss = 0.002049
grad_step = 000177, loss = 0.002028
grad_step = 000178, loss = 0.002018
grad_step = 000179, loss = 0.002004
grad_step = 000180, loss = 0.002013
grad_step = 000181, loss = 0.002028
grad_step = 000182, loss = 0.002087
grad_step = 000183, loss = 0.002086
grad_step = 000184, loss = 0.002108
grad_step = 000185, loss = 0.002113
grad_step = 000186, loss = 0.002088
grad_step = 000187, loss = 0.002024
grad_step = 000188, loss = 0.001965
grad_step = 000189, loss = 0.001933
grad_step = 000190, loss = 0.001937
grad_step = 000191, loss = 0.001982
grad_step = 000192, loss = 0.002013
grad_step = 000193, loss = 0.001984
grad_step = 000194, loss = 0.001941
grad_step = 000195, loss = 0.001908
grad_step = 000196, loss = 0.001891
grad_step = 000197, loss = 0.001872
grad_step = 000198, loss = 0.001879
grad_step = 000199, loss = 0.001895
grad_step = 000200, loss = 0.001939
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001955
grad_step = 000202, loss = 0.001993
grad_step = 000203, loss = 0.001942
grad_step = 000204, loss = 0.001884
grad_step = 000205, loss = 0.001866
grad_step = 000206, loss = 0.001847
grad_step = 000207, loss = 0.001835
grad_step = 000208, loss = 0.001849
grad_step = 000209, loss = 0.001870
grad_step = 000210, loss = 0.001852
grad_step = 000211, loss = 0.001796
grad_step = 000212, loss = 0.001776
grad_step = 000213, loss = 0.001795
grad_step = 000214, loss = 0.001779
grad_step = 000215, loss = 0.001754
grad_step = 000216, loss = 0.001765
grad_step = 000217, loss = 0.001779
grad_step = 000218, loss = 0.001858
grad_step = 000219, loss = 0.001768
grad_step = 000220, loss = 0.001841
grad_step = 000221, loss = 0.001857
grad_step = 000222, loss = 0.001727
grad_step = 000223, loss = 0.001875
grad_step = 000224, loss = 0.001913
grad_step = 000225, loss = 0.001730
grad_step = 000226, loss = 0.001846
grad_step = 000227, loss = 0.001719
grad_step = 000228, loss = 0.001769
grad_step = 000229, loss = 0.001736
grad_step = 000230, loss = 0.001662
grad_step = 000231, loss = 0.001739
grad_step = 000232, loss = 0.001653
grad_step = 000233, loss = 0.001746
grad_step = 000234, loss = 0.001578
grad_step = 000235, loss = 0.001648
grad_step = 000236, loss = 0.001525
grad_step = 000237, loss = 0.001556
grad_step = 000238, loss = 0.001540
grad_step = 000239, loss = 0.001521
grad_step = 000240, loss = 0.001482
grad_step = 000241, loss = 0.001437
grad_step = 000242, loss = 0.001433
grad_step = 000243, loss = 0.001394
grad_step = 000244, loss = 0.001431
grad_step = 000245, loss = 0.001375
grad_step = 000246, loss = 0.001410
grad_step = 000247, loss = 0.001665
grad_step = 000248, loss = 0.001440
grad_step = 000249, loss = 0.001350
grad_step = 000250, loss = 0.001336
grad_step = 000251, loss = 0.001291
grad_step = 000252, loss = 0.001252
grad_step = 000253, loss = 0.001252
grad_step = 000254, loss = 0.001239
grad_step = 000255, loss = 0.001252
grad_step = 000256, loss = 0.001153
grad_step = 000257, loss = 0.001146
grad_step = 000258, loss = 0.001176
grad_step = 000259, loss = 0.001163
grad_step = 000260, loss = 0.001129
grad_step = 000261, loss = 0.001164
grad_step = 000262, loss = 0.001411
grad_step = 000263, loss = 0.001279
grad_step = 000264, loss = 0.001148
grad_step = 000265, loss = 0.001009
grad_step = 000266, loss = 0.001015
grad_step = 000267, loss = 0.001073
grad_step = 000268, loss = 0.001074
grad_step = 000269, loss = 0.001044
grad_step = 000270, loss = 0.001036
grad_step = 000271, loss = 0.001214
grad_step = 000272, loss = 0.000921
grad_step = 000273, loss = 0.001083
grad_step = 000274, loss = 0.001550
grad_step = 000275, loss = 0.001037
grad_step = 000276, loss = 0.001733
grad_step = 000277, loss = 0.001503
grad_step = 000278, loss = 0.001657
grad_step = 000279, loss = 0.001025
grad_step = 000280, loss = 0.001813
grad_step = 000281, loss = 0.000974
grad_step = 000282, loss = 0.001634
grad_step = 000283, loss = 0.000925
grad_step = 000284, loss = 0.001353
grad_step = 000285, loss = 0.000989
grad_step = 000286, loss = 0.001064
grad_step = 000287, loss = 0.001065
grad_step = 000288, loss = 0.000845
grad_step = 000289, loss = 0.001141
grad_step = 000290, loss = 0.000834
grad_step = 000291, loss = 0.000985
grad_step = 000292, loss = 0.000862
grad_step = 000293, loss = 0.000843
grad_step = 000294, loss = 0.000891
grad_step = 000295, loss = 0.000762
grad_step = 000296, loss = 0.000885
grad_step = 000297, loss = 0.000712
grad_step = 000298, loss = 0.000820
grad_step = 000299, loss = 0.000731
grad_step = 000300, loss = 0.000721
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000754
grad_step = 000302, loss = 0.000663
grad_step = 000303, loss = 0.000729
grad_step = 000304, loss = 0.000646
grad_step = 000305, loss = 0.000700
grad_step = 000306, loss = 0.000625
grad_step = 000307, loss = 0.000679
grad_step = 000308, loss = 0.000622
grad_step = 000309, loss = 0.000632
grad_step = 000310, loss = 0.000610
grad_step = 000311, loss = 0.000610
grad_step = 000312, loss = 0.000589
grad_step = 000313, loss = 0.000588
grad_step = 000314, loss = 0.000573
grad_step = 000315, loss = 0.000569
grad_step = 000316, loss = 0.000553
grad_step = 000317, loss = 0.000557
grad_step = 000318, loss = 0.000534
grad_step = 000319, loss = 0.000543
grad_step = 000320, loss = 0.000520
grad_step = 000321, loss = 0.000528
grad_step = 000322, loss = 0.000505
grad_step = 000323, loss = 0.000514
grad_step = 000324, loss = 0.000493
grad_step = 000325, loss = 0.000500
grad_step = 000326, loss = 0.000483
grad_step = 000327, loss = 0.000484
grad_step = 000328, loss = 0.000475
grad_step = 000329, loss = 0.000470
grad_step = 000330, loss = 0.000467
grad_step = 000331, loss = 0.000459
grad_step = 000332, loss = 0.000458
grad_step = 000333, loss = 0.000448
grad_step = 000334, loss = 0.000449
grad_step = 000335, loss = 0.000440
grad_step = 000336, loss = 0.000441
grad_step = 000337, loss = 0.000433
grad_step = 000338, loss = 0.000431
grad_step = 000339, loss = 0.000426
grad_step = 000340, loss = 0.000422
grad_step = 000341, loss = 0.000420
grad_step = 000342, loss = 0.000415
grad_step = 000343, loss = 0.000415
grad_step = 000344, loss = 0.000414
grad_step = 000345, loss = 0.000420
grad_step = 000346, loss = 0.000434
grad_step = 000347, loss = 0.000496
grad_step = 000348, loss = 0.000571
grad_step = 000349, loss = 0.000782
grad_step = 000350, loss = 0.000617
grad_step = 000351, loss = 0.000485
grad_step = 000352, loss = 0.000399
grad_step = 000353, loss = 0.000428
grad_step = 000354, loss = 0.000500
grad_step = 000355, loss = 0.000475
grad_step = 000356, loss = 0.000430
grad_step = 000357, loss = 0.000389
grad_step = 000358, loss = 0.000413
grad_step = 000359, loss = 0.000455
grad_step = 000360, loss = 0.000410
grad_step = 000361, loss = 0.000380
grad_step = 000362, loss = 0.000395
grad_step = 000363, loss = 0.000410
grad_step = 000364, loss = 0.000404
grad_step = 000365, loss = 0.000376
grad_step = 000366, loss = 0.000371
grad_step = 000367, loss = 0.000390
grad_step = 000368, loss = 0.000392
grad_step = 000369, loss = 0.000371
grad_step = 000370, loss = 0.000362
grad_step = 000371, loss = 0.000368
grad_step = 000372, loss = 0.000373
grad_step = 000373, loss = 0.000369
grad_step = 000374, loss = 0.000359
grad_step = 000375, loss = 0.000353
grad_step = 000376, loss = 0.000357
grad_step = 000377, loss = 0.000361
grad_step = 000378, loss = 0.000357
grad_step = 000379, loss = 0.000351
grad_step = 000380, loss = 0.000348
grad_step = 000381, loss = 0.000350
grad_step = 000382, loss = 0.000352
grad_step = 000383, loss = 0.000348
grad_step = 000384, loss = 0.000344
grad_step = 000385, loss = 0.000343
grad_step = 000386, loss = 0.000343
grad_step = 000387, loss = 0.000344
grad_step = 000388, loss = 0.000345
grad_step = 000389, loss = 0.000342
grad_step = 000390, loss = 0.000339
grad_step = 000391, loss = 0.000337
grad_step = 000392, loss = 0.000336
grad_step = 000393, loss = 0.000336
grad_step = 000394, loss = 0.000336
grad_step = 000395, loss = 0.000334
grad_step = 000396, loss = 0.000333
grad_step = 000397, loss = 0.000332
grad_step = 000398, loss = 0.000332
grad_step = 000399, loss = 0.000331
grad_step = 000400, loss = 0.000331
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000331
grad_step = 000402, loss = 0.000330
grad_step = 000403, loss = 0.000330
grad_step = 000404, loss = 0.000329
grad_step = 000405, loss = 0.000327
grad_step = 000406, loss = 0.000326
grad_step = 000407, loss = 0.000325
grad_step = 000408, loss = 0.000325
grad_step = 000409, loss = 0.000324
grad_step = 000410, loss = 0.000324
grad_step = 000411, loss = 0.000323
grad_step = 000412, loss = 0.000323
grad_step = 000413, loss = 0.000323
grad_step = 000414, loss = 0.000323
grad_step = 000415, loss = 0.000324
grad_step = 000416, loss = 0.000325
grad_step = 000417, loss = 0.000326
grad_step = 000418, loss = 0.000329
grad_step = 000419, loss = 0.000332
grad_step = 000420, loss = 0.000337
grad_step = 000421, loss = 0.000343
grad_step = 000422, loss = 0.000354
grad_step = 000423, loss = 0.000365
grad_step = 000424, loss = 0.000392
grad_step = 000425, loss = 0.000418
grad_step = 000426, loss = 0.000478
grad_step = 000427, loss = 0.000487
grad_step = 000428, loss = 0.000510
grad_step = 000429, loss = 0.000430
grad_step = 000430, loss = 0.000355
grad_step = 000431, loss = 0.000319
grad_step = 000432, loss = 0.000344
grad_step = 000433, loss = 0.000383
grad_step = 000434, loss = 0.000383
grad_step = 000435, loss = 0.000344
grad_step = 000436, loss = 0.000315
grad_step = 000437, loss = 0.000324
grad_step = 000438, loss = 0.000350
grad_step = 000439, loss = 0.000362
grad_step = 000440, loss = 0.000351
grad_step = 000441, loss = 0.000327
grad_step = 000442, loss = 0.000312
grad_step = 000443, loss = 0.000316
grad_step = 000444, loss = 0.000331
grad_step = 000445, loss = 0.000340
grad_step = 000446, loss = 0.000336
grad_step = 000447, loss = 0.000323
grad_step = 000448, loss = 0.000311
grad_step = 000449, loss = 0.000308
grad_step = 000450, loss = 0.000314
grad_step = 000451, loss = 0.000321
grad_step = 000452, loss = 0.000323
grad_step = 000453, loss = 0.000322
grad_step = 000454, loss = 0.000314
grad_step = 000455, loss = 0.000307
grad_step = 000456, loss = 0.000305
grad_step = 000457, loss = 0.000307
grad_step = 000458, loss = 0.000310
grad_step = 000459, loss = 0.000312
grad_step = 000460, loss = 0.000311
grad_step = 000461, loss = 0.000308
grad_step = 000462, loss = 0.000305
grad_step = 000463, loss = 0.000304
grad_step = 000464, loss = 0.000303
grad_step = 000465, loss = 0.000303
grad_step = 000466, loss = 0.000304
grad_step = 000467, loss = 0.000305
grad_step = 000468, loss = 0.000305
grad_step = 000469, loss = 0.000305
grad_step = 000470, loss = 0.000304
grad_step = 000471, loss = 0.000303
grad_step = 000472, loss = 0.000301
grad_step = 000473, loss = 0.000301
grad_step = 000474, loss = 0.000300
grad_step = 000475, loss = 0.000299
grad_step = 000476, loss = 0.000299
grad_step = 000477, loss = 0.000299
grad_step = 000478, loss = 0.000299
grad_step = 000479, loss = 0.000299
grad_step = 000480, loss = 0.000299
grad_step = 000481, loss = 0.000299
grad_step = 000482, loss = 0.000299
grad_step = 000483, loss = 0.000300
grad_step = 000484, loss = 0.000302
grad_step = 000485, loss = 0.000303
grad_step = 000486, loss = 0.000306
grad_step = 000487, loss = 0.000309
grad_step = 000488, loss = 0.000313
grad_step = 000489, loss = 0.000316
grad_step = 000490, loss = 0.000321
grad_step = 000491, loss = 0.000325
grad_step = 000492, loss = 0.000331
grad_step = 000493, loss = 0.000338
grad_step = 000494, loss = 0.000345
grad_step = 000495, loss = 0.000352
grad_step = 000496, loss = 0.000355
grad_step = 000497, loss = 0.000354
grad_step = 000498, loss = 0.000346
grad_step = 000499, loss = 0.000334
grad_step = 000500, loss = 0.000319
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000304
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

  date_run                              2020-05-14 00:17:36.895758
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.194968
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 00:17:36.900269
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.078407
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 00:17:36.906461
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.128248
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 00:17:36.910545
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.191423
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
0   2020-05-14 00:17:13.175600  ...    mean_absolute_error
1   2020-05-14 00:17:13.178643  ...     mean_squared_error
2   2020-05-14 00:17:13.181498  ...  median_absolute_error
3   2020-05-14 00:17:13.185092  ...               r2_score
4   2020-05-14 00:17:20.828153  ...    mean_absolute_error
5   2020-05-14 00:17:20.831441  ...     mean_squared_error
6   2020-05-14 00:17:20.833991  ...  median_absolute_error
7   2020-05-14 00:17:20.836497  ...               r2_score
8   2020-05-14 00:17:36.895758  ...    mean_absolute_error
9   2020-05-14 00:17:36.900269  ...     mean_squared_error
10  2020-05-14 00:17:36.906461  ...  median_absolute_error
11  2020-05-14 00:17:36.910545  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6684ebcfd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███       | 3063808/9912422 [00:00<00:00, 30572144.00it/s]9920512it [00:00, 34759544.89it/s]                             
0it [00:00, ?it/s]32768it [00:00, 605700.76it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 472251.21it/s]1654784it [00:00, 10973647.04it/s]                         
0it [00:00, ?it/s]8192it [00:00, 173193.77it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f66378bfeb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6636eea0f0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f66378bfeb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6636e43128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f663467f518> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6634669780> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f66378bfeb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6636e00748> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f663467f518> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6636cbb588> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f9c2e32f208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=a7b48793963ebcffb54a139cbb63d9a11d2c94ee7c8a3c78bc5a55e9929c8282
  Stored in directory: /tmp/pip-ephem-wheel-cache-wq75dkyr/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f9bc612a6d8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 2s
 1032192/17464789 [>.............................] - ETA: 0s
 2719744/17464789 [===>..........................] - ETA: 0s
 4947968/17464789 [=======>......................] - ETA: 0s
12853248/17464789 [=====================>........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 00:18:58.883028: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 00:18:58.887173: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-14 00:18:58.887431: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5654e5456cc0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 00:18:58.887496: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.5286 - accuracy: 0.5090
 2000/25000 [=>............................] - ETA: 7s - loss: 7.8890 - accuracy: 0.4855 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7382 - accuracy: 0.4953
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6321 - accuracy: 0.5023
 5000/25000 [=====>........................] - ETA: 4s - loss: 7.5900 - accuracy: 0.5050
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5746 - accuracy: 0.5060
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6097 - accuracy: 0.5037
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.6283 - accuracy: 0.5025
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6649 - accuracy: 0.5001
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6973 - accuracy: 0.4980
11000/25000 [============>.................] - ETA: 3s - loss: 7.7294 - accuracy: 0.4959
12000/25000 [=============>................] - ETA: 2s - loss: 7.7637 - accuracy: 0.4937
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7433 - accuracy: 0.4950
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7477 - accuracy: 0.4947
15000/25000 [=================>............] - ETA: 2s - loss: 7.7412 - accuracy: 0.4951
16000/25000 [==================>...........] - ETA: 1s - loss: 7.7289 - accuracy: 0.4959
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7198 - accuracy: 0.4965
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7092 - accuracy: 0.4972
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6868 - accuracy: 0.4987
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6758 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6922 - accuracy: 0.4983
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6799 - accuracy: 0.4991
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6793 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6858 - accuracy: 0.4988
25000/25000 [==============================] - 6s 252us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 00:19:10.991142
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 00:19:10.991142  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:31:02, 11.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:17:54, 15.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:45:48, 22.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:32:34, 31.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:16:00, 45.3kB/s].vector_cache/glove.6B.zip:   1%|          | 8.86M/862M [00:01<3:39:56, 64.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:01<2:33:19, 92.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.1M/862M [00:01<1:46:54, 132kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 21.7M/862M [00:01<1:14:31, 188kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.6M/862M [00:01<52:02, 268kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.4M/862M [00:01<36:18, 382kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.2M/862M [00:02<25:25, 543kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.3M/862M [00:02<17:45, 772kB/s].vector_cache/glove.6B.zip:   5%|▍         | 43.1M/862M [00:02<12:29, 1.09MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.8M/862M [00:02<08:47, 1.54MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:02<06:27, 2.09MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<05:00, 2.68MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:04<10:29:49, 21.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<7:20:42, 30.5kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:06<5:09:51, 43.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<3:40:01, 60.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<2:34:35, 86.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.7M/862M [00:06<1:48:04, 123kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:08<1:22:08, 162kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<58:53, 226kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:08<41:30, 320kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:10<32:05, 413kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:10<25:19, 523kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<18:18, 722kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.3M/862M [00:10<13:02, 1.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:12<13:04, 1.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<10:33, 1.25MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:12<07:43, 1.70MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:14<08:27, 1.55MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<07:15, 1.80MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:14<05:25, 2.41MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:16<06:50, 1.90MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<06:09, 2.11MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:16<04:35, 2.84MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:18<06:16, 2.06MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<05:53, 2.20MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:18<04:29, 2.88MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:20<05:55, 2.18MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<05:36, 2.30MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:20<04:13, 3.04MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:20<03:07, 4.11MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:22<44:13, 290kB/s] .vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<32:26, 395kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<23:01, 556kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:24<18:48, 678kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<14:39, 870kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:24<10:32, 1.21MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<07:30, 1.69MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<4:18:57, 49.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<3:04:05, 68.9kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<2:09:22, 98.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<1:30:46, 139kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<1:06:11, 191kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<47:45, 264kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<33:43, 373kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<26:14, 478kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<19:47, 634kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<14:11, 882kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<12:34, 992kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<10:15, 1.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:31, 1.65MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:55, 1.57MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:58, 1.78MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:10, 2.40MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:17, 1.96MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:11, 1.72MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:37, 2.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:17, 2.87MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:40, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:22, 2.28MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:03, 3.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:27, 2.24MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:42, 1.82MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:17, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<03:59, 3.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:39, 2.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:20, 2.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<04:04, 2.97MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:26, 2.22MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:39, 1.81MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:15, 2.29MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<04:01, 2.99MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:25, 2.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:10, 2.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<03:54, 3.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<05:18, 2.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:32, 1.82MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:09, 2.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<03:49, 3.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<06:07, 1.93MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:39, 2.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:14, 2.79MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:30, 2.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:13, 2.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<03:59, 2.94MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:18, 2.21MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:06, 2.30MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<03:52, 3.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:12, 2.24MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:23, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:03, 2.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:44, 3.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:09, 1.88MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:39, 2.05MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:15, 2.71MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:26, 2.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:09, 2.23MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<03:55, 2.92MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:12, 2.20MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:14, 1.83MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:01, 2.27MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<03:40, 3.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<12:42, 894kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<10:10, 1.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<07:26, 1.52MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<07:39, 1.48MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<07:54, 1.43MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:10, 1.83MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<04:28, 2.51MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<14:10, 792kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<11:11, 1.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<08:08, 1.38MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<08:05, 1.38MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:56, 1.61MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:08, 2.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:06, 1.81MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<06:49, 1.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:23, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:54, 2.82MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<15:32, 709kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<12:08, 908kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<08:47, 1.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<08:29, 1.29MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<08:27, 1.29MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:29, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<04:39, 2.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<09:48, 1.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<08:05, 1.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<05:57, 1.82MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:31, 1.66MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<07:03, 1.53MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<05:28, 1.97MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<03:59, 2.70MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:51, 1.57MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<06:00, 1.79MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<04:29, 2.38MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:28, 1.95MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<06:22, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:00, 2.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<03:39, 2.91MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:21, 1.67MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<05:41, 1.86MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<04:16, 2.47MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:13, 2.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<06:05, 1.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:49, 2.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<03:30, 2.98MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<12:07, 863kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<09:38, 1.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<07:01, 1.49MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<07:12, 1.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<07:30, 1.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<05:51, 1.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<04:13, 2.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<13:10, 784kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<10:25, 990kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<07:34, 1.36MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<07:29, 1.37MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<07:39, 1.34MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<05:51, 1.75MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:16, 2.39MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:03, 1.68MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:24, 1.88MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:01, 2.52MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:01, 2.02MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:41, 2.15MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<03:32, 2.85MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:37, 2.17MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:25, 1.85MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<04:19, 2.32MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<03:08, 3.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<23:22, 427kB/s] .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<17:25, 572kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<12:24, 802kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<10:53, 911kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<09:52, 1.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<07:26, 1.33MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<05:19, 1.85MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<12:25, 792kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<09:36, 1.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<07:20, 1.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<05:13, 1.87MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<11:21, 860kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<10:11, 959kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<07:40, 1.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<05:29, 1.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<09:02, 1.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<6:20:01, 25.5kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<4:25:39, 36.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<3:05:12, 52.0kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<3:26:38, 46.6kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<2:26:37, 65.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<1:42:59, 93.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<1:11:56, 133kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<55:38, 172kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<40:02, 239kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<28:13, 338kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<21:42, 437kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<16:17, 583kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<11:36, 815kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<10:07, 931kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<09:12, 1.02MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<06:54, 1.36MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<04:56, 1.90MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<08:00, 1.17MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:41, 1.40MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<04:56, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:26, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:54, 1.57MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:40, 1.98MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<03:22, 2.73MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<11:03, 834kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<08:46, 1.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<06:21, 1.45MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<04:33, 2.01MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<52:57, 173kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<39:13, 233kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<27:52, 328kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<19:38, 464kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<16:11, 561kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<12:18, 738kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<08:50, 1.02MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<08:10, 1.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<07:52, 1.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:57, 1.51MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<04:16, 2.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<07:00, 1.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:56, 1.50MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:24, 2.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:59, 1.78MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:30, 1.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<03:24, 2.60MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<04:16, 2.06MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<03:56, 2.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<02:59, 2.94MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:02, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:56, 1.77MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<03:53, 2.24MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<02:49, 3.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<07:14, 1.20MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<06:04, 1.42MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:29, 1.92MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<04:58, 1.73MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:33, 1.54MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:18, 1.99MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<03:11, 2.68MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:24, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:04, 2.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:05, 2.75MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<03:58, 2.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:37, 1.83MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:41, 2.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<02:39, 3.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<19:46, 424kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<14:44, 569kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<10:30, 795kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<09:10, 906kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<08:24, 989kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<06:17, 1.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:32, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:33, 1.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:50, 1.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:36, 2.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:17, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:51, 1.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<03:48, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<02:45, 2.95MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:58, 1.36MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:07, 1.58MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<03:48, 2.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:23, 1.83MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:52, 1.65MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:48, 2.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<02:46, 2.88MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:47, 1.67MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:13, 1.89MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:09, 2.51MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<03:57, 1.99MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:38, 1.70MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:38, 2.17MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:38, 2.97MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<05:29, 1.43MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:44, 1.65MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:29, 2.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<04:06, 1.89MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:33, 1.71MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:36, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<02:35, 2.97MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<25:07, 306kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<18:27, 417kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<13:03, 587kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<10:44, 709kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<09:09, 832kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<06:44, 1.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<04:49, 1.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<06:03, 1.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<05:04, 1.49MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:44, 2.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<04:16, 1.75MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<04:40, 1.60MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:38, 2.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:38, 2.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<04:49, 1.53MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<04:13, 1.76MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:09, 2.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<03:46, 1.94MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<04:13, 1.74MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:17, 2.22MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<02:23, 3.06MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<06:25, 1.13MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<05:21, 1.36MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:56, 1.84MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:17, 1.68MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:50, 1.88MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<02:50, 2.52MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:32, 2.02MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:17, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:05<02:27, 2.89MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:15, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<02:56, 2.40MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<02:13, 3.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<01:39, 4.22MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<07:11, 974kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<06:41, 1.05MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<05:00, 1.40MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<03:36, 1.93MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:45, 1.46MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:08, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:04, 2.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:37, 1.90MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:19, 2.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<02:29, 2.74MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:11, 2.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:45, 1.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<02:58, 2.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:09, 3.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<05:22, 1.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<04:32, 1.48MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:20, 2.01MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:45, 1.77MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<04:03, 1.64MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:11, 2.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:18, 2.87MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<06:31, 1.01MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<05:19, 1.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:53, 1.69MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<04:06, 1.59MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<04:16, 1.52MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:17, 1.97MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:24, 2.68MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:43, 1.73MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:18, 1.95MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:28, 2.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:08, 2.03MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:34, 1.78MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:47, 2.27MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:04, 3.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<01:58, 3.21MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<4:20:11, 24.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<3:02:09, 34.6kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:28<2:06:38, 49.4kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<1:33:58, 66.5kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<1:07:14, 92.9kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<47:19, 132kB/s]   .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<32:58, 188kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<28:30, 217kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<20:39, 299kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<14:34, 422kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<11:27, 533kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<09:20, 653kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<06:51, 888kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:34<04:50, 1.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<16:14, 372kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<12:00, 503kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<08:32, 704kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<07:16, 821kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<06:23, 935kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<04:43, 1.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<03:23, 1.75MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:41, 1.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:57, 1.49MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:55, 2.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:17, 1.77MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:37, 1.61MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:48, 2.07MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:06, 2.75MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<02:47, 2.06MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:36, 2.21MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<01:59, 2.89MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:36, 2.19MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:10, 1.79MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:30, 2.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<01:48, 3.12MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:21, 1.29MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:41, 1.53MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:43, 2.06MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:05, 1.80MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:28, 1.59MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:42, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<01:58, 2.80MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:25, 1.60MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:02, 1.80MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:16, 2.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:44, 1.97MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:11, 1.69MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:30, 2.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<01:51, 2.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:40, 2.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:28, 2.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<01:52, 2.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:28, 2.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:17, 2.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<01:43, 3.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:22, 2.20MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:47, 1.86MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:14, 2.32MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<01:36, 3.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<15:52, 324kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<11:41, 439kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<08:17, 617kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<06:50, 741kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<05:21, 944kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<03:51, 1.31MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:46, 1.33MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:13, 1.55MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:23, 2.09MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:42, 1.82MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:05, 1.59MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:24, 2.04MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:44, 2.79MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<03:06, 1.56MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:44, 1.77MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:02, 2.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:27, 1.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:51, 1.67MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:15, 2.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<01:38, 2.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<05:00, 942kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<04:03, 1.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:57, 1.58MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:03, 1.52MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:12, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:30, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<01:48, 2.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<05:17, 866kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<04:14, 1.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<03:04, 1.48MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<03:06, 1.45MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:12, 1.41MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:28, 1.82MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:45, 2.52MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<04:05, 1.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:20, 1.33MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:26, 1.80MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:40, 1.64MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:54, 1.50MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:14, 1.94MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:36, 2.68MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<03:06, 1.38MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:40, 1.60MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:58, 2.17MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:26<01:25, 2.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<08:15, 513kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<06:15, 676kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<04:28, 940kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<04:00, 1.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:15, 1.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:22, 1.74MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:34, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:46, 1.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:08, 1.91MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:32, 2.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:30, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:12, 1.83MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:38, 2.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:01, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:18, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:48, 2.19MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:18, 2.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:18, 1.69MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:03, 1.89MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:32, 2.51MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:53, 2.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<01:45, 2.17MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:19, 2.84MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:43, 2.17MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<02:05, 1.79MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:39, 2.26MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:12, 3.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:16, 1.62MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:00, 1.83MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:30, 2.43MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:49, 1.98MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:40, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:16, 2.81MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:38, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:01, 1.74MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:36, 2.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:10, 2.96MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:44, 1.99MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:38, 2.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:15, 2.75MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:33, 2.18MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<01:29, 2.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:08, 2.96MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:29, 2.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:25, 2.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:04, 3.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:26, 2.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:46, 1.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:25, 2.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:02, 3.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<03:50, 830kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<03:02, 1.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:12, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<02:13, 1.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<02:15, 1.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:44, 1.79MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:14, 2.48MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<02:19, 1.31MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:57, 1.56MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:25, 2.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:39, 1.80MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:48, 1.65MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:25, 2.09MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:00, 2.88MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<06:40, 436kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<05:00, 582kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<03:33, 811kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<03:04, 927kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<02:28, 1.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:46, 1.58MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:15, 2.20MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<24:40, 113kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<17:54, 155kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<12:37, 219kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<08:50, 310kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<06:43, 403kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<05:00, 539kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<03:33, 754kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<03:01, 874kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:25, 1.09MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:44, 1.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:45, 1.46MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:47, 1.44MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:22, 1.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<00:58, 2.56MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<08:23, 298kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<06:08, 406kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<04:19, 571kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<03:30, 694kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<02:58, 817kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<02:11, 1.10MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<01:32, 1.54MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<08:09, 290kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<05:57, 395kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<04:11, 557kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<03:01, 760kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<1:44:35, 22.0kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<1:12:58, 31.3kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:23<50:02, 44.7kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 729M/862M [05:25<36:55, 60.3kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<26:20, 84.5kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<18:26, 120kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<12:45, 171kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<09:35, 225kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<06:56, 310kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<04:52, 437kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<03:47, 550kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<03:06, 672kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<02:15, 920kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<01:35, 1.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:43, 1.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:26, 1.40MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:02, 1.91MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:07, 1.72MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:15, 1.55MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:59, 1.96MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:33<00:41, 2.71MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<02:10, 863kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:43, 1.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:14, 1.50MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:15, 1.44MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:04, 1.69MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:47, 2.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:56, 1.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:48, 2.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:37, 2.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<00:26, 3.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<02:21, 709kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:49, 912kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:18, 1.26MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:15, 1.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:03, 1.51MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:46, 2.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:51, 1.79MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:56, 1.64MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:43, 2.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:45<00:30, 2.88MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<02:29, 590kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:52, 780kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:19, 1.09MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:47<00:55, 1.51MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:50, 756kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:35, 874kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:10, 1.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:49, 1.64MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:58, 1.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:49, 1.59MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:36, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<00:41, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:47, 1.60MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:36, 2.05MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<00:27, 2.72MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:34, 2.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:31, 2.23MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:23, 2.97MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:55<00:16, 4.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<17:02, 65.7kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<12:08, 91.9kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<08:28, 130kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<05:44, 186kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<04:25, 237kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<03:11, 327kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<02:12, 461kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<01:41, 577kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<01:24, 698kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<01:00, 953kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:42, 1.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:44, 1.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:37, 1.45MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:26, 1.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:28, 1.75MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:32, 1.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:25, 1.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<00:17, 2.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:31, 1.47MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:27, 1.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:19, 2.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:22, 1.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:25, 1.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:19, 2.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:13, 2.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:20, 1.87MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:18, 2.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:13, 2.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:16, 2.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:14, 2.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:10, 2.93MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:13, 2.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:15, 1.87MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:12, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:08, 3.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<01:18, 326kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:56, 442kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:38, 622kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:28, 746kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:24, 857kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:17, 1.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:12, 1.59MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:11, 1.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:10, 1.66MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:21<00:04, 3.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:44, 296kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:33, 386kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:22, 537kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:14, 756kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:10, 828kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:08, 1.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.43MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:03, 1.42MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:03, 1.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.81MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.49MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.54MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.76MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1065/400000 [00:00<00:37, 10644.15it/s]  1%|          | 2157/400000 [00:00<00:37, 10724.58it/s]  1%|          | 3130/400000 [00:00<00:38, 10405.13it/s]  1%|          | 3808/400000 [00:00<00:50, 7903.11it/s]   1%|          | 4881/400000 [00:00<00:46, 8580.88it/s]  1%|▏         | 5977/400000 [00:00<00:42, 9177.77it/s]  2%|▏         | 6993/400000 [00:00<00:41, 9448.99it/s]  2%|▏         | 8032/400000 [00:00<00:40, 9711.17it/s]  2%|▏         | 8979/400000 [00:00<00:40, 9585.86it/s]  2%|▏         | 9984/400000 [00:01<00:40, 9718.79it/s]  3%|▎         | 11090/400000 [00:01<00:38, 10084.16it/s]  3%|▎         | 12256/400000 [00:01<00:36, 10509.44it/s]  3%|▎         | 13309/400000 [00:01<00:36, 10506.49it/s]  4%|▎         | 14361/400000 [00:01<00:37, 10369.38it/s]  4%|▍         | 15400/400000 [00:01<00:38, 9913.22it/s]   4%|▍         | 16430/400000 [00:01<00:38, 10025.27it/s]  4%|▍         | 17437/400000 [00:01<00:38, 10033.02it/s]  5%|▍         | 18443/400000 [00:01<00:38, 9998.53it/s]   5%|▍         | 19445/400000 [00:01<00:38, 9943.84it/s]  5%|▌         | 20441/400000 [00:02<00:38, 9893.94it/s]  5%|▌         | 21432/400000 [00:02<00:41, 9079.16it/s]  6%|▌         | 22393/400000 [00:02<00:40, 9230.28it/s]  6%|▌         | 23388/400000 [00:02<00:39, 9433.03it/s]  6%|▌         | 24387/400000 [00:02<00:39, 9592.60it/s]  6%|▋         | 25357/400000 [00:02<00:38, 9624.15it/s]  7%|▋         | 26324/400000 [00:02<00:39, 9534.57it/s]  7%|▋         | 27281/400000 [00:02<00:39, 9391.71it/s]  7%|▋         | 28354/400000 [00:02<00:38, 9754.56it/s]  7%|▋         | 29471/400000 [00:03<00:36, 10138.11it/s]  8%|▊         | 30493/400000 [00:03<00:36, 9996.62it/s]   8%|▊         | 31505/400000 [00:03<00:36, 10031.25it/s]  8%|▊         | 32514/400000 [00:03<00:36, 10047.23it/s]  8%|▊         | 33547/400000 [00:03<00:36, 10128.54it/s]  9%|▊         | 34670/400000 [00:03<00:35, 10434.07it/s]  9%|▉         | 35774/400000 [00:03<00:34, 10606.47it/s]  9%|▉         | 36839/400000 [00:03<00:34, 10565.03it/s]  9%|▉         | 37898/400000 [00:03<00:34, 10544.67it/s] 10%|▉         | 38955/400000 [00:03<00:35, 10214.53it/s] 10%|▉         | 39985/400000 [00:04<00:35, 10238.51it/s] 10%|█         | 41061/400000 [00:04<00:34, 10388.50it/s] 11%|█         | 42201/400000 [00:04<00:33, 10672.48it/s] 11%|█         | 43335/400000 [00:04<00:32, 10863.56it/s] 11%|█         | 44425/400000 [00:04<00:32, 10817.82it/s] 11%|█▏        | 45510/400000 [00:04<00:36, 9822.13it/s]  12%|█▏        | 46512/400000 [00:04<00:36, 9778.94it/s] 12%|█▏        | 47504/400000 [00:04<00:37, 9492.63it/s] 12%|█▏        | 48465/400000 [00:04<00:37, 9409.20it/s] 12%|█▏        | 49507/400000 [00:04<00:36, 9690.26it/s] 13%|█▎        | 50484/400000 [00:05<00:38, 9090.41it/s] 13%|█▎        | 51481/400000 [00:05<00:37, 9336.87it/s] 13%|█▎        | 52535/400000 [00:05<00:35, 9666.67it/s] 13%|█▎        | 53536/400000 [00:05<00:35, 9766.28it/s] 14%|█▎        | 54613/400000 [00:05<00:34, 10045.67it/s] 14%|█▍        | 55741/400000 [00:05<00:33, 10386.05it/s] 14%|█▍        | 56788/400000 [00:05<00:33, 10097.53it/s] 14%|█▍        | 57806/400000 [00:05<00:34, 9939.68it/s]  15%|█▍        | 58806/400000 [00:05<00:37, 9064.69it/s] 15%|█▍        | 59732/400000 [00:06<00:39, 8616.29it/s] 15%|█▌        | 60660/400000 [00:06<00:38, 8803.76it/s] 15%|█▌        | 61666/400000 [00:06<00:37, 9143.97it/s] 16%|█▌        | 62594/400000 [00:06<00:37, 8953.95it/s] 16%|█▌        | 63500/400000 [00:06<00:38, 8775.69it/s] 16%|█▌        | 64544/400000 [00:06<00:36, 9213.90it/s] 16%|█▋        | 65521/400000 [00:06<00:35, 9371.98it/s] 17%|█▋        | 66638/400000 [00:06<00:33, 9845.85it/s] 17%|█▋        | 67635/400000 [00:06<00:34, 9656.23it/s] 17%|█▋        | 68652/400000 [00:06<00:33, 9803.29it/s] 17%|█▋        | 69683/400000 [00:07<00:33, 9946.23it/s] 18%|█▊        | 70791/400000 [00:07<00:32, 10259.80it/s] 18%|█▊        | 71824/400000 [00:07<00:32, 10128.18it/s] 18%|█▊        | 72870/400000 [00:07<00:32, 10222.64it/s] 18%|█▊        | 73896/400000 [00:07<00:32, 9897.34it/s]  19%|█▊        | 74982/400000 [00:07<00:31, 10166.17it/s] 19%|█▉        | 76079/400000 [00:07<00:31, 10393.11it/s] 19%|█▉        | 77174/400000 [00:07<00:30, 10552.23it/s] 20%|█▉        | 78234/400000 [00:07<00:31, 10338.49it/s] 20%|█▉        | 79272/400000 [00:08<00:31, 10290.50it/s] 20%|██        | 80304/400000 [00:08<00:32, 9688.49it/s]  20%|██        | 81356/400000 [00:08<00:32, 9922.14it/s] 21%|██        | 82375/400000 [00:08<00:31, 9998.05it/s] 21%|██        | 83418/400000 [00:08<00:31, 10122.07it/s] 21%|██        | 84453/400000 [00:08<00:30, 10187.30it/s] 21%|██▏       | 85532/400000 [00:08<00:30, 10359.24it/s] 22%|██▏       | 86571/400000 [00:08<00:31, 9872.25it/s]  22%|██▏       | 87702/400000 [00:08<00:30, 10261.76it/s] 22%|██▏       | 88742/400000 [00:08<00:30, 10302.03it/s] 22%|██▏       | 89869/400000 [00:09<00:29, 10574.16it/s] 23%|██▎       | 91047/400000 [00:09<00:28, 10908.53it/s] 23%|██▎       | 92145/400000 [00:09<00:28, 10799.06it/s] 23%|██▎       | 93234/400000 [00:09<00:28, 10823.43it/s] 24%|██▎       | 94320/400000 [00:09<00:28, 10580.72it/s] 24%|██▍       | 95382/400000 [00:09<00:28, 10524.26it/s] 24%|██▍       | 96478/400000 [00:09<00:28, 10650.23it/s] 24%|██▍       | 97555/400000 [00:09<00:28, 10685.07it/s] 25%|██▍       | 98626/400000 [00:09<00:29, 10082.45it/s] 25%|██▍       | 99643/400000 [00:09<00:29, 10057.86it/s] 25%|██▌       | 100655/400000 [00:10<00:29, 10056.74it/s] 25%|██▌       | 101695/400000 [00:10<00:29, 10154.21it/s] 26%|██▌       | 102773/400000 [00:10<00:28, 10331.99it/s] 26%|██▌       | 103845/400000 [00:10<00:28, 10443.49it/s] 26%|██▌       | 104908/400000 [00:10<00:28, 10497.97it/s] 27%|██▋       | 106011/400000 [00:10<00:27, 10651.60it/s] 27%|██▋       | 107146/400000 [00:10<00:26, 10849.94it/s] 27%|██▋       | 108234/400000 [00:10<00:27, 10553.65it/s] 27%|██▋       | 109293/400000 [00:10<00:27, 10536.46it/s] 28%|██▊       | 110349/400000 [00:11<00:27, 10514.69it/s] 28%|██▊       | 111403/400000 [00:11<00:27, 10506.21it/s] 28%|██▊       | 112455/400000 [00:11<00:27, 10338.58it/s] 28%|██▊       | 113491/400000 [00:11<00:27, 10243.93it/s] 29%|██▊       | 114517/400000 [00:11<00:29, 9807.35it/s]  29%|██▉       | 115557/400000 [00:11<00:28, 9976.35it/s] 29%|██▉       | 116612/400000 [00:11<00:27, 10141.10it/s] 29%|██▉       | 117693/400000 [00:11<00:27, 10331.05it/s] 30%|██▉       | 118784/400000 [00:11<00:26, 10497.29it/s] 30%|██▉       | 119837/400000 [00:11<00:26, 10404.62it/s] 30%|███       | 120880/400000 [00:12<00:26, 10353.97it/s] 30%|███       | 121988/400000 [00:12<00:26, 10561.40it/s] 31%|███       | 123086/400000 [00:12<00:25, 10683.13it/s] 31%|███       | 124164/400000 [00:12<00:25, 10709.63it/s] 31%|███▏      | 125237/400000 [00:12<00:25, 10665.27it/s] 32%|███▏      | 126305/400000 [00:12<00:26, 10395.47it/s] 32%|███▏      | 127356/400000 [00:12<00:26, 10426.68it/s] 32%|███▏      | 128464/400000 [00:12<00:25, 10613.67it/s] 32%|███▏      | 129528/400000 [00:12<00:25, 10583.49it/s] 33%|███▎      | 130588/400000 [00:12<00:25, 10543.75it/s] 33%|███▎      | 131656/400000 [00:13<00:25, 10581.63it/s] 33%|███▎      | 132735/400000 [00:13<00:25, 10640.23it/s] 33%|███▎      | 133819/400000 [00:13<00:24, 10696.25it/s] 34%|███▎      | 134890/400000 [00:13<00:25, 10409.41it/s] 34%|███▍      | 135933/400000 [00:13<00:26, 10033.95it/s] 34%|███▍      | 136941/400000 [00:13<00:26, 9870.44it/s]  35%|███▍      | 138004/400000 [00:13<00:25, 10083.82it/s] 35%|███▍      | 139117/400000 [00:13<00:25, 10376.23it/s] 35%|███▌      | 140178/400000 [00:13<00:24, 10443.20it/s] 35%|███▌      | 141226/400000 [00:13<00:25, 10167.14it/s] 36%|███▌      | 142276/400000 [00:14<00:25, 10262.26it/s] 36%|███▌      | 143363/400000 [00:14<00:24, 10436.93it/s] 36%|███▌      | 144410/400000 [00:14<00:24, 10298.37it/s] 36%|███▋      | 145443/400000 [00:14<00:25, 10136.39it/s] 37%|███▋      | 146459/400000 [00:14<00:25, 9995.78it/s]  37%|███▋      | 147492/400000 [00:14<00:25, 10093.50it/s] 37%|███▋      | 148503/400000 [00:14<00:25, 9898.77it/s]  37%|███▋      | 149539/400000 [00:14<00:24, 10032.23it/s] 38%|███▊      | 150545/400000 [00:14<00:25, 9888.28it/s]  38%|███▊      | 151543/400000 [00:15<00:25, 9913.10it/s] 38%|███▊      | 152619/400000 [00:15<00:24, 10150.88it/s] 38%|███▊      | 153739/400000 [00:15<00:23, 10442.30it/s] 39%|███▊      | 154827/400000 [00:15<00:23, 10569.73it/s] 39%|███▉      | 155896/400000 [00:15<00:23, 10602.27it/s] 39%|███▉      | 156993/400000 [00:15<00:22, 10708.55it/s] 40%|███▉      | 158099/400000 [00:15<00:22, 10810.96it/s] 40%|███▉      | 159182/400000 [00:15<00:22, 10666.63it/s] 40%|████      | 160251/400000 [00:15<00:22, 10533.12it/s] 40%|████      | 161306/400000 [00:15<00:23, 10319.99it/s] 41%|████      | 162340/400000 [00:16<00:23, 10193.78it/s] 41%|████      | 163362/400000 [00:16<00:23, 10112.47it/s] 41%|████      | 164397/400000 [00:16<00:23, 10179.39it/s] 41%|████▏     | 165434/400000 [00:16<00:22, 10233.16it/s] 42%|████▏     | 166459/400000 [00:16<00:22, 10173.94it/s] 42%|████▏     | 167478/400000 [00:16<00:23, 9878.98it/s]  42%|████▏     | 168482/400000 [00:16<00:23, 9924.17it/s] 42%|████▏     | 169569/400000 [00:16<00:22, 10188.55it/s] 43%|████▎     | 170626/400000 [00:16<00:22, 10298.35it/s] 43%|████▎     | 171659/400000 [00:16<00:22, 10095.67it/s] 43%|████▎     | 172672/400000 [00:17<00:22, 10051.16it/s] 43%|████▎     | 173775/400000 [00:17<00:21, 10325.01it/s] 44%|████▎     | 174883/400000 [00:17<00:21, 10538.39it/s] 44%|████▍     | 175988/400000 [00:17<00:20, 10686.27it/s] 44%|████▍     | 177078/400000 [00:17<00:20, 10748.92it/s] 45%|████▍     | 178155/400000 [00:17<00:21, 10528.66it/s] 45%|████▍     | 179289/400000 [00:17<00:20, 10758.05it/s] 45%|████▌     | 180394/400000 [00:17<00:20, 10843.81it/s] 45%|████▌     | 181481/400000 [00:17<00:20, 10845.76it/s] 46%|████▌     | 182568/400000 [00:17<00:20, 10776.10it/s] 46%|████▌     | 183647/400000 [00:18<00:20, 10678.11it/s] 46%|████▌     | 184720/400000 [00:18<00:20, 10691.68it/s] 46%|████▋     | 185790/400000 [00:18<00:20, 10496.97it/s] 47%|████▋     | 186883/400000 [00:18<00:20, 10621.30it/s] 47%|████▋     | 187983/400000 [00:18<00:19, 10729.60it/s] 47%|████▋     | 189058/400000 [00:18<00:19, 10684.47it/s] 48%|████▊     | 190159/400000 [00:18<00:19, 10779.24it/s] 48%|████▊     | 191267/400000 [00:18<00:19, 10867.30it/s] 48%|████▊     | 192355/400000 [00:18<00:19, 10805.97it/s] 48%|████▊     | 193437/400000 [00:18<00:19, 10425.48it/s] 49%|████▊     | 194483/400000 [00:19<00:19, 10342.65it/s] 49%|████▉     | 195520/400000 [00:19<00:19, 10307.16it/s] 49%|████▉     | 196575/400000 [00:19<00:19, 10378.75it/s] 49%|████▉     | 197615/400000 [00:19<00:19, 10363.72it/s] 50%|████▉     | 198693/400000 [00:19<00:19, 10484.26it/s] 50%|████▉     | 199743/400000 [00:19<00:19, 10306.06it/s] 50%|█████     | 200823/400000 [00:19<00:19, 10447.88it/s] 50%|█████     | 201870/400000 [00:19<00:19, 10312.27it/s] 51%|█████     | 202903/400000 [00:19<00:19, 10085.16it/s] 51%|█████     | 203914/400000 [00:20<00:20, 9690.73it/s]  51%|█████     | 204888/400000 [00:20<00:21, 9270.67it/s] 51%|█████▏    | 205823/400000 [00:20<00:21, 8975.38it/s] 52%|█████▏    | 206855/400000 [00:20<00:20, 9338.85it/s] 52%|█████▏    | 207909/400000 [00:20<00:19, 9667.73it/s] 52%|█████▏    | 208885/400000 [00:20<00:20, 9516.06it/s] 52%|█████▏    | 209904/400000 [00:20<00:19, 9706.85it/s] 53%|█████▎    | 210933/400000 [00:20<00:19, 9872.53it/s] 53%|█████▎    | 211925/400000 [00:20<00:19, 9749.35it/s] 53%|█████▎    | 213009/400000 [00:20<00:18, 10052.23it/s] 54%|█████▎    | 214020/400000 [00:21<00:18, 10036.13it/s] 54%|█████▍    | 215028/400000 [00:21<00:18, 9947.42it/s]  54%|█████▍    | 216146/400000 [00:21<00:17, 10287.01it/s] 54%|█████▍    | 217273/400000 [00:21<00:17, 10561.36it/s] 55%|█████▍    | 218341/400000 [00:21<00:17, 10595.65it/s] 55%|█████▍    | 219414/400000 [00:21<00:16, 10632.65it/s] 55%|█████▌    | 220480/400000 [00:21<00:17, 10298.47it/s] 55%|█████▌    | 221514/400000 [00:21<00:17, 10099.49it/s] 56%|█████▌    | 222528/400000 [00:21<00:17, 9987.37it/s]  56%|█████▌    | 223614/400000 [00:22<00:17, 10233.35it/s] 56%|█████▌    | 224668/400000 [00:22<00:16, 10320.30it/s] 56%|█████▋    | 225737/400000 [00:22<00:16, 10425.79it/s] 57%|█████▋    | 226782/400000 [00:22<00:17, 9990.61it/s]  57%|█████▋    | 227817/400000 [00:22<00:17, 10094.30it/s] 57%|█████▋    | 228877/400000 [00:22<00:16, 10240.75it/s] 57%|█████▋    | 229905/400000 [00:22<00:16, 10058.12it/s] 58%|█████▊    | 230944/400000 [00:22<00:16, 10152.34it/s] 58%|█████▊    | 232012/400000 [00:22<00:16, 10299.95it/s] 58%|█████▊    | 233075/400000 [00:22<00:16, 10394.56it/s] 59%|█████▊    | 234140/400000 [00:23<00:15, 10468.69it/s] 59%|█████▉    | 235189/400000 [00:23<00:15, 10315.71it/s] 59%|█████▉    | 236223/400000 [00:23<00:16, 10050.43it/s] 59%|█████▉    | 237231/400000 [00:23<00:16, 9902.56it/s]  60%|█████▉    | 238224/400000 [00:23<00:16, 9868.50it/s] 60%|█████▉    | 239256/400000 [00:23<00:16, 9999.37it/s] 60%|██████    | 240261/400000 [00:23<00:15, 10013.25it/s] 60%|██████    | 241342/400000 [00:23<00:15, 10239.38it/s] 61%|██████    | 242410/400000 [00:23<00:15, 10366.34it/s] 61%|██████    | 243454/400000 [00:23<00:15, 10386.55it/s] 61%|██████    | 244494/400000 [00:24<00:15, 10362.17it/s] 61%|██████▏   | 245532/400000 [00:24<00:15, 10030.57it/s] 62%|██████▏   | 246552/400000 [00:24<00:15, 10078.89it/s] 62%|██████▏   | 247604/400000 [00:24<00:14, 10205.55it/s] 62%|██████▏   | 248665/400000 [00:24<00:14, 10321.57it/s] 62%|██████▏   | 249749/400000 [00:24<00:14, 10471.41it/s] 63%|██████▎   | 250798/400000 [00:24<00:14, 10306.90it/s] 63%|██████▎   | 251831/400000 [00:24<00:14, 10304.20it/s] 63%|██████▎   | 252907/400000 [00:24<00:14, 10436.13it/s] 63%|██████▎   | 253957/400000 [00:24<00:13, 10454.06it/s] 64%|██████▍   | 255004/400000 [00:25<00:13, 10357.60it/s] 64%|██████▍   | 256041/400000 [00:25<00:14, 10088.33it/s] 64%|██████▍   | 257052/400000 [00:25<00:14, 10001.43it/s] 65%|██████▍   | 258098/400000 [00:25<00:14, 10133.02it/s] 65%|██████▍   | 259157/400000 [00:25<00:13, 10263.28it/s] 65%|██████▌   | 260195/400000 [00:25<00:13, 10296.59it/s] 65%|██████▌   | 261226/400000 [00:25<00:13, 10099.29it/s] 66%|██████▌   | 262238/400000 [00:25<00:13, 9919.32it/s]  66%|██████▌   | 263232/400000 [00:25<00:13, 9876.02it/s] 66%|██████▌   | 264221/400000 [00:25<00:13, 9879.06it/s] 66%|██████▋   | 265245/400000 [00:26<00:13, 9983.84it/s] 67%|██████▋   | 266245/400000 [00:26<00:13, 9936.53it/s] 67%|██████▋   | 267279/400000 [00:26<00:13, 10051.62it/s] 67%|██████▋   | 268285/400000 [00:26<00:13, 9877.49it/s]  67%|██████▋   | 269274/400000 [00:26<00:13, 9829.48it/s] 68%|██████▊   | 270258/400000 [00:26<00:13, 9826.01it/s] 68%|██████▊   | 271242/400000 [00:26<00:13, 9695.46it/s] 68%|██████▊   | 272236/400000 [00:26<00:13, 9767.54it/s] 68%|██████▊   | 273214/400000 [00:26<00:13, 9712.78it/s] 69%|██████▊   | 274239/400000 [00:27<00:12, 9867.83it/s] 69%|██████▉   | 275289/400000 [00:27<00:12, 10047.85it/s] 69%|██████▉   | 276296/400000 [00:27<00:12, 9926.37it/s]  69%|██████▉   | 277356/400000 [00:27<00:12, 10118.68it/s] 70%|██████▉   | 278387/400000 [00:27<00:11, 10173.35it/s] 70%|██████▉   | 279419/400000 [00:27<00:11, 10214.63it/s] 70%|███████   | 280442/400000 [00:27<00:11, 10173.80it/s] 70%|███████   | 281461/400000 [00:27<00:11, 10165.47it/s] 71%|███████   | 282527/400000 [00:27<00:11, 10306.14it/s] 71%|███████   | 283575/400000 [00:27<00:11, 10355.77it/s] 71%|███████   | 284612/400000 [00:28<00:11, 10329.97it/s] 71%|███████▏  | 285646/400000 [00:28<00:11, 10328.11it/s] 72%|███████▏  | 286680/400000 [00:28<00:11, 10217.49it/s] 72%|███████▏  | 287703/400000 [00:28<00:11, 10042.75it/s] 72%|███████▏  | 288709/400000 [00:28<00:11, 9940.68it/s]  72%|███████▏  | 289716/400000 [00:28<00:11, 9976.92it/s] 73%|███████▎  | 290785/400000 [00:28<00:10, 10179.55it/s] 73%|███████▎  | 291805/400000 [00:28<00:10, 10035.95it/s] 73%|███████▎  | 292868/400000 [00:28<00:10, 10206.12it/s] 73%|███████▎  | 293966/400000 [00:28<00:10, 10426.42it/s] 74%|███████▍  | 295012/400000 [00:29<00:10, 10360.96it/s] 74%|███████▍  | 296050/400000 [00:29<00:10, 10294.31it/s] 74%|███████▍  | 297081/400000 [00:29<00:10, 10186.60it/s] 75%|███████▍  | 298151/400000 [00:29<00:09, 10333.88it/s] 75%|███████▍  | 299227/400000 [00:29<00:09, 10456.53it/s] 75%|███████▌  | 300332/400000 [00:29<00:09, 10626.09it/s] 75%|███████▌  | 301427/400000 [00:29<00:09, 10719.49it/s] 76%|███████▌  | 302501/400000 [00:29<00:09, 10598.76it/s] 76%|███████▌  | 303563/400000 [00:29<00:09, 10504.98it/s] 76%|███████▌  | 304643/400000 [00:29<00:09, 10590.60it/s] 76%|███████▋  | 305703/400000 [00:30<00:09, 10332.78it/s] 77%|███████▋  | 306739/400000 [00:30<00:09, 10311.57it/s] 77%|███████▋  | 307772/400000 [00:30<00:09, 9986.00it/s]  77%|███████▋  | 308857/400000 [00:30<00:08, 10228.42it/s] 77%|███████▋  | 309951/400000 [00:30<00:08, 10430.11it/s] 78%|███████▊  | 311064/400000 [00:30<00:08, 10630.63it/s] 78%|███████▊  | 312131/400000 [00:30<00:08, 10608.68it/s] 78%|███████▊  | 313195/400000 [00:30<00:08, 10510.75it/s] 79%|███████▊  | 314257/400000 [00:30<00:08, 10543.16it/s] 79%|███████▉  | 315313/400000 [00:30<00:08, 10191.11it/s] 79%|███████▉  | 316402/400000 [00:31<00:08, 10388.80it/s] 79%|███████▉  | 317475/400000 [00:31<00:07, 10488.44it/s] 80%|███████▉  | 318530/400000 [00:31<00:07, 10506.34it/s] 80%|███████▉  | 319583/400000 [00:31<00:07, 10457.57it/s] 80%|████████  | 320631/400000 [00:31<00:07, 10229.98it/s] 80%|████████  | 321735/400000 [00:31<00:07, 10459.36it/s] 81%|████████  | 322784/400000 [00:31<00:07, 10440.44it/s] 81%|████████  | 323830/400000 [00:31<00:07, 10352.32it/s] 81%|████████  | 324907/400000 [00:31<00:07, 10471.07it/s] 81%|████████▏ | 325956/400000 [00:32<00:07, 10456.91it/s] 82%|████████▏ | 327003/400000 [00:32<00:07, 10369.88it/s] 82%|████████▏ | 328041/400000 [00:32<00:06, 10339.28it/s] 82%|████████▏ | 329076/400000 [00:32<00:06, 10219.86it/s] 83%|████████▎ | 330155/400000 [00:32<00:06, 10381.49it/s] 83%|████████▎ | 331202/400000 [00:32<00:06, 10406.35it/s] 83%|████████▎ | 332244/400000 [00:32<00:06, 10357.38it/s] 83%|████████▎ | 333321/400000 [00:32<00:06, 10477.39it/s] 84%|████████▎ | 334372/400000 [00:32<00:06, 10485.93it/s] 84%|████████▍ | 335457/400000 [00:32<00:06, 10590.52it/s] 84%|████████▍ | 336517/400000 [00:33<00:05, 10583.76it/s] 84%|████████▍ | 337576/400000 [00:33<00:05, 10528.68it/s] 85%|████████▍ | 338637/400000 [00:33<00:05, 10551.77it/s] 85%|████████▍ | 339693/400000 [00:33<00:05, 10474.47it/s] 85%|████████▌ | 340741/400000 [00:33<00:05, 10322.72it/s] 85%|████████▌ | 341779/400000 [00:33<00:05, 10339.76it/s] 86%|████████▌ | 342849/400000 [00:33<00:05, 10443.04it/s] 86%|████████▌ | 343894/400000 [00:33<00:05, 10358.72it/s] 86%|████████▌ | 344931/400000 [00:33<00:05, 10241.34it/s] 87%|████████▋ | 346042/400000 [00:33<00:05, 10485.73it/s] 87%|████████▋ | 347150/400000 [00:34<00:04, 10654.92it/s] 87%|████████▋ | 348218/400000 [00:34<00:04, 10646.78it/s] 87%|████████▋ | 349301/400000 [00:34<00:04, 10699.31it/s] 88%|████████▊ | 350372/400000 [00:34<00:04, 10660.19it/s] 88%|████████▊ | 351439/400000 [00:34<00:04, 10634.74it/s] 88%|████████▊ | 352504/400000 [00:34<00:04, 10625.77it/s] 88%|████████▊ | 353567/400000 [00:34<00:04, 10497.67it/s] 89%|████████▊ | 354618/400000 [00:34<00:04, 10465.36it/s] 89%|████████▉ | 355665/400000 [00:34<00:04, 10318.78it/s] 89%|████████▉ | 356714/400000 [00:34<00:04, 10369.05it/s] 89%|████████▉ | 357779/400000 [00:35<00:04, 10451.35it/s] 90%|████████▉ | 358825/400000 [00:35<00:04, 10143.54it/s] 90%|████████▉ | 359847/400000 [00:35<00:03, 10164.36it/s] 90%|█████████ | 360872/400000 [00:35<00:03, 10189.10it/s] 90%|█████████ | 361904/400000 [00:35<00:03, 10226.88it/s] 91%|█████████ | 362977/400000 [00:35<00:03, 10370.82it/s] 91%|█████████ | 364078/400000 [00:35<00:03, 10554.01it/s] 91%|█████████▏| 365136/400000 [00:35<00:03, 10559.98it/s] 92%|█████████▏| 366194/400000 [00:35<00:03, 10427.20it/s] 92%|█████████▏| 367238/400000 [00:35<00:03, 10377.22it/s] 92%|█████████▏| 368282/400000 [00:36<00:03, 10395.70it/s] 92%|█████████▏| 369323/400000 [00:36<00:02, 10391.60it/s] 93%|█████████▎| 370363/400000 [00:36<00:02, 10235.61it/s] 93%|█████████▎| 371429/400000 [00:36<00:02, 10358.51it/s] 93%|█████████▎| 372466/400000 [00:36<00:02, 10240.24it/s] 93%|█████████▎| 373510/400000 [00:36<00:02, 10296.62it/s] 94%|█████████▎| 374575/400000 [00:36<00:02, 10397.66it/s] 94%|█████████▍| 375616/400000 [00:36<00:02, 10341.29it/s] 94%|█████████▍| 376651/400000 [00:36<00:02, 10050.54it/s] 94%|█████████▍| 377680/400000 [00:36<00:02, 10120.32it/s] 95%|█████████▍| 378742/400000 [00:37<00:02, 10264.73it/s] 95%|█████████▍| 379813/400000 [00:37<00:01, 10393.94it/s] 95%|█████████▌| 380919/400000 [00:37<00:01, 10583.22it/s] 95%|█████████▌| 381980/400000 [00:37<00:01, 10486.26it/s] 96%|█████████▌| 383031/400000 [00:37<00:01, 10346.39it/s] 96%|█████████▌| 384068/400000 [00:37<00:01, 10210.13it/s] 96%|█████████▋| 385100/400000 [00:37<00:01, 10241.01it/s] 97%|█████████▋| 386160/400000 [00:37<00:01, 10344.23it/s] 97%|█████████▋| 387196/400000 [00:37<00:01, 10131.13it/s] 97%|█████████▋| 388211/400000 [00:38<00:01, 9927.94it/s]  97%|█████████▋| 389224/400000 [00:38<00:01, 9985.85it/s] 98%|█████████▊| 390264/400000 [00:38<00:00, 10104.68it/s] 98%|█████████▊| 391276/400000 [00:38<00:00, 9959.14it/s]  98%|█████████▊| 392274/400000 [00:38<00:00, 9953.67it/s] 98%|█████████▊| 393281/400000 [00:38<00:00, 9986.16it/s] 99%|█████████▊| 394281/400000 [00:38<00:00, 9855.10it/s] 99%|█████████▉| 395359/400000 [00:38<00:00, 10113.86it/s] 99%|█████████▉| 396388/400000 [00:38<00:00, 10163.23it/s] 99%|█████████▉| 397443/400000 [00:38<00:00, 10274.54it/s]100%|█████████▉| 398507/400000 [00:39<00:00, 10380.48it/s]100%|█████████▉| 399569/400000 [00:39<00:00, 10449.06it/s]100%|█████████▉| 399999/400000 [00:39<00:00, 10215.19it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f14c7beea58> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011307903113794942 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.011626720827160073 	 Accuracy: 46

  model saves at 46% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15947 out of table with 15889 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15947 out of table with 15889 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-14 00:28:03.709008: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 00:28:03.713130: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-14 00:28:03.713237: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563e8ab536c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 00:28:03.713249: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f14d5e56198> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.6973 - accuracy: 0.4980
 2000/25000 [=>............................] - ETA: 7s - loss: 7.5210 - accuracy: 0.5095 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5542 - accuracy: 0.5073
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.5018 - accuracy: 0.5107
 5000/25000 [=====>........................] - ETA: 4s - loss: 7.5562 - accuracy: 0.5072
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6436 - accuracy: 0.5015
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7017 - accuracy: 0.4977
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.7088 - accuracy: 0.4972
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6939 - accuracy: 0.4982
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6636 - accuracy: 0.5002
11000/25000 [============>.................] - ETA: 3s - loss: 7.6778 - accuracy: 0.4993
12000/25000 [=============>................] - ETA: 2s - loss: 7.6820 - accuracy: 0.4990
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6690 - accuracy: 0.4998
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6874 - accuracy: 0.4986
15000/25000 [=================>............] - ETA: 2s - loss: 7.6860 - accuracy: 0.4987
16000/25000 [==================>...........] - ETA: 1s - loss: 7.6992 - accuracy: 0.4979
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7072 - accuracy: 0.4974
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7050 - accuracy: 0.4975
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7118 - accuracy: 0.4971
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6942 - accuracy: 0.4982
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6893 - accuracy: 0.4985
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6903 - accuracy: 0.4985
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6686 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 6s 243us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f14443291d0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f1444d87160> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 983ms/step - loss: 1.4288 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.3241 - val_crf_viterbi_accuracy: 0.0000e+00

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

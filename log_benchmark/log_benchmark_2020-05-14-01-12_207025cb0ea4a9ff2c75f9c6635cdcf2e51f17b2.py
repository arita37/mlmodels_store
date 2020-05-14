
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f29c5aa3fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 01:12:40.786578
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 01:12:40.790333
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 01:12:40.793492
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 01:12:40.796643
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f29d186d438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352217.4688
Epoch 2/10

1/1 [==============================] - 0s 104ms/step - loss: 257989.8594
Epoch 3/10

1/1 [==============================] - 0s 106ms/step - loss: 172809.1406
Epoch 4/10

1/1 [==============================] - 0s 98ms/step - loss: 105039.4844
Epoch 5/10

1/1 [==============================] - 0s 91ms/step - loss: 60902.4727
Epoch 6/10

1/1 [==============================] - 0s 93ms/step - loss: 36458.4414
Epoch 7/10

1/1 [==============================] - 0s 90ms/step - loss: 23158.1094
Epoch 8/10

1/1 [==============================] - 0s 93ms/step - loss: 15554.5078
Epoch 9/10

1/1 [==============================] - 0s 108ms/step - loss: 11050.5840
Epoch 10/10

1/1 [==============================] - 0s 98ms/step - loss: 8520.9678

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.0806265  -0.8149502  -1.3788736   1.2364511  -1.1664349   0.49712008
  -0.63930976 -1.0869676  -0.54917586  0.5557425   0.2815705   1.0442086
  -0.656208   -1.1423502   0.5814869  -0.2536469   0.25720316  0.40465444
  -0.17112267 -0.10535598 -1.2435961  -2.056985    0.07525259  0.31289715
   1.1749961  -0.5205169  -0.8412032  -0.57661355  1.3780911   1.1636058
  -1.0289614   0.11543128 -0.1697757  -1.0601264   0.92120725 -0.71076465
  -0.31490344  1.7120017  -0.7918754  -1.4650396  -0.26512918  0.455469
  -1.198184    1.6595119  -0.1585487   1.0483198   0.6482826   0.3867542
  -1.140181    0.32905334 -0.43077165 -0.148574   -0.17338192  0.7800633
   0.5884652   1.0577004  -1.1707406  -1.2434978   0.60416186 -0.3605543
   0.05658107  6.510068    7.339386    7.322299    6.380466    7.533894
   6.4208117   6.7723894   6.196767    6.455833    7.4388394   6.933201
   5.017912    8.095036    6.3116217   8.497523    7.1946497   5.6681066
   7.7705474   7.318276    6.4069953   6.908912    6.459507    7.2784324
   6.9929953   8.10223     6.371503    6.548614    5.1649404   7.062907
   6.6053863   7.4836802   8.81637     6.12953     8.649297    7.787384
   7.673636    5.098923    4.7549977   7.200035    6.791771    5.454694
   5.939369    5.97046     7.099166    6.954388    6.990343    5.775554
   7.0977635   9.267586    7.67208     7.406928    6.2622905   5.550081
   7.877356    7.8224044   6.3474293   5.304336    7.105549    5.4650416
  -1.4036174  -1.1393759  -1.633693    0.10091966  0.08364511 -1.0178628
   0.16203088 -0.9220742   1.7638252   0.2267384  -0.47756606  0.9293578
   0.22904858 -1.7475204  -0.18834054  0.8929915   1.362271   -0.03552908
   1.5589185  -1.2890718   1.0248363  -0.63511455  2.2891383  -0.1701101
  -1.2710943  -0.89391816  0.7359972  -0.50285375 -0.5142113  -0.15903738
  -0.2642375   0.5546446   0.16039959  1.1888937   1.8784349  -0.60961497
  -0.3292119  -0.81543654 -1.5572512   0.77718186 -0.77329284 -1.3768861
   1.0206211  -0.95144427  1.3670094  -1.6095471  -0.5742103  -0.35936427
  -1.0445964  -0.3204943  -0.44816625  1.1874287   1.2113413  -0.36973986
  -1.4069088  -0.59600073  1.1095629  -1.7263291  -1.3766013   0.2812168
   0.41091532  1.2995263   1.8218465   1.7602053   0.9739329   1.6788146
   0.6020487   1.0145978   0.77848196  1.7404084   1.4734056   1.5650854
   2.0946074   0.14857143  0.28986746  0.98163104  0.90003324  1.5338435
   0.6152638   0.689913    0.808019    1.0967481   2.293096    0.7706512
   0.5406647   3.3339767   0.21350259  1.3186145   0.9700432   1.5812483
   1.0756053   0.24446476  1.6700847   1.0990622   0.6698683   1.5645065
   0.7326121   0.43574685  0.57887965  2.468958    2.728424    0.89668256
   1.5299988   1.33333     0.17312175  0.12095886  1.7488422   0.4152832
   1.2741064   2.3645425   1.9806814   1.4060428   0.5036962   1.4654597
   1.1273296   0.15945113  0.4708088   1.49525     1.1977459   1.4352721
   0.0161587   7.8901315   7.966807    7.222226    6.99918     8.07279
   6.605452    6.265393    7.5094986   9.280984    8.224959    8.610429
   6.848922    7.329886    7.817033    9.129397    7.0729513   6.4975314
   7.5307055   7.7491145   7.911312    9.480396    7.9103127   7.422011
   8.450869    8.211257    8.287519    6.7531157   6.2384086   8.335176
   6.9991913   6.990698    6.6888785   6.417511    9.008059    6.343385
   8.947332    8.006759    9.063935    8.05062     7.2432547   7.8715234
   7.7677393   8.031303    8.498927    7.2423844   9.034718    8.075108
   8.1048      5.3049407   7.5810695   8.351935    8.378806    7.1113067
   5.769006    6.0067525   8.180689    7.140694    7.3183484   6.1602254
   0.57710564  1.645178    2.4707313   0.49482048  0.36621004  1.2275822
   0.8671854   2.236425    0.5050495   0.47165138  0.6241882   0.54903287
   1.4111848   0.67656803  2.0286694   1.3248651   1.5943804   1.25247
   0.7865045   1.4443167   0.7727653   0.5180756   1.6406457   1.9863878
   0.19470716  2.0042834   0.82172287  0.18540198  0.48360765  1.42312
   1.2224939   0.88824433  2.2981358   1.872335    2.1033816   0.30862153
   0.8235996   0.78030753  0.6440321   1.0167444   0.3012789   0.78126204
   0.6509107   1.6903775   0.6066879   1.5460117   0.2563722   1.2145805
   0.6588895   1.6801779   0.6839031   2.401063    1.9103695   0.5305445
   0.7376041   0.6233496   0.42262685  0.3901049   2.0622618   2.2203887
  -9.40047     1.9090078  -2.2520916 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 01:12:49.508340
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.8497
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 01:12:49.512168
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9016.84
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 01:12:49.514845
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.1676
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 01:12:49.517796
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -806.522
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139817027559552
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139816086004400
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139816086004904
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139816086005408
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139816086005912
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139816086006416

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f29c5aa3240> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.471114
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.434731
grad_step = 000002, loss = 0.403142
grad_step = 000003, loss = 0.370512
grad_step = 000004, loss = 0.336926
grad_step = 000005, loss = 0.310655
grad_step = 000006, loss = 0.295932
grad_step = 000007, loss = 0.280855
grad_step = 000008, loss = 0.268773
grad_step = 000009, loss = 0.253755
grad_step = 000010, loss = 0.238701
grad_step = 000011, loss = 0.227506
grad_step = 000012, loss = 0.218844
grad_step = 000013, loss = 0.210477
grad_step = 000014, loss = 0.199658
grad_step = 000015, loss = 0.186341
grad_step = 000016, loss = 0.172740
grad_step = 000017, loss = 0.162135
grad_step = 000018, loss = 0.153651
grad_step = 000019, loss = 0.142604
grad_step = 000020, loss = 0.130173
grad_step = 000021, loss = 0.119129
grad_step = 000022, loss = 0.108276
grad_step = 000023, loss = 0.097567
grad_step = 000024, loss = 0.089149
grad_step = 000025, loss = 0.082865
grad_step = 000026, loss = 0.076672
grad_step = 000027, loss = 0.070223
grad_step = 000028, loss = 0.064642
grad_step = 000029, loss = 0.059773
grad_step = 000030, loss = 0.054897
grad_step = 000031, loss = 0.050139
grad_step = 000032, loss = 0.045381
grad_step = 000033, loss = 0.040402
grad_step = 000034, loss = 0.035998
grad_step = 000035, loss = 0.032906
grad_step = 000036, loss = 0.030210
grad_step = 000037, loss = 0.027168
grad_step = 000038, loss = 0.024306
grad_step = 000039, loss = 0.021792
grad_step = 000040, loss = 0.019549
grad_step = 000041, loss = 0.017812
grad_step = 000042, loss = 0.016354
grad_step = 000043, loss = 0.014749
grad_step = 000044, loss = 0.013213
grad_step = 000045, loss = 0.011946
grad_step = 000046, loss = 0.010918
grad_step = 000047, loss = 0.010129
grad_step = 000048, loss = 0.009422
grad_step = 000049, loss = 0.008597
grad_step = 000050, loss = 0.007750
grad_step = 000051, loss = 0.007088
grad_step = 000052, loss = 0.006592
grad_step = 000053, loss = 0.006166
grad_step = 000054, loss = 0.005788
grad_step = 000055, loss = 0.005412
grad_step = 000056, loss = 0.005044
grad_step = 000057, loss = 0.004754
grad_step = 000058, loss = 0.004502
grad_step = 000059, loss = 0.004251
grad_step = 000060, loss = 0.004041
grad_step = 000061, loss = 0.003830
grad_step = 000062, loss = 0.003632
grad_step = 000063, loss = 0.003492
grad_step = 000064, loss = 0.003356
grad_step = 000065, loss = 0.003218
grad_step = 000066, loss = 0.003100
grad_step = 000067, loss = 0.002981
grad_step = 000068, loss = 0.002890
grad_step = 000069, loss = 0.002825
grad_step = 000070, loss = 0.002747
grad_step = 000071, loss = 0.002681
grad_step = 000072, loss = 0.002625
grad_step = 000073, loss = 0.002570
grad_step = 000074, loss = 0.002537
grad_step = 000075, loss = 0.002496
grad_step = 000076, loss = 0.002446
grad_step = 000077, loss = 0.002407
grad_step = 000078, loss = 0.002374
grad_step = 000079, loss = 0.002355
grad_step = 000080, loss = 0.002339
grad_step = 000081, loss = 0.002315
grad_step = 000082, loss = 0.002299
grad_step = 000083, loss = 0.002287
grad_step = 000084, loss = 0.002275
grad_step = 000085, loss = 0.002265
grad_step = 000086, loss = 0.002251
grad_step = 000087, loss = 0.002242
grad_step = 000088, loss = 0.002237
grad_step = 000089, loss = 0.002231
grad_step = 000090, loss = 0.002225
grad_step = 000091, loss = 0.002219
grad_step = 000092, loss = 0.002215
grad_step = 000093, loss = 0.002213
grad_step = 000094, loss = 0.002209
grad_step = 000095, loss = 0.002204
grad_step = 000096, loss = 0.002201
grad_step = 000097, loss = 0.002198
grad_step = 000098, loss = 0.002196
grad_step = 000099, loss = 0.002192
grad_step = 000100, loss = 0.002189
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002185
grad_step = 000102, loss = 0.002182
grad_step = 000103, loss = 0.002180
grad_step = 000104, loss = 0.002177
grad_step = 000105, loss = 0.002173
grad_step = 000106, loss = 0.002170
grad_step = 000107, loss = 0.002167
grad_step = 000108, loss = 0.002164
grad_step = 000109, loss = 0.002161
grad_step = 000110, loss = 0.002158
grad_step = 000111, loss = 0.002155
grad_step = 000112, loss = 0.002152
grad_step = 000113, loss = 0.002149
grad_step = 000114, loss = 0.002146
grad_step = 000115, loss = 0.002143
grad_step = 000116, loss = 0.002140
grad_step = 000117, loss = 0.002137
grad_step = 000118, loss = 0.002135
grad_step = 000119, loss = 0.002132
grad_step = 000120, loss = 0.002129
grad_step = 000121, loss = 0.002127
grad_step = 000122, loss = 0.002124
grad_step = 000123, loss = 0.002122
grad_step = 000124, loss = 0.002119
grad_step = 000125, loss = 0.002117
grad_step = 000126, loss = 0.002114
grad_step = 000127, loss = 0.002112
grad_step = 000128, loss = 0.002109
grad_step = 000129, loss = 0.002107
grad_step = 000130, loss = 0.002105
grad_step = 000131, loss = 0.002102
grad_step = 000132, loss = 0.002100
grad_step = 000133, loss = 0.002097
grad_step = 000134, loss = 0.002095
grad_step = 000135, loss = 0.002093
grad_step = 000136, loss = 0.002090
grad_step = 000137, loss = 0.002088
grad_step = 000138, loss = 0.002085
grad_step = 000139, loss = 0.002083
grad_step = 000140, loss = 0.002080
grad_step = 000141, loss = 0.002078
grad_step = 000142, loss = 0.002075
grad_step = 000143, loss = 0.002073
grad_step = 000144, loss = 0.002070
grad_step = 000145, loss = 0.002068
grad_step = 000146, loss = 0.002065
grad_step = 000147, loss = 0.002062
grad_step = 000148, loss = 0.002060
grad_step = 000149, loss = 0.002057
grad_step = 000150, loss = 0.002054
grad_step = 000151, loss = 0.002052
grad_step = 000152, loss = 0.002049
grad_step = 000153, loss = 0.002046
grad_step = 000154, loss = 0.002043
grad_step = 000155, loss = 0.002040
grad_step = 000156, loss = 0.002037
grad_step = 000157, loss = 0.002034
grad_step = 000158, loss = 0.002032
grad_step = 000159, loss = 0.002029
grad_step = 000160, loss = 0.002025
grad_step = 000161, loss = 0.002022
grad_step = 000162, loss = 0.002019
grad_step = 000163, loss = 0.002016
grad_step = 000164, loss = 0.002013
grad_step = 000165, loss = 0.002010
grad_step = 000166, loss = 0.002006
grad_step = 000167, loss = 0.002003
grad_step = 000168, loss = 0.002000
grad_step = 000169, loss = 0.001996
grad_step = 000170, loss = 0.001993
grad_step = 000171, loss = 0.001989
grad_step = 000172, loss = 0.001986
grad_step = 000173, loss = 0.001982
grad_step = 000174, loss = 0.001978
grad_step = 000175, loss = 0.001975
grad_step = 000176, loss = 0.001971
grad_step = 000177, loss = 0.001967
grad_step = 000178, loss = 0.001963
grad_step = 000179, loss = 0.001959
grad_step = 000180, loss = 0.001955
grad_step = 000181, loss = 0.001950
grad_step = 000182, loss = 0.001946
grad_step = 000183, loss = 0.001942
grad_step = 000184, loss = 0.001938
grad_step = 000185, loss = 0.001933
grad_step = 000186, loss = 0.001928
grad_step = 000187, loss = 0.001923
grad_step = 000188, loss = 0.001919
grad_step = 000189, loss = 0.001914
grad_step = 000190, loss = 0.001909
grad_step = 000191, loss = 0.001904
grad_step = 000192, loss = 0.001899
grad_step = 000193, loss = 0.001893
grad_step = 000194, loss = 0.001888
grad_step = 000195, loss = 0.001883
grad_step = 000196, loss = 0.001877
grad_step = 000197, loss = 0.001872
grad_step = 000198, loss = 0.001867
grad_step = 000199, loss = 0.001863
grad_step = 000200, loss = 0.001859
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001855
grad_step = 000202, loss = 0.001851
grad_step = 000203, loss = 0.001842
grad_step = 000204, loss = 0.001833
grad_step = 000205, loss = 0.001824
grad_step = 000206, loss = 0.001818
grad_step = 000207, loss = 0.001813
grad_step = 000208, loss = 0.001808
grad_step = 000209, loss = 0.001804
grad_step = 000210, loss = 0.001805
grad_step = 000211, loss = 0.001819
grad_step = 000212, loss = 0.001879
grad_step = 000213, loss = 0.001989
grad_step = 000214, loss = 0.002075
grad_step = 000215, loss = 0.001888
grad_step = 000216, loss = 0.001776
grad_step = 000217, loss = 0.001843
grad_step = 000218, loss = 0.001887
grad_step = 000219, loss = 0.001865
grad_step = 000220, loss = 0.001763
grad_step = 000221, loss = 0.001774
grad_step = 000222, loss = 0.001858
grad_step = 000223, loss = 0.001795
grad_step = 000224, loss = 0.001735
grad_step = 000225, loss = 0.001759
grad_step = 000226, loss = 0.001777
grad_step = 000227, loss = 0.001770
grad_step = 000228, loss = 0.001724
grad_step = 000229, loss = 0.001707
grad_step = 000230, loss = 0.001743
grad_step = 000231, loss = 0.001741
grad_step = 000232, loss = 0.001712
grad_step = 000233, loss = 0.001691
grad_step = 000234, loss = 0.001684
grad_step = 000235, loss = 0.001700
grad_step = 000236, loss = 0.001707
grad_step = 000237, loss = 0.001693
grad_step = 000238, loss = 0.001676
grad_step = 000239, loss = 0.001659
grad_step = 000240, loss = 0.001647
grad_step = 000241, loss = 0.001649
grad_step = 000242, loss = 0.001653
grad_step = 000243, loss = 0.001660
grad_step = 000244, loss = 0.001680
grad_step = 000245, loss = 0.001702
grad_step = 000246, loss = 0.001748
grad_step = 000247, loss = 0.001789
grad_step = 000248, loss = 0.001857
grad_step = 000249, loss = 0.001798
grad_step = 000250, loss = 0.001719
grad_step = 000251, loss = 0.001614
grad_step = 000252, loss = 0.001608
grad_step = 000253, loss = 0.001678
grad_step = 000254, loss = 0.001717
grad_step = 000255, loss = 0.001722
grad_step = 000256, loss = 0.001644
grad_step = 000257, loss = 0.001583
grad_step = 000258, loss = 0.001562
grad_step = 000259, loss = 0.001578
grad_step = 000260, loss = 0.001617
grad_step = 000261, loss = 0.001647
grad_step = 000262, loss = 0.001671
grad_step = 000263, loss = 0.001651
grad_step = 000264, loss = 0.001626
grad_step = 000265, loss = 0.001574
grad_step = 000266, loss = 0.001537
grad_step = 000267, loss = 0.001516
grad_step = 000268, loss = 0.001514
grad_step = 000269, loss = 0.001528
grad_step = 000270, loss = 0.001546
grad_step = 000271, loss = 0.001573
grad_step = 000272, loss = 0.001591
grad_step = 000273, loss = 0.001621
grad_step = 000274, loss = 0.001611
grad_step = 000275, loss = 0.001606
grad_step = 000276, loss = 0.001552
grad_step = 000277, loss = 0.001502
grad_step = 000278, loss = 0.001463
grad_step = 000279, loss = 0.001452
grad_step = 000280, loss = 0.001463
grad_step = 000281, loss = 0.001488
grad_step = 000282, loss = 0.001534
grad_step = 000283, loss = 0.001580
grad_step = 000284, loss = 0.001666
grad_step = 000285, loss = 0.001685
grad_step = 000286, loss = 0.001707
grad_step = 000287, loss = 0.001577
grad_step = 000288, loss = 0.001456
grad_step = 000289, loss = 0.001396
grad_step = 000290, loss = 0.001434
grad_step = 000291, loss = 0.001507
grad_step = 000292, loss = 0.001504
grad_step = 000293, loss = 0.001449
grad_step = 000294, loss = 0.001382
grad_step = 000295, loss = 0.001363
grad_step = 000296, loss = 0.001387
grad_step = 000297, loss = 0.001423
grad_step = 000298, loss = 0.001461
grad_step = 000299, loss = 0.001452
grad_step = 000300, loss = 0.001431
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001376
grad_step = 000302, loss = 0.001338
grad_step = 000303, loss = 0.001316
grad_step = 000304, loss = 0.001310
grad_step = 000305, loss = 0.001321
grad_step = 000306, loss = 0.001344
grad_step = 000307, loss = 0.001380
grad_step = 000308, loss = 0.001415
grad_step = 000309, loss = 0.001477
grad_step = 000310, loss = 0.001470
grad_step = 000311, loss = 0.001447
grad_step = 000312, loss = 0.001352
grad_step = 000313, loss = 0.001289
grad_step = 000314, loss = 0.001267
grad_step = 000315, loss = 0.001287
grad_step = 000316, loss = 0.001342
grad_step = 000317, loss = 0.001368
grad_step = 000318, loss = 0.001392
grad_step = 000319, loss = 0.001392
grad_step = 000320, loss = 0.001446
grad_step = 000321, loss = 0.001358
grad_step = 000322, loss = 0.001310
grad_step = 000323, loss = 0.001267
grad_step = 000324, loss = 0.001218
grad_step = 000325, loss = 0.001199
grad_step = 000326, loss = 0.001231
grad_step = 000327, loss = 0.001273
grad_step = 000328, loss = 0.001291
grad_step = 000329, loss = 0.001345
grad_step = 000330, loss = 0.001365
grad_step = 000331, loss = 0.001399
grad_step = 000332, loss = 0.001295
grad_step = 000333, loss = 0.001244
grad_step = 000334, loss = 0.001194
grad_step = 000335, loss = 0.001158
grad_step = 000336, loss = 0.001174
grad_step = 000337, loss = 0.001222
grad_step = 000338, loss = 0.001250
grad_step = 000339, loss = 0.001240
grad_step = 000340, loss = 0.001271
grad_step = 000341, loss = 0.001240
grad_step = 000342, loss = 0.001208
grad_step = 000343, loss = 0.001157
grad_step = 000344, loss = 0.001140
grad_step = 000345, loss = 0.001125
grad_step = 000346, loss = 0.001112
grad_step = 000347, loss = 0.001123
grad_step = 000348, loss = 0.001144
grad_step = 000349, loss = 0.001160
grad_step = 000350, loss = 0.001164
grad_step = 000351, loss = 0.001206
grad_step = 000352, loss = 0.001202
grad_step = 000353, loss = 0.001221
grad_step = 000354, loss = 0.001166
grad_step = 000355, loss = 0.001142
grad_step = 000356, loss = 0.001108
grad_step = 000357, loss = 0.001083
grad_step = 000358, loss = 0.001074
grad_step = 000359, loss = 0.001087
grad_step = 000360, loss = 0.001109
grad_step = 000361, loss = 0.001122
grad_step = 000362, loss = 0.001168
grad_step = 000363, loss = 0.001180
grad_step = 000364, loss = 0.001244
grad_step = 000365, loss = 0.001190
grad_step = 000366, loss = 0.001175
grad_step = 000367, loss = 0.001112
grad_step = 000368, loss = 0.001076
grad_step = 000369, loss = 0.001053
grad_step = 000370, loss = 0.001061
grad_step = 000371, loss = 0.001093
grad_step = 000372, loss = 0.001104
grad_step = 000373, loss = 0.001107
grad_step = 000374, loss = 0.001081
grad_step = 000375, loss = 0.001071
grad_step = 000376, loss = 0.001055
grad_step = 000377, loss = 0.001040
grad_step = 000378, loss = 0.001032
grad_step = 000379, loss = 0.001036
grad_step = 000380, loss = 0.001051
grad_step = 000381, loss = 0.001066
grad_step = 000382, loss = 0.001109
grad_step = 000383, loss = 0.001149
grad_step = 000384, loss = 0.001291
grad_step = 000385, loss = 0.001299
grad_step = 000386, loss = 0.001412
grad_step = 000387, loss = 0.001141
grad_step = 000388, loss = 0.001030
grad_step = 000389, loss = 0.001078
grad_step = 000390, loss = 0.001148
grad_step = 000391, loss = 0.001195
grad_step = 000392, loss = 0.001089
grad_step = 000393, loss = 0.001045
grad_step = 000394, loss = 0.001081
grad_step = 000395, loss = 0.001103
grad_step = 000396, loss = 0.001107
grad_step = 000397, loss = 0.001050
grad_step = 000398, loss = 0.001009
grad_step = 000399, loss = 0.001022
grad_step = 000400, loss = 0.001060
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001087
grad_step = 000402, loss = 0.001070
grad_step = 000403, loss = 0.001070
grad_step = 000404, loss = 0.001044
grad_step = 000405, loss = 0.001026
grad_step = 000406, loss = 0.000998
grad_step = 000407, loss = 0.000995
grad_step = 000408, loss = 0.001011
grad_step = 000409, loss = 0.001018
grad_step = 000410, loss = 0.001023
grad_step = 000411, loss = 0.001017
grad_step = 000412, loss = 0.001015
grad_step = 000413, loss = 0.001003
grad_step = 000414, loss = 0.000989
grad_step = 000415, loss = 0.000982
grad_step = 000416, loss = 0.000986
grad_step = 000417, loss = 0.000993
grad_step = 000418, loss = 0.000994
grad_step = 000419, loss = 0.000995
grad_step = 000420, loss = 0.000995
grad_step = 000421, loss = 0.000998
grad_step = 000422, loss = 0.000991
grad_step = 000423, loss = 0.000986
grad_step = 000424, loss = 0.000979
grad_step = 000425, loss = 0.000977
grad_step = 000426, loss = 0.000974
grad_step = 000427, loss = 0.000971
grad_step = 000428, loss = 0.000967
grad_step = 000429, loss = 0.000965
grad_step = 000430, loss = 0.000964
grad_step = 000431, loss = 0.000963
grad_step = 000432, loss = 0.000961
grad_step = 000433, loss = 0.000959
grad_step = 000434, loss = 0.000958
grad_step = 000435, loss = 0.000957
grad_step = 000436, loss = 0.000956
grad_step = 000437, loss = 0.000954
grad_step = 000438, loss = 0.000953
grad_step = 000439, loss = 0.000952
grad_step = 000440, loss = 0.000952
grad_step = 000441, loss = 0.000952
grad_step = 000442, loss = 0.000954
grad_step = 000443, loss = 0.000961
grad_step = 000444, loss = 0.000985
grad_step = 000445, loss = 0.001030
grad_step = 000446, loss = 0.001182
grad_step = 000447, loss = 0.001273
grad_step = 000448, loss = 0.001598
grad_step = 000449, loss = 0.001229
grad_step = 000450, loss = 0.001064
grad_step = 000451, loss = 0.001054
grad_step = 000452, loss = 0.001067
grad_step = 000453, loss = 0.001107
grad_step = 000454, loss = 0.001090
grad_step = 000455, loss = 0.001098
grad_step = 000456, loss = 0.001056
grad_step = 000457, loss = 0.000964
grad_step = 000458, loss = 0.001008
grad_step = 000459, loss = 0.001079
grad_step = 000460, loss = 0.000995
grad_step = 000461, loss = 0.000936
grad_step = 000462, loss = 0.000979
grad_step = 000463, loss = 0.001009
grad_step = 000464, loss = 0.000965
grad_step = 000465, loss = 0.000935
grad_step = 000466, loss = 0.000972
grad_step = 000467, loss = 0.000985
grad_step = 000468, loss = 0.000942
grad_step = 000469, loss = 0.000947
grad_step = 000470, loss = 0.000980
grad_step = 000471, loss = 0.000971
grad_step = 000472, loss = 0.000968
grad_step = 000473, loss = 0.001021
grad_step = 000474, loss = 0.001053
grad_step = 000475, loss = 0.001088
grad_step = 000476, loss = 0.001076
grad_step = 000477, loss = 0.001082
grad_step = 000478, loss = 0.001020
grad_step = 000479, loss = 0.000944
grad_step = 000480, loss = 0.000918
grad_step = 000481, loss = 0.000943
grad_step = 000482, loss = 0.000973
grad_step = 000483, loss = 0.000980
grad_step = 000484, loss = 0.000973
grad_step = 000485, loss = 0.000941
grad_step = 000486, loss = 0.000915
grad_step = 000487, loss = 0.000905
grad_step = 000488, loss = 0.000916
grad_step = 000489, loss = 0.000941
grad_step = 000490, loss = 0.000956
grad_step = 000491, loss = 0.000967
grad_step = 000492, loss = 0.000954
grad_step = 000493, loss = 0.000942
grad_step = 000494, loss = 0.000922
grad_step = 000495, loss = 0.000904
grad_step = 000496, loss = 0.000893
grad_step = 000497, loss = 0.000892
grad_step = 000498, loss = 0.000897
grad_step = 000499, loss = 0.000904
grad_step = 000500, loss = 0.000912
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000919
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

  date_run                              2020-05-14 01:13:07.289642
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.264109
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 01:13:07.295172
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.19839
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 01:13:07.301673
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.138572
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 01:13:07.306070
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -2.0146
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
0   2020-05-14 01:12:40.786578  ...    mean_absolute_error
1   2020-05-14 01:12:40.790333  ...     mean_squared_error
2   2020-05-14 01:12:40.793492  ...  median_absolute_error
3   2020-05-14 01:12:40.796643  ...               r2_score
4   2020-05-14 01:12:49.508340  ...    mean_absolute_error
5   2020-05-14 01:12:49.512168  ...     mean_squared_error
6   2020-05-14 01:12:49.514845  ...  median_absolute_error
7   2020-05-14 01:12:49.517796  ...               r2_score
8   2020-05-14 01:13:07.289642  ...    mean_absolute_error
9   2020-05-14 01:13:07.295172  ...     mean_squared_error
10  2020-05-14 01:13:07.301673  ...  median_absolute_error
11  2020-05-14 01:13:07.306070  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ffa1a881cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:10, 140878.65it/s] 60%|█████▉    | 5939200/9912422 [00:00<00:19, 201049.82it/s]9920512it [00:00, 38351174.01it/s]                           
0it [00:00, ?it/s]32768it [00:00, 582904.42it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 141359.12it/s]1654784it [00:00, 10297236.85it/s]                         
0it [00:00, ?it/s]8192it [00:00, 114041.32it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ffa1a88c940> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff9cc8670f0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff9cd23ceb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff9cc7bf0f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ffa1a88c940> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff9c9fe5c18> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff9cd23ceb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff9cc77d710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ffa1a88c940> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ffa1a88cac8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f1bfd272240> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=c2e538aac6fc84251a43999990fcab4deed639b42e5ee63db49bfd4ce053949a
  Stored in directory: /tmp/pip-ephem-wheel-cache-y6il0pc3/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f1bf33dd080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3342336/17464789 [====>.........................] - ETA: 0s
13295616/17464789 [=====================>........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 01:14:33.007105: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 01:14:33.011074: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-14 01:14:33.011769: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556ebf7ca5b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 01:14:33.011786: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 8.0040 - accuracy: 0.4780
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8736 - accuracy: 0.4865 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6922 - accuracy: 0.4983
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6820 - accuracy: 0.4990
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5874 - accuracy: 0.5052
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6009 - accuracy: 0.5043
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6187 - accuracy: 0.5031
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6240 - accuracy: 0.5028
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6620 - accuracy: 0.5003
11000/25000 [============>.................] - ETA: 3s - loss: 7.6750 - accuracy: 0.4995
12000/25000 [=============>................] - ETA: 3s - loss: 7.6449 - accuracy: 0.5014
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6289 - accuracy: 0.5025
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6261 - accuracy: 0.5026
15000/25000 [=================>............] - ETA: 2s - loss: 7.6216 - accuracy: 0.5029
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6302 - accuracy: 0.5024
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6396 - accuracy: 0.5018
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6445 - accuracy: 0.5014
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6594 - accuracy: 0.5005
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6720 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6812 - accuracy: 0.4990
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6597 - accuracy: 0.5005
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6360 - accuracy: 0.5020
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
25000/25000 [==============================] - 7s 270us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 01:14:46.010461
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 01:14:46.010461  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:43:11, 10.5kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:08:06, 14.8kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:20:53, 21.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:57:05, 30.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.60M/862M [00:01<5:33:05, 43.0kB/s].vector_cache/glove.6B.zip:   1%|          | 7.27M/862M [00:01<3:52:17, 61.3kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.9M/862M [00:01<2:41:36, 87.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.8M/862M [00:01<1:52:53, 125kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.4M/862M [00:01<1:18:34, 178kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.6M/862M [00:01<54:56, 254kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 30.0M/862M [00:01<38:16, 362kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.2M/862M [00:02<26:49, 515kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.4M/862M [00:02<18:44, 733kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.7M/862M [00:02<13:11, 1.04MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.7M/862M [00:02<09:15, 1.47MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.3M/862M [00:02<06:34, 2.06MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:02<05:16, 2.56MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<04:07, 3.27MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:04<11:09:58, 20.1kB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<7:49:34, 28.6kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:04<5:27:58, 40.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:06<3:55:00, 56.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<2:47:43, 79.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<1:58:02, 113kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:06<1:22:37, 161kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:07<58:50, 226kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:08<11:08:11, 19.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<7:47:53, 28.4kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:08<5:26:51, 40.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:10<3:53:50, 56.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<2:46:21, 79.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<1:56:59, 113kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:12<1:23:43, 157kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:12<59:58, 219kB/s]  .vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:12<42:12, 311kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:14<32:29, 403kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<25:23, 516kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<18:21, 713kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:14<12:58, 1.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:16<1:43:02, 126kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<1:13:25, 177kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:16<51:37, 252kB/s]  .vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:18<39:05, 332kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<29:59, 432kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<21:38, 598kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:20<17:10, 751kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<13:19, 967kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:20<09:38, 1.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:22<09:43, 1.32MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:22<09:31, 1.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<07:12, 1.78MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:22<05:13, 2.45MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:24<09:15, 1.38MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<07:48, 1.63MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:24<05:47, 2.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:59, 1.82MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<07:28, 1.70MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:52, 2.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:08, 2.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<05:36, 2.25MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<04:11, 3.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<05:52, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<06:40, 1.88MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:14, 2.39MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<03:46, 3.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<55:20, 226kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<39:58, 312kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<28:12, 442kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<22:36, 549kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<18:20, 676kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<13:27, 922kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<11:42, 1.06MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:36<08:14, 1.49MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<19:50, 619kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<15:07, 811kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<10:49, 1.13MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<10:28, 1.17MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<09:48, 1.24MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:28, 1.63MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<07:11, 1.69MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<06:16, 1.94MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<04:41, 2.58MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<06:06, 1.98MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<06:44, 1.79MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:20, 2.25MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<04:18, 2.79MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<7:27:14, 26.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<5:12:54, 38.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<3:40:27, 54.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<2:36:42, 76.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<1:50:10, 108kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<1:18:43, 151kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<56:20, 211kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<39:39, 299kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<30:24, 388kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<23:46, 496kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<17:15, 683kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<12:11, 964kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<47:57, 245kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<34:45, 337kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<24:32, 477kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<19:52, 587kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<16:36, 703kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<12:16, 950kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:55<08:43, 1.33MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<17:32, 661kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<13:35, 853kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<09:47, 1.18MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<09:17, 1.24MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<09:10, 1.26MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<06:58, 1.65MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<05:03, 2.27MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<07:17, 1.57MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<06:24, 1.79MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<04:47, 2.38MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<05:47, 1.97MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<06:35, 1.73MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<05:12, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<03:47, 2.99MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<13:01, 869kB/s] .vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<10:26, 1.08MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<07:37, 1.48MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<07:43, 1.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<08:00, 1.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:10, 1.82MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:07<04:27, 2.51MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<12:31, 893kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<10:02, 1.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<07:19, 1.52MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<07:29, 1.48MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<07:49, 1.42MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<06:06, 1.82MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<04:23, 2.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<13:22, 826kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<10:36, 1.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:41, 1.43MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<07:43, 1.42MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<07:50, 1.40MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<06:06, 1.80MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<04:23, 2.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<13:23, 814kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<10:24, 1.05MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<07:35, 1.43MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<07:36, 1.42MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<07:50, 1.38MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:03, 1.79MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<04:22, 2.46MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<12:29, 862kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<09:57, 1.08MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<07:15, 1.48MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<07:21, 1.45MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<07:37, 1.40MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:50, 1.83MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<04:14, 2.51MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<07:09, 1.48MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:12, 1.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:36, 2.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:28, 1.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:10, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:52, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<03:32, 2.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<12:05, 868kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<09:38, 1.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:59, 1.50MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<07:07, 1.46MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<07:24, 1.41MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:46, 1.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<04:10, 2.48MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<13:06, 789kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<10:20, 1.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<07:29, 1.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<07:25, 1.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<07:36, 1.35MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:54, 1.74MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<04:15, 2.40MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<13:08, 777kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<10:21, 986kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:29, 1.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<07:24, 1.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<07:25, 1.36MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:42, 1.78MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:08, 2.44MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<06:28, 1.56MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<05:42, 1.77MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:15, 2.35MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:06, 1.96MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<05:48, 1.72MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:32, 2.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:20, 2.99MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:36, 1.77MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:01, 1.97MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:47, 2.61MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:46, 2.07MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<05:37, 1.75MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:26, 2.22MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:12, 3.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<07:57, 1.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<06:27, 1.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:10, 1.89MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<03:47, 2.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:30, 1.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:59, 1.95MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<03:43, 2.61MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:45, 2.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:28, 1.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:18, 2.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<03:09, 3.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:30, 1.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:57, 1.93MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<03:42, 2.58MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:37, 2.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:18, 2.20MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:17, 2.88MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:18, 2.19MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:05, 2.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<03:07, 3.02MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:11, 2.24MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:06, 1.84MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:01, 2.33MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<02:58, 3.13MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:48, 1.93MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<04:26, 2.10MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<03:18, 2.80MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<02:26, 3.79MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<23:04, 401kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<18:11, 508kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<13:14, 696kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<09:21, 981kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<15:33, 589kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<11:57, 767kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<08:34, 1.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<07:55, 1.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<07:41, 1.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:49, 1.56MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:12, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:57, 1.52MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<05:11, 1.74MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<03:53, 2.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:37, 1.94MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:15, 2.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<03:12, 2.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:08, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:53, 1.81MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<03:52, 2.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<02:51, 3.10MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:52, 1.81MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:24, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<03:19, 2.64MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:11, 2.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:58, 1.76MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<03:54, 2.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<02:50, 3.07MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<06:40, 1.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<05:40, 1.53MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:13, 2.05MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:45, 1.81MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<04:18, 2.00MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<03:13, 2.66MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<04:04, 2.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<04:51, 1.76MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<03:49, 2.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<02:45, 3.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<05:27, 1.55MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<5:21:32, 26.4kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<3:44:41, 37.7kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<2:38:07, 53.2kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<1:52:37, 74.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<1:19:08, 106kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<55:18, 151kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<41:44, 200kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<30:08, 277kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<21:16, 391kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<16:35, 499kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<13:26, 615kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<09:52, 837kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<06:59, 1.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<12:36, 651kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<09:45, 841kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<07:02, 1.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<06:38, 1.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<05:34, 1.46MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<04:07, 1.96MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:36, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<05:00, 1.61MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:53, 2.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<02:49, 2.84MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:32, 1.44MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<04:47, 1.67MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<03:32, 2.25MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:10, 1.90MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:45, 1.66MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:43, 2.13MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<02:42, 2.90MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:44, 1.66MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:12, 1.87MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<03:08, 2.50MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:50, 2.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:25, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:29, 2.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<02:32, 3.04MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<08:26, 915kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<06:48, 1.13MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<04:57, 1.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:05, 1.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:20, 1.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:07, 1.85MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<02:57, 2.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<08:56, 848kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<07:06, 1.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<05:08, 1.47MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<03:40, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<47:44, 157kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<35:08, 214kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<24:55, 301kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<17:31, 426kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<14:20, 519kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<10:52, 684kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<07:48, 950kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<07:17, 1.01MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<08:25, 874kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<06:46, 1.09MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<04:54, 1.50MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:32, 2.07MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<08:06, 900kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<06:31, 1.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<04:45, 1.53MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<04:51, 1.49MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<04:14, 1.71MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:08, 2.29MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:44, 1.92MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<04:19, 1.66MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:22, 2.11MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<02:31, 2.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:28, 2.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:07, 2.27MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<02:22, 2.97MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<01:44, 4.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<7:03:31, 16.6kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<4:57:59, 23.6kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<3:28:38, 33.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:08<2:25:10, 48.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<1:48:21, 64.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<1:16:34, 90.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<53:36, 129kB/s]   .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<38:48, 177kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<27:47, 248kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<19:35, 350kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<13:43, 497kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<25:21, 269kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<19:19, 353kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<13:54, 489kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<09:46, 692kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<12:28, 541kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<09:29, 710kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<06:47, 989kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<06:09, 1.08MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:47, 1.15MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:22, 1.52MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<03:07, 2.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<07:45, 851kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<06:11, 1.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:28, 1.47MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<04:31, 1.44MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<04:33, 1.43MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:28, 1.88MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:30, 2.58MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<04:56, 1.31MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<04:05, 1.58MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:02, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:29, 1.83MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:51, 1.66MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:00, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:11, 2.90MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<03:46, 1.68MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:19, 1.90MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:27, 2.56MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<03:07, 2.01MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:30, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<02:47, 2.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<02:00, 3.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<19:15, 321kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<14:10, 436kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<10:03, 613kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<08:18, 737kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<07:07, 860kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<05:17, 1.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<03:44, 1.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<17:13, 352kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<12:44, 475kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<09:01, 668kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<07:33, 792kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<06:40, 896kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<04:57, 1.21MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<03:32, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<04:42, 1.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:57, 1.49MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:54, 2.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<03:17, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:35, 1.62MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:48, 2.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:02, 2.84MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:23, 1.70MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:01, 1.91MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:14, 2.56MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<02:47, 2.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:17, 1.73MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:37, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<01:54, 2.97MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<06:52, 820kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<05:27, 1.03MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:57, 1.42MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<03:57, 1.41MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<03:58, 1.40MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:02, 1.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:12, 2.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<03:23, 1.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<03:02, 1.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:16, 2.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:39, 3.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<05:46, 941kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:32, 1.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:25, 1.58MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<02:27, 2.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<05:08, 1.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<04:11, 1.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:03, 1.74MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<03:19, 1.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<03:34, 1.48MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<03:05, 1.71MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:18, 2.28MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<01:41, 3.10MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<05:02, 1.03MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<04:00, 1.30MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:00<03:08, 1.66MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:15, 2.29MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<03:33, 1.45MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<02:59, 1.72MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:17, 2.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<01:39, 3.07MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<04:37, 1.10MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<03:52, 1.31MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:59, 1.69MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:09, 2.34MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<03:25, 1.46MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:58, 1.69MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:11, 2.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<02:34, 1.92MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:57, 1.67MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:18, 2.14MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:39, 2.94MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<03:21, 1.45MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:54, 1.68MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<02:08, 2.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:31, 1.91MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:47, 1.72MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:09, 2.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:34, 3.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:05, 1.53MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:36, 1.81MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:55, 2.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:24, 3.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<04:36, 1.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<04:19, 1.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:15, 1.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<02:18, 2.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<05:26, 845kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<04:19, 1.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:07, 1.46MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:08, 1.44MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:20<03:09, 1.43MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:25, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:44, 2.58MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<04:02, 1.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<03:20, 1.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:25, 1.83MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:37, 1.67MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<02:46, 1.58MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:10, 2.02MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:33, 2.78MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<14:50, 291kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<10:50, 398kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<07:39, 561kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<06:15, 680kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<05:21, 793kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<03:59, 1.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<02:49, 1.49MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<06:05, 687kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<04:44, 882kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:23, 1.22MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:14, 1.27MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:11, 1.29MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:25, 1.69MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:44, 2.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:12, 1.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:41, 1.50MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:59, 2.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:13, 1.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:28, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<01:55, 2.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:23, 2.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:22, 1.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:46, 1.40MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:02, 1.90MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:16, 1.69MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:28, 1.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<01:55, 1.99MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:22, 2.74MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:36, 1.45MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:15, 1.66MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:39, 2.25MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:56, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:46, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:20, 2.74MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:42, 2.13MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:59, 1.81MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:33, 2.31MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:08, 3.12MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:54, 1.87MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:44, 2.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:18, 2.69MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:39, 2.11MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:56, 1.80MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:31, 2.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:05, 3.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:12, 1.54MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:56, 1.76MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:26, 2.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:42, 1.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:59, 1.68MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:32, 2.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:07, 2.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:54, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:42, 1.92MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:16, 2.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:34, 2.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:27, 2.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:05, 2.91MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:26, 2.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:44, 1.80MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:22, 2.28MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<00:59, 3.13MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:18, 1.33MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:56, 1.58MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:25, 2.15MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:39, 1.82MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:49, 1.64MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:26, 2.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:01, 2.85MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<03:27, 849kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:44, 1.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:58, 1.47MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:00, 1.43MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:04, 1.38MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:35, 1.79MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:07<01:07, 2.49MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<02:41, 1.04MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:12, 1.27MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:35, 1.73MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:41, 1.62MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:49, 1.49MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:24, 1.93MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<01:00, 2.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:08, 1.24MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:48, 1.47MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:19, 1.98MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:28, 1.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:36, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:14, 2.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<00:53, 2.85MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:54, 1.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:36, 1.57MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:10, 2.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:21, 1.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:31, 1.61MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:10, 2.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<00:50, 2.84MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:59, 1.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:39, 1.43MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:12, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:19, 1.73MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:10, 1.97MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:52, 2.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:06, 2.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:15, 1.79MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:59, 2.25MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<00:42, 3.09MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<05:42, 380kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<04:13, 513kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<02:58, 720kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<02:30, 837kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<02:12, 953kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:37, 1.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<01:08, 1.79MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<02:36, 778kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<02:03, 987kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:28, 1.36MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:25, 1.37MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:13, 1.60MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:54, 2.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:01, 1.86MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:09, 1.63MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:54, 2.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:39, 2.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<02:02, 896kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:34, 1.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:11, 1.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:50, 2.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<02:07, 824kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:51, 940kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:22, 1.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:57, 1.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:46, 951kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:24, 1.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:02, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:41<00:44, 2.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<01:47, 900kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<01:36, 1.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:11, 1.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:50, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<05:13, 296kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<03:49, 403kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<02:40, 568kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<02:08, 691kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:49, 810kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:21, 1.08MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:56, 1.52MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:58, 713kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:30, 926kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:05, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<01:01, 1.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<01:01, 1.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:47, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:32, 2.33MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:29, 851kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:11, 1.07MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:50, 1.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:49, 1.45MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:50, 1.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:38, 1.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:27, 2.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:47, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:40, 1.66MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:29, 2.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:33, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:38, 1.67MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:29, 2.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:20, 2.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<01:16, 779kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<01:06, 892kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:49, 1.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:33, 1.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:55, 991kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:45, 1.21MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:32, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:32, 1.57MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:34, 1.49MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:26, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:17, 2.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:44, 1.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:36, 1.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:26, 1.74MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:09<00:26, 1.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:23, 1.83MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:16, 2.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:11<00:19, 2.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:22, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:16, 2.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:11, 3.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:25, 1.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:21, 1.61MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:15, 2.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:16, 1.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<00:17, 1.68MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:13, 2.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:09, 2.94MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<01:00, 435kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:44, 582kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:29, 815kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:23, 924kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:21, 1.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:15, 1.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:10, 1.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:17, 1.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:13, 1.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:09, 1.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:08, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:08, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:06, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:23<00:03, 2.70MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:06, 1.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:05, 1.71MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:02, 1.92MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:02, 2.09MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.76MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 2.14MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 1.78MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 2.26MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 871/400000 [00:00<00:45, 8706.40it/s]  0%|          | 1766/400000 [00:00<00:45, 8774.34it/s]  1%|          | 2571/400000 [00:00<00:46, 8543.30it/s]  1%|          | 3416/400000 [00:00<00:46, 8513.90it/s]  1%|          | 4276/400000 [00:00<00:46, 8537.58it/s]  1%|▏         | 5219/400000 [00:00<00:44, 8786.98it/s]  2%|▏         | 6171/400000 [00:00<00:43, 8992.13it/s]  2%|▏         | 7107/400000 [00:00<00:43, 9097.10it/s]  2%|▏         | 8018/400000 [00:00<00:43, 9100.70it/s]  2%|▏         | 8895/400000 [00:01<00:43, 8912.34it/s]  2%|▏         | 9787/400000 [00:01<00:43, 8912.54it/s]  3%|▎         | 10697/400000 [00:01<00:43, 8967.78it/s]  3%|▎         | 11649/400000 [00:01<00:42, 9124.25it/s]  3%|▎         | 12570/400000 [00:01<00:42, 9148.93it/s]  3%|▎         | 13480/400000 [00:01<00:43, 8928.74it/s]  4%|▎         | 14371/400000 [00:01<00:43, 8848.00it/s]  4%|▍         | 15294/400000 [00:01<00:42, 8957.34it/s]  4%|▍         | 16211/400000 [00:01<00:42, 9018.35it/s]  4%|▍         | 17113/400000 [00:01<00:42, 8947.67it/s]  5%|▍         | 18008/400000 [00:02<00:43, 8745.35it/s]  5%|▍         | 18884/400000 [00:02<00:44, 8521.83it/s]  5%|▍         | 19739/400000 [00:02<00:45, 8384.43it/s]  5%|▌         | 20594/400000 [00:02<00:44, 8433.42it/s]  5%|▌         | 21439/400000 [00:02<00:46, 8165.78it/s]  6%|▌         | 22259/400000 [00:02<00:46, 8066.96it/s]  6%|▌         | 23101/400000 [00:02<00:46, 8167.92it/s]  6%|▌         | 23954/400000 [00:02<00:45, 8272.39it/s]  6%|▌         | 24823/400000 [00:02<00:44, 8391.74it/s]  6%|▋         | 25664/400000 [00:02<00:44, 8387.30it/s]  7%|▋         | 26504/400000 [00:03<00:44, 8325.21it/s]  7%|▋         | 27349/400000 [00:03<00:44, 8360.75it/s]  7%|▋         | 28236/400000 [00:03<00:43, 8507.12it/s]  7%|▋         | 29147/400000 [00:03<00:42, 8678.12it/s]  8%|▊         | 30055/400000 [00:03<00:42, 8794.31it/s]  8%|▊         | 30936/400000 [00:03<00:42, 8664.50it/s]  8%|▊         | 31804/400000 [00:03<00:42, 8635.34it/s]  8%|▊         | 32669/400000 [00:03<00:42, 8549.81it/s]  8%|▊         | 33525/400000 [00:03<00:43, 8366.41it/s]  9%|▊         | 34460/400000 [00:03<00:42, 8638.28it/s]  9%|▉         | 35403/400000 [00:04<00:41, 8860.88it/s]  9%|▉         | 36311/400000 [00:04<00:40, 8922.85it/s]  9%|▉         | 37210/400000 [00:04<00:40, 8940.85it/s] 10%|▉         | 38193/400000 [00:04<00:39, 9188.06it/s] 10%|▉         | 39119/400000 [00:04<00:39, 9206.84it/s] 10%|█         | 40044/400000 [00:04<00:39, 9219.29it/s] 10%|█         | 40968/400000 [00:04<00:40, 8967.45it/s] 10%|█         | 41868/400000 [00:04<00:40, 8849.38it/s] 11%|█         | 42798/400000 [00:04<00:39, 8978.79it/s] 11%|█         | 43765/400000 [00:04<00:38, 9172.86it/s] 11%|█         | 44685/400000 [00:05<00:39, 9056.07it/s] 11%|█▏        | 45593/400000 [00:05<00:40, 8836.74it/s] 12%|█▏        | 46480/400000 [00:05<00:40, 8744.55it/s] 12%|█▏        | 47374/400000 [00:05<00:40, 8800.43it/s] 12%|█▏        | 48260/400000 [00:05<00:39, 8817.28it/s] 12%|█▏        | 49187/400000 [00:05<00:39, 8944.78it/s] 13%|█▎        | 50122/400000 [00:05<00:38, 9061.04it/s] 13%|█▎        | 51077/400000 [00:05<00:37, 9201.02it/s] 13%|█▎        | 52033/400000 [00:05<00:37, 9304.93it/s] 13%|█▎        | 53042/400000 [00:06<00:36, 9526.74it/s] 13%|█▎        | 53997/400000 [00:06<00:37, 9250.36it/s] 14%|█▎        | 54926/400000 [00:06<00:38, 9053.60it/s] 14%|█▍        | 55862/400000 [00:06<00:37, 9143.28it/s] 14%|█▍        | 56801/400000 [00:06<00:37, 9213.55it/s] 14%|█▍        | 57725/400000 [00:06<00:39, 8739.92it/s] 15%|█▍        | 58654/400000 [00:06<00:38, 8896.27it/s] 15%|█▍        | 59623/400000 [00:06<00:37, 9119.46it/s] 15%|█▌        | 60540/400000 [00:06<00:38, 8928.65it/s] 15%|█▌        | 61438/400000 [00:06<00:38, 8759.44it/s] 16%|█▌        | 62341/400000 [00:07<00:38, 8837.53it/s] 16%|█▌        | 63228/400000 [00:07<00:38, 8816.73it/s] 16%|█▌        | 64131/400000 [00:07<00:37, 8877.53it/s] 16%|█▋        | 65074/400000 [00:07<00:37, 9034.91it/s] 17%|█▋        | 66034/400000 [00:07<00:36, 9195.64it/s] 17%|█▋        | 67020/400000 [00:07<00:35, 9383.52it/s] 17%|█▋        | 67961/400000 [00:07<00:35, 9376.61it/s] 17%|█▋        | 68901/400000 [00:07<00:35, 9246.09it/s] 17%|█▋        | 69828/400000 [00:07<00:36, 9086.54it/s] 18%|█▊        | 70809/400000 [00:07<00:35, 9292.04it/s] 18%|█▊        | 71741/400000 [00:08<00:35, 9270.06it/s] 18%|█▊        | 72670/400000 [00:08<00:36, 8991.51it/s] 18%|█▊        | 73573/400000 [00:08<00:36, 8992.54it/s] 19%|█▊        | 74553/400000 [00:08<00:35, 9218.32it/s] 19%|█▉        | 75478/400000 [00:08<00:35, 9184.46it/s] 19%|█▉        | 76399/400000 [00:08<00:35, 9063.03it/s] 19%|█▉        | 77308/400000 [00:08<00:36, 8896.06it/s] 20%|█▉        | 78200/400000 [00:08<00:36, 8809.29it/s] 20%|█▉        | 79149/400000 [00:08<00:35, 9002.98it/s] 20%|██        | 80092/400000 [00:09<00:35, 9125.17it/s] 20%|██        | 81038/400000 [00:09<00:34, 9220.71it/s] 20%|██        | 81962/400000 [00:09<00:34, 9107.18it/s] 21%|██        | 82875/400000 [00:09<00:36, 8777.92it/s] 21%|██        | 83757/400000 [00:09<00:36, 8553.64it/s] 21%|██        | 84617/400000 [00:09<00:36, 8530.72it/s] 21%|██▏       | 85473/400000 [00:09<00:36, 8503.12it/s] 22%|██▏       | 86353/400000 [00:09<00:36, 8588.17it/s] 22%|██▏       | 87214/400000 [00:09<00:36, 8549.22it/s] 22%|██▏       | 88159/400000 [00:09<00:35, 8800.17it/s] 22%|██▏       | 89076/400000 [00:10<00:34, 8905.50it/s] 23%|██▎       | 90002/400000 [00:10<00:34, 9008.63it/s] 23%|██▎       | 90905/400000 [00:10<00:34, 8852.07it/s] 23%|██▎       | 91793/400000 [00:10<00:35, 8738.96it/s] 23%|██▎       | 92669/400000 [00:10<00:35, 8686.79it/s] 23%|██▎       | 93539/400000 [00:10<00:35, 8602.66it/s] 24%|██▎       | 94459/400000 [00:10<00:34, 8772.08it/s] 24%|██▍       | 95360/400000 [00:10<00:34, 8839.17it/s] 24%|██▍       | 96246/400000 [00:10<00:34, 8731.57it/s] 24%|██▍       | 97207/400000 [00:10<00:33, 8976.15it/s] 25%|██▍       | 98110/400000 [00:11<00:33, 8990.95it/s] 25%|██▍       | 99011/400000 [00:11<00:33, 8991.67it/s] 25%|██▍       | 99941/400000 [00:11<00:33, 9078.84it/s] 25%|██▌       | 100850/400000 [00:11<00:33, 9035.35it/s] 25%|██▌       | 101755/400000 [00:11<00:33, 8931.82it/s] 26%|██▌       | 102680/400000 [00:11<00:32, 9023.36it/s] 26%|██▌       | 103600/400000 [00:11<00:32, 9073.25it/s] 26%|██▌       | 104579/400000 [00:11<00:31, 9276.53it/s] 26%|██▋       | 105509/400000 [00:11<00:32, 9188.29it/s] 27%|██▋       | 106430/400000 [00:11<00:32, 9065.09it/s] 27%|██▋       | 107341/400000 [00:12<00:32, 9078.12it/s] 27%|██▋       | 108250/400000 [00:12<00:32, 8921.54it/s] 27%|██▋       | 109144/400000 [00:12<00:32, 8877.83it/s] 28%|██▊       | 110077/400000 [00:12<00:32, 9008.47it/s] 28%|██▊       | 111053/400000 [00:12<00:31, 9220.14it/s] 28%|██▊       | 112034/400000 [00:12<00:30, 9387.91it/s] 28%|██▊       | 112989/400000 [00:12<00:30, 9433.28it/s] 28%|██▊       | 113984/400000 [00:12<00:29, 9580.03it/s] 29%|██▊       | 114944/400000 [00:12<00:30, 9422.59it/s] 29%|██▉       | 115900/400000 [00:12<00:30, 9462.20it/s] 29%|██▉       | 116848/400000 [00:13<00:30, 9166.14it/s] 29%|██▉       | 117768/400000 [00:13<00:31, 9068.54it/s] 30%|██▉       | 118678/400000 [00:13<00:31, 9069.15it/s] 30%|██▉       | 119587/400000 [00:13<00:31, 9033.28it/s] 30%|███       | 120492/400000 [00:13<00:31, 8936.07it/s] 30%|███       | 121410/400000 [00:13<00:30, 9007.21it/s] 31%|███       | 122312/400000 [00:13<00:30, 8989.55it/s] 31%|███       | 123223/400000 [00:13<00:30, 9024.33it/s] 31%|███       | 124139/400000 [00:13<00:30, 9062.77it/s] 31%|███▏      | 125080/400000 [00:14<00:30, 9161.97it/s] 32%|███▏      | 126000/400000 [00:14<00:29, 9171.25it/s] 32%|███▏      | 126918/400000 [00:14<00:29, 9124.75it/s] 32%|███▏      | 127831/400000 [00:14<00:30, 8893.71it/s] 32%|███▏      | 128722/400000 [00:14<00:31, 8715.96it/s] 32%|███▏      | 129598/400000 [00:14<00:30, 8727.25it/s] 33%|███▎      | 130527/400000 [00:14<00:30, 8887.14it/s] 33%|███▎      | 131418/400000 [00:14<00:30, 8833.26it/s] 33%|███▎      | 132303/400000 [00:14<00:31, 8606.56it/s] 33%|███▎      | 133199/400000 [00:14<00:30, 8709.12it/s] 34%|███▎      | 134106/400000 [00:15<00:30, 8814.00it/s] 34%|███▍      | 135044/400000 [00:15<00:29, 8975.56it/s] 34%|███▍      | 136036/400000 [00:15<00:28, 9238.05it/s] 34%|███▍      | 136963/400000 [00:15<00:28, 9247.26it/s] 34%|███▍      | 137896/400000 [00:15<00:28, 9270.10it/s] 35%|███▍      | 138842/400000 [00:15<00:28, 9325.01it/s] 35%|███▍      | 139791/400000 [00:15<00:27, 9372.41it/s] 35%|███▌      | 140730/400000 [00:15<00:28, 9208.08it/s] 35%|███▌      | 141653/400000 [00:15<00:28, 8919.58it/s] 36%|███▌      | 142619/400000 [00:15<00:28, 9128.22it/s] 36%|███▌      | 143536/400000 [00:16<00:28, 9114.74it/s] 36%|███▌      | 144450/400000 [00:16<00:28, 9086.67it/s] 36%|███▋      | 145361/400000 [00:16<00:28, 8933.01it/s] 37%|███▋      | 146256/400000 [00:16<00:28, 8791.96it/s] 37%|███▋      | 147137/400000 [00:16<00:29, 8602.63it/s] 37%|███▋      | 148000/400000 [00:16<00:30, 8346.82it/s] 37%|███▋      | 148843/400000 [00:16<00:30, 8370.73it/s] 37%|███▋      | 149797/400000 [00:16<00:28, 8687.86it/s] 38%|███▊      | 150775/400000 [00:16<00:27, 8987.23it/s] 38%|███▊      | 151710/400000 [00:16<00:27, 9090.09it/s] 38%|███▊      | 152624/400000 [00:17<00:27, 9101.23it/s] 38%|███▊      | 153538/400000 [00:17<00:27, 9101.00it/s] 39%|███▊      | 154451/400000 [00:17<00:27, 9041.52it/s] 39%|███▉      | 155357/400000 [00:17<00:27, 8780.16it/s] 39%|███▉      | 156272/400000 [00:17<00:27, 8887.90it/s] 39%|███▉      | 157164/400000 [00:17<00:27, 8825.94it/s] 40%|███▉      | 158097/400000 [00:17<00:26, 8969.05it/s] 40%|███▉      | 159022/400000 [00:17<00:26, 9051.14it/s] 40%|███▉      | 159929/400000 [00:17<00:26, 9024.49it/s] 40%|████      | 160849/400000 [00:18<00:26, 9075.55it/s] 40%|████      | 161758/400000 [00:18<00:26, 8856.59it/s] 41%|████      | 162646/400000 [00:18<00:27, 8765.69it/s] 41%|████      | 163524/400000 [00:18<00:27, 8606.58it/s] 41%|████      | 164410/400000 [00:18<00:27, 8678.56it/s] 41%|████▏     | 165309/400000 [00:18<00:26, 8767.57it/s] 42%|████▏     | 166223/400000 [00:18<00:26, 8874.05it/s] 42%|████▏     | 167112/400000 [00:18<00:26, 8819.36it/s] 42%|████▏     | 168038/400000 [00:18<00:25, 8946.67it/s] 42%|████▏     | 168998/400000 [00:18<00:25, 9132.41it/s] 42%|████▏     | 169924/400000 [00:19<00:25, 9168.30it/s] 43%|████▎     | 170843/400000 [00:19<00:25, 9123.43it/s] 43%|████▎     | 171757/400000 [00:19<00:25, 8872.06it/s] 43%|████▎     | 172647/400000 [00:19<00:26, 8714.92it/s] 43%|████▎     | 173521/400000 [00:19<00:26, 8628.45it/s] 44%|████▎     | 174386/400000 [00:19<00:27, 8354.20it/s] 44%|████▍     | 175298/400000 [00:19<00:26, 8568.93it/s] 44%|████▍     | 176237/400000 [00:19<00:25, 8797.17it/s] 44%|████▍     | 177121/400000 [00:19<00:25, 8754.21it/s] 44%|████▍     | 178000/400000 [00:19<00:25, 8723.74it/s] 45%|████▍     | 178875/400000 [00:20<00:26, 8442.69it/s] 45%|████▍     | 179771/400000 [00:20<00:25, 8589.80it/s] 45%|████▌     | 180683/400000 [00:20<00:25, 8742.05it/s] 45%|████▌     | 181560/400000 [00:20<00:25, 8608.61it/s] 46%|████▌     | 182424/400000 [00:20<00:25, 8511.12it/s] 46%|████▌     | 183292/400000 [00:20<00:25, 8557.91it/s] 46%|████▌     | 184254/400000 [00:20<00:24, 8849.64it/s] 46%|████▋     | 185162/400000 [00:20<00:24, 8914.97it/s] 47%|████▋     | 186159/400000 [00:20<00:23, 9205.39it/s] 47%|████▋     | 187138/400000 [00:20<00:22, 9371.88it/s] 47%|████▋     | 188098/400000 [00:21<00:22, 9437.41it/s] 47%|████▋     | 189045/400000 [00:21<00:22, 9177.53it/s] 47%|████▋     | 189967/400000 [00:21<00:23, 8852.15it/s] 48%|████▊     | 190858/400000 [00:21<00:24, 8695.96it/s] 48%|████▊     | 191750/400000 [00:21<00:23, 8761.37it/s] 48%|████▊     | 192648/400000 [00:21<00:23, 8824.90it/s] 48%|████▊     | 193540/400000 [00:21<00:23, 8851.07it/s] 49%|████▊     | 194434/400000 [00:21<00:23, 8876.27it/s] 49%|████▉     | 195323/400000 [00:21<00:23, 8749.26it/s] 49%|████▉     | 196224/400000 [00:22<00:23, 8823.02it/s] 49%|████▉     | 197108/400000 [00:22<00:23, 8750.25it/s] 50%|████▉     | 198004/400000 [00:22<00:22, 8811.87it/s] 50%|████▉     | 198886/400000 [00:22<00:23, 8543.07it/s] 50%|████▉     | 199743/400000 [00:22<00:23, 8517.57it/s] 50%|█████     | 200597/400000 [00:22<00:23, 8492.12it/s] 50%|█████     | 201455/400000 [00:22<00:23, 8517.54it/s] 51%|█████     | 202323/400000 [00:22<00:23, 8565.13it/s] 51%|█████     | 203181/400000 [00:22<00:23, 8518.73it/s] 51%|█████     | 204138/400000 [00:22<00:22, 8808.09it/s] 51%|█████▏    | 205022/400000 [00:23<00:22, 8804.06it/s] 51%|█████▏    | 205905/400000 [00:23<00:22, 8537.79it/s] 52%|█████▏    | 206762/400000 [00:23<00:23, 8212.98it/s] 52%|█████▏    | 207645/400000 [00:23<00:22, 8387.73it/s] 52%|█████▏    | 208520/400000 [00:23<00:22, 8490.56it/s] 52%|█████▏    | 209406/400000 [00:23<00:22, 8597.04it/s] 53%|█████▎    | 210269/400000 [00:23<00:22, 8556.09it/s] 53%|█████▎    | 211127/400000 [00:23<00:22, 8344.55it/s] 53%|█████▎    | 211994/400000 [00:23<00:22, 8437.78it/s] 53%|█████▎    | 212966/400000 [00:23<00:21, 8783.98it/s] 53%|█████▎    | 213911/400000 [00:24<00:20, 8971.82it/s] 54%|█████▎    | 214813/400000 [00:24<00:20, 8854.48it/s] 54%|█████▍    | 215707/400000 [00:24<00:20, 8879.28it/s] 54%|█████▍    | 216598/400000 [00:24<00:21, 8637.82it/s] 54%|█████▍    | 217465/400000 [00:24<00:21, 8611.62it/s] 55%|█████▍    | 218329/400000 [00:24<00:21, 8579.24it/s] 55%|█████▍    | 219248/400000 [00:24<00:20, 8752.02it/s] 55%|█████▌    | 220186/400000 [00:24<00:20, 8929.45it/s] 55%|█████▌    | 221120/400000 [00:24<00:19, 9046.23it/s] 56%|█████▌    | 222048/400000 [00:24<00:19, 9113.75it/s] 56%|█████▌    | 222979/400000 [00:25<00:19, 9171.74it/s] 56%|█████▌    | 223914/400000 [00:25<00:19, 9222.68it/s] 56%|█████▌    | 224843/400000 [00:25<00:18, 9240.60it/s] 56%|█████▋    | 225768/400000 [00:25<00:19, 8974.16it/s] 57%|█████▋    | 226668/400000 [00:25<00:19, 8762.33it/s] 57%|█████▋    | 227547/400000 [00:25<00:19, 8647.13it/s] 57%|█████▋    | 228414/400000 [00:25<00:19, 8642.03it/s] 57%|█████▋    | 229348/400000 [00:25<00:19, 8839.11it/s] 58%|█████▊    | 230273/400000 [00:25<00:18, 8957.38it/s] 58%|█████▊    | 231211/400000 [00:26<00:18, 9078.65it/s] 58%|█████▊    | 232127/400000 [00:26<00:18, 9100.31it/s] 58%|█████▊    | 233039/400000 [00:26<00:18, 9092.66it/s] 59%|█████▊    | 234006/400000 [00:26<00:17, 9256.86it/s] 59%|█████▊    | 234940/400000 [00:26<00:17, 9280.75it/s] 59%|█████▉    | 235870/400000 [00:26<00:17, 9208.22it/s] 59%|█████▉    | 236792/400000 [00:26<00:18, 8984.91it/s] 59%|█████▉    | 237702/400000 [00:26<00:17, 9018.83it/s] 60%|█████▉    | 238606/400000 [00:26<00:18, 8931.75it/s] 60%|█████▉    | 239501/400000 [00:26<00:18, 8875.86it/s] 60%|██████    | 240390/400000 [00:27<00:18, 8696.87it/s] 60%|██████    | 241262/400000 [00:27<00:18, 8658.24it/s] 61%|██████    | 242129/400000 [00:27<00:18, 8362.30it/s] 61%|██████    | 242969/400000 [00:27<00:18, 8341.10it/s] 61%|██████    | 243847/400000 [00:27<00:18, 8466.54it/s] 61%|██████    | 244764/400000 [00:27<00:17, 8664.62it/s] 61%|██████▏   | 245650/400000 [00:27<00:17, 8719.61it/s] 62%|██████▏   | 246551/400000 [00:27<00:17, 8802.47it/s] 62%|██████▏   | 247433/400000 [00:27<00:17, 8772.71it/s] 62%|██████▏   | 248351/400000 [00:27<00:17, 8890.89it/s] 62%|██████▏   | 249254/400000 [00:28<00:16, 8930.27it/s] 63%|██████▎   | 250148/400000 [00:28<00:17, 8766.20it/s] 63%|██████▎   | 251095/400000 [00:28<00:16, 8962.97it/s] 63%|██████▎   | 252043/400000 [00:28<00:16, 9111.35it/s] 63%|██████▎   | 252957/400000 [00:28<00:16, 9084.46it/s] 63%|██████▎   | 253939/400000 [00:28<00:15, 9290.39it/s] 64%|██████▎   | 254888/400000 [00:28<00:15, 9349.01it/s] 64%|██████▍   | 255825/400000 [00:28<00:16, 8892.65it/s] 64%|██████▍   | 256721/400000 [00:28<00:16, 8886.92it/s] 64%|██████▍   | 257635/400000 [00:28<00:15, 8960.10it/s] 65%|██████▍   | 258534/400000 [00:29<00:15, 8930.53it/s] 65%|██████▍   | 259430/400000 [00:29<00:16, 8745.53it/s] 65%|██████▌   | 260386/400000 [00:29<00:15, 8973.07it/s] 65%|██████▌   | 261351/400000 [00:29<00:15, 9164.63it/s] 66%|██████▌   | 262271/400000 [00:29<00:15, 9114.52it/s] 66%|██████▌   | 263185/400000 [00:29<00:15, 9040.19it/s] 66%|██████▌   | 264091/400000 [00:29<00:15, 8931.67it/s] 66%|██████▌   | 264986/400000 [00:29<00:15, 8784.31it/s] 66%|██████▋   | 265914/400000 [00:29<00:15, 8925.47it/s] 67%|██████▋   | 266862/400000 [00:30<00:14, 9083.73it/s] 67%|██████▋   | 267779/400000 [00:30<00:14, 9108.06it/s] 67%|██████▋   | 268692/400000 [00:30<00:14, 9076.70it/s] 67%|██████▋   | 269601/400000 [00:30<00:14, 9063.70it/s] 68%|██████▊   | 270523/400000 [00:30<00:14, 9108.47it/s] 68%|██████▊   | 271435/400000 [00:30<00:14, 9045.59it/s] 68%|██████▊   | 272387/400000 [00:30<00:13, 9180.46it/s] 68%|██████▊   | 273323/400000 [00:30<00:13, 9231.96it/s] 69%|██████▊   | 274247/400000 [00:30<00:13, 9101.23it/s] 69%|██████▉   | 275163/400000 [00:30<00:13, 9118.27it/s] 69%|██████▉   | 276125/400000 [00:31<00:13, 9262.64it/s] 69%|██████▉   | 277053/400000 [00:31<00:13, 9140.78it/s] 69%|██████▉   | 277979/400000 [00:31<00:13, 9175.12it/s] 70%|██████▉   | 278920/400000 [00:31<00:13, 9241.58it/s] 70%|██████▉   | 279907/400000 [00:31<00:12, 9419.04it/s] 70%|███████   | 280851/400000 [00:31<00:12, 9306.72it/s] 70%|███████   | 281783/400000 [00:31<00:12, 9221.84it/s] 71%|███████   | 282732/400000 [00:31<00:12, 9298.85it/s] 71%|███████   | 283663/400000 [00:31<00:12, 9291.75it/s] 71%|███████   | 284600/400000 [00:31<00:12, 9312.96it/s] 71%|███████▏  | 285532/400000 [00:32<00:12, 9289.86it/s] 72%|███████▏  | 286462/400000 [00:32<00:12, 8884.54it/s] 72%|███████▏  | 287365/400000 [00:32<00:12, 8926.77it/s] 72%|███████▏  | 288261/400000 [00:32<00:12, 8910.36it/s] 72%|███████▏  | 289159/400000 [00:32<00:12, 8929.73it/s] 73%|███████▎  | 290102/400000 [00:32<00:12, 9072.70it/s] 73%|███████▎  | 291031/400000 [00:32<00:11, 9133.70it/s] 73%|███████▎  | 291946/400000 [00:32<00:12, 8990.21it/s] 73%|███████▎  | 292847/400000 [00:32<00:12, 8904.59it/s] 73%|███████▎  | 293772/400000 [00:32<00:11, 9004.08it/s] 74%|███████▎  | 294674/400000 [00:33<00:11, 8853.08it/s] 74%|███████▍  | 295561/400000 [00:33<00:11, 8756.09it/s] 74%|███████▍  | 296486/400000 [00:33<00:11, 8897.74it/s] 74%|███████▍  | 297405/400000 [00:33<00:11, 8981.07it/s] 75%|███████▍  | 298305/400000 [00:33<00:11, 8803.96it/s] 75%|███████▍  | 299201/400000 [00:33<00:11, 8848.24it/s] 75%|███████▌  | 300150/400000 [00:33<00:11, 9031.38it/s] 75%|███████▌  | 301055/400000 [00:33<00:10, 9018.24it/s] 75%|███████▌  | 301959/400000 [00:33<00:10, 8965.37it/s] 76%|███████▌  | 302929/400000 [00:33<00:10, 9173.37it/s] 76%|███████▌  | 303866/400000 [00:34<00:10, 9228.41it/s] 76%|███████▌  | 304804/400000 [00:34<00:10, 9272.43it/s] 76%|███████▋  | 305733/400000 [00:34<00:10, 9268.99it/s] 77%|███████▋  | 306696/400000 [00:34<00:09, 9372.66it/s] 77%|███████▋  | 307635/400000 [00:34<00:09, 9334.48it/s] 77%|███████▋  | 308570/400000 [00:34<00:09, 9149.75it/s] 77%|███████▋  | 309487/400000 [00:34<00:09, 9120.44it/s] 78%|███████▊  | 310400/400000 [00:34<00:09, 9122.31it/s] 78%|███████▊  | 311313/400000 [00:34<00:09, 8966.50it/s] 78%|███████▊  | 312211/400000 [00:34<00:09, 8922.53it/s] 78%|███████▊  | 313198/400000 [00:35<00:09, 9185.67it/s] 79%|███████▊  | 314120/400000 [00:35<00:09, 8997.10it/s] 79%|███████▉  | 315023/400000 [00:35<00:09, 8925.06it/s] 79%|███████▉  | 315918/400000 [00:35<00:09, 8815.07it/s] 79%|███████▉  | 316845/400000 [00:35<00:09, 8944.80it/s] 79%|███████▉  | 317756/400000 [00:35<00:09, 8991.99it/s] 80%|███████▉  | 318702/400000 [00:35<00:08, 9126.24it/s] 80%|███████▉  | 319616/400000 [00:35<00:08, 9098.90it/s] 80%|████████  | 320599/400000 [00:35<00:08, 9305.14it/s] 80%|████████  | 321571/400000 [00:36<00:08, 9423.82it/s] 81%|████████  | 322516/400000 [00:36<00:08, 9410.37it/s] 81%|████████  | 323471/400000 [00:36<00:08, 9449.82it/s] 81%|████████  | 324476/400000 [00:36<00:07, 9621.94it/s] 81%|████████▏ | 325440/400000 [00:36<00:07, 9355.12it/s] 82%|████████▏ | 326379/400000 [00:36<00:08, 9173.42it/s] 82%|████████▏ | 327322/400000 [00:36<00:07, 9247.09it/s] 82%|████████▏ | 328295/400000 [00:36<00:07, 9384.62it/s] 82%|████████▏ | 329308/400000 [00:36<00:07, 9593.94it/s] 83%|████████▎ | 330314/400000 [00:36<00:07, 9728.65it/s] 83%|████████▎ | 331290/400000 [00:37<00:07, 9616.63it/s] 83%|████████▎ | 332254/400000 [00:37<00:07, 9575.66it/s] 83%|████████▎ | 333213/400000 [00:37<00:07, 9417.67it/s] 84%|████████▎ | 334166/400000 [00:37<00:06, 9449.93it/s] 84%|████████▍ | 335113/400000 [00:37<00:06, 9331.76it/s] 84%|████████▍ | 336067/400000 [00:37<00:06, 9392.96it/s] 84%|████████▍ | 337066/400000 [00:37<00:06, 9563.17it/s] 85%|████████▍ | 338024/400000 [00:37<00:06, 9415.83it/s] 85%|████████▍ | 338968/400000 [00:37<00:06, 9218.83it/s] 85%|████████▍ | 339892/400000 [00:37<00:06, 9194.93it/s] 85%|████████▌ | 340833/400000 [00:38<00:06, 9255.89it/s] 85%|████████▌ | 341760/400000 [00:38<00:06, 9215.27it/s] 86%|████████▌ | 342683/400000 [00:38<00:06, 9050.33it/s] 86%|████████▌ | 343592/400000 [00:38<00:06, 9060.04it/s] 86%|████████▌ | 344499/400000 [00:38<00:06, 9023.66it/s] 86%|████████▋ | 345410/400000 [00:38<00:06, 9046.47it/s] 87%|████████▋ | 346396/400000 [00:38<00:05, 9274.73it/s] 87%|████████▋ | 347344/400000 [00:38<00:05, 9332.39it/s] 87%|████████▋ | 348279/400000 [00:38<00:05, 8632.29it/s] 87%|████████▋ | 349192/400000 [00:38<00:05, 8774.50it/s] 88%|████████▊ | 350157/400000 [00:39<00:05, 9018.25it/s] 88%|████████▊ | 351109/400000 [00:39<00:05, 9162.82it/s] 88%|████████▊ | 352061/400000 [00:39<00:05, 9265.56it/s] 88%|████████▊ | 353010/400000 [00:39<00:05, 9328.91it/s] 88%|████████▊ | 353947/400000 [00:39<00:04, 9224.48it/s] 89%|████████▊ | 354900/400000 [00:39<00:04, 9313.73it/s] 89%|████████▉ | 355836/400000 [00:39<00:04, 9325.49it/s] 89%|████████▉ | 356770/400000 [00:39<00:04, 9112.94it/s] 89%|████████▉ | 357752/400000 [00:39<00:04, 9311.96it/s] 90%|████████▉ | 358724/400000 [00:40<00:04, 9430.32it/s] 90%|████████▉ | 359721/400000 [00:40<00:04, 9584.67it/s] 90%|█████████ | 360682/400000 [00:40<00:04, 9461.55it/s] 90%|█████████ | 361630/400000 [00:40<00:04, 9413.03it/s] 91%|█████████ | 362578/400000 [00:40<00:03, 9431.37it/s] 91%|█████████ | 363523/400000 [00:40<00:03, 9273.64it/s] 91%|█████████ | 364452/400000 [00:40<00:03, 9174.45it/s] 91%|█████████▏| 365371/400000 [00:40<00:03, 9048.48it/s] 92%|█████████▏| 366278/400000 [00:40<00:03, 8945.73it/s] 92%|█████████▏| 367188/400000 [00:40<00:03, 8990.86it/s] 92%|█████████▏| 368093/400000 [00:41<00:03, 9008.32it/s] 92%|█████████▏| 368995/400000 [00:41<00:03, 8976.80it/s] 92%|█████████▏| 369894/400000 [00:41<00:03, 8817.39it/s] 93%|█████████▎| 370805/400000 [00:41<00:03, 8901.17it/s] 93%|█████████▎| 371696/400000 [00:41<00:03, 8899.83it/s] 93%|█████████▎| 372590/400000 [00:41<00:03, 8910.63it/s] 93%|█████████▎| 373482/400000 [00:41<00:03, 8771.17it/s] 94%|█████████▎| 374363/400000 [00:41<00:02, 8781.44it/s] 94%|█████████▍| 375305/400000 [00:41<00:02, 8962.23it/s] 94%|█████████▍| 376203/400000 [00:41<00:02, 8824.38it/s] 94%|█████████▍| 377097/400000 [00:42<00:02, 8857.39it/s] 94%|█████████▍| 377984/400000 [00:42<00:02, 8664.59it/s] 95%|█████████▍| 378867/400000 [00:42<00:02, 8710.22it/s] 95%|█████████▍| 379822/400000 [00:42<00:02, 8943.71it/s] 95%|█████████▌| 380734/400000 [00:42<00:02, 8993.73it/s] 95%|█████████▌| 381636/400000 [00:42<00:02, 8807.24it/s] 96%|█████████▌| 382537/400000 [00:42<00:01, 8865.78it/s] 96%|█████████▌| 383446/400000 [00:42<00:01, 8928.85it/s] 96%|█████████▌| 384372/400000 [00:42<00:01, 9024.24it/s] 96%|█████████▋| 385355/400000 [00:42<00:01, 9250.74it/s] 97%|█████████▋| 386297/400000 [00:43<00:01, 9300.33it/s] 97%|█████████▋| 387229/400000 [00:43<00:01, 9290.41it/s] 97%|█████████▋| 388160/400000 [00:43<00:01, 9287.20it/s] 97%|█████████▋| 389090/400000 [00:43<00:01, 9230.45it/s] 98%|█████████▊| 390014/400000 [00:43<00:01, 9230.66it/s] 98%|█████████▊| 390964/400000 [00:43<00:00, 9309.57it/s] 98%|█████████▊| 391901/400000 [00:43<00:00, 9324.95it/s] 98%|█████████▊| 392841/400000 [00:43<00:00, 9345.30it/s] 98%|█████████▊| 393776/400000 [00:43<00:00, 9268.09it/s] 99%|█████████▊| 394704/400000 [00:43<00:00, 9239.89it/s] 99%|█████████▉| 395629/400000 [00:44<00:00, 9162.22it/s] 99%|█████████▉| 396546/400000 [00:44<00:00, 8734.87it/s] 99%|█████████▉| 397424/400000 [00:44<00:00, 8464.18it/s]100%|█████████▉| 398309/400000 [00:44<00:00, 8574.67it/s]100%|█████████▉| 399238/400000 [00:44<00:00, 8775.96it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8971.17it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f26a854ad30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01099969823984629 	 Accuracy: 55
Train Epoch: 1 	 Loss: 0.011103414372855605 	 Accuracy: 57

  model saves at 57% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15780 out of table with 15776 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15780 out of table with 15776 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-14 01:23:44.458922: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 01:23:44.463683: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-14 01:23:44.463835: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555900a5d850 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 01:23:44.463849: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f264e052160> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6820 - accuracy: 0.4990
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7433 - accuracy: 0.4950 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7842 - accuracy: 0.4923
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7318 - accuracy: 0.4958
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7862 - accuracy: 0.4922
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6845 - accuracy: 0.4988
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6494 - accuracy: 0.5011
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6768 - accuracy: 0.4993
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
11000/25000 [============>.................] - ETA: 3s - loss: 7.6513 - accuracy: 0.5010
12000/25000 [=============>................] - ETA: 3s - loss: 7.6858 - accuracy: 0.4988
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6784 - accuracy: 0.4992
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6622 - accuracy: 0.5003
15000/25000 [=================>............] - ETA: 2s - loss: 7.6503 - accuracy: 0.5011
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6446 - accuracy: 0.5014
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6278 - accuracy: 0.5025
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6164 - accuracy: 0.5033
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6230 - accuracy: 0.5028
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6337 - accuracy: 0.5021
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6381 - accuracy: 0.5019
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6360 - accuracy: 0.5020
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6600 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 7s 270us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f26a854ad30> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f26256e3160> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4367 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.3522 - val_crf_viterbi_accuracy: 0.0000e+00

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

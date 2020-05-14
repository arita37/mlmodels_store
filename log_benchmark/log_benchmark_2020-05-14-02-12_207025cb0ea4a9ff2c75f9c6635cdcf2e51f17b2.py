
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7efdd5605fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 02:12:30.024809
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 02:12:30.028518
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 02:12:30.031544
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 02:12:30.034718
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7efde13cf470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 357704.4375
Epoch 2/10

1/1 [==============================] - 0s 103ms/step - loss: 326612.8750
Epoch 3/10

1/1 [==============================] - 0s 99ms/step - loss: 270312.2188
Epoch 4/10

1/1 [==============================] - 0s 96ms/step - loss: 186185.6562
Epoch 5/10

1/1 [==============================] - 0s 96ms/step - loss: 116994.7969
Epoch 6/10

1/1 [==============================] - 0s 95ms/step - loss: 69902.4766
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 42272.8828
Epoch 8/10

1/1 [==============================] - 0s 102ms/step - loss: 26579.4805
Epoch 9/10

1/1 [==============================] - 0s 95ms/step - loss: 17537.4160
Epoch 10/10

1/1 [==============================] - 0s 104ms/step - loss: 12162.0879

  #### Inference Need return ypred, ytrue ######################### 
[[ 4.12986159e-01  1.02449369e+00  1.13472283e-01  2.75889248e-01
   3.16266902e-02  8.32399189e-01  6.13213301e-01  2.70823002e-01
   3.67808282e-01 -8.95428717e-01 -7.29651153e-01  4.07391608e-01
  -2.92391717e-01  7.33218908e-01  7.90662527e-01 -1.27187693e+00
  -6.91814840e-01  1.90558389e-01 -2.50672251e-01  3.69522691e-01
   2.17832923e-02  1.01353586e-01  5.84514260e-01 -1.36008787e+00
  -5.82920074e-01 -3.33567321e-01 -2.66707391e-01  1.10726821e+00
  -2.97328472e-01 -1.55187607e-01 -4.25628930e-01 -3.71297002e-01
   6.08297586e-01  1.17146432e+00  6.34704649e-01 -6.95581853e-01
  -4.46284175e-01  8.11015308e-01 -1.93493754e-01 -1.73032093e+00
   7.50720501e-02 -1.38285294e-01  7.52550423e-01  2.54881024e-01
  -1.42691776e-01  1.30839527e-01  5.65763891e-01 -3.64760160e-01
  -3.40757705e-02 -6.48363113e-01 -5.12928188e-01 -7.37046480e-01
   1.78998441e-01 -3.42328012e-01 -2.48244554e-01 -4.85140532e-01
   5.90651810e-01  6.46253049e-01  2.81214446e-01 -9.70584899e-02
   1.62726200e+00  2.16975927e-01  5.40168822e-01  1.28586888e-02
  -2.24519297e-01 -6.06025279e-01  5.94946206e-01 -1.41815692e-01
  -5.83713830e-01  1.67594850e-01 -2.22918987e-02  1.61650264e+00
  -7.22528994e-01 -9.35728103e-02  4.28885490e-01 -1.64537096e+00
  -1.53851449e-01 -3.35412085e-01  3.33158046e-01  9.11259651e-01
   6.04138613e-01 -1.20555162e+00 -9.08637345e-02 -4.10352349e-01
  -1.78085431e-01  8.74747634e-02  9.53175500e-02 -7.90563226e-03
   2.33954608e-01  8.26663494e-01 -1.39728278e-01  2.61470258e-01
  -6.07042253e-01  4.54514265e-01 -9.35423017e-01 -3.53169501e-01
  -4.04654086e-01 -5.04001617e-01  1.47708848e-01  9.70366180e-01
   5.37962168e-02 -2.45970696e-01 -1.29588616e+00  3.62696111e-01
   3.51516634e-01  9.23120499e-01  2.29771018e-01 -7.02954173e-01
  -4.87580001e-02 -6.41810536e-01 -6.18560553e-01  2.68536925e-01
  -7.84723163e-01  8.94790590e-02  9.47295725e-02 -1.08701313e+00
   3.05426925e-01  7.63323903e-01  1.68692839e+00 -3.87245536e-01
  -3.00941288e-01  5.69397116e+00  6.65721989e+00  5.59482527e+00
   6.35450697e+00  5.38642836e+00  5.93760967e+00  6.07276678e+00
   6.11617279e+00  5.24626541e+00  5.27651358e+00  5.09208870e+00
   6.46279240e+00  4.92404413e+00  6.52115726e+00  5.15361118e+00
   5.23031473e+00  4.43621063e+00  6.48906755e+00  6.07060051e+00
   4.96363115e+00  5.40431833e+00  4.81084013e+00  6.12383127e+00
   5.99674702e+00  5.08825541e+00  4.87852240e+00  4.93255138e+00
   5.09406900e+00  6.78574991e+00  7.12542200e+00  5.24920845e+00
   5.48711157e+00  5.40967178e+00  5.52266264e+00  4.50003386e+00
   6.42051649e+00  5.85802126e+00  6.52146816e+00  6.73430586e+00
   5.74884176e+00  5.91922855e+00  5.76608038e+00  6.09263420e+00
   7.43979931e+00  4.36867332e+00  5.76175213e+00  4.61902189e+00
   5.15112925e+00  4.63235521e+00  5.02954245e+00  5.35871458e+00
   5.85359240e+00  5.02086592e+00  6.78633547e+00  5.88420248e+00
   4.53844595e+00  4.35940742e+00  4.44965124e+00  5.24295378e+00
   9.70763803e-01  1.21075749e+00  1.06851840e+00  1.47571683e+00
   4.79152143e-01  6.62469685e-01  2.40026116e-01  9.37585473e-01
   7.90006161e-01  1.65715003e+00  1.47925377e+00  3.39353085e-01
   1.53644967e+00  8.12996507e-01  2.34577799e+00  1.20835996e+00
   5.76784492e-01  4.14358318e-01  2.10564661e+00  1.87749302e+00
   1.56291485e+00  1.64842868e+00  9.19131041e-01  9.51234698e-01
   1.00694704e+00  7.88112640e-01  8.06891441e-01  7.26208985e-01
   2.69968987e-01  1.58385801e+00  4.21459317e-01  1.61832237e+00
   5.80969989e-01  5.85475028e-01  1.02067947e+00  2.94612288e-01
   7.27050424e-01  1.51733875e+00  1.80270231e+00  1.55872321e+00
   1.55146027e+00  7.99459457e-01  6.93791926e-01  1.41351855e+00
   3.58307838e-01  4.76161122e-01  1.32132339e+00  1.52773809e+00
   5.14648318e-01  3.88608932e-01  8.45157027e-01  1.16051757e+00
   5.89720011e-01  1.12220538e+00  1.58512735e+00  5.51388443e-01
   1.04105639e+00  1.79470968e+00  1.20256424e+00  2.07739043e+00
   4.11002219e-01  1.57502460e+00  5.45725167e-01  5.39012671e-01
   7.59162545e-01  2.13120413e+00  1.08403242e+00  9.10778046e-01
   5.40829003e-01  6.10512495e-01  1.62350845e+00  1.21767604e+00
   1.34930754e+00  1.03060484e+00  1.23285794e+00  1.30507588e+00
   1.50090277e+00  7.53037333e-01  9.19492841e-01  5.35630107e-01
   1.53959107e+00  9.88470316e-01  1.63903475e+00  3.95287633e-01
   1.40844727e+00  5.54794908e-01  6.67900324e-01  2.67771530e+00
   2.70336986e-01  1.78373373e+00  7.38717198e-01  1.04482687e+00
   1.03436589e+00  3.98766875e-01  3.24098229e-01  4.94841695e-01
   7.30179429e-01  1.29722273e+00  7.48141766e-01  5.09295225e-01
   1.61933374e+00  1.39305425e+00  1.30602038e+00  3.90555978e-01
   1.03927588e+00  6.88358009e-01  1.04015446e+00  9.83353257e-01
   1.24829364e+00  1.24578273e+00  1.41590190e+00  6.78973675e-01
   1.38438964e+00  6.93229020e-01  1.56872272e+00  8.32555771e-01
   1.78160989e+00  9.14967835e-01  1.48862767e+00  6.23887956e-01
   7.17287064e-02  5.56432962e+00  5.23789597e+00  6.87721634e+00
   6.96592712e+00  6.06905317e+00  7.59871197e+00  6.70507908e+00
   5.82479334e+00  6.35670137e+00  6.44914770e+00  6.83052063e+00
   6.04179811e+00  6.64688969e+00  6.31425905e+00  5.74320507e+00
   7.07707691e+00  6.49973583e+00  7.19423866e+00  7.09408283e+00
   6.90549850e+00  5.90488434e+00  6.80182219e+00  6.29943991e+00
   5.99042130e+00  6.39516068e+00  5.01160002e+00  5.72893906e+00
   5.02131557e+00  5.96488142e+00  6.40724564e+00  6.75141191e+00
   7.23497343e+00  6.65555620e+00  5.94355392e+00  5.92093658e+00
   5.94204283e+00  5.06944990e+00  5.83919907e+00  6.49292564e+00
   6.53731203e+00  6.94502401e+00  7.12849522e+00  6.66966486e+00
   5.83698416e+00  6.28758669e+00  7.08709288e+00  7.72082233e+00
   7.54983282e+00  5.95332003e+00  5.94418287e+00  6.86658049e+00
   7.12120199e+00  5.53383923e+00  6.77978849e+00  7.91021729e+00
   6.17697382e+00  5.93140459e+00  6.80118656e+00  6.86852503e+00
  -7.99191046e+00 -8.71861076e+00  4.07626390e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 02:12:38.656025
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    96.663
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 02:12:38.659781
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9364.57
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 02:12:38.662893
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   97.2733
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 02:12:38.665932
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -837.664
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139628312539600
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139625800065712
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139625800066216
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139625800066720
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139625800067224
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139625800067728

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7efdd72a5fd0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.474537
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.445401
grad_step = 000002, loss = 0.424574
grad_step = 000003, loss = 0.400509
grad_step = 000004, loss = 0.373634
grad_step = 000005, loss = 0.348162
grad_step = 000006, loss = 0.330642
grad_step = 000007, loss = 0.318425
grad_step = 000008, loss = 0.306726
grad_step = 000009, loss = 0.291946
grad_step = 000010, loss = 0.277251
grad_step = 000011, loss = 0.267149
grad_step = 000012, loss = 0.259872
grad_step = 000013, loss = 0.251459
grad_step = 000014, loss = 0.241681
grad_step = 000015, loss = 0.232077
grad_step = 000016, loss = 0.223009
grad_step = 000017, loss = 0.213858
grad_step = 000018, loss = 0.204184
grad_step = 000019, loss = 0.194363
grad_step = 000020, loss = 0.185806
grad_step = 000021, loss = 0.178695
grad_step = 000022, loss = 0.171305
grad_step = 000023, loss = 0.163467
grad_step = 000024, loss = 0.156022
grad_step = 000025, loss = 0.148998
grad_step = 000026, loss = 0.142102
grad_step = 000027, loss = 0.135353
grad_step = 000028, loss = 0.128743
grad_step = 000029, loss = 0.122278
grad_step = 000030, loss = 0.116075
grad_step = 000031, loss = 0.110102
grad_step = 000032, loss = 0.104241
grad_step = 000033, loss = 0.098537
grad_step = 000034, loss = 0.093148
grad_step = 000035, loss = 0.088197
grad_step = 000036, loss = 0.083490
grad_step = 000037, loss = 0.078775
grad_step = 000038, loss = 0.074136
grad_step = 000039, loss = 0.069799
grad_step = 000040, loss = 0.065755
grad_step = 000041, loss = 0.061848
grad_step = 000042, loss = 0.058098
grad_step = 000043, loss = 0.054582
grad_step = 000044, loss = 0.051248
grad_step = 000045, loss = 0.048089
grad_step = 000046, loss = 0.045144
grad_step = 000047, loss = 0.042369
grad_step = 000048, loss = 0.039721
grad_step = 000049, loss = 0.037223
grad_step = 000050, loss = 0.034899
grad_step = 000051, loss = 0.032692
grad_step = 000052, loss = 0.030569
grad_step = 000053, loss = 0.028620
grad_step = 000054, loss = 0.026842
grad_step = 000055, loss = 0.025124
grad_step = 000056, loss = 0.023486
grad_step = 000057, loss = 0.021984
grad_step = 000058, loss = 0.020570
grad_step = 000059, loss = 0.019229
grad_step = 000060, loss = 0.017978
grad_step = 000061, loss = 0.016803
grad_step = 000062, loss = 0.015683
grad_step = 000063, loss = 0.014633
grad_step = 000064, loss = 0.013659
grad_step = 000065, loss = 0.012739
grad_step = 000066, loss = 0.011878
grad_step = 000067, loss = 0.011076
grad_step = 000068, loss = 0.010318
grad_step = 000069, loss = 0.009606
grad_step = 000070, loss = 0.008946
grad_step = 000071, loss = 0.008340
grad_step = 000072, loss = 0.007771
grad_step = 000073, loss = 0.007236
grad_step = 000074, loss = 0.006748
grad_step = 000075, loss = 0.006298
grad_step = 000076, loss = 0.005880
grad_step = 000077, loss = 0.005499
grad_step = 000078, loss = 0.005151
grad_step = 000079, loss = 0.004829
grad_step = 000080, loss = 0.004536
grad_step = 000081, loss = 0.004270
grad_step = 000082, loss = 0.004031
grad_step = 000083, loss = 0.003816
grad_step = 000084, loss = 0.003619
grad_step = 000085, loss = 0.003441
grad_step = 000086, loss = 0.003282
grad_step = 000087, loss = 0.003140
grad_step = 000088, loss = 0.003014
grad_step = 000089, loss = 0.002901
grad_step = 000090, loss = 0.002800
grad_step = 000091, loss = 0.002711
grad_step = 000092, loss = 0.002633
grad_step = 000093, loss = 0.002565
grad_step = 000094, loss = 0.002504
grad_step = 000095, loss = 0.002452
grad_step = 000096, loss = 0.002406
grad_step = 000097, loss = 0.002366
grad_step = 000098, loss = 0.002332
grad_step = 000099, loss = 0.002302
grad_step = 000100, loss = 0.002276
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002254
grad_step = 000102, loss = 0.002236
grad_step = 000103, loss = 0.002219
grad_step = 000104, loss = 0.002206
grad_step = 000105, loss = 0.002193
grad_step = 000106, loss = 0.002183
grad_step = 000107, loss = 0.002174
grad_step = 000108, loss = 0.002167
grad_step = 000109, loss = 0.002160
grad_step = 000110, loss = 0.002154
grad_step = 000111, loss = 0.002148
grad_step = 000112, loss = 0.002143
grad_step = 000113, loss = 0.002138
grad_step = 000114, loss = 0.002134
grad_step = 000115, loss = 0.002130
grad_step = 000116, loss = 0.002126
grad_step = 000117, loss = 0.002123
grad_step = 000118, loss = 0.002119
grad_step = 000119, loss = 0.002115
grad_step = 000120, loss = 0.002111
grad_step = 000121, loss = 0.002107
grad_step = 000122, loss = 0.002103
grad_step = 000123, loss = 0.002099
grad_step = 000124, loss = 0.002094
grad_step = 000125, loss = 0.002090
grad_step = 000126, loss = 0.002086
grad_step = 000127, loss = 0.002083
grad_step = 000128, loss = 0.002079
grad_step = 000129, loss = 0.002074
grad_step = 000130, loss = 0.002069
grad_step = 000131, loss = 0.002064
grad_step = 000132, loss = 0.002059
grad_step = 000133, loss = 0.002055
grad_step = 000134, loss = 0.002052
grad_step = 000135, loss = 0.002051
grad_step = 000136, loss = 0.002058
grad_step = 000137, loss = 0.002071
grad_step = 000138, loss = 0.002070
grad_step = 000139, loss = 0.002046
grad_step = 000140, loss = 0.002028
grad_step = 000141, loss = 0.002037
grad_step = 000142, loss = 0.002047
grad_step = 000143, loss = 0.002034
grad_step = 000144, loss = 0.002015
grad_step = 000145, loss = 0.002016
grad_step = 000146, loss = 0.002025
grad_step = 000147, loss = 0.002021
grad_step = 000148, loss = 0.002006
grad_step = 000149, loss = 0.001998
grad_step = 000150, loss = 0.002002
grad_step = 000151, loss = 0.002005
grad_step = 000152, loss = 0.001999
grad_step = 000153, loss = 0.001989
grad_step = 000154, loss = 0.001981
grad_step = 000155, loss = 0.001981
grad_step = 000156, loss = 0.001984
grad_step = 000157, loss = 0.001984
grad_step = 000158, loss = 0.001978
grad_step = 000159, loss = 0.001971
grad_step = 000160, loss = 0.001964
grad_step = 000161, loss = 0.001959
grad_step = 000162, loss = 0.001958
grad_step = 000163, loss = 0.001960
grad_step = 000164, loss = 0.001966
grad_step = 000165, loss = 0.001983
grad_step = 000166, loss = 0.001989
grad_step = 000167, loss = 0.001981
grad_step = 000168, loss = 0.001975
grad_step = 000169, loss = 0.001954
grad_step = 000170, loss = 0.001939
grad_step = 000171, loss = 0.001946
grad_step = 000172, loss = 0.001953
grad_step = 000173, loss = 0.001947
grad_step = 000174, loss = 0.001943
grad_step = 000175, loss = 0.001948
grad_step = 000176, loss = 0.001938
grad_step = 000177, loss = 0.001928
grad_step = 000178, loss = 0.001933
grad_step = 000179, loss = 0.001935
grad_step = 000180, loss = 0.001929
grad_step = 000181, loss = 0.001931
grad_step = 000182, loss = 0.001934
grad_step = 000183, loss = 0.001928
grad_step = 000184, loss = 0.001925
grad_step = 000185, loss = 0.001926
grad_step = 000186, loss = 0.001921
grad_step = 000187, loss = 0.001916
grad_step = 000188, loss = 0.001916
grad_step = 000189, loss = 0.001916
grad_step = 000190, loss = 0.001912
grad_step = 000191, loss = 0.001910
grad_step = 000192, loss = 0.001910
grad_step = 000193, loss = 0.001909
grad_step = 000194, loss = 0.001906
grad_step = 000195, loss = 0.001905
grad_step = 000196, loss = 0.001905
grad_step = 000197, loss = 0.001905
grad_step = 000198, loss = 0.001905
grad_step = 000199, loss = 0.001908
grad_step = 000200, loss = 0.001917
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001934
grad_step = 000202, loss = 0.001965
grad_step = 000203, loss = 0.002003
grad_step = 000204, loss = 0.001999
grad_step = 000205, loss = 0.001968
grad_step = 000206, loss = 0.001911
grad_step = 000207, loss = 0.001897
grad_step = 000208, loss = 0.001911
grad_step = 000209, loss = 0.001925
grad_step = 000210, loss = 0.001925
grad_step = 000211, loss = 0.001889
grad_step = 000212, loss = 0.001892
grad_step = 000213, loss = 0.001908
grad_step = 000214, loss = 0.001910
grad_step = 000215, loss = 0.001900
grad_step = 000216, loss = 0.001878
grad_step = 000217, loss = 0.001879
grad_step = 000218, loss = 0.001892
grad_step = 000219, loss = 0.001889
grad_step = 000220, loss = 0.001881
grad_step = 000221, loss = 0.001873
grad_step = 000222, loss = 0.001865
grad_step = 000223, loss = 0.001870
grad_step = 000224, loss = 0.001877
grad_step = 000225, loss = 0.001869
grad_step = 000226, loss = 0.001863
grad_step = 000227, loss = 0.001858
grad_step = 000228, loss = 0.001855
grad_step = 000229, loss = 0.001858
grad_step = 000230, loss = 0.001860
grad_step = 000231, loss = 0.001856
grad_step = 000232, loss = 0.001850
grad_step = 000233, loss = 0.001848
grad_step = 000234, loss = 0.001845
grad_step = 000235, loss = 0.001841
grad_step = 000236, loss = 0.001840
grad_step = 000237, loss = 0.001841
grad_step = 000238, loss = 0.001840
grad_step = 000239, loss = 0.001837
grad_step = 000240, loss = 0.001835
grad_step = 000241, loss = 0.001834
grad_step = 000242, loss = 0.001832
grad_step = 000243, loss = 0.001830
grad_step = 000244, loss = 0.001831
grad_step = 000245, loss = 0.001832
grad_step = 000246, loss = 0.001830
grad_step = 000247, loss = 0.001829
grad_step = 000248, loss = 0.001831
grad_step = 000249, loss = 0.001834
grad_step = 000250, loss = 0.001838
grad_step = 000251, loss = 0.001841
grad_step = 000252, loss = 0.001835
grad_step = 000253, loss = 0.001826
grad_step = 000254, loss = 0.001818
grad_step = 000255, loss = 0.001810
grad_step = 000256, loss = 0.001804
grad_step = 000257, loss = 0.001798
grad_step = 000258, loss = 0.001794
grad_step = 000259, loss = 0.001789
grad_step = 000260, loss = 0.001786
grad_step = 000261, loss = 0.001783
grad_step = 000262, loss = 0.001781
grad_step = 000263, loss = 0.001779
grad_step = 000264, loss = 0.001779
grad_step = 000265, loss = 0.001782
grad_step = 000266, loss = 0.001794
grad_step = 000267, loss = 0.001812
grad_step = 000268, loss = 0.001828
grad_step = 000269, loss = 0.001863
grad_step = 000270, loss = 0.001885
grad_step = 000271, loss = 0.001887
grad_step = 000272, loss = 0.001832
grad_step = 000273, loss = 0.001772
grad_step = 000274, loss = 0.001748
grad_step = 000275, loss = 0.001755
grad_step = 000276, loss = 0.001774
grad_step = 000277, loss = 0.001792
grad_step = 000278, loss = 0.001785
grad_step = 000279, loss = 0.001749
grad_step = 000280, loss = 0.001727
grad_step = 000281, loss = 0.001730
grad_step = 000282, loss = 0.001749
grad_step = 000283, loss = 0.001759
grad_step = 000284, loss = 0.001748
grad_step = 000285, loss = 0.001727
grad_step = 000286, loss = 0.001708
grad_step = 000287, loss = 0.001701
grad_step = 000288, loss = 0.001703
grad_step = 000289, loss = 0.001709
grad_step = 000290, loss = 0.001718
grad_step = 000291, loss = 0.001726
grad_step = 000292, loss = 0.001724
grad_step = 000293, loss = 0.001711
grad_step = 000294, loss = 0.001695
grad_step = 000295, loss = 0.001681
grad_step = 000296, loss = 0.001668
grad_step = 000297, loss = 0.001657
grad_step = 000298, loss = 0.001652
grad_step = 000299, loss = 0.001649
grad_step = 000300, loss = 0.001647
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001649
grad_step = 000302, loss = 0.001671
grad_step = 000303, loss = 0.001718
grad_step = 000304, loss = 0.001731
grad_step = 000305, loss = 0.001745
grad_step = 000306, loss = 0.001725
grad_step = 000307, loss = 0.001698
grad_step = 000308, loss = 0.001643
grad_step = 000309, loss = 0.001602
grad_step = 000310, loss = 0.001613
grad_step = 000311, loss = 0.001648
grad_step = 000312, loss = 0.001654
grad_step = 000313, loss = 0.001641
grad_step = 000314, loss = 0.001634
grad_step = 000315, loss = 0.001622
grad_step = 000316, loss = 0.001585
grad_step = 000317, loss = 0.001558
grad_step = 000318, loss = 0.001562
grad_step = 000319, loss = 0.001581
grad_step = 000320, loss = 0.001592
grad_step = 000321, loss = 0.001593
grad_step = 000322, loss = 0.001596
grad_step = 000323, loss = 0.001592
grad_step = 000324, loss = 0.001564
grad_step = 000325, loss = 0.001526
grad_step = 000326, loss = 0.001505
grad_step = 000327, loss = 0.001514
grad_step = 000328, loss = 0.001531
grad_step = 000329, loss = 0.001542
grad_step = 000330, loss = 0.001560
grad_step = 000331, loss = 0.001607
grad_step = 000332, loss = 0.001688
grad_step = 000333, loss = 0.001604
grad_step = 000334, loss = 0.001528
grad_step = 000335, loss = 0.001504
grad_step = 000336, loss = 0.001496
grad_step = 000337, loss = 0.001483
grad_step = 000338, loss = 0.001502
grad_step = 000339, loss = 0.001515
grad_step = 000340, loss = 0.001464
grad_step = 000341, loss = 0.001415
grad_step = 000342, loss = 0.001428
grad_step = 000343, loss = 0.001444
grad_step = 000344, loss = 0.001423
grad_step = 000345, loss = 0.001403
grad_step = 000346, loss = 0.001405
grad_step = 000347, loss = 0.001403
grad_step = 000348, loss = 0.001383
grad_step = 000349, loss = 0.001383
grad_step = 000350, loss = 0.001438
grad_step = 000351, loss = 0.001570
grad_step = 000352, loss = 0.001886
grad_step = 000353, loss = 0.002052
grad_step = 000354, loss = 0.001837
grad_step = 000355, loss = 0.001503
grad_step = 000356, loss = 0.001453
grad_step = 000357, loss = 0.001545
grad_step = 000358, loss = 0.001544
grad_step = 000359, loss = 0.001454
grad_step = 000360, loss = 0.001409
grad_step = 000361, loss = 0.001467
grad_step = 000362, loss = 0.001395
grad_step = 000363, loss = 0.001324
grad_step = 000364, loss = 0.001406
grad_step = 000365, loss = 0.001382
grad_step = 000366, loss = 0.001281
grad_step = 000367, loss = 0.001305
grad_step = 000368, loss = 0.001343
grad_step = 000369, loss = 0.001275
grad_step = 000370, loss = 0.001253
grad_step = 000371, loss = 0.001278
grad_step = 000372, loss = 0.001269
grad_step = 000373, loss = 0.001245
grad_step = 000374, loss = 0.001231
grad_step = 000375, loss = 0.001213
grad_step = 000376, loss = 0.001214
grad_step = 000377, loss = 0.001218
grad_step = 000378, loss = 0.001187
grad_step = 000379, loss = 0.001174
grad_step = 000380, loss = 0.001195
grad_step = 000381, loss = 0.001204
grad_step = 000382, loss = 0.001235
grad_step = 000383, loss = 0.001377
grad_step = 000384, loss = 0.001732
grad_step = 000385, loss = 0.001759
grad_step = 000386, loss = 0.001384
grad_step = 000387, loss = 0.001159
grad_step = 000388, loss = 0.001233
grad_step = 000389, loss = 0.001357
grad_step = 000390, loss = 0.001300
grad_step = 000391, loss = 0.001187
grad_step = 000392, loss = 0.001158
grad_step = 000393, loss = 0.001202
grad_step = 000394, loss = 0.001214
grad_step = 000395, loss = 0.001147
grad_step = 000396, loss = 0.001106
grad_step = 000397, loss = 0.001136
grad_step = 000398, loss = 0.001152
grad_step = 000399, loss = 0.001095
grad_step = 000400, loss = 0.001051
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001084
grad_step = 000402, loss = 0.001095
grad_step = 000403, loss = 0.001043
grad_step = 000404, loss = 0.001023
grad_step = 000405, loss = 0.001046
grad_step = 000406, loss = 0.001030
grad_step = 000407, loss = 0.001007
grad_step = 000408, loss = 0.001024
grad_step = 000409, loss = 0.001022
grad_step = 000410, loss = 0.000991
grad_step = 000411, loss = 0.000980
grad_step = 000412, loss = 0.000982
grad_step = 000413, loss = 0.000968
grad_step = 000414, loss = 0.000952
grad_step = 000415, loss = 0.000948
grad_step = 000416, loss = 0.000946
grad_step = 000417, loss = 0.000938
grad_step = 000418, loss = 0.000931
grad_step = 000419, loss = 0.000941
grad_step = 000420, loss = 0.000982
grad_step = 000421, loss = 0.001106
grad_step = 000422, loss = 0.001346
grad_step = 000423, loss = 0.001549
grad_step = 000424, loss = 0.001282
grad_step = 000425, loss = 0.000965
grad_step = 000426, loss = 0.000946
grad_step = 000427, loss = 0.001088
grad_step = 000428, loss = 0.001093
grad_step = 000429, loss = 0.000987
grad_step = 000430, loss = 0.000934
grad_step = 000431, loss = 0.000964
grad_step = 000432, loss = 0.000995
grad_step = 000433, loss = 0.000957
grad_step = 000434, loss = 0.000899
grad_step = 000435, loss = 0.000891
grad_step = 000436, loss = 0.000920
grad_step = 000437, loss = 0.000915
grad_step = 000438, loss = 0.000857
grad_step = 000439, loss = 0.000832
grad_step = 000440, loss = 0.000874
grad_step = 000441, loss = 0.000865
grad_step = 000442, loss = 0.000819
grad_step = 000443, loss = 0.000812
grad_step = 000444, loss = 0.000822
grad_step = 000445, loss = 0.000796
grad_step = 000446, loss = 0.000791
grad_step = 000447, loss = 0.000820
grad_step = 000448, loss = 0.000806
grad_step = 000449, loss = 0.000776
grad_step = 000450, loss = 0.000774
grad_step = 000451, loss = 0.000770
grad_step = 000452, loss = 0.000748
grad_step = 000453, loss = 0.000743
grad_step = 000454, loss = 0.000755
grad_step = 000455, loss = 0.000751
grad_step = 000456, loss = 0.000747
grad_step = 000457, loss = 0.000753
grad_step = 000458, loss = 0.000760
grad_step = 000459, loss = 0.000756
grad_step = 000460, loss = 0.000749
grad_step = 000461, loss = 0.000744
grad_step = 000462, loss = 0.000735
grad_step = 000463, loss = 0.000715
grad_step = 000464, loss = 0.000698
grad_step = 000465, loss = 0.000688
grad_step = 000466, loss = 0.000680
grad_step = 000467, loss = 0.000673
grad_step = 000468, loss = 0.000671
grad_step = 000469, loss = 0.000673
grad_step = 000470, loss = 0.000675
grad_step = 000471, loss = 0.000681
grad_step = 000472, loss = 0.000693
grad_step = 000473, loss = 0.000719
grad_step = 000474, loss = 0.000761
grad_step = 000475, loss = 0.000821
grad_step = 000476, loss = 0.000862
grad_step = 000477, loss = 0.000868
grad_step = 000478, loss = 0.000788
grad_step = 000479, loss = 0.000688
grad_step = 000480, loss = 0.000630
grad_step = 000481, loss = 0.000649
grad_step = 000482, loss = 0.000703
grad_step = 000483, loss = 0.000714
grad_step = 000484, loss = 0.000672
grad_step = 000485, loss = 0.000614
grad_step = 000486, loss = 0.000609
grad_step = 000487, loss = 0.000647
grad_step = 000488, loss = 0.000671
grad_step = 000489, loss = 0.000655
grad_step = 000490, loss = 0.000604
grad_step = 000491, loss = 0.000581
grad_step = 000492, loss = 0.000595
grad_step = 000493, loss = 0.000616
grad_step = 000494, loss = 0.000625
grad_step = 000495, loss = 0.000606
grad_step = 000496, loss = 0.000581
grad_step = 000497, loss = 0.000560
grad_step = 000498, loss = 0.000556
grad_step = 000499, loss = 0.000567
grad_step = 000500, loss = 0.000578
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000587
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

  date_run                              2020-05-14 02:12:57.822793
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.202358
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 02:12:57.828829
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0991478
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 02:12:57.836794
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.12193
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 02:12:57.842182
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.506587
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
0   2020-05-14 02:12:30.024809  ...    mean_absolute_error
1   2020-05-14 02:12:30.028518  ...     mean_squared_error
2   2020-05-14 02:12:30.031544  ...  median_absolute_error
3   2020-05-14 02:12:30.034718  ...               r2_score
4   2020-05-14 02:12:38.656025  ...    mean_absolute_error
5   2020-05-14 02:12:38.659781  ...     mean_squared_error
6   2020-05-14 02:12:38.662893  ...  median_absolute_error
7   2020-05-14 02:12:38.665932  ...               r2_score
8   2020-05-14 02:12:57.822793  ...    mean_absolute_error
9   2020-05-14 02:12:57.828829  ...     mean_squared_error
10  2020-05-14 02:12:57.836794  ...  median_absolute_error
11  2020-05-14 02:12:57.842182  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb81f361cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:14, 133551.79it/s] 86%|████████▌ | 8527872/9912422 [00:00<00:07, 190655.50it/s]9920512it [00:00, 42209524.93it/s]                           
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 219993.84it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 2753593.77it/s]          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 24604.60it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7d1d1ceb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7d13470f0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb81f36c940> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7d12a00f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7d1d1ceb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7ceac6c50> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb81f36c940> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7d125e710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7d1d1ceb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb7ceadd4e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f4080836208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=45791ed78ef3d8f6f1764b3677084e0d5e4dc34221b7f09356704d5b28272f72
  Stored in directory: /tmp/pip-ephem-wheel-cache-4g4hxupq/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f4076bbc080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3964928/17464789 [=====>........................] - ETA: 0s
11255808/17464789 [==================>...........] - ETA: 0s
16252928/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 02:14:24.940288: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 02:14:24.944877: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095090000 Hz
2020-05-14 02:14:24.945027: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5573b44007d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 02:14:24.945041: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5440 - accuracy: 0.5080
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5056 - accuracy: 0.5105 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5440 - accuracy: 0.5080
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5095 - accuracy: 0.5102
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5532 - accuracy: 0.5074
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5542 - accuracy: 0.5073
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5965 - accuracy: 0.5046
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6283 - accuracy: 0.5025
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6343 - accuracy: 0.5021
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6268 - accuracy: 0.5026
11000/25000 [============>.................] - ETA: 3s - loss: 7.6541 - accuracy: 0.5008
12000/25000 [=============>................] - ETA: 3s - loss: 7.6590 - accuracy: 0.5005
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6454 - accuracy: 0.5014
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6381 - accuracy: 0.5019
15000/25000 [=================>............] - ETA: 2s - loss: 7.6564 - accuracy: 0.5007
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6590 - accuracy: 0.5005
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6567 - accuracy: 0.5006
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6479 - accuracy: 0.5012
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6529 - accuracy: 0.5009
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6398 - accuracy: 0.5017
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6374 - accuracy: 0.5019
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6471 - accuracy: 0.5013
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6446 - accuracy: 0.5014
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6519 - accuracy: 0.5010
25000/25000 [==============================] - 8s 300us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 02:14:39.185931
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 02:14:39.185931  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:26:58, 10.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:38:49, 14.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:42:24, 20.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:12:09, 29.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<5:43:37, 41.6kB/s].vector_cache/glove.6B.zip:   1%|          | 9.68M/862M [00:01<3:58:56, 59.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.2M/862M [00:01<2:46:15, 84.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.2M/862M [00:01<1:56:07, 121kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.5M/862M [00:01<1:20:51, 173kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.8M/862M [00:01<56:30, 246kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.9M/862M [00:01<39:23, 351kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.3M/862M [00:02<27:35, 500kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.0M/862M [00:02<19:17, 710kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.0M/862M [00:02<13:32, 1.01MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.7M/862M [00:02<09:30, 1.42MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:03<07:22, 1.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<07:03, 1.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<07:08, 1.88MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<05:32, 2.42MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<06:12, 2.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<06:09, 2.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:07<04:40, 2.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:07<03:25, 3.87MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<27:15, 487kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:09<20:39, 643kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:09<14:46, 897kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<13:03, 1.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:11<10:30, 1.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:11<07:41, 1.72MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<08:25, 1.56MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:13<08:34, 1.53MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:13<06:33, 2.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:13<04:46, 2.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<08:45, 1.49MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:15<07:28, 1.75MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:15<05:33, 2.35MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<06:55, 1.88MB/s].vector_cache/glove.6B.zip:  10%|▉         | 81.9M/862M [00:17<06:10, 2.10MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:17<04:38, 2.79MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<06:18, 2.05MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:19<05:43, 2.26MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:19<04:17, 3.01MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<06:04, 2.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<06:52, 1.87MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:21<05:21, 2.40MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.2M/862M [00:21<04:00, 3.21MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<06:31, 1.96MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<06:06, 2.10MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:23<04:38, 2.75MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<05:52, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<07:14, 1.76MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:25<05:42, 2.23MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<04:13, 3.01MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:37, 1.91MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:08, 2.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<04:40, 2.70MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:52, 2.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:39, 2.23MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<04:19, 2.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:37, 2.23MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:51, 1.82MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:34, 2.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<04:04, 3.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<11:52, 1.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<09:47, 1.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<07:13, 1.72MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:35, 1.63MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<09:54, 1.25MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<07:26, 1.66MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<05:21, 2.30MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<24:13, 508kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<18:25, 668kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<13:11, 932kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<11:44, 1.04MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<11:14, 1.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<08:30, 1.44MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<06:06, 2.00MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<09:06, 1.34MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<07:52, 1.55MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<05:51, 2.07MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:35, 1.84MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<07:25, 1.63MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:49, 2.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<04:16, 2.82MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<06:41, 1.80MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<06:07, 1.97MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<04:38, 2.59MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:42, 2.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:25, 2.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<04:06, 2.91MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:19, 2.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:31, 1.82MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<05:16, 2.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:48<03:51, 3.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<11:17, 1.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<09:18, 1.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<06:48, 1.73MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:11, 1.64MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<06:26, 1.83MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<04:48, 2.44MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:45, 2.03MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:55, 1.69MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:26, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<03:56, 2.95MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<08:37, 1.35MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<07:27, 1.56MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<05:30, 2.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<06:14, 1.85MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<07:03, 1.64MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<05:38, 2.05MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<04:06, 2.80MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<11:12, 1.02MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<09:13, 1.25MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<06:47, 1.69MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<07:05, 1.61MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<07:45, 1.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<06:06, 1.87MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<04:24, 2.58MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<10:50, 1.05MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<08:55, 1.27MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<06:32, 1.73MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<06:53, 1.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<07:34, 1.49MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<05:58, 1.88MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:06<04:19, 2.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<10:34, 1.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<08:46, 1.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<06:24, 1.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<06:47, 1.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<07:20, 1.52MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<05:45, 1.93MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<04:11, 2.65MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<09:49, 1.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<08:12, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<06:00, 1.84MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:28, 1.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<07:13, 1.52MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<05:42, 1.93MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<04:08, 2.64MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<10:44, 1.02MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<08:49, 1.24MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<06:26, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:16<04:39, 2.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<16:16, 668kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<12:43, 853kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<09:13, 1.18MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<08:38, 1.25MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<08:33, 1.26MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<06:32, 1.65MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<04:43, 2.27MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<07:10, 1.50MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:17, 1.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<04:40, 2.29MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:28, 1.95MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:25, 1.66MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<05:07, 2.07MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:24<03:44, 2.84MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<10:12, 1.04MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<08:11, 1.29MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<06:01, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<04:21, 2.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<09:50, 1.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<08:10, 1.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<05:58, 1.76MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<04:17, 2.43MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<47:48, 218kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<35:53, 291kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<25:44, 405kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<18:05, 574kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<20:10, 514kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<15:22, 675kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<11:02, 937kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<09:50, 1.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<09:24, 1.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<07:08, 1.44MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<05:07, 2.00MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<10:24, 984kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<08:30, 1.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<06:14, 1.64MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<06:27, 1.58MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<07:00, 1.45MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<05:30, 1.84MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:38<03:59, 2.53MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<10:01, 1.01MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<08:14, 1.22MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<06:01, 1.67MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<06:16, 1.60MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<05:35, 1.79MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:12, 2.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:59, 2.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:48, 1.71MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:33, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<03:20, 2.97MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:49, 1.70MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:15, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<03:55, 2.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:46, 2.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<05:44, 1.71MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:36, 2.13MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:21, 2.91MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<09:20, 1.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<07:41, 1.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:39, 1.72MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:57, 1.62MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<06:35, 1.47MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:12, 1.86MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<03:45, 2.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<09:02, 1.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<07:28, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:30, 1.74MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<05:48, 1.64MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<06:17, 1.52MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<04:54, 1.94MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<03:33, 2.67MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<06:06, 1.55MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<05:24, 1.75MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:01, 2.35MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<04:44, 1.98MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<04:27, 2.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:21, 2.79MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<04:16, 2.18MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<05:16, 1.77MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:15, 2.19MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:02<03:06, 2.99MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<08:53, 1.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<07:18, 1.27MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:22, 1.72MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<05:40, 1.62MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<06:07, 1.50MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:50, 1.90MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<03:30, 2.61MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<09:04, 1.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<07:26, 1.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<05:28, 1.66MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<05:42, 1.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<06:12, 1.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:47, 1.89MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:29, 2.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<05:28, 1.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<04:54, 1.83MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:39, 2.45MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:23, 2.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<05:15, 1.70MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:07, 2.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:00, 2.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:47, 1.53MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<05:08, 1.72MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:51, 2.29MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:30, 1.95MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:11, 2.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:09, 2.77MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:00, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<03:50, 2.26MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<02:54, 2.98MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:48, 2.27MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:46, 1.81MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<03:47, 2.28MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<02:44, 3.13MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<06:38, 1.29MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<05:40, 1.51MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<04:10, 2.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:40, 1.82MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:17, 1.98MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:12, 2.64MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:58, 2.12MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:39, 1.81MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:44, 2.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<02:43, 3.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<10:17, 813kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<08:11, 1.02MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<05:56, 1.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:51, 1.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<06:01, 1.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:38, 1.79MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:21, 2.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<05:11, 1.58MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:26, 1.85MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:21, 2.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:01, 2.02MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:48, 1.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:47, 2.15MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<02:44, 2.96MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<07:04, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<05:56, 1.36MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<04:21, 1.85MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:41, 1.71MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:59, 1.61MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:55, 2.04MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<02:49, 2.81MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<12:50, 619kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<09:55, 800kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<07:10, 1.11MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:36, 1.19MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<06:33, 1.20MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:59, 1.57MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<03:34, 2.18MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<06:12, 1.26MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<05:16, 1.48MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:54, 1.99MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:19, 1.79MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:49, 1.60MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<03:50, 2.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<02:46, 2.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<07:10, 1.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:56, 1.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<04:22, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:37, 1.65MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:05, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:56, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<02:51, 2.64MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:37, 1.63MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<04:09, 1.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<03:05, 2.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:42, 2.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:19, 1.72MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:25, 2.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:55<02:28, 2.99MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<06:09, 1.20MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:11, 1.42MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<03:48, 1.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:10, 1.76MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:43, 1.55MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:40, 1.99MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<02:40, 2.73MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<04:39, 1.56MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<04:07, 1.76MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<03:05, 2.33MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<03:38, 1.97MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<04:18, 1.67MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:26, 2.08MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:03<02:29, 2.86MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<06:35, 1.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<05:28, 1.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<04:01, 1.76MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:15, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:37, 1.52MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:35, 1.95MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:07<02:37, 2.67MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<04:07, 1.69MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:43, 1.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:47, 2.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:22, 2.05MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:02, 1.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:12, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<02:20, 2.93MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<06:11, 1.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<05:10, 1.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<03:49, 1.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:03, 1.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:16, 1.58MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:21, 2.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:15<02:24, 2.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<10:52, 616kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<08:23, 797kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<06:03, 1.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<05:35, 1.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<05:31, 1.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:15, 1.55MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:19<03:03, 2.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<06:50, 959kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<05:35, 1.17MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<04:03, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:10, 1.55MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:26, 1.46MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:29, 1.85MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:23<02:30, 2.56MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<05:46, 1.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:48, 1.33MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<03:31, 1.82MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:45, 1.69MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:23, 1.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:31, 2.50MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:03, 2.05MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:40, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:53, 2.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:07, 2.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:23, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:07, 1.99MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:22, 2.61MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:55, 2.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:22, 1.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:41, 2.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:33<01:57, 3.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<10:53, 558kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<08:20, 727kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<06:00, 1.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<05:24, 1.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:31, 1.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<03:18, 1.81MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:32, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:01, 1.96MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:15, 2.62MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<01:39, 3.55MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<07:16, 807kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<05:43, 1.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<04:07, 1.41MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<04:09, 1.40MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:19, 1.34MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:18, 1.75MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<02:23, 2.41MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:54, 1.47MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:25, 1.67MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:33, 2.22MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:56, 1.92MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:23, 1.67MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:40, 2.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<01:55, 2.91MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:54, 1.43MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:20, 1.67MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<02:29, 2.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:56, 1.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:43, 2.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:03, 2.66MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:33, 2.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:07, 1.74MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:28, 2.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<01:48, 3.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:10, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:49, 1.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:07, 2.53MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:37, 2.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:00, 1.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:23, 2.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<01:43, 3.05MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<07:47, 673kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<06:04, 862kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:21, 1.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<03:05, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<16:16, 318kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<12:00, 430kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<08:31, 604kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<06:58, 732kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<06:03, 841kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<04:30, 1.13MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<03:11, 1.58MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<04:33, 1.11MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:47, 1.33MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:46, 1.81MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:56, 1.68MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:39, 1.87MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<01:58, 2.50MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:23, 2.05MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:45, 1.77MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:08, 2.27MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<01:34, 3.07MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:41, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:27, 1.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<01:50, 2.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:15, 2.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:41, 1.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:10, 2.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:33, 3.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<04:22, 1.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:36, 1.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:38, 1.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:46, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:05, 1.50MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:25, 1.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<01:45, 2.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<04:28, 1.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:40, 1.24MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:40, 1.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:47, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:30, 1.79MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<01:52, 2.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:12, 1.99MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:34, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:04, 2.12MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:29, 2.91MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<03:57, 1.09MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<03:17, 1.32MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:25, 1.78MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<02:33, 1.67MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<02:50, 1.50MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:12, 1.93MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:35, 2.66MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<03:08, 1.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:41, 1.56MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:59, 2.10MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<02:14, 1.85MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:00, 2.05MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:29, 2.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:56, 2.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:21, 1.73MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<01:51, 2.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:21, 2.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<03:30, 1.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:56, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:09, 1.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:18, 1.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:31, 1.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:59, 1.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:26, 2.68MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:48, 1.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<03:07, 1.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:16, 1.68MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:21, 1.60MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<02:34, 1.47MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:59, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:25, 2.61MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:24, 1.54MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:07, 1.74MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:34, 2.34MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:50, 1.98MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:11, 1.67MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:42, 2.12MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:14, 2.90MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:09, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:56, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:26, 2.46MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:43, 2.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:37, 2.16MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:13, 2.82MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:33, 2.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:30, 2.28MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:07, 3.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:29, 2.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:52, 1.79MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:29, 2.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:06, 3.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:35, 2.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:30, 2.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:08, 2.86MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:27, 2.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:24, 2.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:03, 3.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:23, 2.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:42, 1.85MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:21, 2.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<00:58, 3.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<02:13, 1.39MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:55, 1.60MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:24, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:36, 1.88MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:51, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:27, 2.07MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:02, 2.85MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:18, 1.28MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:04<01:57, 1.50MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:26, 2.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:35, 1.81MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<01:27, 1.97MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<01:05, 2.62MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:20, 2.11MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:16, 2.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<00:57, 2.89MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:13, 2.23MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:31, 1.79MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:12, 2.25MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<00:52, 3.09MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:05, 1.28MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:46, 1.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:18, 2.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:26, 1.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:38, 1.58MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:16, 2.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<00:55, 2.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:28, 1.71MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:18, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<00:58, 2.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:12, 2.04MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:25, 1.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:08, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:18<00:48, 2.96MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:12, 1.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:49, 1.31MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<01:20, 1.77MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:23, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:28, 1.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:09, 2.01MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<00:49, 2.76MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<04:01, 560kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<03:03, 735kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<02:10, 1.02MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:58, 1.11MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:51, 1.17MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:24, 1.55MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<00:59, 2.16MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:57, 1.08MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:50, 1.15MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:23, 1.52MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:28<00:58, 2.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<03:05, 664kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<02:22, 859kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<01:41, 1.19MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:35, 1.24MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:20, 1.46MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:59, 1.97MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:04, 1.78MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:11, 1.60MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:55, 2.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<00:39, 2.81MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:36, 1.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:20, 1.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:58, 1.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:02, 1.71MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:09, 1.53MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:53, 1.96MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:38, 2.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:09, 1.47MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:00, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:44, 2.25MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:50, 1.93MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:58, 1.68MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:46, 2.09MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<00:32, 2.87MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:25, 1.09MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:10, 1.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:51, 1.79MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:53, 1.67MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:59, 1.50MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:45, 1.94MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:33, 2.62MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:45, 1.89MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:40, 2.08MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:30, 2.74MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:38, 2.11MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:46, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:36, 2.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:25, 3.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:54, 1.43MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:46, 1.64MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:34, 2.21MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:38, 1.91MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:44, 1.64MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:35, 2.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:24, 2.81MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:00, 1.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:50, 1.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:36, 1.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:37, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:33, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:25, 2.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:29, 2.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:35, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:27, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:19, 2.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:50, 1.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:42, 1.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:30, 1.79MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:31, 1.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:26, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:19, 2.53MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:23, 2.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:27, 1.75MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:21, 2.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:14, 3.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:25, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:23, 1.87MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:16, 2.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:19, 2.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:23, 1.71MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:10<00:17, 2.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:12, 2.96MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:23, 1.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:20, 1.71MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:14, 2.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:16, 1.95MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:18, 1.66MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:14, 2.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:09, 2.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:18, 1.49MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:11, 2.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:11, 1.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.68MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:10, 2.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:06, 2.94MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:13, 1.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:11, 1.65MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:07, 2.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:07, 1.92MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:08, 1.64MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:04, 2.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:06, 1.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:03, 2.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 2.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.18MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:00, 2.98MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:02, 1.04MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.26MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.71MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 814/400000 [00:00<00:49, 8139.23it/s]  0%|          | 1644/400000 [00:00<00:48, 8185.85it/s]  1%|          | 2465/400000 [00:00<00:48, 8190.49it/s]  1%|          | 3254/400000 [00:00<00:48, 8097.14it/s]  1%|          | 4107/400000 [00:00<00:48, 8221.13it/s]  1%|          | 4957/400000 [00:00<00:47, 8300.80it/s]  1%|▏         | 5797/400000 [00:00<00:47, 8327.42it/s]  2%|▏         | 6661/400000 [00:00<00:46, 8416.83it/s]  2%|▏         | 7486/400000 [00:00<00:46, 8365.34it/s]  2%|▏         | 8342/400000 [00:01<00:46, 8420.10it/s]  2%|▏         | 9201/400000 [00:01<00:46, 8469.48it/s]  3%|▎         | 10069/400000 [00:01<00:45, 8529.55it/s]  3%|▎         | 10913/400000 [00:01<00:45, 8501.77it/s]  3%|▎         | 11760/400000 [00:01<00:45, 8490.21it/s]  3%|▎         | 12604/400000 [00:01<00:46, 8418.91it/s]  3%|▎         | 13443/400000 [00:01<00:46, 8284.31it/s]  4%|▎         | 14270/400000 [00:01<00:47, 8185.50it/s]  4%|▍         | 15095/400000 [00:01<00:46, 8202.12it/s]  4%|▍         | 15935/400000 [00:01<00:46, 8258.50it/s]  4%|▍         | 16788/400000 [00:02<00:45, 8335.76it/s]  4%|▍         | 17655/400000 [00:02<00:45, 8431.58it/s]  5%|▍         | 18502/400000 [00:02<00:45, 8441.00it/s]  5%|▍         | 19363/400000 [00:02<00:44, 8488.72it/s]  5%|▌         | 20213/400000 [00:02<00:44, 8461.35it/s]  5%|▌         | 21070/400000 [00:02<00:44, 8493.14it/s]  5%|▌         | 21920/400000 [00:02<00:44, 8471.05it/s]  6%|▌         | 22768/400000 [00:02<00:44, 8450.18it/s]  6%|▌         | 23614/400000 [00:02<00:44, 8438.93it/s]  6%|▌         | 24458/400000 [00:02<00:44, 8430.91it/s]  6%|▋         | 25302/400000 [00:03<00:45, 8308.07it/s]  7%|▋         | 26152/400000 [00:03<00:44, 8363.92it/s]  7%|▋         | 27010/400000 [00:03<00:44, 8424.35it/s]  7%|▋         | 27877/400000 [00:03<00:43, 8496.56it/s]  7%|▋         | 28733/400000 [00:03<00:43, 8514.26it/s]  7%|▋         | 29590/400000 [00:03<00:43, 8527.88it/s]  8%|▊         | 30458/400000 [00:03<00:43, 8572.92it/s]  8%|▊         | 31318/400000 [00:03<00:42, 8578.77it/s]  8%|▊         | 32183/400000 [00:03<00:42, 8600.01it/s]  8%|▊         | 33044/400000 [00:03<00:42, 8545.25it/s]  8%|▊         | 33899/400000 [00:04<00:43, 8465.18it/s]  9%|▊         | 34746/400000 [00:04<00:45, 8070.71it/s]  9%|▉         | 35603/400000 [00:04<00:44, 8212.84it/s]  9%|▉         | 36463/400000 [00:04<00:43, 8322.83it/s]  9%|▉         | 37309/400000 [00:04<00:43, 8362.14it/s] 10%|▉         | 38170/400000 [00:04<00:42, 8434.65it/s] 10%|▉         | 39036/400000 [00:04<00:42, 8499.51it/s] 10%|▉         | 39888/400000 [00:04<00:43, 8292.17it/s] 10%|█         | 40743/400000 [00:04<00:42, 8366.67it/s] 10%|█         | 41599/400000 [00:04<00:42, 8421.52it/s] 11%|█         | 42458/400000 [00:05<00:42, 8469.65it/s] 11%|█         | 43308/400000 [00:05<00:42, 8475.72it/s] 11%|█         | 44157/400000 [00:05<00:42, 8458.04it/s] 11%|█▏        | 45004/400000 [00:05<00:41, 8458.68it/s] 11%|█▏        | 45860/400000 [00:05<00:41, 8487.79it/s] 12%|█▏        | 46727/400000 [00:05<00:41, 8539.77it/s] 12%|█▏        | 47590/400000 [00:05<00:41, 8566.20it/s] 12%|█▏        | 48457/400000 [00:05<00:40, 8595.72it/s] 12%|█▏        | 49320/400000 [00:05<00:40, 8603.95it/s] 13%|█▎        | 50181/400000 [00:05<00:40, 8547.45it/s] 13%|█▎        | 51036/400000 [00:06<00:40, 8538.71it/s] 13%|█▎        | 51893/400000 [00:06<00:40, 8547.78it/s] 13%|█▎        | 52748/400000 [00:06<00:40, 8537.25it/s] 13%|█▎        | 53602/400000 [00:06<00:40, 8505.89it/s] 14%|█▎        | 54453/400000 [00:06<00:41, 8369.39it/s] 14%|█▍        | 55291/400000 [00:06<00:42, 8019.70it/s] 14%|█▍        | 56097/400000 [00:06<00:46, 7449.39it/s] 14%|█▍        | 56932/400000 [00:06<00:44, 7696.51it/s] 14%|█▍        | 57769/400000 [00:06<00:43, 7885.45it/s] 15%|█▍        | 58604/400000 [00:07<00:42, 8017.70it/s] 15%|█▍        | 59462/400000 [00:07<00:41, 8176.64it/s] 15%|█▌        | 60313/400000 [00:07<00:41, 8273.31it/s] 15%|█▌        | 61170/400000 [00:07<00:40, 8359.66it/s] 16%|█▌        | 62012/400000 [00:07<00:40, 8375.26it/s] 16%|█▌        | 62869/400000 [00:07<00:39, 8430.27it/s] 16%|█▌        | 63714/400000 [00:07<00:40, 8398.59it/s] 16%|█▌        | 64585/400000 [00:07<00:39, 8489.11it/s] 16%|█▋        | 65436/400000 [00:07<00:39, 8492.30it/s] 17%|█▋        | 66286/400000 [00:07<00:39, 8450.09it/s] 17%|█▋        | 67137/400000 [00:08<00:39, 8467.76it/s] 17%|█▋        | 67989/400000 [00:08<00:39, 8482.08it/s] 17%|█▋        | 68838/400000 [00:08<00:39, 8473.42it/s] 17%|█▋        | 69698/400000 [00:08<00:38, 8509.05it/s] 18%|█▊        | 70553/400000 [00:08<00:38, 8520.93it/s] 18%|█▊        | 71419/400000 [00:08<00:38, 8561.96it/s] 18%|█▊        | 72289/400000 [00:08<00:38, 8601.51it/s] 18%|█▊        | 73150/400000 [00:08<00:38, 8589.84it/s] 19%|█▊        | 74016/400000 [00:08<00:37, 8609.16it/s] 19%|█▊        | 74882/400000 [00:08<00:37, 8623.60it/s] 19%|█▉        | 75745/400000 [00:09<00:37, 8575.45it/s] 19%|█▉        | 76603/400000 [00:09<00:37, 8571.61it/s] 19%|█▉        | 77471/400000 [00:09<00:37, 8603.36it/s] 20%|█▉        | 78333/400000 [00:09<00:37, 8605.54it/s] 20%|█▉        | 79194/400000 [00:09<00:37, 8553.82it/s] 20%|██        | 80050/400000 [00:09<00:38, 8299.43it/s] 20%|██        | 80899/400000 [00:09<00:38, 8354.70it/s] 20%|██        | 81749/400000 [00:09<00:37, 8395.19it/s] 21%|██        | 82595/400000 [00:09<00:37, 8413.47it/s] 21%|██        | 83438/400000 [00:09<00:38, 8268.73it/s] 21%|██        | 84277/400000 [00:10<00:38, 8302.62it/s] 21%|██▏       | 85123/400000 [00:10<00:37, 8347.16it/s] 21%|██▏       | 85987/400000 [00:10<00:37, 8430.87it/s] 22%|██▏       | 86843/400000 [00:10<00:36, 8468.67it/s] 22%|██▏       | 87697/400000 [00:10<00:36, 8487.81it/s] 22%|██▏       | 88553/400000 [00:10<00:36, 8507.59it/s] 22%|██▏       | 89412/400000 [00:10<00:36, 8531.41it/s] 23%|██▎       | 90273/400000 [00:10<00:36, 8554.47it/s] 23%|██▎       | 91134/400000 [00:10<00:36, 8568.43it/s] 23%|██▎       | 91991/400000 [00:10<00:36, 8536.37it/s] 23%|██▎       | 92852/400000 [00:11<00:35, 8557.83it/s] 23%|██▎       | 93717/400000 [00:11<00:35, 8583.99it/s] 24%|██▎       | 94582/400000 [00:11<00:35, 8600.92it/s] 24%|██▍       | 95447/400000 [00:11<00:35, 8613.62it/s] 24%|██▍       | 96315/400000 [00:11<00:35, 8631.18it/s] 24%|██▍       | 97179/400000 [00:11<00:35, 8615.70it/s] 25%|██▍       | 98045/400000 [00:11<00:35, 8626.53it/s] 25%|██▍       | 98908/400000 [00:11<00:34, 8624.30it/s] 25%|██▍       | 99776/400000 [00:11<00:34, 8639.63it/s] 25%|██▌       | 100642/400000 [00:11<00:34, 8645.00it/s] 25%|██▌       | 101507/400000 [00:12<00:34, 8603.50it/s] 26%|██▌       | 102370/400000 [00:12<00:34, 8609.44it/s] 26%|██▌       | 103231/400000 [00:12<00:34, 8583.43it/s] 26%|██▌       | 104090/400000 [00:12<00:34, 8578.89it/s] 26%|██▌       | 104960/400000 [00:12<00:34, 8612.48it/s] 26%|██▋       | 105822/400000 [00:12<00:34, 8608.29it/s] 27%|██▋       | 106685/400000 [00:12<00:34, 8612.39it/s] 27%|██▋       | 107547/400000 [00:12<00:33, 8612.71it/s] 27%|██▋       | 108416/400000 [00:12<00:33, 8634.47it/s] 27%|██▋       | 109280/400000 [00:12<00:33, 8610.64it/s] 28%|██▊       | 110142/400000 [00:13<00:33, 8607.25it/s] 28%|██▊       | 111003/400000 [00:13<00:33, 8605.42it/s] 28%|██▊       | 111864/400000 [00:13<00:33, 8580.64it/s] 28%|██▊       | 112724/400000 [00:13<00:33, 8586.42it/s] 28%|██▊       | 113593/400000 [00:13<00:33, 8616.60it/s] 29%|██▊       | 114459/400000 [00:13<00:33, 8628.68it/s] 29%|██▉       | 115323/400000 [00:13<00:32, 8631.50it/s] 29%|██▉       | 116187/400000 [00:13<00:32, 8618.28it/s] 29%|██▉       | 117059/400000 [00:13<00:32, 8646.33it/s] 29%|██▉       | 117924/400000 [00:13<00:32, 8639.29it/s] 30%|██▉       | 118790/400000 [00:14<00:32, 8644.98it/s] 30%|██▉       | 119656/400000 [00:14<00:32, 8647.33it/s] 30%|███       | 120521/400000 [00:14<00:32, 8618.64it/s] 30%|███       | 121383/400000 [00:14<00:32, 8618.76it/s] 31%|███       | 122245/400000 [00:14<00:32, 8564.03it/s] 31%|███       | 123107/400000 [00:14<00:32, 8578.10it/s] 31%|███       | 123975/400000 [00:14<00:32, 8606.50it/s] 31%|███       | 124836/400000 [00:14<00:31, 8599.72it/s] 31%|███▏      | 125700/400000 [00:14<00:31, 8611.19it/s] 32%|███▏      | 126562/400000 [00:14<00:31, 8594.55it/s] 32%|███▏      | 127422/400000 [00:15<00:31, 8575.20it/s] 32%|███▏      | 128280/400000 [00:15<00:31, 8547.40it/s] 32%|███▏      | 129135/400000 [00:15<00:31, 8527.01it/s] 32%|███▏      | 129991/400000 [00:15<00:31, 8535.68it/s] 33%|███▎      | 130848/400000 [00:15<00:31, 8543.80it/s] 33%|███▎      | 131704/400000 [00:15<00:31, 8545.89it/s] 33%|███▎      | 132561/400000 [00:15<00:31, 8552.38it/s] 33%|███▎      | 133420/400000 [00:15<00:31, 8562.65it/s] 34%|███▎      | 134277/400000 [00:15<00:31, 8551.06it/s] 34%|███▍      | 135133/400000 [00:15<00:31, 8509.09it/s] 34%|███▍      | 135990/400000 [00:16<00:30, 8524.85it/s] 34%|███▍      | 136843/400000 [00:16<00:30, 8515.27it/s] 34%|███▍      | 137702/400000 [00:16<00:30, 8535.14it/s] 35%|███▍      | 138562/400000 [00:16<00:30, 8551.61it/s] 35%|███▍      | 139418/400000 [00:16<00:30, 8530.56it/s] 35%|███▌      | 140272/400000 [00:16<00:30, 8520.07it/s] 35%|███▌      | 141133/400000 [00:16<00:30, 8545.71it/s] 35%|███▌      | 141998/400000 [00:16<00:30, 8575.71it/s] 36%|███▌      | 142867/400000 [00:16<00:29, 8607.42it/s] 36%|███▌      | 143740/400000 [00:16<00:29, 8643.12it/s] 36%|███▌      | 144605/400000 [00:17<00:29, 8605.78it/s] 36%|███▋      | 145472/400000 [00:17<00:29, 8624.30it/s] 37%|███▋      | 146337/400000 [00:17<00:29, 8630.04it/s] 37%|███▋      | 147201/400000 [00:17<00:29, 8610.07it/s] 37%|███▋      | 148063/400000 [00:17<00:29, 8603.98it/s] 37%|███▋      | 148924/400000 [00:17<00:29, 8574.44it/s] 37%|███▋      | 149792/400000 [00:17<00:29, 8604.95it/s] 38%|███▊      | 150663/400000 [00:17<00:28, 8634.72it/s] 38%|███▊      | 151527/400000 [00:17<00:28, 8615.94it/s] 38%|███▊      | 152389/400000 [00:17<00:28, 8596.50it/s] 38%|███▊      | 153249/400000 [00:18<00:28, 8590.17it/s] 39%|███▊      | 154113/400000 [00:18<00:28, 8603.29it/s] 39%|███▊      | 154974/400000 [00:18<00:28, 8602.47it/s] 39%|███▉      | 155846/400000 [00:18<00:28, 8636.26it/s] 39%|███▉      | 156710/400000 [00:18<00:29, 8341.56it/s] 39%|███▉      | 157554/400000 [00:18<00:28, 8369.03it/s] 40%|███▉      | 158418/400000 [00:18<00:28, 8447.63it/s] 40%|███▉      | 159283/400000 [00:18<00:28, 8506.92it/s] 40%|████      | 160150/400000 [00:18<00:28, 8553.40it/s] 40%|████      | 161009/400000 [00:18<00:27, 8562.44it/s] 40%|████      | 161868/400000 [00:19<00:27, 8570.61it/s] 41%|████      | 162726/400000 [00:19<00:27, 8557.03it/s] 41%|████      | 163582/400000 [00:19<00:27, 8545.33it/s] 41%|████      | 164450/400000 [00:19<00:27, 8582.92it/s] 41%|████▏     | 165309/400000 [00:19<00:27, 8558.40it/s] 42%|████▏     | 166171/400000 [00:19<00:27, 8575.06it/s] 42%|████▏     | 167029/400000 [00:19<00:27, 8556.00it/s] 42%|████▏     | 167889/400000 [00:19<00:27, 8566.65it/s] 42%|████▏     | 168758/400000 [00:19<00:26, 8600.44it/s] 42%|████▏     | 169619/400000 [00:19<00:26, 8600.90it/s] 43%|████▎     | 170480/400000 [00:20<00:26, 8586.67it/s] 43%|████▎     | 171346/400000 [00:20<00:26, 8607.19it/s] 43%|████▎     | 172217/400000 [00:20<00:26, 8636.96it/s] 43%|████▎     | 173081/400000 [00:20<00:26, 8620.30it/s] 43%|████▎     | 173944/400000 [00:20<00:26, 8574.84it/s] 44%|████▎     | 174805/400000 [00:20<00:26, 8584.06it/s] 44%|████▍     | 175666/400000 [00:20<00:26, 8590.21it/s] 44%|████▍     | 176526/400000 [00:20<00:26, 8587.77it/s] 44%|████▍     | 177395/400000 [00:20<00:25, 8616.24it/s] 45%|████▍     | 178257/400000 [00:20<00:25, 8595.42it/s] 45%|████▍     | 179122/400000 [00:21<00:25, 8611.53it/s] 45%|████▍     | 179984/400000 [00:21<00:25, 8584.26it/s] 45%|████▌     | 180847/400000 [00:21<00:25, 8597.16it/s] 45%|████▌     | 181707/400000 [00:21<00:25, 8593.03it/s] 46%|████▌     | 182577/400000 [00:21<00:25, 8622.35it/s] 46%|████▌     | 183449/400000 [00:21<00:25, 8649.02it/s] 46%|████▌     | 184320/400000 [00:21<00:24, 8666.18it/s] 46%|████▋     | 185187/400000 [00:21<00:24, 8654.81it/s] 47%|████▋     | 186053/400000 [00:21<00:25, 8462.08it/s] 47%|████▋     | 186901/400000 [00:21<00:25, 8416.71it/s] 47%|████▋     | 187754/400000 [00:22<00:25, 8447.45it/s] 47%|████▋     | 188615/400000 [00:22<00:24, 8494.65it/s] 47%|████▋     | 189478/400000 [00:22<00:24, 8532.25it/s] 48%|████▊     | 190338/400000 [00:22<00:24, 8549.85it/s] 48%|████▊     | 191194/400000 [00:22<00:24, 8538.80it/s] 48%|████▊     | 192067/400000 [00:22<00:24, 8592.94it/s] 48%|████▊     | 192930/400000 [00:22<00:24, 8602.38it/s] 48%|████▊     | 193791/400000 [00:22<00:23, 8600.96it/s] 49%|████▊     | 194652/400000 [00:22<00:23, 8594.13it/s] 49%|████▉     | 195512/400000 [00:23<00:24, 8411.59it/s] 49%|████▉     | 196365/400000 [00:23<00:24, 8443.93it/s] 49%|████▉     | 197230/400000 [00:23<00:23, 8502.83it/s] 50%|████▉     | 198089/400000 [00:23<00:23, 8527.20it/s] 50%|████▉     | 198957/400000 [00:23<00:23, 8571.78it/s] 50%|████▉     | 199820/400000 [00:23<00:23, 8587.68it/s] 50%|█████     | 200688/400000 [00:23<00:23, 8612.85it/s] 50%|█████     | 201555/400000 [00:23<00:22, 8628.30it/s] 51%|█████     | 202418/400000 [00:23<00:22, 8614.89it/s] 51%|█████     | 203281/400000 [00:23<00:22, 8619.27it/s] 51%|█████     | 204143/400000 [00:24<00:22, 8612.87it/s] 51%|█████▏    | 205008/400000 [00:24<00:22, 8621.48it/s] 51%|█████▏    | 205871/400000 [00:24<00:22, 8619.95it/s] 52%|█████▏    | 206734/400000 [00:24<00:22, 8567.73it/s] 52%|█████▏    | 207594/400000 [00:24<00:22, 8576.59it/s] 52%|█████▏    | 208460/400000 [00:24<00:22, 8599.55it/s] 52%|█████▏    | 209327/400000 [00:24<00:22, 8619.66it/s] 53%|█████▎    | 210190/400000 [00:24<00:22, 8618.61it/s] 53%|█████▎    | 211052/400000 [00:24<00:21, 8601.81it/s] 53%|█████▎    | 211926/400000 [00:24<00:21, 8641.35it/s] 53%|█████▎    | 212791/400000 [00:25<00:21, 8575.81it/s] 53%|█████▎    | 213654/400000 [00:25<00:21, 8589.66it/s] 54%|█████▎    | 214514/400000 [00:25<00:22, 8326.37it/s] 54%|█████▍    | 215366/400000 [00:25<00:22, 8382.00it/s] 54%|█████▍    | 216206/400000 [00:25<00:21, 8379.18it/s] 54%|█████▍    | 217065/400000 [00:25<00:21, 8441.04it/s] 54%|█████▍    | 217922/400000 [00:25<00:21, 8478.81it/s] 55%|█████▍    | 218771/400000 [00:25<00:21, 8466.85it/s] 55%|█████▍    | 219619/400000 [00:25<00:21, 8463.85it/s] 55%|█████▌    | 220473/400000 [00:25<00:21, 8484.82it/s] 55%|█████▌    | 221339/400000 [00:26<00:20, 8534.70it/s] 56%|█████▌    | 222202/400000 [00:26<00:20, 8561.32it/s] 56%|█████▌    | 223071/400000 [00:26<00:20, 8597.60it/s] 56%|█████▌    | 223931/400000 [00:26<00:20, 8443.24it/s] 56%|█████▌    | 224793/400000 [00:26<00:20, 8492.64it/s] 56%|█████▋    | 225661/400000 [00:26<00:20, 8545.60it/s] 57%|█████▋    | 226524/400000 [00:26<00:20, 8568.48it/s] 57%|█████▋    | 227387/400000 [00:26<00:20, 8585.90it/s] 57%|█████▋    | 228251/400000 [00:26<00:19, 8600.22it/s] 57%|█████▋    | 229121/400000 [00:26<00:19, 8629.16it/s] 57%|█████▋    | 229985/400000 [00:27<00:19, 8620.34it/s] 58%|█████▊    | 230848/400000 [00:27<00:19, 8603.72it/s] 58%|█████▊    | 231709/400000 [00:27<00:19, 8574.12it/s] 58%|█████▊    | 232567/400000 [00:27<00:19, 8553.76it/s] 58%|█████▊    | 233423/400000 [00:27<00:20, 8323.88it/s] 59%|█████▊    | 234280/400000 [00:27<00:19, 8395.18it/s] 59%|█████▉    | 235121/400000 [00:27<00:19, 8318.90it/s] 59%|█████▉    | 235978/400000 [00:27<00:19, 8392.33it/s] 59%|█████▉    | 236824/400000 [00:27<00:19, 8410.81it/s] 59%|█████▉    | 237684/400000 [00:27<00:19, 8465.99it/s] 60%|█████▉    | 238554/400000 [00:28<00:18, 8534.22it/s] 60%|█████▉    | 239417/400000 [00:28<00:18, 8559.45it/s] 60%|██████    | 240274/400000 [00:28<00:18, 8557.40it/s] 60%|██████    | 241130/400000 [00:28<00:18, 8508.01it/s] 60%|██████    | 241989/400000 [00:28<00:18, 8529.39it/s] 61%|██████    | 242848/400000 [00:28<00:18, 8545.05it/s] 61%|██████    | 243703/400000 [00:28<00:18, 8543.35it/s] 61%|██████    | 244558/400000 [00:28<00:18, 8535.14it/s] 61%|██████▏   | 245433/400000 [00:28<00:17, 8596.41it/s] 62%|██████▏   | 246303/400000 [00:28<00:17, 8626.01it/s] 62%|██████▏   | 247174/400000 [00:29<00:17, 8649.90it/s] 62%|██████▏   | 248040/400000 [00:29<00:17, 8638.96it/s] 62%|██████▏   | 248904/400000 [00:29<00:17, 8636.64it/s] 62%|██████▏   | 249772/400000 [00:29<00:17, 8647.10it/s] 63%|██████▎   | 250637/400000 [00:29<00:17, 8638.52it/s] 63%|██████▎   | 251501/400000 [00:29<00:17, 8620.61it/s] 63%|██████▎   | 252370/400000 [00:29<00:17, 8641.09it/s] 63%|██████▎   | 253245/400000 [00:29<00:16, 8671.48it/s] 64%|██████▎   | 254113/400000 [00:29<00:16, 8660.72it/s] 64%|██████▎   | 254982/400000 [00:29<00:16, 8666.66it/s] 64%|██████▍   | 255849/400000 [00:30<00:16, 8665.33it/s] 64%|██████▍   | 256716/400000 [00:30<00:16, 8622.10it/s] 64%|██████▍   | 257579/400000 [00:30<00:16, 8500.51it/s] 65%|██████▍   | 258434/400000 [00:30<00:16, 8514.48it/s] 65%|██████▍   | 259296/400000 [00:30<00:16, 8544.92it/s] 65%|██████▌   | 260154/400000 [00:30<00:16, 8554.60it/s] 65%|██████▌   | 261018/400000 [00:30<00:16, 8578.69it/s] 65%|██████▌   | 261877/400000 [00:30<00:16, 8370.09it/s] 66%|██████▌   | 262727/400000 [00:30<00:16, 8406.75it/s] 66%|██████▌   | 263582/400000 [00:30<00:16, 8448.76it/s] 66%|██████▌   | 264438/400000 [00:31<00:15, 8479.83it/s] 66%|██████▋   | 265290/400000 [00:31<00:15, 8490.09it/s] 67%|██████▋   | 266154/400000 [00:31<00:15, 8533.61it/s] 67%|██████▋   | 267018/400000 [00:31<00:15, 8562.50it/s] 67%|██████▋   | 267879/400000 [00:31<00:15, 8575.82it/s] 67%|██████▋   | 268745/400000 [00:31<00:15, 8598.92it/s] 67%|██████▋   | 269620/400000 [00:31<00:15, 8641.42it/s] 68%|██████▊   | 270488/400000 [00:31<00:14, 8650.62it/s] 68%|██████▊   | 271354/400000 [00:31<00:14, 8640.93it/s] 68%|██████▊   | 272219/400000 [00:31<00:14, 8619.66it/s] 68%|██████▊   | 273082/400000 [00:32<00:14, 8578.77it/s] 68%|██████▊   | 273940/400000 [00:32<00:14, 8417.64it/s] 69%|██████▊   | 274802/400000 [00:32<00:14, 8476.23it/s] 69%|██████▉   | 275674/400000 [00:32<00:14, 8545.22it/s] 69%|██████▉   | 276539/400000 [00:32<00:14, 8573.73it/s] 69%|██████▉   | 277404/400000 [00:32<00:14, 8596.18it/s] 70%|██████▉   | 278276/400000 [00:32<00:14, 8630.40it/s] 70%|██████▉   | 279147/400000 [00:32<00:13, 8652.28it/s] 70%|███████   | 280013/400000 [00:32<00:13, 8653.74it/s] 70%|███████   | 280885/400000 [00:32<00:13, 8670.74it/s] 70%|███████   | 281753/400000 [00:33<00:14, 8409.90it/s] 71%|███████   | 282618/400000 [00:33<00:13, 8478.12it/s] 71%|███████   | 283480/400000 [00:33<00:13, 8518.44it/s] 71%|███████   | 284334/400000 [00:33<00:13, 8523.68it/s] 71%|███████▏  | 285188/400000 [00:33<00:13, 8526.65it/s] 72%|███████▏  | 286042/400000 [00:33<00:13, 8525.76it/s] 72%|███████▏  | 286905/400000 [00:33<00:13, 8554.21it/s] 72%|███████▏  | 287761/400000 [00:33<00:13, 8543.73it/s] 72%|███████▏  | 288627/400000 [00:33<00:12, 8575.49it/s] 72%|███████▏  | 289485/400000 [00:33<00:12, 8573.04it/s] 73%|███████▎  | 290357/400000 [00:34<00:12, 8614.01it/s] 73%|███████▎  | 291219/400000 [00:34<00:12, 8614.69it/s] 73%|███████▎  | 292081/400000 [00:34<00:12, 8606.73it/s] 73%|███████▎  | 292950/400000 [00:34<00:12, 8631.45it/s] 73%|███████▎  | 293814/400000 [00:34<00:12, 8584.00it/s] 74%|███████▎  | 294674/400000 [00:34<00:12, 8588.76it/s] 74%|███████▍  | 295533/400000 [00:34<00:12, 8562.00it/s] 74%|███████▍  | 296390/400000 [00:34<00:12, 8544.77it/s] 74%|███████▍  | 297245/400000 [00:34<00:12, 8541.85it/s] 75%|███████▍  | 298100/400000 [00:34<00:11, 8529.15it/s] 75%|███████▍  | 298953/400000 [00:35<00:11, 8520.45it/s] 75%|███████▍  | 299807/400000 [00:35<00:11, 8524.56it/s] 75%|███████▌  | 300660/400000 [00:35<00:11, 8496.90it/s] 75%|███████▌  | 301510/400000 [00:35<00:11, 8361.09it/s] 76%|███████▌  | 302347/400000 [00:35<00:11, 8303.59it/s] 76%|███████▌  | 303208/400000 [00:35<00:11, 8393.11it/s] 76%|███████▌  | 304072/400000 [00:35<00:11, 8463.29it/s] 76%|███████▌  | 304934/400000 [00:35<00:11, 8507.18it/s] 76%|███████▋  | 305794/400000 [00:35<00:11, 8532.77it/s] 77%|███████▋  | 306651/400000 [00:36<00:10, 8541.92it/s] 77%|███████▋  | 307506/400000 [00:36<00:11, 8377.68it/s] 77%|███████▋  | 308367/400000 [00:36<00:10, 8444.85it/s] 77%|███████▋  | 309232/400000 [00:36<00:10, 8503.44it/s] 78%|███████▊  | 310083/400000 [00:36<00:10, 8503.04it/s] 78%|███████▊  | 310934/400000 [00:36<00:10, 8472.31it/s] 78%|███████▊  | 311789/400000 [00:36<00:10, 8493.70it/s] 78%|███████▊  | 312639/400000 [00:36<00:10, 8484.00it/s] 78%|███████▊  | 313501/400000 [00:36<00:10, 8522.14it/s] 79%|███████▊  | 314354/400000 [00:36<00:10, 8519.91it/s] 79%|███████▉  | 315207/400000 [00:37<00:10, 8439.62it/s] 79%|███████▉  | 316054/400000 [00:37<00:09, 8447.58it/s] 79%|███████▉  | 316920/400000 [00:37<00:09, 8507.91it/s] 79%|███████▉  | 317791/400000 [00:37<00:09, 8565.62it/s] 80%|███████▉  | 318655/400000 [00:37<00:09, 8585.72it/s] 80%|███████▉  | 319514/400000 [00:37<00:09, 8573.50it/s] 80%|████████  | 320373/400000 [00:37<00:09, 8575.80it/s] 80%|████████  | 321233/400000 [00:37<00:09, 8580.96it/s] 81%|████████  | 322100/400000 [00:37<00:09, 8604.71it/s] 81%|████████  | 322970/400000 [00:37<00:08, 8631.16it/s] 81%|████████  | 323834/400000 [00:38<00:09, 8395.72it/s] 81%|████████  | 324697/400000 [00:38<00:08, 8463.44it/s] 81%|████████▏ | 325570/400000 [00:38<00:08, 8541.12it/s] 82%|████████▏ | 326431/400000 [00:38<00:08, 8558.78it/s] 82%|████████▏ | 327297/400000 [00:38<00:08, 8587.65it/s] 82%|████████▏ | 328160/400000 [00:38<00:08, 8598.43it/s] 82%|████████▏ | 329024/400000 [00:38<00:08, 8609.33it/s] 82%|████████▏ | 329886/400000 [00:38<00:08, 8538.20it/s] 83%|████████▎ | 330750/400000 [00:38<00:08, 8566.00it/s] 83%|████████▎ | 331616/400000 [00:38<00:07, 8591.46it/s] 83%|████████▎ | 332483/400000 [00:39<00:07, 8613.56it/s] 83%|████████▎ | 333347/400000 [00:39<00:07, 8620.98it/s] 84%|████████▎ | 334218/400000 [00:39<00:07, 8645.59it/s] 84%|████████▍ | 335085/400000 [00:39<00:07, 8652.18it/s] 84%|████████▍ | 335951/400000 [00:39<00:07, 8643.30it/s] 84%|████████▍ | 336816/400000 [00:39<00:07, 8613.50it/s] 84%|████████▍ | 337678/400000 [00:39<00:07, 8602.40it/s] 85%|████████▍ | 338539/400000 [00:39<00:07, 8594.20it/s] 85%|████████▍ | 339401/400000 [00:39<00:07, 8600.09it/s] 85%|████████▌ | 340270/400000 [00:39<00:06, 8626.60it/s] 85%|████████▌ | 341133/400000 [00:40<00:06, 8568.59it/s] 86%|████████▌ | 342006/400000 [00:40<00:06, 8614.59it/s] 86%|████████▌ | 342868/400000 [00:40<00:06, 8593.16it/s] 86%|████████▌ | 343728/400000 [00:40<00:06, 8585.72it/s] 86%|████████▌ | 344587/400000 [00:40<00:06, 8572.58it/s] 86%|████████▋ | 345445/400000 [00:40<00:06, 8557.89it/s] 87%|████████▋ | 346310/400000 [00:40<00:06, 8585.23it/s] 87%|████████▋ | 347172/400000 [00:40<00:06, 8593.68it/s] 87%|████████▋ | 348034/400000 [00:40<00:06, 8601.13it/s] 87%|████████▋ | 348901/400000 [00:40<00:05, 8619.82it/s] 87%|████████▋ | 349764/400000 [00:41<00:05, 8588.16it/s] 88%|████████▊ | 350623/400000 [00:41<00:05, 8584.27it/s] 88%|████████▊ | 351482/400000 [00:41<00:05, 8492.94it/s] 88%|████████▊ | 352332/400000 [00:41<00:05, 8484.63it/s] 88%|████████▊ | 353200/400000 [00:41<00:05, 8540.86it/s] 89%|████████▊ | 354065/400000 [00:41<00:05, 8569.33it/s] 89%|████████▊ | 354926/400000 [00:41<00:05, 8579.63it/s] 89%|████████▉ | 355786/400000 [00:41<00:05, 8584.02it/s] 89%|████████▉ | 356651/400000 [00:41<00:05, 8603.41it/s] 89%|████████▉ | 357517/400000 [00:41<00:04, 8617.44it/s] 90%|████████▉ | 358383/400000 [00:42<00:04, 8627.05it/s] 90%|████████▉ | 359246/400000 [00:42<00:04, 8576.53it/s] 90%|█████████ | 360121/400000 [00:42<00:04, 8627.41it/s] 90%|█████████ | 360986/400000 [00:42<00:04, 8631.08it/s] 90%|█████████ | 361853/400000 [00:42<00:04, 8642.00it/s] 91%|█████████ | 362718/400000 [00:42<00:04, 8568.50it/s] 91%|█████████ | 363576/400000 [00:42<00:04, 8515.30it/s] 91%|█████████ | 364440/400000 [00:42<00:04, 8550.93it/s] 91%|█████████▏| 365312/400000 [00:42<00:04, 8598.94it/s] 92%|█████████▏| 366176/400000 [00:42<00:03, 8610.11it/s] 92%|█████████▏| 367050/400000 [00:43<00:03, 8648.61it/s] 92%|█████████▏| 367915/400000 [00:43<00:03, 8612.72it/s] 92%|█████████▏| 368780/400000 [00:43<00:03, 8621.32it/s] 92%|█████████▏| 369650/400000 [00:43<00:03, 8642.46it/s] 93%|█████████▎| 370515/400000 [00:43<00:03, 8643.51it/s] 93%|█████████▎| 371380/400000 [00:43<00:03, 8636.35it/s] 93%|█████████▎| 372244/400000 [00:43<00:03, 8633.53it/s] 93%|█████████▎| 373108/400000 [00:43<00:03, 8623.32it/s] 93%|█████████▎| 373984/400000 [00:43<00:03, 8662.72it/s] 94%|█████████▎| 374851/400000 [00:43<00:02, 8654.99it/s] 94%|█████████▍| 375717/400000 [00:44<00:02, 8499.15it/s] 94%|█████████▍| 376568/400000 [00:44<00:02, 8496.20it/s] 94%|█████████▍| 377419/400000 [00:44<00:02, 8493.64it/s] 95%|█████████▍| 378272/400000 [00:44<00:02, 8501.70it/s] 95%|█████████▍| 379128/400000 [00:44<00:02, 8517.77it/s] 95%|█████████▍| 379983/400000 [00:44<00:02, 8525.77it/s] 95%|█████████▌| 380841/400000 [00:44<00:02, 8539.43it/s] 95%|█████████▌| 381703/400000 [00:44<00:02, 8562.71it/s] 96%|█████████▌| 382560/400000 [00:44<00:02, 8556.56it/s] 96%|█████████▌| 383416/400000 [00:44<00:01, 8550.56it/s] 96%|█████████▌| 384272/400000 [00:45<00:01, 8544.88it/s] 96%|█████████▋| 385127/400000 [00:45<00:01, 8528.53it/s] 96%|█████████▋| 385995/400000 [00:45<00:01, 8571.56it/s] 97%|█████████▋| 386853/400000 [00:45<00:01, 8531.20it/s] 97%|█████████▋| 387723/400000 [00:45<00:01, 8578.83it/s] 97%|█████████▋| 388584/400000 [00:45<00:01, 8585.14it/s] 97%|█████████▋| 389443/400000 [00:45<00:01, 8568.17it/s] 98%|█████████▊| 390312/400000 [00:45<00:01, 8604.17it/s] 98%|█████████▊| 391188/400000 [00:45<00:01, 8649.87it/s] 98%|█████████▊| 392054/400000 [00:45<00:00, 8651.06it/s] 98%|█████████▊| 392920/400000 [00:46<00:00, 8341.36it/s] 98%|█████████▊| 393757/400000 [00:46<00:00, 8344.64it/s] 99%|█████████▊| 394618/400000 [00:46<00:00, 8422.28it/s] 99%|█████████▉| 395481/400000 [00:46<00:00, 8481.81it/s] 99%|█████████▉| 396347/400000 [00:46<00:00, 8532.91it/s] 99%|█████████▉| 397215/400000 [00:46<00:00, 8575.76it/s]100%|█████████▉| 398074/400000 [00:46<00:00, 8540.14it/s]100%|█████████▉| 398947/400000 [00:46<00:00, 8594.43it/s]100%|█████████▉| 399810/400000 [00:46<00:00, 8602.61it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8528.59it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f1597040d30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011210699279250334 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.010960681183282348 	 Accuracy: 69

  model saves at 69% accuracy 

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
2020-05-14 02:23:51.731542: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 02:23:51.735615: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095090000 Hz
2020-05-14 02:23:51.736951: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56413ff757c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 02:23:51.736967: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f1540570390> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8506 - accuracy: 0.4880
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6436 - accuracy: 0.5015 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7535 - accuracy: 0.4943
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7203 - accuracy: 0.4965
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7525 - accuracy: 0.4944
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7816 - accuracy: 0.4925
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6885 - accuracy: 0.4986
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7471 - accuracy: 0.4947
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7518 - accuracy: 0.4944
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7249 - accuracy: 0.4962
11000/25000 [============>.................] - ETA: 3s - loss: 7.7182 - accuracy: 0.4966
12000/25000 [=============>................] - ETA: 3s - loss: 7.7254 - accuracy: 0.4962
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7291 - accuracy: 0.4959
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6809 - accuracy: 0.4991
15000/25000 [=================>............] - ETA: 2s - loss: 7.7147 - accuracy: 0.4969
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6973 - accuracy: 0.4980
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6910 - accuracy: 0.4984
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6692 - accuracy: 0.4998
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6828 - accuracy: 0.4989
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6628 - accuracy: 0.5002
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6703 - accuracy: 0.4998
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6757 - accuracy: 0.4994
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6693 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6564 - accuracy: 0.5007
25000/25000 [==============================] - 8s 307us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f14fbf6d710> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f152c271128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.7088 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.6875 - val_crf_viterbi_accuracy: 0.0133

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


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fda56dd7fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 03:13:21.653170
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 03:13:21.656965
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 03:13:21.660314
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 03:13:21.666340
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fda62ba1470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353096.0938
Epoch 2/10

1/1 [==============================] - 0s 102ms/step - loss: 266602.4062
Epoch 3/10

1/1 [==============================] - 0s 95ms/step - loss: 178850.5312
Epoch 4/10

1/1 [==============================] - 0s 99ms/step - loss: 108017.6172
Epoch 5/10

1/1 [==============================] - 0s 103ms/step - loss: 63283.6602
Epoch 6/10

1/1 [==============================] - 0s 95ms/step - loss: 39016.3320
Epoch 7/10

1/1 [==============================] - 0s 103ms/step - loss: 25524.5664
Epoch 8/10

1/1 [==============================] - 0s 102ms/step - loss: 17654.8027
Epoch 9/10

1/1 [==============================] - 0s 106ms/step - loss: 12845.9453
Epoch 10/10

1/1 [==============================] - 0s 101ms/step - loss: 9786.4980

  #### Inference Need return ypred, ytrue ######################### 
[[-0.06616488  0.26441443  0.6111342  -0.24461654  0.13668427 -0.77513015
  -0.18881202 -0.24323463 -0.5520743  -0.90080976  0.9287618  -0.4344293
   1.0876468   0.49813277  0.3392669   1.1773862   0.34403577  1.3296816
  -0.01518652  0.62768793 -1.0235234   0.04812112  0.03107242  1.1193728
   0.37721315 -0.07409906  0.38977215 -0.6940767   1.5653201   0.67582643
   1.5592914   0.51177466  0.31220174 -1.4207691  -1.4630017  -0.3002165
  -0.29111227 -0.2948629  -0.32043952 -1.0145763   0.10561472  1.4367144
  -0.36592633  0.33501935  0.3832366  -0.43489715  0.9009788   0.2904465
  -0.16299391 -0.32449284  1.125349   -0.6833075  -1.1790432   0.5651237
   0.7145611   0.81509686 -1.6033154  -0.17803198  1.8496618  -0.56057686
   1.0830986  -0.9070192  -0.3129031   1.5211388   0.78136957  0.46297738
  -0.18696639  0.11028836 -1.655678   -1.1053973   0.3708498   0.80117553
  -1.083169    1.2624674  -0.28314328 -0.04110637 -0.2663241   0.54489934
  -0.5706147  -0.05649859  1.5849696  -0.16147262 -0.73060524  0.07026729
   0.24883988  0.03172168  0.49177876  1.2454169  -0.46471298 -0.6044084
  -0.81151414 -0.4570378  -0.26943007  0.43421084  1.2275879  -0.95443773
  -0.7400476  -1.1586161   0.10785106  0.57271224  0.50425905 -0.3990801
   0.38631555  0.41694075 -0.36186606 -0.93500537  0.5785209  -0.814636
  -0.83386016  0.24231285 -0.7631724  -1.7900937  -1.9782202  -1.2975135
  -0.26383522 -0.11478958 -0.44242635 -0.07958919  0.17288737 -0.31317976
  -0.04589248  6.680791    7.0278907   7.3638673   7.1011944   5.0046215
   6.6709247   7.6142907   6.5709805   6.543688    6.5366883   8.060137
   7.116242    4.6154137   5.373384    5.503984    6.663184    5.8421974
   5.9452276   6.609889    5.3239183   6.8704233   5.0550537   5.292838
   5.733771    7.1235037   7.1471214   7.1549616   5.665458    5.220244
   5.1572795   6.274342    6.927991    5.4392877   4.93573     6.0864124
   7.463133    8.269121    6.2745686   7.405292    5.4608507   6.74992
   7.3236513   5.0369043   6.626606    6.6159787   5.9941034   6.322258
   5.6205955   6.0292387   6.979906    6.954465    5.347898    6.6123986
   5.2954307   6.1474485   7.217134    5.98655     5.4846377   5.4860797
   0.2100035   1.2882736   1.5696862   0.75377905  1.8737111   0.4625348
   0.19289011  0.74018914  1.6167852   0.32238758  0.54911405  0.9588247
   0.9623923   0.90182805  2.6444874   1.7253886   1.1717424   0.4530986
   0.98484445  0.15956998  0.71428716  1.098271    0.21523213  0.55798686
   1.7549713   0.577703    1.8311815   1.6867579   1.057211    1.9466888
   2.831542    1.4303952   0.1816625   0.6156206   2.403348    0.6474875
   0.37631714  0.5475493   0.14739811  1.4155095   0.7307849   0.7011114
   0.55550754  0.29238605  0.6878907   1.3175671   0.90541875  0.20046467
   1.2570668   1.6751757   1.9175074   0.25329626  0.1446681   0.33533227
   1.3270854   0.1779924   0.21048254  1.1358535   1.3153278   1.4895543
   0.8036802   1.409297    0.340721    2.178946    0.5954164   2.4334192
   0.11196953  0.7253559   1.2301607   1.6289667   1.3460205   0.49565482
   0.99319863  0.9658679   0.53048265  1.0647204   2.3624787   2.269527
   1.5044614   1.88933     0.70185727  0.7270018   0.25967407  1.4481237
   0.29304016  3.0988307   0.66639805  2.2761762   0.2781887   1.1604278
   1.0558254   1.5028143   0.48453897  0.6227061   0.2615782   1.4625303
   0.9500998   0.4911697   1.1152209   0.42614502  0.39388424  0.36345553
   1.5967672   2.2098951   1.8974357   0.22121882  0.10268462  1.0816172
   0.20123136  0.647974    1.159977    0.67951745  1.9405266   0.49532187
   1.7495964   2.595019    1.3091495   0.39336038  1.5446721   1.397337
   0.12532479  7.163763    6.8521132   7.4400654   7.092747    7.591221
   5.733766    6.882182    5.991876    7.0211134   7.271619    8.068989
   6.5188694   8.030525    6.673275    6.186973    7.513407    7.221203
   6.7233458   6.890765    7.996724    7.4368873   7.7586265   7.160246
   7.343089    7.1484146   6.62963     7.1031437   6.1276913   6.7445507
   6.0855026   5.4835634   6.312713    6.4886317   7.387283    6.304556
   6.1295505   6.457179    7.86602     7.152264    6.5619993   5.9687185
   7.3815947   7.0123997   5.6597905   6.6978      7.1989183   6.758548
   6.581436    6.0377274   6.513971    5.985227    7.650607    6.7725105
   6.897044    7.2878613   6.470035    6.71266     6.600105    7.210328
  -5.0957193  -6.211282   -0.2731759 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 03:13:30.738802
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.1961
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 03:13:30.744455
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9271.37
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 03:13:30.748220
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.0352
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 03:13:30.751443
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -829.316
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140575377932416
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140574436401896
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140574436402400
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140574436402904
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140574436403408
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140574436403912

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fda58a77f98> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.499018
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.467044
grad_step = 000002, loss = 0.444615
grad_step = 000003, loss = 0.422229
grad_step = 000004, loss = 0.397285
grad_step = 000005, loss = 0.369376
grad_step = 000006, loss = 0.346032
grad_step = 000007, loss = 0.331281
grad_step = 000008, loss = 0.325654
grad_step = 000009, loss = 0.312550
grad_step = 000010, loss = 0.295650
grad_step = 000011, loss = 0.282907
grad_step = 000012, loss = 0.274616
grad_step = 000013, loss = 0.267703
grad_step = 000014, loss = 0.259954
grad_step = 000015, loss = 0.250817
grad_step = 000016, loss = 0.240703
grad_step = 000017, loss = 0.230571
grad_step = 000018, loss = 0.221311
grad_step = 000019, loss = 0.212734
grad_step = 000020, loss = 0.204089
grad_step = 000021, loss = 0.195234
grad_step = 000022, loss = 0.186613
grad_step = 000023, loss = 0.178840
grad_step = 000024, loss = 0.171713
grad_step = 000025, loss = 0.164410
grad_step = 000026, loss = 0.156851
grad_step = 000027, loss = 0.149541
grad_step = 000028, loss = 0.142660
grad_step = 000029, loss = 0.135998
grad_step = 000030, loss = 0.129390
grad_step = 000031, loss = 0.122945
grad_step = 000032, loss = 0.116693
grad_step = 000033, loss = 0.110541
grad_step = 000034, loss = 0.104762
grad_step = 000035, loss = 0.099459
grad_step = 000036, loss = 0.094295
grad_step = 000037, loss = 0.089124
grad_step = 000038, loss = 0.084199
grad_step = 000039, loss = 0.079677
grad_step = 000040, loss = 0.075327
grad_step = 000041, loss = 0.071053
grad_step = 000042, loss = 0.066961
grad_step = 000043, loss = 0.063107
grad_step = 000044, loss = 0.059513
grad_step = 000045, loss = 0.056096
grad_step = 000046, loss = 0.052780
grad_step = 000047, loss = 0.049673
grad_step = 000048, loss = 0.046824
grad_step = 000049, loss = 0.044069
grad_step = 000050, loss = 0.041397
grad_step = 000051, loss = 0.038928
grad_step = 000052, loss = 0.036611
grad_step = 000053, loss = 0.034381
grad_step = 000054, loss = 0.032237
grad_step = 000055, loss = 0.030230
grad_step = 000056, loss = 0.028390
grad_step = 000057, loss = 0.026605
grad_step = 000058, loss = 0.024884
grad_step = 000059, loss = 0.023317
grad_step = 000060, loss = 0.021828
grad_step = 000061, loss = 0.020401
grad_step = 000062, loss = 0.019082
grad_step = 000063, loss = 0.017830
grad_step = 000064, loss = 0.016639
grad_step = 000065, loss = 0.015521
grad_step = 000066, loss = 0.014473
grad_step = 000067, loss = 0.013483
grad_step = 000068, loss = 0.012556
grad_step = 000069, loss = 0.011699
grad_step = 000070, loss = 0.010892
grad_step = 000071, loss = 0.010137
grad_step = 000072, loss = 0.009432
grad_step = 000073, loss = 0.008779
grad_step = 000074, loss = 0.008164
grad_step = 000075, loss = 0.007601
grad_step = 000076, loss = 0.007080
grad_step = 000077, loss = 0.006595
grad_step = 000078, loss = 0.006157
grad_step = 000079, loss = 0.005749
grad_step = 000080, loss = 0.005378
grad_step = 000081, loss = 0.005041
grad_step = 000082, loss = 0.004732
grad_step = 000083, loss = 0.004451
grad_step = 000084, loss = 0.004200
grad_step = 000085, loss = 0.003970
grad_step = 000086, loss = 0.003766
grad_step = 000087, loss = 0.003581
grad_step = 000088, loss = 0.003416
grad_step = 000089, loss = 0.003269
grad_step = 000090, loss = 0.003139
grad_step = 000091, loss = 0.003022
grad_step = 000092, loss = 0.002920
grad_step = 000093, loss = 0.002829
grad_step = 000094, loss = 0.002748
grad_step = 000095, loss = 0.002678
grad_step = 000096, loss = 0.002616
grad_step = 000097, loss = 0.002562
grad_step = 000098, loss = 0.002514
grad_step = 000099, loss = 0.002473
grad_step = 000100, loss = 0.002436
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002405
grad_step = 000102, loss = 0.002377
grad_step = 000103, loss = 0.002352
grad_step = 000104, loss = 0.002330
grad_step = 000105, loss = 0.002311
grad_step = 000106, loss = 0.002294
grad_step = 000107, loss = 0.002279
grad_step = 000108, loss = 0.002266
grad_step = 000109, loss = 0.002253
grad_step = 000110, loss = 0.002242
grad_step = 000111, loss = 0.002232
grad_step = 000112, loss = 0.002222
grad_step = 000113, loss = 0.002212
grad_step = 000114, loss = 0.002204
grad_step = 000115, loss = 0.002195
grad_step = 000116, loss = 0.002187
grad_step = 000117, loss = 0.002179
grad_step = 000118, loss = 0.002171
grad_step = 000119, loss = 0.002163
grad_step = 000120, loss = 0.002155
grad_step = 000121, loss = 0.002148
grad_step = 000122, loss = 0.002144
grad_step = 000123, loss = 0.002152
grad_step = 000124, loss = 0.002189
grad_step = 000125, loss = 0.002194
grad_step = 000126, loss = 0.002136
grad_step = 000127, loss = 0.002115
grad_step = 000128, loss = 0.002155
grad_step = 000129, loss = 0.002132
grad_step = 000130, loss = 0.002094
grad_step = 000131, loss = 0.002127
grad_step = 000132, loss = 0.002118
grad_step = 000133, loss = 0.002080
grad_step = 000134, loss = 0.002106
grad_step = 000135, loss = 0.002101
grad_step = 000136, loss = 0.002067
grad_step = 000137, loss = 0.002088
grad_step = 000138, loss = 0.002086
grad_step = 000139, loss = 0.002055
grad_step = 000140, loss = 0.002070
grad_step = 000141, loss = 0.002072
grad_step = 000142, loss = 0.002044
grad_step = 000143, loss = 0.002052
grad_step = 000144, loss = 0.002060
grad_step = 000145, loss = 0.002036
grad_step = 000146, loss = 0.002033
grad_step = 000147, loss = 0.002044
grad_step = 000148, loss = 0.002031
grad_step = 000149, loss = 0.002017
grad_step = 000150, loss = 0.002023
grad_step = 000151, loss = 0.002024
grad_step = 000152, loss = 0.002012
grad_step = 000153, loss = 0.002004
grad_step = 000154, loss = 0.002007
grad_step = 000155, loss = 0.002009
grad_step = 000156, loss = 0.002000
grad_step = 000157, loss = 0.001991
grad_step = 000158, loss = 0.001988
grad_step = 000159, loss = 0.001990
grad_step = 000160, loss = 0.001991
grad_step = 000161, loss = 0.001987
grad_step = 000162, loss = 0.001980
grad_step = 000163, loss = 0.001973
grad_step = 000164, loss = 0.001967
grad_step = 000165, loss = 0.001964
grad_step = 000166, loss = 0.001963
grad_step = 000167, loss = 0.001963
grad_step = 000168, loss = 0.001967
grad_step = 000169, loss = 0.001978
grad_step = 000170, loss = 0.002000
grad_step = 000171, loss = 0.002040
grad_step = 000172, loss = 0.002035
grad_step = 000173, loss = 0.001996
grad_step = 000174, loss = 0.001940
grad_step = 000175, loss = 0.001952
grad_step = 000176, loss = 0.001991
grad_step = 000177, loss = 0.001966
grad_step = 000178, loss = 0.001929
grad_step = 000179, loss = 0.001934
grad_step = 000180, loss = 0.001954
grad_step = 000181, loss = 0.001942
grad_step = 000182, loss = 0.001915
grad_step = 000183, loss = 0.001921
grad_step = 000184, loss = 0.001936
grad_step = 000185, loss = 0.001920
grad_step = 000186, loss = 0.001901
grad_step = 000187, loss = 0.001906
grad_step = 000188, loss = 0.001913
grad_step = 000189, loss = 0.001905
grad_step = 000190, loss = 0.001890
grad_step = 000191, loss = 0.001889
grad_step = 000192, loss = 0.001894
grad_step = 000193, loss = 0.001891
grad_step = 000194, loss = 0.001880
grad_step = 000195, loss = 0.001872
grad_step = 000196, loss = 0.001872
grad_step = 000197, loss = 0.001874
grad_step = 000198, loss = 0.001871
grad_step = 000199, loss = 0.001863
grad_step = 000200, loss = 0.001855
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001851
grad_step = 000202, loss = 0.001851
grad_step = 000203, loss = 0.001850
grad_step = 000204, loss = 0.001848
grad_step = 000205, loss = 0.001843
grad_step = 000206, loss = 0.001838
grad_step = 000207, loss = 0.001831
grad_step = 000208, loss = 0.001825
grad_step = 000209, loss = 0.001820
grad_step = 000210, loss = 0.001815
grad_step = 000211, loss = 0.001810
grad_step = 000212, loss = 0.001805
grad_step = 000213, loss = 0.001801
grad_step = 000214, loss = 0.001798
grad_step = 000215, loss = 0.001804
grad_step = 000216, loss = 0.001837
grad_step = 000217, loss = 0.001953
grad_step = 000218, loss = 0.002098
grad_step = 000219, loss = 0.002238
grad_step = 000220, loss = 0.001886
grad_step = 000221, loss = 0.001836
grad_step = 000222, loss = 0.002011
grad_step = 000223, loss = 0.001875
grad_step = 000224, loss = 0.001878
grad_step = 000225, loss = 0.001890
grad_step = 000226, loss = 0.001795
grad_step = 000227, loss = 0.001941
grad_step = 000228, loss = 0.001827
grad_step = 000229, loss = 0.001762
grad_step = 000230, loss = 0.001872
grad_step = 000231, loss = 0.001773
grad_step = 000232, loss = 0.001781
grad_step = 000233, loss = 0.001785
grad_step = 000234, loss = 0.001725
grad_step = 000235, loss = 0.001771
grad_step = 000236, loss = 0.001727
grad_step = 000237, loss = 0.001724
grad_step = 000238, loss = 0.001731
grad_step = 000239, loss = 0.001684
grad_step = 000240, loss = 0.001719
grad_step = 000241, loss = 0.001705
grad_step = 000242, loss = 0.001686
grad_step = 000243, loss = 0.001701
grad_step = 000244, loss = 0.001676
grad_step = 000245, loss = 0.001673
grad_step = 000246, loss = 0.001672
grad_step = 000247, loss = 0.001655
grad_step = 000248, loss = 0.001671
grad_step = 000249, loss = 0.001683
grad_step = 000250, loss = 0.001726
grad_step = 000251, loss = 0.001760
grad_step = 000252, loss = 0.001802
grad_step = 000253, loss = 0.001723
grad_step = 000254, loss = 0.001655
grad_step = 000255, loss = 0.001632
grad_step = 000256, loss = 0.001667
grad_step = 000257, loss = 0.001696
grad_step = 000258, loss = 0.001636
grad_step = 000259, loss = 0.001596
grad_step = 000260, loss = 0.001625
grad_step = 000261, loss = 0.001655
grad_step = 000262, loss = 0.001676
grad_step = 000263, loss = 0.001649
grad_step = 000264, loss = 0.001609
grad_step = 000265, loss = 0.001578
grad_step = 000266, loss = 0.001586
grad_step = 000267, loss = 0.001600
grad_step = 000268, loss = 0.001603
grad_step = 000269, loss = 0.001605
grad_step = 000270, loss = 0.001586
grad_step = 000271, loss = 0.001564
grad_step = 000272, loss = 0.001559
grad_step = 000273, loss = 0.001566
grad_step = 000274, loss = 0.001573
grad_step = 000275, loss = 0.001571
grad_step = 000276, loss = 0.001563
grad_step = 000277, loss = 0.001549
grad_step = 000278, loss = 0.001540
grad_step = 000279, loss = 0.001536
grad_step = 000280, loss = 0.001536
grad_step = 000281, loss = 0.001539
grad_step = 000282, loss = 0.001542
grad_step = 000283, loss = 0.001542
grad_step = 000284, loss = 0.001541
grad_step = 000285, loss = 0.001542
grad_step = 000286, loss = 0.001536
grad_step = 000287, loss = 0.001531
grad_step = 000288, loss = 0.001525
grad_step = 000289, loss = 0.001519
grad_step = 000290, loss = 0.001511
grad_step = 000291, loss = 0.001505
grad_step = 000292, loss = 0.001500
grad_step = 000293, loss = 0.001495
grad_step = 000294, loss = 0.001490
grad_step = 000295, loss = 0.001487
grad_step = 000296, loss = 0.001484
grad_step = 000297, loss = 0.001483
grad_step = 000298, loss = 0.001485
grad_step = 000299, loss = 0.001493
grad_step = 000300, loss = 0.001510
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001556
grad_step = 000302, loss = 0.001604
grad_step = 000303, loss = 0.001701
grad_step = 000304, loss = 0.001655
grad_step = 000305, loss = 0.001584
grad_step = 000306, loss = 0.001467
grad_step = 000307, loss = 0.001453
grad_step = 000308, loss = 0.001515
grad_step = 000309, loss = 0.001518
grad_step = 000310, loss = 0.001470
grad_step = 000311, loss = 0.001429
grad_step = 000312, loss = 0.001449
grad_step = 000313, loss = 0.001474
grad_step = 000314, loss = 0.001446
grad_step = 000315, loss = 0.001409
grad_step = 000316, loss = 0.001407
grad_step = 000317, loss = 0.001430
grad_step = 000318, loss = 0.001450
grad_step = 000319, loss = 0.001430
grad_step = 000320, loss = 0.001401
grad_step = 000321, loss = 0.001378
grad_step = 000322, loss = 0.001372
grad_step = 000323, loss = 0.001373
grad_step = 000324, loss = 0.001377
grad_step = 000325, loss = 0.001389
grad_step = 000326, loss = 0.001400
grad_step = 000327, loss = 0.001414
grad_step = 000328, loss = 0.001401
grad_step = 000329, loss = 0.001393
grad_step = 000330, loss = 0.001372
grad_step = 000331, loss = 0.001354
grad_step = 000332, loss = 0.001336
grad_step = 000333, loss = 0.001325
grad_step = 000334, loss = 0.001324
grad_step = 000335, loss = 0.001327
grad_step = 000336, loss = 0.001332
grad_step = 000337, loss = 0.001336
grad_step = 000338, loss = 0.001347
grad_step = 000339, loss = 0.001356
grad_step = 000340, loss = 0.001375
grad_step = 000341, loss = 0.001378
grad_step = 000342, loss = 0.001391
grad_step = 000343, loss = 0.001374
grad_step = 000344, loss = 0.001356
grad_step = 000345, loss = 0.001319
grad_step = 000346, loss = 0.001295
grad_step = 000347, loss = 0.001292
grad_step = 000348, loss = 0.001304
grad_step = 000349, loss = 0.001317
grad_step = 000350, loss = 0.001312
grad_step = 000351, loss = 0.001299
grad_step = 000352, loss = 0.001283
grad_step = 000353, loss = 0.001275
grad_step = 000354, loss = 0.001277
grad_step = 000355, loss = 0.001281
grad_step = 000356, loss = 0.001287
grad_step = 000357, loss = 0.001287
grad_step = 000358, loss = 0.001288
grad_step = 000359, loss = 0.001284
grad_step = 000360, loss = 0.001281
grad_step = 000361, loss = 0.001273
grad_step = 000362, loss = 0.001266
grad_step = 000363, loss = 0.001259
grad_step = 000364, loss = 0.001253
grad_step = 000365, loss = 0.001249
grad_step = 000366, loss = 0.001245
grad_step = 000367, loss = 0.001242
grad_step = 000368, loss = 0.001239
grad_step = 000369, loss = 0.001237
grad_step = 000370, loss = 0.001235
grad_step = 000371, loss = 0.001234
grad_step = 000372, loss = 0.001233
grad_step = 000373, loss = 0.001234
grad_step = 000374, loss = 0.001237
grad_step = 000375, loss = 0.001250
grad_step = 000376, loss = 0.001275
grad_step = 000377, loss = 0.001344
grad_step = 000378, loss = 0.001439
grad_step = 000379, loss = 0.001631
grad_step = 000380, loss = 0.001555
grad_step = 000381, loss = 0.001433
grad_step = 000382, loss = 0.001258
grad_step = 000383, loss = 0.001284
grad_step = 000384, loss = 0.001365
grad_step = 000385, loss = 0.001310
grad_step = 000386, loss = 0.001274
grad_step = 000387, loss = 0.001280
grad_step = 000388, loss = 0.001250
grad_step = 000389, loss = 0.001274
grad_step = 000390, loss = 0.001276
grad_step = 000391, loss = 0.001204
grad_step = 000392, loss = 0.001230
grad_step = 000393, loss = 0.001261
grad_step = 000394, loss = 0.001210
grad_step = 000395, loss = 0.001214
grad_step = 000396, loss = 0.001236
grad_step = 000397, loss = 0.001201
grad_step = 000398, loss = 0.001196
grad_step = 000399, loss = 0.001208
grad_step = 000400, loss = 0.001195
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001188
grad_step = 000402, loss = 0.001198
grad_step = 000403, loss = 0.001195
grad_step = 000404, loss = 0.001171
grad_step = 000405, loss = 0.001171
grad_step = 000406, loss = 0.001178
grad_step = 000407, loss = 0.001165
grad_step = 000408, loss = 0.001159
grad_step = 000409, loss = 0.001163
grad_step = 000410, loss = 0.001158
grad_step = 000411, loss = 0.001154
grad_step = 000412, loss = 0.001155
grad_step = 000413, loss = 0.001152
grad_step = 000414, loss = 0.001147
grad_step = 000415, loss = 0.001141
grad_step = 000416, loss = 0.001141
grad_step = 000417, loss = 0.001142
grad_step = 000418, loss = 0.001136
grad_step = 000419, loss = 0.001131
grad_step = 000420, loss = 0.001131
grad_step = 000421, loss = 0.001130
grad_step = 000422, loss = 0.001127
grad_step = 000423, loss = 0.001125
grad_step = 000424, loss = 0.001125
grad_step = 000425, loss = 0.001126
grad_step = 000426, loss = 0.001130
grad_step = 000427, loss = 0.001138
grad_step = 000428, loss = 0.001162
grad_step = 000429, loss = 0.001200
grad_step = 000430, loss = 0.001281
grad_step = 000431, loss = 0.001367
grad_step = 000432, loss = 0.001501
grad_step = 000433, loss = 0.001423
grad_step = 000434, loss = 0.001285
grad_step = 000435, loss = 0.001128
grad_step = 000436, loss = 0.001132
grad_step = 000437, loss = 0.001229
grad_step = 000438, loss = 0.001222
grad_step = 000439, loss = 0.001147
grad_step = 000440, loss = 0.001114
grad_step = 000441, loss = 0.001146
grad_step = 000442, loss = 0.001169
grad_step = 000443, loss = 0.001134
grad_step = 000444, loss = 0.001093
grad_step = 000445, loss = 0.001101
grad_step = 000446, loss = 0.001132
grad_step = 000447, loss = 0.001128
grad_step = 000448, loss = 0.001084
grad_step = 000449, loss = 0.001069
grad_step = 000450, loss = 0.001092
grad_step = 000451, loss = 0.001104
grad_step = 000452, loss = 0.001090
grad_step = 000453, loss = 0.001074
grad_step = 000454, loss = 0.001069
grad_step = 000455, loss = 0.001061
grad_step = 000456, loss = 0.001060
grad_step = 000457, loss = 0.001069
grad_step = 000458, loss = 0.001069
grad_step = 000459, loss = 0.001055
grad_step = 000460, loss = 0.001043
grad_step = 000461, loss = 0.001044
grad_step = 000462, loss = 0.001047
grad_step = 000463, loss = 0.001045
grad_step = 000464, loss = 0.001041
grad_step = 000465, loss = 0.001038
grad_step = 000466, loss = 0.001032
grad_step = 000467, loss = 0.001027
grad_step = 000468, loss = 0.001027
grad_step = 000469, loss = 0.001029
grad_step = 000470, loss = 0.001027
grad_step = 000471, loss = 0.001021
grad_step = 000472, loss = 0.001018
grad_step = 000473, loss = 0.001016
grad_step = 000474, loss = 0.001013
grad_step = 000475, loss = 0.001009
grad_step = 000476, loss = 0.001008
grad_step = 000477, loss = 0.001007
grad_step = 000478, loss = 0.001006
grad_step = 000479, loss = 0.001003
grad_step = 000480, loss = 0.001002
grad_step = 000481, loss = 0.001001
grad_step = 000482, loss = 0.000999
grad_step = 000483, loss = 0.000997
grad_step = 000484, loss = 0.000996
grad_step = 000485, loss = 0.000994
grad_step = 000486, loss = 0.000993
grad_step = 000487, loss = 0.000991
grad_step = 000488, loss = 0.000990
grad_step = 000489, loss = 0.000989
grad_step = 000490, loss = 0.000989
grad_step = 000491, loss = 0.000989
grad_step = 000492, loss = 0.000989
grad_step = 000493, loss = 0.000990
grad_step = 000494, loss = 0.000993
grad_step = 000495, loss = 0.000996
grad_step = 000496, loss = 0.001004
grad_step = 000497, loss = 0.001011
grad_step = 000498, loss = 0.001025
grad_step = 000499, loss = 0.001033
grad_step = 000500, loss = 0.001046
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001038
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

  date_run                              2020-05-14 03:13:49.520714
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.23169
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 03:13:49.526455
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.143647
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 03:13:49.534335
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.12922
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 03:13:49.539617
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.18277
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
0   2020-05-14 03:13:21.653170  ...    mean_absolute_error
1   2020-05-14 03:13:21.656965  ...     mean_squared_error
2   2020-05-14 03:13:21.660314  ...  median_absolute_error
3   2020-05-14 03:13:21.666340  ...               r2_score
4   2020-05-14 03:13:30.738802  ...    mean_absolute_error
5   2020-05-14 03:13:30.744455  ...     mean_squared_error
6   2020-05-14 03:13:30.748220  ...  median_absolute_error
7   2020-05-14 03:13:30.751443  ...               r2_score
8   2020-05-14 03:13:49.520714  ...    mean_absolute_error
9   2020-05-14 03:13:49.526455  ...     mean_squared_error
10  2020-05-14 03:13:49.534335  ...  median_absolute_error
11  2020-05-14 03:13:49.539617  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7eff646b0d30> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 157296.72it/s] 70%|██████▉   | 6914048/9912422 [00:00<00:13, 224489.95it/s]9920512it [00:00, 42529122.80it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1012195.59it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 471953.90it/s]1654784it [00:00, 11672205.92it/s]                         
0it [00:00, ?it/s]8192it [00:00, 161226.28it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7eff1706ceb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7eff1669a0f0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7eff1706ceb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7eff165f0128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7eff13e2d518> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7eff13e16780> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7eff1706ceb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7eff165ad748> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7eff13e2d518> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7eff16468588> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f731a405208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=e78e64061bca538820d3bad09f794d43f457dd690f436c9bdbe106dbf0a9a70b
  Stored in directory: /tmp/pip-ephem-wheel-cache-1fhdq1qj/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f72b2200780> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2465792/17464789 [===>..........................] - ETA: 0s
11698176/17464789 [===================>..........] - ETA: 0s
16654336/17464789 [===========================>..] - ETA: 0s
16867328/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 03:15:22.244871: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 03:15:22.249396: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-14 03:15:22.249552: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5562676e6dc0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 03:15:22.249565: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.4520 - accuracy: 0.5140
 2000/25000 [=>............................] - ETA: 8s - loss: 7.3676 - accuracy: 0.5195 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.4622 - accuracy: 0.5133
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5133 - accuracy: 0.5100
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5930 - accuracy: 0.5048
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5976 - accuracy: 0.5045
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5943 - accuracy: 0.5047
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5861 - accuracy: 0.5052
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6019 - accuracy: 0.5042
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5946 - accuracy: 0.5047
11000/25000 [============>.................] - ETA: 3s - loss: 7.5635 - accuracy: 0.5067
12000/25000 [=============>................] - ETA: 3s - loss: 7.5912 - accuracy: 0.5049
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6135 - accuracy: 0.5035
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6436 - accuracy: 0.5015
15000/25000 [=================>............] - ETA: 2s - loss: 7.6421 - accuracy: 0.5016
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6446 - accuracy: 0.5014
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6450 - accuracy: 0.5014
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6726 - accuracy: 0.4996
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6739 - accuracy: 0.4995
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6728 - accuracy: 0.4996
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6885 - accuracy: 0.4986
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6701 - accuracy: 0.4998
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6760 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 7s 284us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 03:15:35.966825
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 03:15:35.966825  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:05<149:09:22, 1.61kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:05<104:38:37, 2.29kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:05<73:17:28, 3.27kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:05<51:16:39, 4.67kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:05<35:47:00, 6.66kB/s].vector_cache/glove.6B.zip:   1%|          | 8.14M/862M [00:05<24:55:07, 9.52kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.6M/862M [00:05<17:22:25, 13.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.8M/862M [00:05<12:07:05, 19.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.9M/862M [00:06<8:27:14, 27.7kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.3M/862M [00:06<5:53:46, 39.6kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.3M/862M [00:06<4:06:34, 56.6kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:06<2:51:50, 80.8kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.7M/862M [00:06<1:59:46, 115kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 38.0M/862M [00:06<1:23:30, 164kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.3M/862M [00:06<58:14, 235kB/s]  .vector_cache/glove.6B.zip:   5%|▌         | 46.6M/862M [00:06<40:39, 334kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:06<28:23, 476kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:07<22:18, 605kB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.2M/862M [00:07<17:58, 750kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.3M/862M [00:08<12:37, 1.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:08<15:43, 853kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:10<19:49:31, 11.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:11<14:02:37, 15.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:11<9:53:20, 22.6kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:11<6:57:36, 32.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.1M/862M [00:11<4:53:41, 45.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:11<3:25:56, 65.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:11<2:24:06, 92.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:13<1:47:17, 124kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:13<1:19:32, 168kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:13<57:11, 233kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 62.6M/862M [00:13<40:40, 328kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:13<28:43, 463kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:15<23:32, 564kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:15<19:14, 690kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:15<15:03, 881kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.4M/862M [00:15<12:36, 1.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:15<10:02, 1.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.9M/862M [00:15<08:48, 1.51MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.3M/862M [00:15<07:14, 1.83MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.6M/862M [00:16<06:34, 2.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:16<05:32, 2.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:16<05:10, 2.56MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:16<04:25, 2.99MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:16<04:13, 3.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:16<07:53, 1.67MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:16<06:19, 2.09MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:16<05:23, 2.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:17<05:10, 2.55MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.4M/862M [00:17<04:15, 3.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:17<03:42, 3.56MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:17<03:19, 3.96MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:17<02:57, 4.44MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:19<29:08, 451kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:19<36:38, 359kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:19<32:57, 399kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:19<28:07, 467kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:20<23:45, 553kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:20<17:29, 751kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:20<13:01, 1.01MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.4M/862M [00:20<09:41, 1.35MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:20<07:25, 1.77MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:20<05:43, 2.29MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:20<04:33, 2.87MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:21<52:29, 249kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:21<38:56, 336kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:21<28:34, 457kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:21<21:32, 606kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:22<15:42, 831kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.7M/862M [00:22<11:32, 1.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:22<08:37, 1.51MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:22<06:30, 2.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:23<14:08, 920kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:23<14:49, 877kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:23<12:17, 1.06MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:23<09:16, 1.40MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:24<07:03, 1.84MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.1M/862M [00:24<05:26, 2.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:24<04:17, 3.02MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:24<03:26, 3.75MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:24<02:49, 4.57MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.7M/862M [00:24<02:22, 5.44MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.2M/862M [00:27<25:05, 514kB/s] .vector_cache/glove.6B.zip:  10%|█         | 88.3M/862M [00:27<28:44, 449kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:28<25:37, 503kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:28<21:43, 594kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:28<18:01, 715kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:28<14:04, 916kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:28<10:33, 1.22MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:28<07:48, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:28<05:53, 2.18MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.1M/862M [00:28<04:34, 2.81MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:28<04:04, 3.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:28<03:54, 3.28MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:29<04:14, 3.02MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:29<04:22, 2.93MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:29<04:01, 3.18MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:29<03:16, 3.91MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.4M/862M [00:30<06:42, 1.90MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.0M/862M [00:30<05:34, 2.29MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:30<04:19, 2.94MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:30<03:25, 3.72MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:32<07:57, 1.60MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:32<13:19, 954kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:32<11:10, 1.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:32<08:29, 1.49MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:32<06:29, 1.95MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:32<05:02, 2.51MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:32<03:57, 3.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:32<03:15, 3.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:34<52:50, 239kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:34<40:36, 311kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:34<29:38, 426kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:34<21:43, 581kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:34<15:35, 809kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:34<11:28, 1.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:34<09:21, 1.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:34<07:59, 1.57MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:34<06:54, 1.82MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:35<05:52, 2.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:35<05:05, 2.47MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:37<3:16:54, 63.9kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:37<2:30:12, 83.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:37<1:49:07, 115kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:38<1:18:31, 160kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:38<55:28, 226kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:38<39:25, 318kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:38<28:03, 447kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:38<20:03, 624kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:39<16:49, 743kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:39<14:06, 886kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:39<14:35, 857kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:39<14:54, 839kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:39<14:16, 876kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:39<13:57, 895kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:39<13:19, 938kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:39<11:52, 1.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:39<10:33, 1.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:40<08:02, 1.55MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:40<05:59, 2.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:40<04:30, 2.76MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:42<19:21, 642kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:42<26:14, 474kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:42<22:15, 559kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 116M/862M [00:42<17:04, 728kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:42<12:29, 994kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:42<09:03, 1.37MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:42<06:44, 1.84MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:42<05:14, 2.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:45<40:28, 305kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:45<38:31, 321kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:45<31:00, 399kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:45<23:15, 531kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:45<16:42, 739kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:45<11:58, 1.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:47<11:33, 1.06MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:47<13:03, 941kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:47<15:07, 813kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:47<13:25, 916kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:47<12:45, 964kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:47<12:32, 979kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:47<12:40, 970kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:48<13:32, 908kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:48<12:57, 948kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:48<10:22, 1.18MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:48<07:40, 1.60MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:48<05:35, 2.19MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:49<08:43, 1.40MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:49<07:05, 1.72MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:49<05:38, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:49<04:34, 2.67MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:49<03:42, 3.29MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:49<03:12, 3.79MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:49<02:36, 4.66MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:52<1:08:03, 179kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:52<59:37, 204kB/s]  .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:52<45:14, 269kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:52<33:09, 367kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:52<23:35, 515kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:52<16:44, 723kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:52<12:39, 956kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:52<09:33, 1.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:52<07:41, 1.57MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:53<50:16, 240kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:54<36:09, 334kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:54<25:51, 467kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:54<18:55, 638kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:54<13:59, 862kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:54<10:20, 1.17MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:54<07:54, 1.52MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:54<06:09, 1.95MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:57<27:34, 436kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:57<31:15, 385kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:57<25:53, 464kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:57<19:24, 619kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:57<14:03, 854kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:57<09:59, 1.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:59<13:32, 883kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:59<11:50, 1.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:59<09:19, 1.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:59<07:22, 1.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:59<05:58, 2.00MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:59<04:42, 2.53MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [01:01<05:49, 2.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [01:01<05:38, 2.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [01:01<04:27, 2.66MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [01:01<03:23, 3.49MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [01:03<05:29, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [01:03<06:29, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [01:03<05:33, 2.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [01:03<04:12, 2.80MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [01:03<03:05, 3.81MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [01:05<32:16, 364kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:05<28:25, 413kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:05<21:50, 538kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:05<16:15, 722kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:05<11:44, 998kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [01:05<08:22, 1.39MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [01:07<13:06, 891kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [01:07<12:24, 940kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [01:07<09:38, 1.21MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:07<07:05, 1.64MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:07<05:21, 2.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:07<03:52, 2.99MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:11<06:59, 1.65MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:11<16:18, 708kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:11<13:44, 840kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:11<10:18, 1.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:11<07:26, 1.55MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:13<09:10, 1.25MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:13<14:26, 795kB/s] .vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:13<12:11, 942kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:13<08:57, 1.28MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:14<06:23, 1.79MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:15<15:00, 760kB/s] .vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:15<12:35, 906kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:15<09:24, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:15<06:50, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:16<04:57, 2.28MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:17<27:51, 407kB/s] .vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:17<21:11, 535kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:17<15:45, 719kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:17<11:12, 1.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:19<11:27, 983kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:19<16:07, 699kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:20<12:58, 869kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:20<10:16, 1.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:20<08:02, 1.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:20<05:56, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:20<04:19, 2.59MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:21<23:10, 483kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:21<17:11, 651kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:22<12:15, 910kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:23<11:49, 941kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:24<12:12, 912kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:24<09:23, 1.18MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:24<06:55, 1.60MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:28<10:40, 1.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:29<17:35, 629kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:29<17:19, 639kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:29<17:38, 627kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:29<17:31, 631kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:29<16:42, 662kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:29<14:27, 765kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:29<12:57, 853kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:29<11:05, 997kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:29<08:19, 1.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:30<05:59, 1.84MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:30<05:12, 2.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:30<03:52, 2.83MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:32<05:24, 2.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:32<07:35, 1.43MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:33<06:06, 1.78MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:33<04:42, 2.31MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:33<03:27, 3.14MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:34<08:11, 1.32MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:34<07:23, 1.46MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:35<05:35, 1.93MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:35<04:17, 2.51MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:35<03:19, 3.24MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:35<03:26, 3.12MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:36<10:40, 1.01MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:36<09:22, 1.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:37<07:02, 1.53MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:37<05:04, 2.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:38<12:04, 886kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:38<13:31, 791kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:38<11:21, 941kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:39<08:44, 1.22MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:39<06:25, 1.66MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:39<04:38, 2.29MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:40<14:19, 742kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:40<13:38, 779kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:40<11:23, 933kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:41<08:42, 1.22MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:41<06:19, 1.67MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:41<05:15, 2.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:41<03:47, 2.78MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:41<02:51, 3.68MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:43<1:59:42, 87.8kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:44<1:34:11, 112kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:44<1:08:22, 154kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:44<48:19, 217kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:44<34:01, 308kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:44<24:06, 434kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:44<17:34, 595kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:44<12:59, 804kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:45<35:35, 293kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:46<26:47, 390kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:46<19:11, 543kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:46<13:31, 767kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:47<22:16, 466kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:48<18:36, 558kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:48<13:36, 761kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:48<09:41, 1.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:48<06:56, 1.48MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:49<1:17:29, 133kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:50<57:24, 179kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:50<40:48, 252kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:50<28:41, 358kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:50<20:11, 507kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:51<1:03:54, 160kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:52<47:31, 215kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:52<33:50, 302kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:52<23:47, 429kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:52<16:47, 605kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:53<9:24:47, 18.0kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:54<6:38:03, 25.5kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:54<4:39:04, 36.4kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:54<3:15:24, 51.9kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:54<2:16:44, 74.0kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:55<1:39:58, 101kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:56<1:16:09, 133kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:56<54:52, 184kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:56<38:41, 260kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:56<27:19, 368kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:58<24:32, 409kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:58<27:02, 371kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:59<23:01, 436kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:59<20:57, 478kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:59<18:37, 538kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:59<15:55, 630kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:59<13:03, 768kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:59<10:45, 931kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:59<09:12, 1.09MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:59<06:51, 1.46MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:59<05:02, 1.98MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [02:00<03:48, 2.62MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [02:00<14:36, 682kB/s] .vector_cache/glove.6B.zip:  31%|███       | 265M/862M [02:01<11:33, 862kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [02:01<09:14, 1.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [02:01<08:29, 1.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [02:01<08:17, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [02:01<08:04, 1.23MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [02:01<07:10, 1.39MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [02:01<05:23, 1.84MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [02:01<04:02, 2.45MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [02:02<06:14, 1.59MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [02:03<08:15, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [02:03<08:36, 1.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [02:03<09:04, 1.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [02:03<10:02, 985kB/s] .vector_cache/glove.6B.zip:  31%|███       | 269M/862M [02:03<11:48, 838kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [02:03<10:57, 902kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 269M/862M [02:03<09:19, 1.06MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [02:03<07:36, 1.30MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [02:03<06:19, 1.56MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [02:03<04:47, 2.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [02:04<03:35, 2.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [02:04<05:33, 1.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [02:04<04:46, 2.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [02:05<03:51, 2.54MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [02:05<03:26, 2.85MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [02:05<03:13, 3.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [02:05<02:35, 3.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [02:05<02:05, 4.65MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [02:07<10:40, 913kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [02:07<17:10, 568kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [02:07<15:16, 638kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [02:07<12:00, 812kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [02:07<09:25, 1.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [02:07<07:01, 1.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [02:07<05:52, 1.65MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [02:08<05:03, 1.92MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [02:08<04:32, 2.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [02:08<04:08, 2.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [02:08<03:35, 2.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [02:08<02:49, 3.43MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [02:09<19:50, 488kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [02:09<14:22, 673kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [02:09<11:00, 878kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [02:09<08:22, 1.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [02:09<06:37, 1.46MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [02:09<05:39, 1.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:09<04:42, 2.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:09<04:21, 2.21MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:10<03:55, 2.46MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:10<03:30, 2.75MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:10<02:51, 3.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:12<12:03:49, 13.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:12<8:36:26, 18.6kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:12<6:05:05, 26.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [02:12<4:17:59, 37.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [02:12<3:01:22, 53.0kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [02:12<2:07:07, 75.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:12<1:29:03, 107kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:13<1:02:25, 153kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:14<56:21, 169kB/s]  .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:14<41:55, 228kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:14<30:20, 314kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:14<23:08, 412kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:14<17:44, 537kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:14<12:45, 746kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:14<09:11, 1.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:15<06:47, 1.40MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:15<05:24, 1.75MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:17<31:11, 304kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:17<31:12, 304kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:17<25:08, 377kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:17<18:48, 503kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:17<13:35, 696kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:18<09:46, 966kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:18<07:02, 1.34MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:20<13:23, 703kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:20<18:52, 498kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:20<16:03, 586kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:20<12:09, 773kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:20<08:49, 1.06MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:20<06:25, 1.46MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:21<04:41, 1.99MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:23<22:45, 410kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:23<25:09, 371kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:23<20:44, 450kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:23<15:28, 603kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:24<11:11, 833kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:24<08:02, 1.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:24<05:50, 1.59MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:24<04:16, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:24<03:12, 2.87MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:26<11:54, 775kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:26<17:55, 514kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:26<15:56, 579kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:26<12:29, 738kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:26<09:07, 1.01MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:26<06:40, 1.38MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:26<04:57, 1.85MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:27<03:41, 2.48MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:28<6:48:57, 22.4kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:28<4:48:26, 31.7kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:28<3:22:13, 45.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:28<2:21:23, 64.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:28<1:38:59, 91.9kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:30<1:17:20, 117kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:31<1:03:36, 143kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:31<47:59, 189kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:31<36:29, 249kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:31<28:43, 316kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:31<21:15, 427kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:31<15:25, 588kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:31<11:06, 815kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:31<08:04, 1.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:31<06:02, 1.49MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:34<14:18, 630kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:34<18:52, 478kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:34<16:15, 555kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:34<12:32, 719kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:34<09:07, 986kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:34<07:03, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:35<05:59, 1.50MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:35<04:51, 1.85MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:35<04:11, 2.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:35<03:46, 2.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:35<03:30, 2.55MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:35<03:15, 2.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:35<02:47, 3.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:35<02:11, 4.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:36<04:59, 1.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:36<03:46, 2.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:36<02:47, 3.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:38<08:15, 1.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:38<12:03, 735kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:38<10:15, 863kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:38<07:31, 1.17MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:38<05:31, 1.60MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:41<06:52, 1.28MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:41<12:13, 719kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:41<11:31, 763kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:41<09:35, 915kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:41<07:06, 1.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:41<05:09, 1.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:41<13:42, 638kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:42<09:44, 896kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:42<06:55, 1.25MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:44<14:03, 617kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:44<16:57, 511kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:44<14:02, 618kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:44<10:14, 846kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:44<07:21, 1.17MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:46<08:07, 1.06MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:47<14:10, 607kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:47<12:33, 685kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:47<09:46, 879kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:47<07:35, 1.13MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:47<05:54, 1.45MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:47<05:07, 1.67MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:47<04:23, 1.96MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:47<03:48, 2.25MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:47<03:09, 2.72MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:48<04:07, 2.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:49<03:40, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:49<02:59, 2.85MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:49<02:50, 3.00MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:49<02:34, 3.30MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:49<01:59, 4.26MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:50<04:46, 1.78MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:51<06:02, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:51<05:53, 1.44MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:51<05:18, 1.59MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:51<04:01, 2.10MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:51<02:56, 2.86MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:52<05:52, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:53<06:02, 1.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:53<06:32, 1.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:53<06:15, 1.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:53<06:11, 1.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:53<05:45, 1.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:53<04:46, 1.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:53<03:49, 2.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:53<02:54, 2.87MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:55<05:10, 1.61MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:55<07:19, 1.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:55<06:03, 1.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:56<04:25, 1.87MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:56<03:13, 2.56MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:57<08:32, 966kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:57<08:05, 1.02MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:57<06:17, 1.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:57<04:37, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:58<03:31, 2.33MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:58<02:40, 3.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:59<10:38, 770kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:59<09:36, 853kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:59<07:34, 1.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:59<05:54, 1.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [03:00<04:38, 1.76MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [03:00<03:39, 2.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [03:00<03:04, 2.65MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [03:00<02:37, 3.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [03:00<02:12, 3.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [03:01<07:11, 1.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [03:01<06:21, 1.28MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [03:01<04:59, 1.62MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [03:02<04:03, 2.00MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [03:02<03:39, 2.22MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [03:02<02:54, 2.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [03:03<03:46, 2.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [03:03<03:43, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [03:03<03:02, 2.64MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [03:03<02:45, 2.91MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [03:04<02:47, 2.88MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [03:04<02:40, 3.01MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [03:04<02:33, 3.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [03:04<02:26, 3.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [03:04<02:16, 3.53MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [03:04<02:06, 3.78MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [03:05<05:52, 1.36MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [03:05<04:55, 1.62MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [03:05<03:35, 2.21MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [03:05<02:43, 2.90MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [03:06<02:08, 3.69MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [03:08<46:42, 169kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [03:08<39:27, 201kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [03:08<30:44, 257kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [03:08<22:44, 348kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [03:08<16:18, 484kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [03:09<11:30, 684kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [03:10<10:15, 765kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [03:10<08:20, 941kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [03:10<06:20, 1.24MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [03:10<04:52, 1.61MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [03:10<03:41, 2.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [03:11<02:43, 2.85MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [03:13<11:34, 672kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [03:13<15:47, 493kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [03:13<13:25, 579kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [03:13<10:22, 749kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [03:13<07:31, 1.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [03:13<05:23, 1.44MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [03:15<06:46, 1.14MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [03:15<06:13, 1.24MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [03:15<04:56, 1.56MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [03:15<03:59, 1.92MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [03:15<03:19, 2.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [03:15<02:41, 2.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [03:17<03:32, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [03:17<03:01, 2.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [03:17<02:29, 3.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [03:17<02:09, 3.54MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [03:17<02:01, 3.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [03:17<01:53, 4.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [03:17<01:44, 4.38MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [03:17<01:44, 4.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [03:17<01:47, 4.22MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [03:18<35:53, 211kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [03:19<25:53, 292kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:19<18:22, 411kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:19<13:09, 573kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:19<09:27, 796kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:19<06:49, 1.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:21<16:22, 458kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:21<15:11, 494kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:21<11:54, 630kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:21<08:51, 845kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:21<06:19, 1.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:23<06:11, 1.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:23<07:21, 1.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:23<07:37, 974kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:23<07:03, 1.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:23<07:16, 1.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:23<07:17, 1.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:23<07:08, 1.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:23<06:09, 1.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:24<04:37, 1.60MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:24<03:27, 2.13MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:24<02:39, 2.77MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:25<04:50, 1.52MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:25<03:48, 1.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:25<02:49, 2.59MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:25<02:06, 3.46MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:27<09:12, 792kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:27<08:10, 892kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:27<06:22, 1.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:27<04:51, 1.50MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:27<03:46, 1.92MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:27<02:47, 2.60MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:29<05:36, 1.29MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:29<06:12, 1.16MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:29<05:02, 1.43MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:29<03:46, 1.91MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:29<02:42, 2.64MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:31<14:49, 483kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:31<11:53, 602kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:31<08:51, 807kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:31<06:29, 1.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:31<04:35, 1.55MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:33<2:14:27, 52.7kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:33<1:35:35, 74.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:33<1:07:29, 105kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:33<47:37, 149kB/s]  .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:33<33:26, 211kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:33<23:31, 299kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:34<16:45, 419kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:35<39:57, 176kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:35<28:43, 244kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:35<20:23, 344kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:35<14:27, 484kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:35<10:23, 670kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:40<09:23, 733kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:40<12:13, 564kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:40<10:56, 629kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:40<08:30, 808kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:40<06:11, 1.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:40<04:25, 1.54MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:43<08:31, 800kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:43<12:50, 531kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:43<10:34, 644kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:43<08:10, 833kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:43<05:53, 1.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:43<04:12, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:44<04:18, 1.57MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:44<03:07, 2.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:44<02:25, 2.76MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:44<01:59, 3.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:44<01:40, 4.00MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:46<22:34, 296kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:46<16:56, 394kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:46<12:18, 542kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:46<08:44, 760kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:48<07:33, 874kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:48<07:03, 937kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:48<06:02, 1.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:48<04:55, 1.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:48<03:48, 1.73MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:48<02:51, 2.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:48<02:06, 3.11MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:50<19:27, 336kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:50<14:04, 464kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:50<10:15, 636kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:50<07:16, 894kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:52<07:52, 822kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:52<10:09, 637kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:52<09:14, 700kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:53<09:13, 702kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:53<08:17, 779kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:53<06:14, 1.03MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:53<04:38, 1.39MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:53<03:22, 1.90MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:54<04:11, 1.53MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:54<03:35, 1.78MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:54<02:58, 2.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:54<02:24, 2.66MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:55<01:57, 3.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:55<01:47, 3.56MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:55<01:26, 4.39MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:56<06:24, 989kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:56<05:16, 1.20MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:56<03:58, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:56<02:51, 2.20MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:58<04:22, 1.43MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:59<05:28, 1.14MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:59<07:08, 877kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:59<06:59, 895kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:59<06:22, 980kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:59<05:11, 1.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:59<03:55, 1.59MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:59<02:58, 2.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:59<02:13, 2.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [04:02<07:33, 819kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [04:02<11:24, 542kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [04:02<10:10, 608kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [04:02<08:36, 719kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [04:02<06:29, 951kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [04:03<04:42, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [04:07<06:41, 914kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [04:07<10:41, 572kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [04:07<09:28, 646kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [04:07<07:51, 778kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [04:07<06:51, 891kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [04:08<05:19, 1.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [04:08<04:13, 1.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [04:08<03:37, 1.68MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [04:08<03:13, 1.88MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [04:08<02:58, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [04:08<02:50, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [04:08<02:15, 2.69MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [04:08<01:39, 3.63MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [04:08<01:40, 3.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [04:09<01:22, 4.38MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [04:09<01:00, 5.93MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [04:12<07:23, 801kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [04:12<10:59, 539kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [04:12<12:17, 481kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [04:12<11:29, 515kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [04:12<10:53, 543kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [04:12<08:44, 677kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [04:12<06:36, 894kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [04:12<04:54, 1.20MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [04:12<03:33, 1.65MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [04:12<02:43, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [04:15<08:22, 698kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [04:15<11:45, 498kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [04:15<10:13, 572kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [04:15<09:00, 650kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [04:16<06:56, 843kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [04:16<05:12, 1.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [04:16<03:50, 1.52MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [04:16<02:47, 2.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [04:17<04:48, 1.20MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [04:17<04:03, 1.43MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [04:17<03:15, 1.77MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [04:17<02:26, 2.35MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [04:17<01:47, 3.19MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [04:21<13:03, 437kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [04:21<13:50, 413kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [04:21<11:05, 515kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [04:21<08:12, 695kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [04:21<05:56, 958kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [04:22<04:20, 1.31MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [04:22<03:10, 1.78MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [04:24<09:21, 604kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [04:24<11:13, 503kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [04:24<09:28, 596kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [04:24<07:19, 769kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [04:24<05:17, 1.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [04:24<03:46, 1.48MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [04:26<05:39, 985kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [04:26<05:27, 1.02MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [04:26<05:58, 932kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [04:26<06:16, 888kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [04:26<06:13, 896kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [04:26<05:49, 955kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [04:26<05:30, 1.01MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [04:27<05:15, 1.06MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [04:27<04:16, 1.30MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [04:27<03:21, 1.65MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [04:27<02:28, 2.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:28<02:50, 1.93MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:28<02:18, 2.39MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:28<01:44, 3.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:29<01:54, 2.87MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [04:31<02:25, 2.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [04:32<07:00, 769kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [04:32<06:36, 816kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [04:32<05:10, 1.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:32<03:49, 1.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:32<02:47, 1.91MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:32<02:04, 2.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:33<06:52, 774kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:33<05:44, 927kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:34<05:01, 1.06MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:34<04:24, 1.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:34<04:05, 1.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:34<03:54, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:34<03:08, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:34<02:35, 2.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:34<02:17, 2.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:34<01:59, 2.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:34<01:44, 3.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:34<01:23, 3.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:35<04:44, 1.11MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:35<03:25, 1.52MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:36<02:32, 2.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:36<01:56, 2.68MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:37<06:04, 853kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:37<05:20, 969kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:37<04:10, 1.24MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:38<03:14, 1.59MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:38<02:39, 1.94MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:38<02:05, 2.47MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:38<01:39, 3.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:38<01:21, 3.77MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:39<06:17, 814kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:39<05:14, 976kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:39<03:58, 1.29MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:40<02:54, 1.75MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:40<02:05, 2.41MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:43<23:35, 214kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:43<21:43, 232kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:43<16:39, 303kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:43<12:11, 413kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:43<08:45, 574kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:43<06:11, 808kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:48<09:32, 521kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:48<11:39, 427kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:49<09:28, 525kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:49<07:11, 690kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:49<05:11, 954kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:49<03:40, 1.34MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:49<03:07, 1.57MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:49<02:12, 2.20MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:51<05:54, 819kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:51<06:38, 729kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:52<05:48, 833kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:52<04:44, 1.02MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:52<03:32, 1.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:52<02:32, 1.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:53<03:45, 1.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:53<03:05, 1.54MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:53<02:31, 1.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:54<02:03, 2.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:54<01:37, 2.91MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:54<01:15, 3.74MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:55<03:09, 1.49MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:55<03:13, 1.46MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:56<02:38, 1.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:56<02:02, 2.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:56<01:31, 3.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:58<03:24, 1.36MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:58<07:05, 654kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:58<06:19, 732kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:58<04:58, 930kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:59<03:37, 1.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:59<02:35, 1.77MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [05:00<04:08, 1.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [05:00<03:29, 1.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [05:00<02:37, 1.73MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [05:00<01:53, 2.39MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [05:07<09:57, 452kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [05:07<11:25, 394kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [05:07<09:25, 476kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [05:07<07:10, 626kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [05:07<05:13, 858kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [05:07<03:46, 1.18MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [05:08<02:46, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [05:08<02:03, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [05:12<53:56, 82.1kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [05:12<41:44, 106kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [05:12<30:47, 144kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [05:12<22:26, 197kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [05:12<16:10, 273kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [05:12<11:22, 386kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [05:12<07:55, 549kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [05:13<06:22, 680kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [05:13<04:30, 956kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [05:16<04:40, 913kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [05:16<07:26, 573kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [05:16<06:31, 654kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [05:16<05:04, 840kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [05:16<03:41, 1.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [05:16<02:41, 1.57MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [05:16<01:59, 2.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [05:18<04:54, 854kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [05:19<06:53, 609kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [05:19<06:27, 650kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [05:19<05:08, 816kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [05:19<03:47, 1.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [05:19<02:45, 1.51MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [05:19<01:59, 2.08MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [05:20<04:03, 1.02MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [05:20<02:57, 1.39MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [05:20<02:14, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [05:20<01:39, 2.47MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [05:24<06:17, 645kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [05:25<08:24, 483kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [05:25<07:22, 551kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [05:25<06:18, 644kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [05:25<04:45, 850kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [05:25<03:26, 1.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [05:27<03:15, 1.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [05:27<03:14, 1.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [05:27<02:39, 1.50MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [05:27<02:09, 1.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [05:27<01:43, 2.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [05:27<01:20, 2.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [05:32<03:42, 1.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [05:32<06:28, 606kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [05:32<05:40, 691kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [05:32<04:25, 886kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [05:32<03:13, 1.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [05:32<02:17, 1.69MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [05:33<03:26, 1.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [05:34<02:56, 1.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [05:34<02:37, 1.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [05:34<02:17, 1.68MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [05:34<01:58, 1.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [05:34<01:42, 2.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [05:34<01:18, 2.92MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [05:40<04:36, 820kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [05:40<07:00, 540kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [05:40<06:08, 615kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [05:40<04:40, 809kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [05:40<03:23, 1.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [05:40<02:24, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [05:47<15:15, 244kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [05:47<13:47, 269kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [05:48<11:05, 335kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [05:48<08:58, 414kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [05:48<07:34, 490kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [05:48<05:34, 664kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [05:48<03:59, 922kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [05:50<03:54, 934kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [05:51<06:36, 552kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [05:51<05:34, 653kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [05:51<04:19, 843kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [05:51<03:07, 1.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [05:51<02:13, 1.62MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [05:53<04:45, 753kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [05:53<04:20, 824kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [05:53<04:18, 831kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [05:53<04:06, 871kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [05:53<04:04, 876kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [05:53<03:51, 925kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [05:54<03:34, 998kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [05:54<03:04, 1.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [05:54<02:28, 1.44MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [05:54<01:58, 1.80MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [05:54<01:29, 2.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [05:54<01:08, 3.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [05:56<05:04, 692kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [05:56<06:25, 546kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [05:56<05:20, 655kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [05:57<03:54, 894kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [05:57<02:47, 1.24MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [05:57<02:00, 1.72MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [06:00<24:43, 139kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [06:00<20:35, 167kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [06:00<16:41, 206kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [06:00<13:38, 252kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [06:00<10:28, 328kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [06:00<07:33, 454kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [06:01<05:21, 637kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [06:02<04:16, 787kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [06:02<03:11, 1.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [06:02<02:24, 1.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [06:02<01:48, 1.84MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [06:02<01:23, 2.38MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [06:02<01:04, 3.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [06:04<04:46, 690kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [06:04<05:09, 640kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [06:05<04:17, 768kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [06:05<03:16, 1.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [06:05<02:21, 1.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [06:08<03:07, 1.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [06:09<05:25, 596kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [06:09<04:50, 667kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [06:09<03:48, 846kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [06:09<02:46, 1.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [06:09<01:58, 1.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [06:13<04:42, 673kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [06:13<06:37, 478kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [06:13<05:49, 543kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [06:13<05:14, 604kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [06:13<04:23, 719kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [06:13<03:13, 974kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [06:13<02:19, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [06:14<02:15, 1.38MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [06:15<01:44, 1.76MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [06:15<01:17, 2.37MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [06:15<00:59, 3.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [06:17<02:29, 1.21MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [06:17<04:53, 619kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [06:18<04:33, 664kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [06:18<03:49, 791kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [06:18<02:51, 1.05MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [06:18<02:02, 1.46MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [06:20<02:13, 1.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [06:20<01:56, 1.52MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [06:20<01:33, 1.89MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [06:20<01:18, 2.24MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [06:20<01:07, 2.62MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [06:20<00:56, 3.11MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [06:20<00:42, 4.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [06:21<02:50, 1.02MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [06:22<02:14, 1.29MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [06:22<01:39, 1.73MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [06:22<01:13, 2.32MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [06:22<00:54, 3.11MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [06:25<31:14, 90.2kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [06:25<24:37, 114kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [06:25<18:02, 156kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [06:26<12:54, 218kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [06:26<09:04, 308kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [06:26<06:18, 437kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [06:27<05:47, 475kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [06:27<04:08, 661kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [06:27<02:58, 916kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [06:27<02:10, 1.25MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [06:27<01:38, 1.65MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [06:27<01:14, 2.15MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [06:29<07:34, 353kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [06:29<06:13, 431kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [06:29<05:23, 497kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [06:29<04:45, 562kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [06:29<04:28, 598kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [06:29<03:58, 673kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [06:29<02:59, 893kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [06:29<02:08, 1.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [06:33<02:42, 964kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [06:33<04:32, 576kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [06:33<04:16, 610kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [06:33<03:59, 652kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [06:33<03:46, 689kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [06:33<03:39, 712kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [06:33<03:00, 867kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [06:33<02:16, 1.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [06:34<01:38, 1.57MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [06:34<01:12, 2.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [06:35<02:07, 1.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [06:35<01:42, 1.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [06:35<01:22, 1.85MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [06:35<01:11, 2.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [06:35<01:01, 2.46MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [06:35<00:53, 2.82MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [06:35<00:47, 3.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [06:35<00:45, 3.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [06:35<00:42, 3.47MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [06:36<00:40, 3.69MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [06:37<02:16, 1.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [06:37<02:15, 1.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [06:37<02:28, 997kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [06:37<02:43, 908kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [06:37<02:41, 917kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [06:37<02:46, 887kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [06:37<02:34, 959kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [06:37<02:20, 1.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [06:37<02:10, 1.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [06:38<02:07, 1.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [06:38<02:03, 1.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [06:38<01:50, 1.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [06:38<01:21, 1.79MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [06:39<01:20, 1.79MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [06:39<01:00, 2.35MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [06:39<00:45, 3.13MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [06:39<00:35, 4.00MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [06:41<03:16, 713kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [06:41<02:44, 849kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [06:41<02:05, 1.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [06:41<01:35, 1.45MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [06:41<01:11, 1.94MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [06:41<00:53, 2.55MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [06:42<00:41, 3.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [06:43<36:46, 61.6kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [06:43<26:07, 86.6kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [06:43<18:24, 123kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [06:43<12:58, 173kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [06:43<09:04, 246kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [06:43<06:22, 348kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [06:43<04:30, 488kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [06:45<08:56, 246kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [06:45<06:32, 335kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [06:45<04:39, 467kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [06:45<03:19, 652kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [06:45<02:23, 900kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [06:45<01:43, 1.23MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [06:47<02:53, 737kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [06:47<02:18, 918kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [06:47<01:43, 1.23MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [06:47<01:13, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [06:50<01:56, 1.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [06:50<03:15, 632kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [06:51<03:07, 659kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [06:51<02:52, 715kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [06:51<02:33, 800kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [06:51<02:05, 980kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [06:51<01:33, 1.31MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [06:51<01:06, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [06:53<01:51, 1.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [06:53<02:31, 786kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [06:53<02:12, 899kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [06:54<01:42, 1.16MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [06:54<01:14, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [06:54<00:53, 2.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [06:55<02:08, 899kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [06:55<01:45, 1.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [06:55<01:16, 1.49MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [06:56<00:53, 2.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [06:57<02:26, 758kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [06:57<01:56, 952kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [06:57<01:23, 1.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [06:58<00:58, 1.82MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [06:59<02:59, 596kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [07:00<02:30, 710kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [07:00<02:12, 807kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [07:00<01:56, 918kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [07:00<01:41, 1.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [07:00<01:32, 1.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [07:00<01:09, 1.52MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [07:00<00:50, 2.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [07:00<00:38, 2.68MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [07:01<01:26, 1.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [07:02<01:10, 1.46MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [07:02<00:58, 1.74MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [07:02<00:52, 1.93MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [07:02<00:47, 2.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [07:02<00:45, 2.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [07:02<00:43, 2.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [07:02<00:41, 2.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [07:02<00:37, 2.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [07:02<00:31, 3.15MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [07:02<00:25, 3.90MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [07:03<01:38, 999kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [07:04<01:23, 1.18MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [07:04<01:15, 1.30MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [07:04<01:08, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [07:04<01:05, 1.50MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [07:04<00:58, 1.66MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [07:04<00:48, 2.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [07:04<00:37, 2.55MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [07:04<00:29, 3.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [07:04<00:23, 3.99MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [07:08<08:25, 187kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [07:08<07:28, 211kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [07:08<05:36, 280kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [07:09<04:10, 376kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [07:09<02:58, 523kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [07:09<02:04, 738kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [07:11<02:09, 697kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [07:11<02:20, 645kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [07:11<01:49, 822kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [07:11<01:20, 1.11MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [07:11<00:59, 1.49MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [07:11<00:45, 1.95MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [07:12<00:36, 2.42MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [07:12<00:29, 2.92MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [07:13<01:33, 927kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [07:13<01:10, 1.22MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [07:13<00:51, 1.64MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [07:13<00:36, 2.28MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [07:15<02:47, 492kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [07:15<02:16, 599kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [07:15<01:39, 819kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [07:15<01:10, 1.14MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [07:17<01:05, 1.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [07:17<01:03, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [07:17<01:01, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [07:17<00:56, 1.37MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [07:17<00:56, 1.38MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [07:17<00:46, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [07:17<00:34, 2.21MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [07:17<00:25, 2.93MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [07:19<00:44, 1.64MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [07:19<00:46, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [07:19<00:50, 1.45MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [07:19<00:51, 1.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [07:19<00:52, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [07:19<00:52, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [07:19<00:52, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [07:19<00:43, 1.66MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [07:20<00:34, 2.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [07:20<00:24, 2.84MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [07:21<00:54, 1.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [07:21<00:55, 1.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [07:21<00:52, 1.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [07:21<00:42, 1.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [07:22<00:32, 2.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [07:22<00:23, 2.82MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [07:22<00:17, 3.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [07:25<04:02, 270kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [07:25<03:54, 280kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [07:25<03:00, 362kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [07:25<02:13, 488kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [07:25<01:34, 682kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [07:25<01:04, 961kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [07:27<01:31, 675kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [07:27<01:18, 781kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [07:27<01:12, 842kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [07:27<01:07, 906kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [07:27<01:02, 978kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [07:27<00:54, 1.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [07:27<00:41, 1.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [07:28<00:29, 1.97MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [07:29<00:32, 1.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [07:29<00:27, 2.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [07:29<00:21, 2.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [07:29<00:14, 3.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [07:31<03:34, 247kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [07:31<02:46, 318kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [07:31<02:01, 435kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [07:31<01:25, 607kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [07:31<00:57, 858kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [07:33<01:25, 571kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [07:33<01:12, 669kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [07:33<00:57, 842kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [07:33<00:44, 1.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [07:33<00:32, 1.45MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [07:33<00:23, 1.99MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [07:35<00:33, 1.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [07:35<00:35, 1.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [07:35<00:35, 1.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [07:35<00:30, 1.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [07:35<00:27, 1.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [07:35<00:24, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [07:35<00:22, 1.97MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [07:36<00:19, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [07:36<00:14, 2.83MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [07:37<00:18, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [07:37<00:13, 2.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [07:37<00:10, 3.59MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [07:37<00:15, 2.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [07:38<03:14, 195kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [07:40<02:07, 266kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [07:40<02:02, 276kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [07:40<01:33, 361kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [07:40<01:07, 493kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [07:40<00:47, 688kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [07:41<00:31, 970kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [07:43<01:23, 357kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [07:43<01:28, 338kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [07:43<01:10, 419kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [07:44<00:52, 557kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [07:44<00:37, 771kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [07:44<00:24, 1.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [07:45<00:27, 947kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [07:45<00:24, 1.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [07:45<00:23, 1.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [07:46<00:23, 1.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [07:46<00:22, 1.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [07:46<00:22, 1.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [07:46<00:21, 1.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [07:46<00:20, 1.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [07:46<00:15, 1.59MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [07:46<00:10, 2.13MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [07:49<00:17, 1.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [07:49<00:35, 612kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [07:49<00:29, 733kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [07:49<00:22, 922kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [07:49<00:16, 1.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [07:49<00:10, 1.72MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [07:49<00:07, 2.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [07:52<07:54, 36.6kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [07:52<05:45, 50.2kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [07:52<04:03, 70.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [07:52<02:44, 100kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [07:52<01:45, 143kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [07:52<01:05, 203kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [07:54<05:13, 42.4kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [07:54<03:41, 59.3kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [07:54<02:32, 84.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [07:54<01:44, 119kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [07:54<01:04, 169kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [07:56<00:40, 226kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [07:56<00:31, 291kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [07:56<00:23, 383kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [07:56<00:18, 477kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [07:56<00:14, 577kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [07:57<00:12, 692kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [07:57<00:09, 869kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [07:57<00:06, 1.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [07:57<00:04, 1.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [07:57<00:03, 1.95MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [07:57<00:02, 2.60MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [07:58<00:03, 1.26MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [07:58<00:02, 1.68MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [07:58<00:01, 2.24MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [07:58<00:00, 2.86MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [07:58<00:00, 3.55MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [08:00<00:00, 1.11MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [08:00<00:00, 1.09MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [08:00<00:00, 1.11MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [08:00<00:00, 1.12MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [08:00<00:00, 1.11MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [08:00<00:00, 1.07MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [08:00<00:00, 1.08MB/s].vector_cache/glove.6B.zip: 862MB [08:00, 1.14MB/s]                           .vector_cache/glove.6B.zip: 862MB [08:00, 1.79MB/s]
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 865/400000 [00:00<00:46, 8646.90it/s]  0%|          | 1710/400000 [00:00<00:46, 8583.48it/s]  1%|          | 2572/400000 [00:00<00:46, 8592.51it/s]  1%|          | 3435/400000 [00:00<00:46, 8602.71it/s]  1%|          | 4214/400000 [00:00<00:47, 8340.45it/s]  1%|▏         | 5064/400000 [00:00<00:47, 8387.06it/s]  1%|▏         | 5870/400000 [00:00<00:47, 8284.52it/s]  2%|▏         | 6725/400000 [00:00<00:47, 8361.17it/s]  2%|▏         | 7526/400000 [00:00<00:47, 8252.34it/s]  2%|▏         | 8406/400000 [00:01<00:46, 8409.11it/s]  2%|▏         | 9224/400000 [00:01<00:47, 8194.16it/s]  3%|▎         | 10036/400000 [00:01<00:47, 8171.07it/s]  3%|▎         | 10843/400000 [00:01<00:48, 8077.97it/s]  3%|▎         | 11720/400000 [00:01<00:46, 8273.10it/s]  3%|▎         | 12544/400000 [00:01<00:46, 8247.74it/s]  3%|▎         | 13367/400000 [00:01<00:46, 8239.39it/s]  4%|▎         | 14190/400000 [00:01<00:48, 8015.87it/s]  4%|▍         | 15051/400000 [00:01<00:47, 8183.45it/s]  4%|▍         | 15871/400000 [00:01<00:48, 7940.73it/s]  4%|▍         | 16712/400000 [00:02<00:47, 8074.05it/s]  4%|▍         | 17522/400000 [00:02<00:47, 7995.28it/s]  5%|▍         | 18324/400000 [00:02<00:48, 7812.64it/s]  5%|▍         | 19192/400000 [00:02<00:47, 8053.36it/s]  5%|▌         | 20056/400000 [00:02<00:46, 8219.43it/s]  5%|▌         | 20936/400000 [00:02<00:45, 8384.34it/s]  5%|▌         | 21818/400000 [00:02<00:44, 8508.52it/s]  6%|▌         | 22678/400000 [00:02<00:44, 8533.68it/s]  6%|▌         | 23534/400000 [00:02<00:44, 8526.18it/s]  6%|▌         | 24414/400000 [00:02<00:43, 8604.63it/s]  6%|▋         | 25276/400000 [00:03<00:43, 8536.26it/s]  7%|▋         | 26154/400000 [00:03<00:43, 8606.93it/s]  7%|▋         | 27032/400000 [00:03<00:43, 8655.48it/s]  7%|▋         | 27907/400000 [00:03<00:42, 8681.89it/s]  7%|▋         | 28785/400000 [00:03<00:42, 8708.27it/s]  7%|▋         | 29657/400000 [00:03<00:43, 8524.99it/s]  8%|▊         | 30511/400000 [00:03<00:44, 8299.73it/s]  8%|▊         | 31347/400000 [00:03<00:44, 8317.59it/s]  8%|▊         | 32219/400000 [00:03<00:43, 8432.43it/s]  8%|▊         | 33088/400000 [00:03<00:43, 8507.47it/s]  8%|▊         | 33966/400000 [00:04<00:42, 8585.67it/s]  9%|▊         | 34846/400000 [00:04<00:42, 8646.74it/s]  9%|▉         | 35712/400000 [00:04<00:42, 8644.17it/s]  9%|▉         | 36585/400000 [00:04<00:41, 8667.14it/s]  9%|▉         | 37456/400000 [00:04<00:41, 8677.15it/s] 10%|▉         | 38330/400000 [00:04<00:41, 8693.03it/s] 10%|▉         | 39213/400000 [00:04<00:41, 8732.28it/s] 10%|█         | 40087/400000 [00:04<00:41, 8733.35it/s] 10%|█         | 40961/400000 [00:04<00:41, 8707.19it/s] 10%|█         | 41832/400000 [00:04<00:41, 8631.36it/s] 11%|█         | 42708/400000 [00:05<00:41, 8668.19it/s] 11%|█         | 43607/400000 [00:05<00:40, 8760.80it/s] 11%|█         | 44484/400000 [00:05<00:41, 8662.67it/s] 11%|█▏        | 45351/400000 [00:05<00:41, 8453.41it/s] 12%|█▏        | 46258/400000 [00:05<00:41, 8627.37it/s] 12%|█▏        | 47123/400000 [00:05<00:41, 8572.35it/s] 12%|█▏        | 48040/400000 [00:05<00:40, 8741.30it/s] 12%|█▏        | 48943/400000 [00:05<00:39, 8825.29it/s] 12%|█▏        | 49840/400000 [00:05<00:39, 8866.84it/s] 13%|█▎        | 50745/400000 [00:05<00:39, 8918.36it/s] 13%|█▎        | 51664/400000 [00:06<00:38, 8998.03it/s] 13%|█▎        | 52575/400000 [00:06<00:38, 9027.28it/s] 13%|█▎        | 53479/400000 [00:06<00:38, 9008.97it/s] 14%|█▎        | 54382/400000 [00:06<00:38, 9013.15it/s] 14%|█▍        | 55296/400000 [00:06<00:38, 9049.01it/s] 14%|█▍        | 56202/400000 [00:06<00:38, 9026.60it/s] 14%|█▍        | 57105/400000 [00:06<00:38, 8968.01it/s] 15%|█▍        | 58003/400000 [00:06<00:38, 8919.38it/s] 15%|█▍        | 58896/400000 [00:06<00:39, 8659.84it/s] 15%|█▍        | 59764/400000 [00:06<00:39, 8652.22it/s] 15%|█▌        | 60646/400000 [00:07<00:39, 8700.70it/s] 15%|█▌        | 61546/400000 [00:07<00:38, 8786.21it/s] 16%|█▌        | 62426/400000 [00:07<00:39, 8622.66it/s] 16%|█▌        | 63291/400000 [00:07<00:39, 8629.25it/s] 16%|█▌        | 64205/400000 [00:07<00:38, 8773.76it/s] 16%|█▋        | 65104/400000 [00:07<00:37, 8835.24it/s] 16%|█▋        | 65989/400000 [00:07<00:38, 8732.65it/s] 17%|█▋        | 66890/400000 [00:07<00:37, 8811.11it/s] 17%|█▋        | 67772/400000 [00:07<00:38, 8691.51it/s] 17%|█▋        | 68678/400000 [00:08<00:37, 8796.94it/s] 17%|█▋        | 69616/400000 [00:08<00:36, 8962.95it/s] 18%|█▊        | 70514/400000 [00:08<00:37, 8841.46it/s] 18%|█▊        | 71408/400000 [00:08<00:37, 8870.56it/s] 18%|█▊        | 72297/400000 [00:08<00:37, 8816.45it/s] 18%|█▊        | 73188/400000 [00:08<00:36, 8843.72it/s] 19%|█▊        | 74073/400000 [00:08<00:37, 8730.50it/s] 19%|█▊        | 74947/400000 [00:08<00:38, 8433.72it/s] 19%|█▉        | 75797/400000 [00:08<00:38, 8450.98it/s] 19%|█▉        | 76665/400000 [00:08<00:37, 8516.20it/s] 19%|█▉        | 77529/400000 [00:09<00:37, 8552.76it/s] 20%|█▉        | 78406/400000 [00:09<00:37, 8614.06it/s] 20%|█▉        | 79269/400000 [00:09<00:38, 8415.70it/s] 20%|██        | 80126/400000 [00:09<00:37, 8460.66it/s] 20%|██        | 80997/400000 [00:09<00:37, 8532.71it/s] 20%|██        | 81885/400000 [00:09<00:36, 8631.20it/s] 21%|██        | 82752/400000 [00:09<00:36, 8641.85it/s] 21%|██        | 83617/400000 [00:09<00:36, 8623.60it/s] 21%|██        | 84483/400000 [00:09<00:36, 8633.43it/s] 21%|██▏       | 85347/400000 [00:09<00:36, 8625.40it/s] 22%|██▏       | 86210/400000 [00:10<00:37, 8471.87it/s] 22%|██▏       | 87058/400000 [00:10<00:37, 8401.63it/s] 22%|██▏       | 87906/400000 [00:10<00:37, 8423.02it/s] 22%|██▏       | 88781/400000 [00:10<00:36, 8515.87it/s] 22%|██▏       | 89634/400000 [00:10<00:37, 8338.86it/s] 23%|██▎       | 90510/400000 [00:10<00:36, 8460.25it/s] 23%|██▎       | 91383/400000 [00:10<00:36, 8539.39it/s] 23%|██▎       | 92255/400000 [00:10<00:35, 8590.20it/s] 23%|██▎       | 93115/400000 [00:10<00:36, 8524.40it/s] 23%|██▎       | 93971/400000 [00:10<00:35, 8532.33it/s] 24%|██▎       | 94825/400000 [00:11<00:36, 8440.82it/s] 24%|██▍       | 95670/400000 [00:11<00:36, 8364.93it/s] 24%|██▍       | 96538/400000 [00:11<00:35, 8456.47it/s] 24%|██▍       | 97412/400000 [00:11<00:35, 8538.65it/s] 25%|██▍       | 98282/400000 [00:11<00:35, 8586.11it/s] 25%|██▍       | 99142/400000 [00:11<00:35, 8586.68it/s] 25%|██▌       | 100002/400000 [00:11<00:36, 8330.42it/s] 25%|██▌       | 100845/400000 [00:11<00:35, 8359.85it/s] 25%|██▌       | 101709/400000 [00:11<00:35, 8440.31it/s] 26%|██▌       | 102557/400000 [00:11<00:35, 8451.61it/s] 26%|██▌       | 103434/400000 [00:12<00:34, 8544.29it/s] 26%|██▌       | 104295/400000 [00:12<00:34, 8562.40it/s] 26%|██▋       | 105160/400000 [00:12<00:34, 8586.16it/s] 27%|██▋       | 106020/400000 [00:12<00:35, 8395.74it/s] 27%|██▋       | 106884/400000 [00:12<00:34, 8467.33it/s] 27%|██▋       | 107753/400000 [00:12<00:34, 8531.29it/s] 27%|██▋       | 108607/400000 [00:12<00:34, 8509.60it/s] 27%|██▋       | 109459/400000 [00:12<00:34, 8414.57it/s] 28%|██▊       | 110327/400000 [00:12<00:34, 8492.41it/s] 28%|██▊       | 111177/400000 [00:12<00:34, 8435.18it/s] 28%|██▊       | 112049/400000 [00:13<00:33, 8517.69it/s] 28%|██▊       | 112902/400000 [00:13<00:34, 8285.65it/s] 28%|██▊       | 113777/400000 [00:13<00:33, 8418.44it/s] 29%|██▊       | 114652/400000 [00:13<00:33, 8512.99it/s] 29%|██▉       | 115532/400000 [00:13<00:33, 8594.84it/s] 29%|██▉       | 116395/400000 [00:13<00:32, 8605.10it/s] 29%|██▉       | 117257/400000 [00:13<00:33, 8521.46it/s] 30%|██▉       | 118110/400000 [00:13<00:33, 8520.15it/s] 30%|██▉       | 118963/400000 [00:13<00:33, 8418.33it/s] 30%|██▉       | 119850/400000 [00:14<00:32, 8546.27it/s] 30%|███       | 120706/400000 [00:14<00:32, 8546.79it/s] 30%|███       | 121583/400000 [00:14<00:32, 8611.30it/s] 31%|███       | 122456/400000 [00:14<00:32, 8645.89it/s] 31%|███       | 123347/400000 [00:14<00:31, 8721.97it/s] 31%|███       | 124220/400000 [00:14<00:31, 8672.59it/s] 31%|███▏      | 125106/400000 [00:14<00:31, 8727.51it/s] 31%|███▏      | 125980/400000 [00:14<00:32, 8548.21it/s] 32%|███▏      | 126836/400000 [00:14<00:32, 8506.70it/s] 32%|███▏      | 127698/400000 [00:14<00:31, 8537.80it/s] 32%|███▏      | 128560/400000 [00:15<00:31, 8561.94it/s] 32%|███▏      | 129417/400000 [00:15<00:31, 8513.46it/s] 33%|███▎      | 130292/400000 [00:15<00:31, 8580.61it/s] 33%|███▎      | 131166/400000 [00:15<00:31, 8627.37it/s] 33%|███▎      | 132038/400000 [00:15<00:30, 8653.13it/s] 33%|███▎      | 132913/400000 [00:15<00:30, 8680.26it/s] 33%|███▎      | 133791/400000 [00:15<00:30, 8709.13it/s] 34%|███▎      | 134671/400000 [00:15<00:30, 8733.37it/s] 34%|███▍      | 135545/400000 [00:15<00:30, 8640.22it/s] 34%|███▍      | 136429/400000 [00:15<00:30, 8695.96it/s] 34%|███▍      | 137308/400000 [00:16<00:30, 8722.87it/s] 35%|███▍      | 138194/400000 [00:16<00:29, 8761.82it/s] 35%|███▍      | 139076/400000 [00:16<00:29, 8776.66it/s] 35%|███▍      | 139971/400000 [00:16<00:29, 8827.68it/s] 35%|███▌      | 140883/400000 [00:16<00:29, 8912.90it/s] 35%|███▌      | 141775/400000 [00:16<00:28, 8905.23it/s] 36%|███▌      | 142666/400000 [00:16<00:29, 8828.84it/s] 36%|███▌      | 143550/400000 [00:16<00:29, 8827.02it/s] 36%|███▌      | 144434/400000 [00:16<00:28, 8829.76it/s] 36%|███▋      | 145318/400000 [00:16<00:28, 8782.80it/s] 37%|███▋      | 146197/400000 [00:17<00:29, 8607.52it/s] 37%|███▋      | 147075/400000 [00:17<00:29, 8657.63it/s] 37%|███▋      | 147947/400000 [00:17<00:29, 8675.47it/s] 37%|███▋      | 148816/400000 [00:17<00:28, 8679.03it/s] 37%|███▋      | 149697/400000 [00:17<00:28, 8716.06it/s] 38%|███▊      | 150569/400000 [00:17<00:28, 8615.60it/s] 38%|███▊      | 151432/400000 [00:17<00:30, 8207.28it/s] 38%|███▊      | 152302/400000 [00:17<00:29, 8346.72it/s] 38%|███▊      | 153184/400000 [00:17<00:29, 8483.12it/s] 39%|███▊      | 154074/400000 [00:17<00:28, 8602.40it/s] 39%|███▊      | 154944/400000 [00:18<00:28, 8629.03it/s] 39%|███▉      | 155809/400000 [00:18<00:28, 8630.92it/s] 39%|███▉      | 156674/400000 [00:18<00:28, 8636.28it/s] 39%|███▉      | 157568/400000 [00:18<00:27, 8722.60it/s] 40%|███▉      | 158459/400000 [00:18<00:27, 8776.20it/s] 40%|███▉      | 159341/400000 [00:18<00:27, 8787.53it/s] 40%|████      | 160221/400000 [00:18<00:27, 8788.66it/s] 40%|████      | 161101/400000 [00:18<00:27, 8769.63it/s] 40%|████      | 162000/400000 [00:18<00:26, 8834.19it/s] 41%|████      | 162884/400000 [00:18<00:26, 8789.84it/s] 41%|████      | 163764/400000 [00:19<00:26, 8778.22it/s] 41%|████      | 164642/400000 [00:19<00:26, 8738.56it/s] 41%|████▏     | 165517/400000 [00:19<00:26, 8737.03it/s] 42%|████▏     | 166391/400000 [00:19<00:26, 8702.88it/s] 42%|████▏     | 167262/400000 [00:19<00:27, 8483.48it/s] 42%|████▏     | 168135/400000 [00:19<00:27, 8555.56it/s] 42%|████▏     | 168992/400000 [00:19<00:27, 8539.67it/s] 42%|████▏     | 169865/400000 [00:19<00:26, 8595.29it/s] 43%|████▎     | 170738/400000 [00:19<00:26, 8632.90it/s] 43%|████▎     | 171610/400000 [00:19<00:26, 8657.12it/s] 43%|████▎     | 172477/400000 [00:20<00:26, 8628.33it/s] 43%|████▎     | 173347/400000 [00:20<00:26, 8649.39it/s] 44%|████▎     | 174224/400000 [00:20<00:26, 8682.63it/s] 44%|████▍     | 175103/400000 [00:20<00:25, 8714.25it/s] 44%|████▍     | 175985/400000 [00:20<00:25, 8744.60it/s] 44%|████▍     | 176860/400000 [00:20<00:27, 8104.01it/s] 44%|████▍     | 177739/400000 [00:20<00:26, 8297.36it/s] 45%|████▍     | 178593/400000 [00:20<00:26, 8365.95it/s] 45%|████▍     | 179436/400000 [00:20<00:26, 8367.96it/s] 45%|████▌     | 180277/400000 [00:21<00:27, 7987.80it/s] 45%|████▌     | 181083/400000 [00:21<00:27, 7825.30it/s] 45%|████▌     | 181871/400000 [00:21<00:27, 7838.11it/s] 46%|████▌     | 182688/400000 [00:21<00:27, 7933.82it/s] 46%|████▌     | 183528/400000 [00:21<00:26, 8066.37it/s] 46%|████▌     | 184364/400000 [00:21<00:26, 8149.94it/s] 46%|████▋     | 185181/400000 [00:21<00:27, 7951.14it/s] 46%|████▋     | 185979/400000 [00:21<00:28, 7587.45it/s] 47%|████▋     | 186819/400000 [00:21<00:27, 7804.99it/s] 47%|████▋     | 187682/400000 [00:21<00:26, 8033.50it/s] 47%|████▋     | 188560/400000 [00:22<00:25, 8241.88it/s] 47%|████▋     | 189431/400000 [00:22<00:25, 8374.38it/s] 48%|████▊     | 190273/400000 [00:22<00:25, 8385.04it/s] 48%|████▊     | 191128/400000 [00:22<00:24, 8433.46it/s] 48%|████▊     | 191974/400000 [00:22<00:26, 7952.00it/s] 48%|████▊     | 192822/400000 [00:22<00:25, 8100.49it/s] 48%|████▊     | 193673/400000 [00:22<00:25, 8217.22it/s] 49%|████▊     | 194500/400000 [00:22<00:26, 7642.42it/s] 49%|████▉     | 195276/400000 [00:22<00:27, 7507.71it/s] 49%|████▉     | 196144/400000 [00:23<00:26, 7822.67it/s] 49%|████▉     | 196936/400000 [00:23<00:26, 7793.94it/s] 49%|████▉     | 197812/400000 [00:23<00:25, 8060.11it/s] 50%|████▉     | 198693/400000 [00:23<00:24, 8269.03it/s] 50%|████▉     | 199527/400000 [00:23<00:25, 7742.16it/s] 50%|█████     | 200404/400000 [00:23<00:24, 8023.42it/s] 50%|█████     | 201265/400000 [00:23<00:24, 8189.03it/s] 51%|█████     | 202136/400000 [00:23<00:23, 8337.33it/s] 51%|█████     | 203024/400000 [00:23<00:23, 8492.24it/s] 51%|█████     | 203922/400000 [00:23<00:22, 8632.13it/s] 51%|█████     | 204815/400000 [00:24<00:22, 8719.19it/s] 51%|█████▏    | 205719/400000 [00:24<00:22, 8812.68it/s] 52%|█████▏    | 206615/400000 [00:24<00:21, 8853.20it/s] 52%|█████▏    | 207524/400000 [00:24<00:21, 8921.19it/s] 52%|█████▏    | 208418/400000 [00:24<00:21, 8779.72it/s] 52%|█████▏    | 209298/400000 [00:24<00:22, 8565.37it/s] 53%|█████▎    | 210185/400000 [00:24<00:21, 8653.60it/s] 53%|█████▎    | 211097/400000 [00:24<00:21, 8787.65it/s] 53%|█████▎    | 212021/400000 [00:24<00:21, 8915.96it/s] 53%|█████▎    | 212915/400000 [00:24<00:21, 8756.87it/s] 53%|█████▎    | 213798/400000 [00:25<00:21, 8778.44it/s] 54%|█████▎    | 214678/400000 [00:25<00:21, 8767.53it/s] 54%|█████▍    | 215556/400000 [00:25<00:21, 8742.18it/s] 54%|█████▍    | 216436/400000 [00:25<00:20, 8758.43it/s] 54%|█████▍    | 217313/400000 [00:25<00:20, 8711.43it/s] 55%|█████▍    | 218207/400000 [00:25<00:20, 8778.19it/s] 55%|█████▍    | 219086/400000 [00:25<00:20, 8734.27it/s] 55%|█████▍    | 219963/400000 [00:25<00:20, 8743.23it/s] 55%|█████▌    | 220849/400000 [00:25<00:20, 8775.44it/s] 55%|█████▌    | 221727/400000 [00:25<00:20, 8717.37it/s] 56%|█████▌    | 222608/400000 [00:26<00:20, 8744.31it/s] 56%|█████▌    | 223483/400000 [00:26<00:20, 8447.58it/s] 56%|█████▌    | 224361/400000 [00:26<00:20, 8543.16it/s] 56%|█████▋    | 225247/400000 [00:26<00:20, 8633.01it/s] 57%|█████▋    | 226125/400000 [00:26<00:20, 8674.27it/s] 57%|█████▋    | 227015/400000 [00:26<00:19, 8739.41it/s] 57%|█████▋    | 227890/400000 [00:26<00:19, 8713.88it/s] 57%|█████▋    | 228770/400000 [00:26<00:19, 8737.54it/s] 57%|█████▋    | 229645/400000 [00:26<00:19, 8732.78it/s] 58%|█████▊    | 230519/400000 [00:26<00:19, 8711.09it/s] 58%|█████▊    | 231391/400000 [00:27<00:19, 8629.89it/s] 58%|█████▊    | 232264/400000 [00:27<00:19, 8657.38it/s] 58%|█████▊    | 233138/400000 [00:27<00:19, 8681.20it/s] 59%|█████▊    | 234007/400000 [00:27<00:19, 8585.32it/s] 59%|█████▊    | 234891/400000 [00:27<00:19, 8658.23it/s] 59%|█████▉    | 235772/400000 [00:27<00:18, 8701.20it/s] 59%|█████▉    | 236643/400000 [00:27<00:19, 8545.92it/s] 59%|█████▉    | 237499/400000 [00:27<00:19, 8438.27it/s] 60%|█████▉    | 238344/400000 [00:27<00:19, 8249.34it/s] 60%|█████▉    | 239180/400000 [00:28<00:19, 8279.87it/s] 60%|██████    | 240061/400000 [00:28<00:18, 8432.00it/s] 60%|██████    | 240939/400000 [00:28<00:18, 8532.10it/s] 60%|██████    | 241832/400000 [00:28<00:18, 8647.43it/s] 61%|██████    | 242721/400000 [00:28<00:18, 8716.95it/s] 61%|██████    | 243605/400000 [00:28<00:17, 8752.89it/s] 61%|██████    | 244497/400000 [00:28<00:17, 8800.47it/s] 61%|██████▏   | 245378/400000 [00:28<00:17, 8768.22it/s] 62%|██████▏   | 246266/400000 [00:28<00:17, 8801.02it/s] 62%|██████▏   | 247147/400000 [00:28<00:17, 8773.72it/s] 62%|██████▏   | 248031/400000 [00:29<00:17, 8791.27it/s] 62%|██████▏   | 248938/400000 [00:29<00:17, 8870.73it/s] 62%|██████▏   | 249826/400000 [00:29<00:17, 8759.37it/s] 63%|██████▎   | 250705/400000 [00:29<00:17, 8767.25it/s] 63%|██████▎   | 251609/400000 [00:29<00:16, 8844.94it/s] 63%|██████▎   | 252506/400000 [00:29<00:16, 8881.83it/s] 63%|██████▎   | 253397/400000 [00:29<00:16, 8887.51it/s] 64%|██████▎   | 254286/400000 [00:29<00:16, 8727.41it/s] 64%|██████▍   | 255160/400000 [00:29<00:16, 8545.48it/s] 64%|██████▍   | 256033/400000 [00:29<00:16, 8597.09it/s] 64%|██████▍   | 256905/400000 [00:30<00:16, 8630.71it/s] 64%|██████▍   | 257769/400000 [00:30<00:16, 8579.41it/s] 65%|██████▍   | 258634/400000 [00:30<00:16, 8599.32it/s] 65%|██████▍   | 259495/400000 [00:30<00:16, 8533.00it/s] 65%|██████▌   | 260388/400000 [00:30<00:16, 8646.70it/s] 65%|██████▌   | 261304/400000 [00:30<00:15, 8792.02it/s] 66%|██████▌   | 262185/400000 [00:30<00:15, 8659.35it/s] 66%|██████▌   | 263053/400000 [00:30<00:15, 8662.95it/s] 66%|██████▌   | 263938/400000 [00:30<00:15, 8717.89it/s] 66%|██████▌   | 264811/400000 [00:30<00:15, 8681.67it/s] 66%|██████▋   | 265696/400000 [00:31<00:15, 8730.64it/s] 67%|██████▋   | 266570/400000 [00:31<00:15, 8680.95it/s] 67%|██████▋   | 267439/400000 [00:31<00:15, 8431.47it/s] 67%|██████▋   | 268284/400000 [00:31<00:15, 8402.83it/s] 67%|██████▋   | 269165/400000 [00:31<00:15, 8519.40it/s] 68%|██████▊   | 270019/400000 [00:31<00:15, 8444.05it/s] 68%|██████▊   | 270883/400000 [00:31<00:15, 8501.61it/s] 68%|██████▊   | 271734/400000 [00:31<00:15, 8406.19it/s] 68%|██████▊   | 272583/400000 [00:31<00:15, 8430.81it/s] 68%|██████▊   | 273459/400000 [00:31<00:14, 8524.42it/s] 69%|██████▊   | 274328/400000 [00:32<00:14, 8573.23it/s] 69%|██████▉   | 275186/400000 [00:32<00:14, 8560.04it/s] 69%|██████▉   | 276043/400000 [00:32<00:14, 8456.26it/s] 69%|██████▉   | 276924/400000 [00:32<00:14, 8556.96it/s] 69%|██████▉   | 277781/400000 [00:32<00:14, 8506.95it/s] 70%|██████▉   | 278667/400000 [00:32<00:14, 8607.49it/s] 70%|██████▉   | 279529/400000 [00:32<00:14, 8600.64it/s] 70%|███████   | 280422/400000 [00:32<00:13, 8696.85it/s] 70%|███████   | 281293/400000 [00:32<00:13, 8648.18it/s] 71%|███████   | 282159/400000 [00:32<00:13, 8651.31it/s] 71%|███████   | 283035/400000 [00:33<00:13, 8682.66it/s] 71%|███████   | 283912/400000 [00:33<00:13, 8708.58it/s] 71%|███████   | 284786/400000 [00:33<00:13, 8717.08it/s] 71%|███████▏  | 285671/400000 [00:33<00:13, 8755.74it/s] 72%|███████▏  | 286553/400000 [00:33<00:12, 8772.81it/s] 72%|███████▏  | 287436/400000 [00:33<00:12, 8787.06it/s] 72%|███████▏  | 288321/400000 [00:33<00:12, 8804.29it/s] 72%|███████▏  | 289203/400000 [00:33<00:12, 8808.44it/s] 73%|███████▎  | 290084/400000 [00:33<00:12, 8779.20it/s] 73%|███████▎  | 290962/400000 [00:33<00:12, 8746.77it/s] 73%|███████▎  | 291837/400000 [00:34<00:12, 8682.48it/s] 73%|███████▎  | 292706/400000 [00:34<00:12, 8609.66it/s] 73%|███████▎  | 293568/400000 [00:34<00:12, 8436.86it/s] 74%|███████▎  | 294441/400000 [00:34<00:12, 8520.49it/s] 74%|███████▍  | 295304/400000 [00:34<00:12, 8551.63it/s] 74%|███████▍  | 296160/400000 [00:34<00:13, 7784.83it/s] 74%|███████▍  | 296997/400000 [00:34<00:12, 7950.68it/s] 74%|███████▍  | 297879/400000 [00:34<00:12, 8191.95it/s] 75%|███████▍  | 298753/400000 [00:34<00:12, 8348.76it/s] 75%|███████▍  | 299622/400000 [00:35<00:11, 8447.85it/s] 75%|███████▌  | 300473/400000 [00:35<00:11, 8456.61it/s] 75%|███████▌  | 301343/400000 [00:35<00:11, 8526.57it/s] 76%|███████▌  | 302216/400000 [00:35<00:11, 8585.48it/s] 76%|███████▌  | 303077/400000 [00:35<00:11, 8572.77it/s] 76%|███████▌  | 303952/400000 [00:35<00:11, 8623.01it/s] 76%|███████▌  | 304816/400000 [00:35<00:11, 8608.18it/s] 76%|███████▋  | 305682/400000 [00:35<00:10, 8623.45it/s] 77%|███████▋  | 306545/400000 [00:35<00:11, 8450.65it/s] 77%|███████▋  | 307392/400000 [00:35<00:10, 8422.99it/s] 77%|███████▋  | 308264/400000 [00:36<00:10, 8509.83it/s] 77%|███████▋  | 309128/400000 [00:36<00:10, 8546.67it/s] 77%|███████▋  | 309993/400000 [00:36<00:10, 8577.12it/s] 78%|███████▊  | 310852/400000 [00:36<00:10, 8575.81it/s] 78%|███████▊  | 311727/400000 [00:36<00:10, 8625.28it/s] 78%|███████▊  | 312593/400000 [00:36<00:10, 8633.90it/s] 78%|███████▊  | 313471/400000 [00:36<00:09, 8675.48it/s] 79%|███████▊  | 314346/400000 [00:36<00:09, 8695.25it/s] 79%|███████▉  | 315216/400000 [00:36<00:09, 8651.96it/s] 79%|███████▉  | 316082/400000 [00:36<00:09, 8616.94it/s] 79%|███████▉  | 316964/400000 [00:37<00:09, 8676.04it/s] 79%|███████▉  | 317840/400000 [00:37<00:09, 8698.88it/s] 80%|███████▉  | 318728/400000 [00:37<00:09, 8749.94it/s] 80%|███████▉  | 319621/400000 [00:37<00:09, 8802.61it/s] 80%|████████  | 320502/400000 [00:37<00:09, 8753.87it/s] 80%|████████  | 321391/400000 [00:37<00:08, 8792.12it/s] 81%|████████  | 322275/400000 [00:37<00:08, 8804.68it/s] 81%|████████  | 323156/400000 [00:37<00:08, 8784.55it/s] 81%|████████  | 324035/400000 [00:37<00:08, 8755.63it/s] 81%|████████  | 324911/400000 [00:37<00:08, 8615.52it/s] 81%|████████▏ | 325774/400000 [00:38<00:08, 8471.83it/s] 82%|████████▏ | 326623/400000 [00:38<00:08, 8444.13it/s] 82%|████████▏ | 327496/400000 [00:38<00:08, 8525.95it/s] 82%|████████▏ | 328364/400000 [00:38<00:08, 8569.70it/s] 82%|████████▏ | 329222/400000 [00:38<00:08, 8469.73it/s] 83%|████████▎ | 330078/400000 [00:38<00:08, 8495.69it/s] 83%|████████▎ | 330954/400000 [00:38<00:08, 8573.15it/s] 83%|████████▎ | 331828/400000 [00:38<00:07, 8620.33it/s] 83%|████████▎ | 332691/400000 [00:38<00:07, 8606.11it/s] 83%|████████▎ | 333567/400000 [00:38<00:07, 8651.25it/s] 84%|████████▎ | 334437/400000 [00:39<00:07, 8665.13it/s] 84%|████████▍ | 335311/400000 [00:39<00:07, 8686.72it/s] 84%|████████▍ | 336180/400000 [00:39<00:07, 8681.37it/s] 84%|████████▍ | 337056/400000 [00:39<00:07, 8701.98it/s] 84%|████████▍ | 337928/400000 [00:39<00:07, 8707.00it/s] 85%|████████▍ | 338799/400000 [00:39<00:07, 8663.46it/s] 85%|████████▍ | 339666/400000 [00:39<00:07, 8394.69it/s] 85%|████████▌ | 340508/400000 [00:39<00:07, 8209.05it/s] 85%|████████▌ | 341391/400000 [00:39<00:06, 8385.61it/s] 86%|████████▌ | 342236/400000 [00:39<00:06, 8402.85it/s] 86%|████████▌ | 343109/400000 [00:40<00:06, 8495.94it/s] 86%|████████▌ | 343988/400000 [00:40<00:06, 8579.82it/s] 86%|████████▌ | 344853/400000 [00:40<00:06, 8600.51it/s] 86%|████████▋ | 345728/400000 [00:40<00:06, 8644.61it/s] 87%|████████▋ | 346604/400000 [00:40<00:06, 8676.86it/s] 87%|████████▋ | 347484/400000 [00:40<00:06, 8711.96it/s] 87%|████████▋ | 348361/400000 [00:40<00:05, 8727.13it/s] 87%|████████▋ | 349234/400000 [00:40<00:05, 8677.38it/s] 88%|████████▊ | 350121/400000 [00:40<00:05, 8733.90it/s] 88%|████████▊ | 350995/400000 [00:40<00:05, 8589.82it/s] 88%|████████▊ | 351876/400000 [00:41<00:05, 8653.18it/s] 88%|████████▊ | 352744/400000 [00:41<00:05, 8660.36it/s] 88%|████████▊ | 353614/400000 [00:41<00:05, 8671.47it/s] 89%|████████▊ | 354503/400000 [00:41<00:05, 8733.46it/s] 89%|████████▉ | 355377/400000 [00:41<00:05, 8583.50it/s] 89%|████████▉ | 356253/400000 [00:41<00:05, 8633.51it/s] 89%|████████▉ | 357131/400000 [00:41<00:04, 8674.60it/s] 89%|████████▉ | 357999/400000 [00:41<00:04, 8480.13it/s] 90%|████████▉ | 358849/400000 [00:41<00:04, 8463.92it/s] 90%|████████▉ | 359697/400000 [00:42<00:04, 8313.96it/s] 90%|█████████ | 360568/400000 [00:42<00:04, 8426.77it/s] 90%|█████████ | 361436/400000 [00:42<00:04, 8497.70it/s] 91%|█████████ | 362294/400000 [00:42<00:04, 8522.04it/s] 91%|█████████ | 363147/400000 [00:42<00:04, 8518.88it/s] 91%|█████████ | 364015/400000 [00:42<00:04, 8564.81it/s] 91%|█████████ | 364889/400000 [00:42<00:04, 8616.58it/s] 91%|█████████▏| 365758/400000 [00:42<00:03, 8637.89it/s] 92%|█████████▏| 366634/400000 [00:42<00:03, 8672.95it/s] 92%|█████████▏| 367517/400000 [00:42<00:03, 8717.18it/s] 92%|█████████▏| 368389/400000 [00:43<00:03, 8508.91it/s] 92%|█████████▏| 369253/400000 [00:43<00:03, 8544.39it/s] 93%|█████████▎| 370125/400000 [00:43<00:03, 8594.61it/s] 93%|█████████▎| 371002/400000 [00:43<00:03, 8645.98it/s] 93%|█████████▎| 371881/400000 [00:43<00:03, 8687.83it/s] 93%|█████████▎| 372751/400000 [00:43<00:03, 8690.39it/s] 93%|█████████▎| 373621/400000 [00:43<00:03, 8598.78it/s] 94%|█████████▎| 374499/400000 [00:43<00:02, 8649.89it/s] 94%|█████████▍| 375365/400000 [00:43<00:02, 8454.05it/s] 94%|█████████▍| 376212/400000 [00:43<00:02, 8280.94it/s] 94%|█████████▍| 377042/400000 [00:44<00:02, 8056.13it/s] 94%|█████████▍| 377898/400000 [00:44<00:02, 8199.56it/s] 95%|█████████▍| 378752/400000 [00:44<00:02, 8298.79it/s] 95%|█████████▍| 379595/400000 [00:44<00:02, 8335.75it/s] 95%|█████████▌| 380460/400000 [00:44<00:02, 8427.38it/s] 95%|█████████▌| 381304/400000 [00:44<00:02, 8415.54it/s] 96%|█████████▌| 382161/400000 [00:44<00:02, 8459.02it/s] 96%|█████████▌| 383008/400000 [00:44<00:02, 8371.77it/s] 96%|█████████▌| 383853/400000 [00:44<00:01, 8392.95it/s] 96%|█████████▌| 384727/400000 [00:44<00:01, 8492.64it/s] 96%|█████████▋| 385586/400000 [00:45<00:01, 8518.67it/s] 97%|█████████▋| 386468/400000 [00:45<00:01, 8605.12it/s] 97%|█████████▋| 387346/400000 [00:45<00:01, 8655.20it/s] 97%|█████████▋| 388240/400000 [00:45<00:01, 8738.62it/s] 97%|█████████▋| 389127/400000 [00:45<00:01, 8774.65it/s] 98%|█████████▊| 390007/400000 [00:45<00:01, 8781.87it/s] 98%|█████████▊| 390886/400000 [00:45<00:01, 8756.64it/s] 98%|█████████▊| 391762/400000 [00:45<00:00, 8741.95it/s] 98%|█████████▊| 392637/400000 [00:45<00:00, 8619.56it/s] 98%|█████████▊| 393520/400000 [00:45<00:00, 8681.46it/s] 99%|█████████▊| 394405/400000 [00:46<00:00, 8731.06it/s] 99%|█████████▉| 395279/400000 [00:46<00:00, 8724.15it/s] 99%|█████████▉| 396152/400000 [00:46<00:00, 8688.67it/s] 99%|█████████▉| 397026/400000 [00:46<00:00, 8702.18it/s] 99%|█████████▉| 397897/400000 [00:46<00:00, 8521.61it/s]100%|█████████▉| 398751/400000 [00:46<00:00, 8385.33it/s]100%|█████████▉| 399633/400000 [00:46<00:00, 8509.73it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8559.66it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f0e79bdaac8> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011445969471092388 	 Accuracy: 48
Train Epoch: 1 	 Loss: 0.011242197508796003 	 Accuracy: 53

  model saves at 53% accuracy 

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
2020-05-14 03:26:22.008522: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 03:26:22.013009: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-14 03:26:22.013140: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5564c04c0ab0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 03:26:22.013154: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f0e252960b8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8506 - accuracy: 0.4880
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5516 - accuracy: 0.5075 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5440 - accuracy: 0.5080
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5670 - accuracy: 0.5065
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5869 - accuracy: 0.5052
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6002 - accuracy: 0.5043
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5549 - accuracy: 0.5073
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5516 - accuracy: 0.5075
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5780 - accuracy: 0.5058
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6329 - accuracy: 0.5022
11000/25000 [============>.................] - ETA: 3s - loss: 7.6304 - accuracy: 0.5024
12000/25000 [=============>................] - ETA: 3s - loss: 7.6526 - accuracy: 0.5009
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6454 - accuracy: 0.5014
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6184 - accuracy: 0.5031
15000/25000 [=================>............] - ETA: 2s - loss: 7.6513 - accuracy: 0.5010
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6475 - accuracy: 0.5013
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6405 - accuracy: 0.5017
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6232 - accuracy: 0.5028
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6045 - accuracy: 0.5041
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6199 - accuracy: 0.5031
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6206 - accuracy: 0.5030
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6290 - accuracy: 0.5025
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6446 - accuracy: 0.5014
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6596 - accuracy: 0.5005
25000/25000 [==============================] - 7s 284us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f0df6d74978> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f0dde849eb8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 2.0118 - crf_viterbi_accuracy: 0.1067 - val_loss: 1.9488 - val_crf_viterbi_accuracy: 0.0667

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

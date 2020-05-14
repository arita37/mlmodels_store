
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f01b5b77fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 04:13:52.874561
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 04:13:52.878327
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 04:13:52.881574
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 04:13:52.884738
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f01c1b8f470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356553.7500
Epoch 2/10

1/1 [==============================] - 0s 96ms/step - loss: 267473.6250
Epoch 3/10

1/1 [==============================] - 0s 87ms/step - loss: 177686.5469
Epoch 4/10

1/1 [==============================] - 0s 86ms/step - loss: 110471.3672
Epoch 5/10

1/1 [==============================] - 0s 86ms/step - loss: 68879.4531
Epoch 6/10

1/1 [==============================] - 0s 90ms/step - loss: 44615.8711
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 30436.0254
Epoch 8/10

1/1 [==============================] - 0s 92ms/step - loss: 21808.0332
Epoch 9/10

1/1 [==============================] - 0s 91ms/step - loss: 16324.6240
Epoch 10/10

1/1 [==============================] - 0s 101ms/step - loss: 12706.6807

  #### Inference Need return ypred, ytrue ######################### 
[[ 8.14594030e-01 -3.88410360e-01  6.61369264e-01 -9.85198975e-01
   6.97595060e-01  1.03015864e+00 -8.33776832e-01  3.81028563e-01
  -5.01534045e-01 -8.50786209e-01 -9.89111066e-02 -3.11617553e-01
  -1.25085831e-01  1.01330185e+00 -1.35714817e+00 -1.40702724e-01
   4.57711101e-01 -7.29393482e-01 -8.54209423e-01  5.11830375e-02
   1.08340383e-02 -6.76056921e-01 -1.79966539e-01 -1.76373959e-01
  -3.32145423e-01  4.20037150e-01 -5.36252022e-01 -1.02144194e+00
   6.96538806e-01  1.56768650e-01  1.02246571e+00  1.01990044e-01
   2.65621215e-01 -1.24775159e+00 -7.06518352e-01  5.79775214e-01
  -4.01533067e-01  1.37128663e+00 -7.45615959e-02 -5.07057071e-01
  -4.87190068e-01 -1.41505718e+00  5.51151514e-01 -2.24310160e-03
  -4.08807129e-01 -6.94822669e-01 -8.05863976e-01  8.14716756e-01
   8.49243164e-01 -1.23110497e+00 -1.65445447e-01 -2.26971865e-01
  -2.93736875e-01 -3.16385150e-01  1.37372637e+00 -4.08507884e-02
   6.77202284e-01 -8.42975199e-01 -3.20480406e-01  7.01820254e-02
  -1.53046757e-01  4.84324598e+00  5.97601604e+00  5.66556025e+00
   5.56623077e+00  6.44993973e+00  4.26734734e+00  5.42392969e+00
   5.13031387e+00  5.21615934e+00  5.27015686e+00  5.97511435e+00
   5.32186413e+00  6.03566790e+00  3.77012587e+00  4.05099916e+00
   4.23667812e+00  5.94123125e+00  5.01531792e+00  6.57432652e+00
   5.08661079e+00  4.68737125e+00  5.08180571e+00  5.64079189e+00
   3.90943384e+00  4.19192171e+00  4.11412621e+00  4.83962822e+00
   4.00859451e+00  5.65188503e+00  6.04755831e+00  5.95649672e+00
   4.67495775e+00  5.73175240e+00  4.87192631e+00  5.05392790e+00
   4.56094933e+00  5.11342716e+00  5.53958416e+00  6.57828188e+00
   5.65531588e+00  6.76972198e+00  4.97738886e+00  4.87684870e+00
   5.38122511e+00  5.47122574e+00  4.02464199e+00  4.45076227e+00
   4.32877541e+00  3.94718242e+00  4.96441936e+00  4.97700500e+00
   5.85883093e+00  4.41041327e+00  3.99311662e+00  4.50343657e+00
   6.51636600e+00  6.64082432e+00  6.90612745e+00  4.98712587e+00
  -5.13739586e-02 -7.52946794e-01  5.59291691e-02 -2.00962037e-01
   3.19491148e-01  1.34299207e+00 -8.18722099e-02 -2.59702235e-01
   2.16333181e-01 -6.72389448e-01 -7.29041696e-02 -3.77715766e-01
  -8.78271163e-01 -1.46100855e+00 -5.25251448e-01 -8.85270059e-01
  -3.12314659e-01  3.43789876e-01  1.66765451e-02  7.78824985e-01
  -1.59152186e+00  5.47349870e-01  1.11400509e+00  9.02873397e-01
  -1.06942451e+00  4.95183051e-01 -3.10106754e-01  9.69959021e-01
  -5.72976947e-01  3.03573817e-01  1.28732145e-01 -3.85128111e-01
   5.28574474e-02  1.14209354e+00 -1.17263460e+00 -1.65096417e-01
   5.61818719e-01  3.45764071e-01 -4.21053052e-01 -2.40006834e-01
   3.78167033e-01  1.44950151e-01 -1.15829277e+00  1.34059811e+00
   1.24864638e+00 -6.04610920e-01 -2.40912437e-02 -4.06027228e-01
   3.96714807e-01 -1.13044530e-02  1.96290314e-02  1.77033663e-01
   7.69167840e-01  1.17107272e+00  7.22516179e-02  3.94848697e-02
  -4.43080962e-02  6.19284689e-01 -1.53599453e+00 -1.43765092e+00
   4.10060942e-01  2.89306521e-01  1.56533825e+00  7.88807929e-01
   1.25276232e+00  4.25347924e-01  1.12561154e+00  8.84602726e-01
   6.92652225e-01  1.11522245e+00  3.15466642e-01  1.97022641e+00
   1.25720072e+00  3.59220326e-01  2.33355427e+00  1.58436894e+00
   7.93413341e-01  9.24321771e-01  1.26176775e+00  2.93071926e-01
   1.77843487e+00  1.74341393e+00  7.81520963e-01  2.24939585e+00
   3.49070370e-01  2.13876843e-01  3.04672718e-01  2.94650793e-01
   1.29459047e+00  3.14654350e-01  2.02632236e+00  1.15973508e+00
   1.27409399e+00  1.91839814e+00  9.87207949e-01  1.83681643e+00
   1.81861758e+00  1.00423074e+00  7.53945947e-01  2.48114705e-01
   3.88968825e-01  1.48287773e+00  6.92947030e-01  1.90592122e+00
   1.32856870e+00  4.49192524e-01  6.01139784e-01  2.16268206e+00
   1.98144221e+00  4.10847545e-01  7.39412665e-01  2.34881926e+00
   8.38486314e-01  8.50937188e-01  5.40909886e-01  6.48983717e-01
   1.40071034e+00  1.68093491e+00  3.92913699e-01  7.36595273e-01
   5.48153520e-02  6.69408607e+00  6.63913918e+00  5.78083944e+00
   5.59908628e+00  6.83747005e+00  7.01013470e+00  6.40867186e+00
   6.33934832e+00  4.86482048e+00  5.15417480e+00  6.50602531e+00
   6.33395720e+00  6.57150316e+00  5.96033812e+00  6.70496368e+00
   5.88645124e+00  5.25620937e+00  5.97095871e+00  5.16396284e+00
   5.82936525e+00  6.72480106e+00  5.22662830e+00  5.89011240e+00
   6.49390268e+00  6.15748787e+00  4.29265451e+00  5.00168419e+00
   6.15830088e+00  6.92049265e+00  5.93176603e+00  6.08565998e+00
   6.60089540e+00  6.50653601e+00  6.73887396e+00  4.76996708e+00
   6.04272890e+00  6.14498901e+00  5.08786058e+00  4.96818495e+00
   5.94426298e+00  6.55755043e+00  6.83143711e+00  6.96482038e+00
   5.74622297e+00  5.44386816e+00  5.41172552e+00  6.45899296e+00
   5.82153034e+00  6.42437124e+00  5.34324074e+00  6.31964540e+00
   6.14988184e+00  4.97960901e+00  4.75449467e+00  6.12159967e+00
   6.39461899e+00  5.48176956e+00  6.38839245e+00  6.82448006e+00
   1.17884779e+00  4.51630712e-01  3.34300399e-01  2.78518057e+00
   1.71927488e+00  2.35630035e+00  1.28456879e+00  1.32942176e+00
   1.98426390e+00  5.13905823e-01  1.36419868e+00  6.22082233e-01
   5.69773614e-01  9.26902771e-01  5.50850868e-01  2.23417401e-01
   2.47295141e+00  1.09127724e+00  2.22478342e+00  7.66547799e-01
   6.07479036e-01  1.44463491e+00  1.02538300e+00  1.47273934e+00
   2.44349909e+00  3.09021294e-01  5.76111555e-01  1.07815576e+00
   3.57319355e-01  3.66171479e-01  2.35172415e+00  1.61159718e+00
   8.53988647e-01  1.55859971e+00  1.50324631e+00  5.71970463e-01
   3.72113883e-01  6.78855836e-01  9.13374960e-01  2.10265398e+00
   1.09903741e+00  9.52233195e-01  1.54341173e+00  2.05977142e-01
   1.16873884e+00  3.72325897e-01  9.94610190e-01  6.98050559e-01
   2.07959938e+00  1.07133698e+00  2.24831629e+00  1.83239126e+00
   2.17888975e+00  2.44426870e+00  9.61568117e-01  6.91485643e-01
   1.73055696e+00  4.66420650e-01  7.49298751e-01  2.66151190e-01
  -4.39589405e+00  3.15834260e+00 -3.70536423e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 04:14:01.394364
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   97.3877
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 04:14:01.397650
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9501.06
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 04:14:01.400680
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   97.7246
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 04:14:01.403583
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -849.887
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139644963636560
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139644022043200
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139644022043704
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139644022044208
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139644022044712
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139644022045216

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f01b5a85be0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.551050
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.519990
grad_step = 000002, loss = 0.497189
grad_step = 000003, loss = 0.472174
grad_step = 000004, loss = 0.444204
grad_step = 000005, loss = 0.417165
grad_step = 000006, loss = 0.396350
grad_step = 000007, loss = 0.383886
grad_step = 000008, loss = 0.369590
grad_step = 000009, loss = 0.349121
grad_step = 000010, loss = 0.330489
grad_step = 000011, loss = 0.316375
grad_step = 000012, loss = 0.305198
grad_step = 000013, loss = 0.293699
grad_step = 000014, loss = 0.279172
grad_step = 000015, loss = 0.263550
grad_step = 000016, loss = 0.247728
grad_step = 000017, loss = 0.233287
grad_step = 000018, loss = 0.221101
grad_step = 000019, loss = 0.210554
grad_step = 000020, loss = 0.200000
grad_step = 000021, loss = 0.188471
grad_step = 000022, loss = 0.177503
grad_step = 000023, loss = 0.167261
grad_step = 000024, loss = 0.157542
grad_step = 000025, loss = 0.148127
grad_step = 000026, loss = 0.139056
grad_step = 000027, loss = 0.130405
grad_step = 000028, loss = 0.122078
grad_step = 000029, loss = 0.114112
grad_step = 000030, loss = 0.106924
grad_step = 000031, loss = 0.100503
grad_step = 000032, loss = 0.093798
grad_step = 000033, loss = 0.087019
grad_step = 000034, loss = 0.080983
grad_step = 000035, loss = 0.075521
grad_step = 000036, loss = 0.070490
grad_step = 000037, loss = 0.065671
grad_step = 000038, loss = 0.060880
grad_step = 000039, loss = 0.056472
grad_step = 000040, loss = 0.052538
grad_step = 000041, loss = 0.048816
grad_step = 000042, loss = 0.045126
grad_step = 000043, loss = 0.041630
grad_step = 000044, loss = 0.038579
grad_step = 000045, loss = 0.035746
grad_step = 000046, loss = 0.032928
grad_step = 000047, loss = 0.030203
grad_step = 000048, loss = 0.027658
grad_step = 000049, loss = 0.025331
grad_step = 000050, loss = 0.023185
grad_step = 000051, loss = 0.021152
grad_step = 000052, loss = 0.019255
grad_step = 000053, loss = 0.017519
grad_step = 000054, loss = 0.015902
grad_step = 000055, loss = 0.014443
grad_step = 000056, loss = 0.013058
grad_step = 000057, loss = 0.011688
grad_step = 000058, loss = 0.010498
grad_step = 000059, loss = 0.009480
grad_step = 000060, loss = 0.008581
grad_step = 000061, loss = 0.007735
grad_step = 000062, loss = 0.006930
grad_step = 000063, loss = 0.006207
grad_step = 000064, loss = 0.005604
grad_step = 000065, loss = 0.005133
grad_step = 000066, loss = 0.004704
grad_step = 000067, loss = 0.004268
grad_step = 000068, loss = 0.003896
grad_step = 000069, loss = 0.003616
grad_step = 000070, loss = 0.003397
grad_step = 000071, loss = 0.003203
grad_step = 000072, loss = 0.003030
grad_step = 000073, loss = 0.002864
grad_step = 000074, loss = 0.002749
grad_step = 000075, loss = 0.002672
grad_step = 000076, loss = 0.002623
grad_step = 000077, loss = 0.002568
grad_step = 000078, loss = 0.002510
grad_step = 000079, loss = 0.002455
grad_step = 000080, loss = 0.002427
grad_step = 000081, loss = 0.002408
grad_step = 000082, loss = 0.002397
grad_step = 000083, loss = 0.002391
grad_step = 000084, loss = 0.002379
grad_step = 000085, loss = 0.002359
grad_step = 000086, loss = 0.002336
grad_step = 000087, loss = 0.002313
grad_step = 000088, loss = 0.002292
grad_step = 000089, loss = 0.002276
grad_step = 000090, loss = 0.002263
grad_step = 000091, loss = 0.002255
grad_step = 000092, loss = 0.002246
grad_step = 000093, loss = 0.002241
grad_step = 000094, loss = 0.002245
grad_step = 000095, loss = 0.002257
grad_step = 000096, loss = 0.002284
grad_step = 000097, loss = 0.002312
grad_step = 000098, loss = 0.002323
grad_step = 000099, loss = 0.002274
grad_step = 000100, loss = 0.002202
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002138
grad_step = 000102, loss = 0.002120
grad_step = 000103, loss = 0.002144
grad_step = 000104, loss = 0.002174
grad_step = 000105, loss = 0.002179
grad_step = 000106, loss = 0.002146
grad_step = 000107, loss = 0.002101
grad_step = 000108, loss = 0.002073
grad_step = 000109, loss = 0.002074
grad_step = 000110, loss = 0.002092
grad_step = 000111, loss = 0.002105
grad_step = 000112, loss = 0.002101
grad_step = 000113, loss = 0.002080
grad_step = 000114, loss = 0.002053
grad_step = 000115, loss = 0.002034
grad_step = 000116, loss = 0.002028
grad_step = 000117, loss = 0.002032
grad_step = 000118, loss = 0.002041
grad_step = 000119, loss = 0.002050
grad_step = 000120, loss = 0.002055
grad_step = 000121, loss = 0.002056
grad_step = 000122, loss = 0.002050
grad_step = 000123, loss = 0.002041
grad_step = 000124, loss = 0.002029
grad_step = 000125, loss = 0.002016
grad_step = 000126, loss = 0.002003
grad_step = 000127, loss = 0.001993
grad_step = 000128, loss = 0.001985
grad_step = 000129, loss = 0.001979
grad_step = 000130, loss = 0.001975
grad_step = 000131, loss = 0.001972
grad_step = 000132, loss = 0.001969
grad_step = 000133, loss = 0.001966
grad_step = 000134, loss = 0.001963
grad_step = 000135, loss = 0.001961
grad_step = 000136, loss = 0.001960
grad_step = 000137, loss = 0.001961
grad_step = 000138, loss = 0.001970
grad_step = 000139, loss = 0.002000
grad_step = 000140, loss = 0.002095
grad_step = 000141, loss = 0.002349
grad_step = 000142, loss = 0.002798
grad_step = 000143, loss = 0.003135
grad_step = 000144, loss = 0.002644
grad_step = 000145, loss = 0.001963
grad_step = 000146, loss = 0.002303
grad_step = 000147, loss = 0.002672
grad_step = 000148, loss = 0.002178
grad_step = 000149, loss = 0.002001
grad_step = 000150, loss = 0.002435
grad_step = 000151, loss = 0.002230
grad_step = 000152, loss = 0.001949
grad_step = 000153, loss = 0.002253
grad_step = 000154, loss = 0.002201
grad_step = 000155, loss = 0.001931
grad_step = 000156, loss = 0.002141
grad_step = 000157, loss = 0.002134
grad_step = 000158, loss = 0.001933
grad_step = 000159, loss = 0.002063
grad_step = 000160, loss = 0.002088
grad_step = 000161, loss = 0.001922
grad_step = 000162, loss = 0.002019
grad_step = 000163, loss = 0.002042
grad_step = 000164, loss = 0.001923
grad_step = 000165, loss = 0.001982
grad_step = 000166, loss = 0.002012
grad_step = 000167, loss = 0.001917
grad_step = 000168, loss = 0.001954
grad_step = 000169, loss = 0.001983
grad_step = 000170, loss = 0.001916
grad_step = 000171, loss = 0.001930
grad_step = 000172, loss = 0.001963
grad_step = 000173, loss = 0.001913
grad_step = 000174, loss = 0.001914
grad_step = 000175, loss = 0.001943
grad_step = 000176, loss = 0.001913
grad_step = 000177, loss = 0.001900
grad_step = 000178, loss = 0.001926
grad_step = 000179, loss = 0.001912
grad_step = 000180, loss = 0.001893
grad_step = 000181, loss = 0.001908
grad_step = 000182, loss = 0.001909
grad_step = 000183, loss = 0.001890
grad_step = 000184, loss = 0.001893
grad_step = 000185, loss = 0.001902
grad_step = 000186, loss = 0.001891
grad_step = 000187, loss = 0.001883
grad_step = 000188, loss = 0.001891
grad_step = 000189, loss = 0.001890
grad_step = 000190, loss = 0.001879
grad_step = 000191, loss = 0.001879
grad_step = 000192, loss = 0.001884
grad_step = 000193, loss = 0.001879
grad_step = 000194, loss = 0.001873
grad_step = 000195, loss = 0.001874
grad_step = 000196, loss = 0.001876
grad_step = 000197, loss = 0.001872
grad_step = 000198, loss = 0.001867
grad_step = 000199, loss = 0.001868
grad_step = 000200, loss = 0.001869
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001866
grad_step = 000202, loss = 0.001862
grad_step = 000203, loss = 0.001862
grad_step = 000204, loss = 0.001862
grad_step = 000205, loss = 0.001861
grad_step = 000206, loss = 0.001858
grad_step = 000207, loss = 0.001856
grad_step = 000208, loss = 0.001856
grad_step = 000209, loss = 0.001855
grad_step = 000210, loss = 0.001853
grad_step = 000211, loss = 0.001851
grad_step = 000212, loss = 0.001849
grad_step = 000213, loss = 0.001849
grad_step = 000214, loss = 0.001848
grad_step = 000215, loss = 0.001847
grad_step = 000216, loss = 0.001845
grad_step = 000217, loss = 0.001843
grad_step = 000218, loss = 0.001842
grad_step = 000219, loss = 0.001841
grad_step = 000220, loss = 0.001840
grad_step = 000221, loss = 0.001839
grad_step = 000222, loss = 0.001838
grad_step = 000223, loss = 0.001836
grad_step = 000224, loss = 0.001835
grad_step = 000225, loss = 0.001834
grad_step = 000226, loss = 0.001833
grad_step = 000227, loss = 0.001832
grad_step = 000228, loss = 0.001831
grad_step = 000229, loss = 0.001829
grad_step = 000230, loss = 0.001828
grad_step = 000231, loss = 0.001827
grad_step = 000232, loss = 0.001826
grad_step = 000233, loss = 0.001824
grad_step = 000234, loss = 0.001823
grad_step = 000235, loss = 0.001822
grad_step = 000236, loss = 0.001821
grad_step = 000237, loss = 0.001819
grad_step = 000238, loss = 0.001818
grad_step = 000239, loss = 0.001817
grad_step = 000240, loss = 0.001816
grad_step = 000241, loss = 0.001815
grad_step = 000242, loss = 0.001814
grad_step = 000243, loss = 0.001813
grad_step = 000244, loss = 0.001811
grad_step = 000245, loss = 0.001810
grad_step = 000246, loss = 0.001809
grad_step = 000247, loss = 0.001809
grad_step = 000248, loss = 0.001808
grad_step = 000249, loss = 0.001810
grad_step = 000250, loss = 0.001814
grad_step = 000251, loss = 0.001827
grad_step = 000252, loss = 0.001862
grad_step = 000253, loss = 0.001948
grad_step = 000254, loss = 0.002166
grad_step = 000255, loss = 0.002549
grad_step = 000256, loss = 0.003156
grad_step = 000257, loss = 0.003096
grad_step = 000258, loss = 0.002386
grad_step = 000259, loss = 0.001817
grad_step = 000260, loss = 0.002193
grad_step = 000261, loss = 0.002584
grad_step = 000262, loss = 0.002140
grad_step = 000263, loss = 0.001829
grad_step = 000264, loss = 0.002196
grad_step = 000265, loss = 0.002243
grad_step = 000266, loss = 0.001854
grad_step = 000267, loss = 0.001924
grad_step = 000268, loss = 0.002150
grad_step = 000269, loss = 0.001933
grad_step = 000270, loss = 0.001822
grad_step = 000271, loss = 0.002014
grad_step = 000272, loss = 0.001975
grad_step = 000273, loss = 0.001804
grad_step = 000274, loss = 0.001892
grad_step = 000275, loss = 0.001957
grad_step = 000276, loss = 0.001825
grad_step = 000277, loss = 0.001818
grad_step = 000278, loss = 0.001911
grad_step = 000279, loss = 0.001847
grad_step = 000280, loss = 0.001786
grad_step = 000281, loss = 0.001852
grad_step = 000282, loss = 0.001852
grad_step = 000283, loss = 0.001787
grad_step = 000284, loss = 0.001804
grad_step = 000285, loss = 0.001838
grad_step = 000286, loss = 0.001801
grad_step = 000287, loss = 0.001779
grad_step = 000288, loss = 0.001811
grad_step = 000289, loss = 0.001809
grad_step = 000290, loss = 0.001777
grad_step = 000291, loss = 0.001784
grad_step = 000292, loss = 0.001802
grad_step = 000293, loss = 0.001784
grad_step = 000294, loss = 0.001770
grad_step = 000295, loss = 0.001784
grad_step = 000296, loss = 0.001786
grad_step = 000297, loss = 0.001770
grad_step = 000298, loss = 0.001767
grad_step = 000299, loss = 0.001778
grad_step = 000300, loss = 0.001774
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001763
grad_step = 000302, loss = 0.001764
grad_step = 000303, loss = 0.001770
grad_step = 000304, loss = 0.001765
grad_step = 000305, loss = 0.001758
grad_step = 000306, loss = 0.001759
grad_step = 000307, loss = 0.001762
grad_step = 000308, loss = 0.001759
grad_step = 000309, loss = 0.001754
grad_step = 000310, loss = 0.001754
grad_step = 000311, loss = 0.001756
grad_step = 000312, loss = 0.001753
grad_step = 000313, loss = 0.001749
grad_step = 000314, loss = 0.001749
grad_step = 000315, loss = 0.001750
grad_step = 000316, loss = 0.001749
grad_step = 000317, loss = 0.001746
grad_step = 000318, loss = 0.001744
grad_step = 000319, loss = 0.001744
grad_step = 000320, loss = 0.001744
grad_step = 000321, loss = 0.001742
grad_step = 000322, loss = 0.001740
grad_step = 000323, loss = 0.001739
grad_step = 000324, loss = 0.001739
grad_step = 000325, loss = 0.001738
grad_step = 000326, loss = 0.001737
grad_step = 000327, loss = 0.001735
grad_step = 000328, loss = 0.001733
grad_step = 000329, loss = 0.001733
grad_step = 000330, loss = 0.001732
grad_step = 000331, loss = 0.001731
grad_step = 000332, loss = 0.001730
grad_step = 000333, loss = 0.001728
grad_step = 000334, loss = 0.001727
grad_step = 000335, loss = 0.001726
grad_step = 000336, loss = 0.001725
grad_step = 000337, loss = 0.001725
grad_step = 000338, loss = 0.001723
grad_step = 000339, loss = 0.001722
grad_step = 000340, loss = 0.001721
grad_step = 000341, loss = 0.001720
grad_step = 000342, loss = 0.001718
grad_step = 000343, loss = 0.001717
grad_step = 000344, loss = 0.001716
grad_step = 000345, loss = 0.001715
grad_step = 000346, loss = 0.001714
grad_step = 000347, loss = 0.001713
grad_step = 000348, loss = 0.001712
grad_step = 000349, loss = 0.001711
grad_step = 000350, loss = 0.001711
grad_step = 000351, loss = 0.001713
grad_step = 000352, loss = 0.001717
grad_step = 000353, loss = 0.001724
grad_step = 000354, loss = 0.001731
grad_step = 000355, loss = 0.001738
grad_step = 000356, loss = 0.001744
grad_step = 000357, loss = 0.001746
grad_step = 000358, loss = 0.001746
grad_step = 000359, loss = 0.001740
grad_step = 000360, loss = 0.001736
grad_step = 000361, loss = 0.001734
grad_step = 000362, loss = 0.001735
grad_step = 000363, loss = 0.001736
grad_step = 000364, loss = 0.001736
grad_step = 000365, loss = 0.001737
grad_step = 000366, loss = 0.001740
grad_step = 000367, loss = 0.001745
grad_step = 000368, loss = 0.001751
grad_step = 000369, loss = 0.001755
grad_step = 000370, loss = 0.001760
grad_step = 000371, loss = 0.001762
grad_step = 000372, loss = 0.001767
grad_step = 000373, loss = 0.001766
grad_step = 000374, loss = 0.001765
grad_step = 000375, loss = 0.001759
grad_step = 000376, loss = 0.001754
grad_step = 000377, loss = 0.001744
grad_step = 000378, loss = 0.001734
grad_step = 000379, loss = 0.001721
grad_step = 000380, loss = 0.001710
grad_step = 000381, loss = 0.001701
grad_step = 000382, loss = 0.001693
grad_step = 000383, loss = 0.001686
grad_step = 000384, loss = 0.001681
grad_step = 000385, loss = 0.001677
grad_step = 000386, loss = 0.001674
grad_step = 000387, loss = 0.001672
grad_step = 000388, loss = 0.001671
grad_step = 000389, loss = 0.001670
grad_step = 000390, loss = 0.001668
grad_step = 000391, loss = 0.001668
grad_step = 000392, loss = 0.001667
grad_step = 000393, loss = 0.001667
grad_step = 000394, loss = 0.001667
grad_step = 000395, loss = 0.001668
grad_step = 000396, loss = 0.001671
grad_step = 000397, loss = 0.001678
grad_step = 000398, loss = 0.001693
grad_step = 000399, loss = 0.001726
grad_step = 000400, loss = 0.001792
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001930
grad_step = 000402, loss = 0.002170
grad_step = 000403, loss = 0.002601
grad_step = 000404, loss = 0.002972
grad_step = 000405, loss = 0.003133
grad_step = 000406, loss = 0.002488
grad_step = 000407, loss = 0.001792
grad_step = 000408, loss = 0.001757
grad_step = 000409, loss = 0.002212
grad_step = 000410, loss = 0.002353
grad_step = 000411, loss = 0.001908
grad_step = 000412, loss = 0.001685
grad_step = 000413, loss = 0.001976
grad_step = 000414, loss = 0.002093
grad_step = 000415, loss = 0.001822
grad_step = 000416, loss = 0.001689
grad_step = 000417, loss = 0.001867
grad_step = 000418, loss = 0.001915
grad_step = 000419, loss = 0.001730
grad_step = 000420, loss = 0.001695
grad_step = 000421, loss = 0.001816
grad_step = 000422, loss = 0.001787
grad_step = 000423, loss = 0.001679
grad_step = 000424, loss = 0.001702
grad_step = 000425, loss = 0.001762
grad_step = 000426, loss = 0.001718
grad_step = 000427, loss = 0.001661
grad_step = 000428, loss = 0.001693
grad_step = 000429, loss = 0.001722
grad_step = 000430, loss = 0.001679
grad_step = 000431, loss = 0.001652
grad_step = 000432, loss = 0.001680
grad_step = 000433, loss = 0.001686
grad_step = 000434, loss = 0.001657
grad_step = 000435, loss = 0.001645
grad_step = 000436, loss = 0.001664
grad_step = 000437, loss = 0.001664
grad_step = 000438, loss = 0.001642
grad_step = 000439, loss = 0.001638
grad_step = 000440, loss = 0.001650
grad_step = 000441, loss = 0.001649
grad_step = 000442, loss = 0.001633
grad_step = 000443, loss = 0.001631
grad_step = 000444, loss = 0.001639
grad_step = 000445, loss = 0.001637
grad_step = 000446, loss = 0.001627
grad_step = 000447, loss = 0.001624
grad_step = 000448, loss = 0.001630
grad_step = 000449, loss = 0.001629
grad_step = 000450, loss = 0.001621
grad_step = 000451, loss = 0.001617
grad_step = 000452, loss = 0.001621
grad_step = 000453, loss = 0.001622
grad_step = 000454, loss = 0.001616
grad_step = 000455, loss = 0.001612
grad_step = 000456, loss = 0.001613
grad_step = 000457, loss = 0.001614
grad_step = 000458, loss = 0.001612
grad_step = 000459, loss = 0.001608
grad_step = 000460, loss = 0.001606
grad_step = 000461, loss = 0.001607
grad_step = 000462, loss = 0.001607
grad_step = 000463, loss = 0.001604
grad_step = 000464, loss = 0.001601
grad_step = 000465, loss = 0.001601
grad_step = 000466, loss = 0.001601
grad_step = 000467, loss = 0.001599
grad_step = 000468, loss = 0.001597
grad_step = 000469, loss = 0.001596
grad_step = 000470, loss = 0.001594
grad_step = 000471, loss = 0.001593
grad_step = 000472, loss = 0.001592
grad_step = 000473, loss = 0.001590
grad_step = 000474, loss = 0.001589
grad_step = 000475, loss = 0.001588
grad_step = 000476, loss = 0.001587
grad_step = 000477, loss = 0.001586
grad_step = 000478, loss = 0.001585
grad_step = 000479, loss = 0.001585
grad_step = 000480, loss = 0.001585
grad_step = 000481, loss = 0.001585
grad_step = 000482, loss = 0.001583
grad_step = 000483, loss = 0.001581
grad_step = 000484, loss = 0.001577
grad_step = 000485, loss = 0.001574
grad_step = 000486, loss = 0.001572
grad_step = 000487, loss = 0.001571
grad_step = 000488, loss = 0.001571
grad_step = 000489, loss = 0.001570
grad_step = 000490, loss = 0.001569
grad_step = 000491, loss = 0.001567
grad_step = 000492, loss = 0.001565
grad_step = 000493, loss = 0.001563
grad_step = 000494, loss = 0.001561
grad_step = 000495, loss = 0.001559
grad_step = 000496, loss = 0.001558
grad_step = 000497, loss = 0.001558
grad_step = 000498, loss = 0.001558
grad_step = 000499, loss = 0.001560
grad_step = 000500, loss = 0.001564
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001571
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

  date_run                              2020-05-14 04:14:18.940098
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.244002
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 04:14:18.945740
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.149558
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 04:14:18.952921
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.148948
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 04:14:18.957750
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.27258
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
0   2020-05-14 04:13:52.874561  ...    mean_absolute_error
1   2020-05-14 04:13:52.878327  ...     mean_squared_error
2   2020-05-14 04:13:52.881574  ...  median_absolute_error
3   2020-05-14 04:13:52.884738  ...               r2_score
4   2020-05-14 04:14:01.394364  ...    mean_absolute_error
5   2020-05-14 04:14:01.397650  ...     mean_squared_error
6   2020-05-14 04:14:01.400680  ...  median_absolute_error
7   2020-05-14 04:14:01.403583  ...               r2_score
8   2020-05-14 04:14:18.940098  ...    mean_absolute_error
9   2020-05-14 04:14:18.945740  ...     mean_squared_error
10  2020-05-14 04:14:18.952921  ...  median_absolute_error
11  2020-05-14 04:14:18.957750  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f21ae096fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:32, 301203.05it/s]  2%|▏         | 212992/9912422 [00:00<00:24, 388879.24it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 538244.30it/s] 29%|██▉       | 2908160/9912422 [00:00<00:09, 759171.21it/s] 54%|█████▍    | 5390336/9912422 [00:00<00:04, 1066828.12it/s] 82%|████████▏ | 8093696/9912422 [00:01<00:01, 1491293.30it/s]9920512it [00:01, 8733113.30it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 310761.54it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 304552.26it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 392573.41it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 543196.61it/s]1654784it [00:00, 2704266.31it/s]                           
0it [00:00, ?it/s]8192it [00:00, 91998.13it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2160a98e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f215d8e40b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2160a98e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f216001e0b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f215d8594a8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f215d844c18> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2160a98e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f215ffdd6d8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f215d8594a8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f21ae0a1ef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f4c3e7381d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=7ca0d455d1851cc332412353b836166d3bd7de03cf4ec5d8ebd73a2e59387f61
  Stored in directory: /tmp/pip-ephem-wheel-cache-cpm088_z/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f4c348a2048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 46s
   57344/17464789 [..............................] - ETA: 39s
   90112/17464789 [..............................] - ETA: 38s
  212992/17464789 [..............................] - ETA: 21s
  442368/17464789 [..............................] - ETA: 12s
  909312/17464789 [>.............................] - ETA: 7s 
 1826816/17464789 [==>...........................] - ETA: 3s
 3629056/17464789 [=====>........................] - ETA: 2s
 6709248/17464789 [==========>...................] - ETA: 0s
 9625600/17464789 [===============>..............] - ETA: 0s
12607488/17464789 [====================>.........] - ETA: 0s
15556608/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 04:15:48.028546: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 04:15:48.033175: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095094999 Hz
2020-05-14 04:15:48.033314: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a22189b070 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 04:15:48.033328: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.9273 - accuracy: 0.4830
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5746 - accuracy: 0.5060 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6615 - accuracy: 0.5003
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7586 - accuracy: 0.4940
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6912 - accuracy: 0.4984
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6922 - accuracy: 0.4983
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6798 - accuracy: 0.4991
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6417 - accuracy: 0.5016
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6598 - accuracy: 0.5004
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6329 - accuracy: 0.5022
11000/25000 [============>.................] - ETA: 3s - loss: 7.6220 - accuracy: 0.5029
12000/25000 [=============>................] - ETA: 3s - loss: 7.6360 - accuracy: 0.5020
13000/25000 [==============>...............] - ETA: 2s - loss: 7.5911 - accuracy: 0.5049
14000/25000 [===============>..............] - ETA: 2s - loss: 7.5965 - accuracy: 0.5046
15000/25000 [=================>............] - ETA: 2s - loss: 7.6308 - accuracy: 0.5023
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6465 - accuracy: 0.5013
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6414 - accuracy: 0.5016
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6564 - accuracy: 0.5007
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6529 - accuracy: 0.5009
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6705 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6798 - accuracy: 0.4991
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6764 - accuracy: 0.4994
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6606 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6583 - accuracy: 0.5005
25000/25000 [==============================] - 7s 277us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 04:16:01.493484
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 04:16:01.493484  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<12:38:54, 18.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<8:53:15, 26.9kB/s]  .vector_cache/glove.6B.zip:   0%|          | 1.41M/862M [00:00<6:13:07, 38.4kB/s].vector_cache/glove.6B.zip:   0%|          | 3.78M/862M [00:00<4:20:39, 54.9kB/s].vector_cache/glove.6B.zip:   1%|          | 7.13M/862M [00:00<3:01:52, 78.4kB/s].vector_cache/glove.6B.zip:   1%|          | 10.6M/862M [00:00<2:06:55, 112kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 14.0M/862M [00:01<1:28:36, 160kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.9M/862M [00:01<1:01:57, 227kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.2M/862M [00:01<43:20, 324kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 23.2M/862M [00:01<30:22, 460kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.0M/862M [00:01<21:20, 653kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.2M/862M [00:01<15:00, 925kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.5M/862M [00:01<10:35, 1.31MB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.8M/862M [00:01<07:34, 1.82MB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.0M/862M [00:01<05:28, 2.51MB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.2M/862M [00:01<04:00, 3.42MB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.3M/862M [00:02<02:56, 4.66MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.6M/862M [00:02<02:10, 6.27MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.9M/862M [00:02<01:38, 8.29MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:02<01:16, 10.7MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:02<17:09, 786kB/s] .vector_cache/glove.6B.zip:   6%|▋         | 55.3M/862M [00:02<12:06, 1.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<15:57, 842kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<12:51, 1.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:04<09:08, 1.46MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<11:11, 1.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:06<08:59, 1.49MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:06<06:23, 2.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<15:22, 865kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<11:52, 1.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:08<08:23, 1.58MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<21:14, 623kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<16:44, 789kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.3M/862M [00:10<11:51, 1.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<13:03, 1.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<13:09, 1.00MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:12<09:27, 1.39MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<09:21, 1.40MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<08:32, 1.53MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.6M/862M [00:14<06:07, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<08:54, 1.46MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<08:14, 1.58MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<06:36, 1.97MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:16<04:43, 2.74MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:17<04:08, 3.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:19<16:40:41, 12.9kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.1M/862M [00:19<11:42:07, 18.4kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:19<8:10:23, 26.3kB/s] .vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:19<5:42:55, 37.5kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:21<4:13:23, 50.7kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:21<2:59:09, 71.7kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:21<2:05:10, 102kB/s] .vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:23<1:32:02, 139kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.4M/862M [00:23<1:05:37, 195kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:23<45:58, 277kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 99.2M/862M [00:25<36:38, 347kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:25<27:28, 463kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<19:20, 655kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<18:00, 702kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<14:30, 872kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<10:16, 1.23MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<12:09, 1.03MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<09:33, 1.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<06:51, 1.83MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<09:17, 1.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<08:18, 1.51MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<05:56, 2.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<09:03, 1.37MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<06:28, 1.91MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<08:13, 1.50MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<06:51, 1.80MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<04:56, 2.49MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<08:11, 1.50MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<07:03, 1.74MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<08:28, 1.45MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:37<06:00, 2.03MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<06:09, 1.98MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<10:05:04, 20.2kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<7:04:09, 28.8kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:39<4:55:41, 41.1kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<3:42:19, 54.7kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<2:37:23, 77.2kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:41<1:49:57, 110kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<1:21:08, 149kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<58:34, 206kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<41:00, 294kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<34:07, 352kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<25:39, 468kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<18:03, 663kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<16:54, 707kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<13:33, 881kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<09:40, 1.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<09:40, 1.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<08:27, 1.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:03, 1.96MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:52<09:36, 1.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:52<08:23, 1.41MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:00, 1.96MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:54<08:18, 1.41MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:54<07:32, 1.56MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:24, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:56<08:05, 1.44MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:56<07:15, 1.61MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:12, 2.23MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:58<07:45, 1.49MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:58<07:05, 1.64MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:05, 2.27MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:00<07:51, 1.47MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:00<07:13, 1.59MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:10, 2.22MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:02<08:02, 1.42MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<06:52, 1.67MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<04:55, 2.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:04<08:37, 1.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:04<07:46, 1.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:32, 2.05MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:06<08:45, 1.29MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:06<07:44, 1.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<05:30, 2.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:08<09:09, 1.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:08<08:05, 1.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:45, 1.95MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:10<09:32, 1.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:10<08:15, 1.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<05:53, 1.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:12<08:55, 1.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:12<07:24, 1.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:17, 2.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:14<09:36, 1.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:14<08:19, 1.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:56, 1.85MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:16<08:14, 1.33MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:16<07:24, 1.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<05:18, 2.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:18<07:34, 1.44MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<06:51, 1.59MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<04:55, 2.21MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<03:46, 2.88MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:20<11:13, 967kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:20<09:02, 1.20MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:26, 1.68MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<04:55, 2.19MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:22<13:13, 815kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:22<10:47, 998kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<07:46, 1.38MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<05:44, 1.87MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<04:22, 2.45MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:24<14:41, 729kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:24<11:47, 908kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<08:28, 1.26MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:25<06:11, 1.72MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:26<07:57, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:26<09:59, 1.06MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:26<10:35, 1.00MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:26<10:14, 1.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:27<10:11, 1.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<08:04, 1.32MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:27<05:49, 1.82MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:28<06:35, 1.61MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 228M/862M [01:28<05:11, 2.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<03:50, 2.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<03:06, 3.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<02:29, 4.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:30<24:57, 421kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:30<19:01, 552kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<13:35, 771kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<09:53, 1.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:31<07:16, 1.44MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<05:23, 1.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:33<1:12:54, 143kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:33<52:36, 198kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<36:59, 281kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:33<26:17, 395kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<18:44, 554kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<18:01, 575kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:35<14:04, 736kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<10:10, 1.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:35<07:25, 1.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:36<05:32, 1.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:37<08:46, 1.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:37<08:15, 1.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:37<06:30, 1.58MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:37<05:02, 2.04MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<04:00, 2.57MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<03:15, 3.15MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:38<02:42, 3.77MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:39<06:50, 1.49MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:39<06:02, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:39<04:59, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<04:15, 2.40MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<03:36, 2.82MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:40<02:56, 3.46MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:40<02:31, 4.03MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:40<02:04, 4.91MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:41<52:47, 192kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:41<37:57, 267kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:41<26:47, 378kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:41<19:00, 532kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<13:30, 747kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:43<27:50, 362kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:43<21:06, 478kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:43<15:06, 667kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:43<10:54, 922kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:44<07:58, 1.26MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:44<05:54, 1.70MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:46<21:17, 471kB/s] .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:46<16:19, 613kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:46<11:38, 859kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:46<08:17, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:48<11:26, 869kB/s] .vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:48<09:35, 1.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<06:57, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:48<05:04, 1.95MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:50<07:03, 1.40MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<06:01, 1.64MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<23:06, 427kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:51<3:40:41, 44.8kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<2:34:23, 63.8kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:52<1:48:40, 90.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:52<1:16:23, 128kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:52<53:15, 183kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:54<1:14:23, 131kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:54<53:37, 182kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:54<37:31, 258kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:56<29:15, 331kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:56<21:47, 444kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:56<15:25, 625kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:56<10:55, 880kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:58<15:04, 637kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:58<11:56, 803kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:58<08:35, 1.11MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:58<06:11, 1.54MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:00<08:13, 1.16MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:00<07:12, 1.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:00<05:15, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:00<03:50, 2.46MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<06:16, 1.51MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<05:48, 1.63MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 297M/862M [02:02<04:10, 2.26MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:05<07:44, 1.21MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:05<06:45, 1.39MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<04:54, 1.90MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:05<03:40, 2.54MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:07<06:10, 1.51MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:07<08:14, 1.13MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:07<06:54, 1.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:07<05:37, 1.66MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:07<05:02, 1.85MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:07<04:24, 2.11MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<03:39, 2.53MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:08<02:59, 3.09MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:08<02:45, 3.37MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:08<02:19, 3.99MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:08<02:20, 3.95MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:09<40:36, 228kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:09<32:33, 284kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:09<1:19:10, 117kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:09<55:13, 167kB/s]  .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:12<43:49, 209kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:12<31:58, 287kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:12<22:32, 406kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:12<16:03, 569kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:14<14:35, 624kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:14<11:33, 788kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:14<08:16, 1.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:14<06:00, 1.51MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:15<04:25, 2.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:16<22:54, 394kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:16<17:15, 523kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:16<12:17, 733kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:16<08:50, 1.02MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:17<06:28, 1.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:18<10:28, 856kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:18<08:39, 1.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:18<06:16, 1.43MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:18<04:41, 1.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:20<05:47, 1.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:20<06:34, 1.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:20<06:04, 1.46MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:20<05:30, 1.61MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:21<04:06, 2.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:21<03:05, 2.87MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:21<02:22, 3.72MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:22<38:22, 230kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:22<29:18, 301kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:22<21:04, 418kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:22<14:58, 588kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:22<10:40, 823kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:25<12:30, 700kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:25<10:06, 866kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:25<07:22, 1.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:25<05:23, 1.62MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:25<03:57, 2.20MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:27<08:38, 1.01MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:27<06:45, 1.28MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:27<05:00, 1.73MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:27<03:46, 2.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:27<02:52, 3.00MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:29<07:13, 1.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:29<07:29, 1.15MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:29<05:30, 1.56MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:29<04:04, 2.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:29<03:06, 2.76MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:30<06:49, 1.25MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:31<07:31, 1.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:31<06:12, 1.38MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:31<04:38, 1.84MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:31<03:30, 2.43MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:31<02:42, 3.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:32<05:57, 1.42MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:33<04:55, 1.72MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:33<03:32, 2.39MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:33<02:40, 3.14MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:34<54:05, 156kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:35<39:49, 211kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:35<28:13, 298kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:35<20:05, 418kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:35<14:25, 581kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:35<10:25, 803kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:35<07:35, 1.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:36<17:31, 476kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:37<13:07, 636kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:37<09:24, 885kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:37<06:52, 1.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:37<05:04, 1.63MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:38<07:29, 1.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:39<06:49, 1.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:39<05:01, 1.65MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:39<03:46, 2.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:39<02:55, 2.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:39<02:23, 3.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:41<13:48, 595kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:41<10:49, 758kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:41<07:44, 1.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:41<05:37, 1.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:41<04:08, 1.97MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:43<30:10, 270kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:43<22:03, 369kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:43<15:34, 521kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:43<11:05, 729kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:43<07:59, 1.01MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:45<19:04, 423kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:45<14:31, 555kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:45<10:24, 773kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:45<07:30, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:45<05:30, 1.45MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:47<07:29, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:47<07:00, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:47<05:12, 1.54MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:47<03:58, 2.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:47<03:04, 2.59MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:47<02:23, 3.32MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:49<07:02, 1.13MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:49<07:56, 999kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:49<07:31, 1.05MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:49<06:50, 1.16MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:49<06:14, 1.27MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:49<05:12, 1.52MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:49<04:25, 1.79MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:49<03:24, 2.32MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:50<02:35, 3.05MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:52<06:14, 1.26MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:52<05:31, 1.42MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:52<03:59, 1.96MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:53<04:39, 1.67MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:54<06:30, 1.20MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:54<04:52, 1.60MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:54<03:42, 2.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:54<02:57, 2.62MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:54<02:26, 3.17MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:54<02:05, 3.70MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:55<07:25, 1.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:56<06:13, 1.24MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:56<04:32, 1.70MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:56<03:23, 2.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:56<02:34, 2.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:57<08:22, 915kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:58<07:45, 988kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:58<05:37, 1.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:58<04:06, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:58<03:00, 2.52MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [03:00<5:06:31, 24.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [03:00<3:35:31, 35.2kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [03:00<2:30:30, 50.2kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:00<1:45:20, 71.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:02<1:16:17, 98.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:02<54:16, 139kB/s]   .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:02<38:00, 197kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:02<26:46, 279kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:02<18:51, 395kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:04<3:47:09, 32.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:04<2:39:42, 46.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:04<1:51:22, 66.6kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:04<1:17:49, 94.9kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:06<1:28:17, 83.6kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:06<1:03:31, 116kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:07<44:57, 164kB/s]  .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:07<32:51, 224kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:07<23:28, 314kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:07<16:53, 436kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:07<12:44, 577kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:07<10:16, 715kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:07<08:12, 896kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:07<07:05, 1.04MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:07<05:50, 1.26MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:08<04:52, 1.50MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:08<03:51, 1.90MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:08<03:07, 2.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:09<20:09, 363kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:09<15:11, 481kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:09<11:10, 654kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:09<09:36, 760kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:09<07:42, 948kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:09<06:22, 1.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:09<05:37, 1.30MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:09<04:56, 1.48MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:10<04:34, 1.59MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:10<03:43, 1.96MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:10<02:58, 2.44MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:10<02:29, 2.92MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:10<02:06, 3.45MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:11<17:31, 414kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:11<13:09, 550kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:11<09:46, 740kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:11<08:00, 903kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:11<06:09, 1.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:11<04:48, 1.50MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:11<03:53, 1.85MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:11<03:14, 2.22MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:11<02:38, 2.73MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:12<02:12, 3.26MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:13<06:40, 1.08MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:13<05:45, 1.25MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:13<04:31, 1.58MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:13<04:50, 1.48MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:13<04:58, 1.44MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:13<04:29, 1.59MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:13<04:10, 1.71MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:13<03:56, 1.81MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:13<03:39, 1.96MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:14<03:17, 2.17MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:14<02:52, 2.49MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:14<02:21, 3.01MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:14<01:59, 3.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:15<05:39, 1.26MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:15<04:25, 1.60MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:15<03:30, 2.02MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:15<03:20, 2.12MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:15<03:05, 2.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:15<02:58, 2.38MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:15<03:02, 2.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:15<02:54, 2.43MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:15<02:25, 2.92MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:16<02:00, 3.51MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:18<08:58, 784kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:18<07:17, 965kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:18<05:30, 1.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:18<04:27, 1.57MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:18<03:25, 2.05MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:18<02:41, 2.60MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:18<02:10, 3.20MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:18<01:49, 3.82MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:20<30:16, 230kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:20<22:12, 314kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:20<15:47, 440kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:21<11:18, 614kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:21<08:09, 849kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:21<05:57, 1.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:22<06:40, 1.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:22<05:15, 1.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:22<04:04, 1.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:22<03:29, 1.97MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:22<02:47, 2.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:22<02:11, 3.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:22<01:47, 3.84MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:24<05:38, 1.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:24<06:51, 996kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:24<06:48, 1.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:24<07:01, 972kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:24<07:07, 958kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:24<06:40, 1.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:24<04:59, 1.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:24<03:43, 1.82MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:25<02:50, 2.39MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:25<02:12, 3.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:27<13:56, 485kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:27<10:43, 631kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:27<07:52, 858kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:27<05:55, 1.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:27<04:22, 1.54MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:27<03:17, 2.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:28<02:31, 2.65MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:29<12:14, 547kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:29<11:23, 588kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:29<08:37, 776kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:29<07:24, 903kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:29<06:41, 1.00MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:29<06:10, 1.08MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:30<05:34, 1.20MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:30<04:58, 1.34MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:30<04:03, 1.64MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:30<03:09, 2.11MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:30<02:22, 2.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:31<03:55, 1.69MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:31<03:21, 1.98MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:31<02:36, 2.54MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:31<02:33, 2.59MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:31<02:32, 2.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:31<02:18, 2.85MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:31<02:05, 3.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:32<01:56, 3.40MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:32<01:47, 3.66MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:32<01:32, 4.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:33<24:13, 271kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:33<19:00, 345kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:33<14:23, 455kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:33<11:52, 552kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:33<10:26, 627kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:33<09:07, 718kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:33<08:02, 815kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:34<07:07, 918kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:34<06:15, 1.04MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:34<05:36, 1.17MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:34<04:48, 1.36MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:34<03:28, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:35<03:50, 1.69MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:35<03:08, 2.06MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:35<02:43, 2.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:35<02:33, 2.53MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:35<02:29, 2.60MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:35<02:25, 2.66MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:35<02:35, 2.50MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:36<02:15, 2.86MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:36<02:04, 3.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:36<01:51, 3.47MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:36<01:42, 3.76MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:37<06:03, 1.06MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:37<05:00, 1.28MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:37<03:49, 1.67MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:37<04:53, 1.31MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:38<04:09, 1.54MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:38<03:06, 2.06MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:38<02:20, 2.73MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:38<01:47, 3.55MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:39<06:59, 906kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:40<07:48, 812kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:40<07:41, 823kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:40<06:38, 952kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:40<05:01, 1.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:40<03:40, 1.72MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:40<02:42, 2.32MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:43<07:52, 796kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:43<06:22, 981kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:43<04:40, 1.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:43<03:35, 1.74MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:43<03:03, 2.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:43<02:38, 2.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:43<02:18, 2.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:43<02:03, 3.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:44<01:43, 3.60MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:46<20:09, 307kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:46<14:59, 413kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:46<10:41, 578kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:46<08:05, 764kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:46<05:54, 1.04MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:46<04:31, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:46<03:22, 1.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:48<06:07, 1.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:48<05:37, 1.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:48<04:12, 1.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:48<03:41, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:48<03:12, 1.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:48<02:35, 2.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:48<02:02, 2.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:48<01:42, 3.55MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:48<01:29, 4.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:50<16:54, 358kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:50<13:05, 463kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:50<09:20, 647kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:50<06:49, 885kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:50<05:02, 1.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:50<03:57, 1.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:50<02:57, 2.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:52<08:00, 748kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:52<06:51, 874kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:52<04:58, 1.20MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:52<03:39, 1.63MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:52<02:44, 2.17MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:52<02:14, 2.64MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:52<01:58, 3.00MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:54<50:23, 118kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:54<36:08, 164kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:54<25:24, 232kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:54<17:53, 329kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:54<12:38, 464kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:54<09:09, 639kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:55<30:23, 193kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:56<22:05, 265kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:56<15:33, 375kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:56<10:59, 528kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:56<07:48, 741kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:58<1:14:21, 77.8kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:58<52:26, 110kB/s]   .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:58<36:44, 157kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:58<25:46, 223kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:58<18:10, 315kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [04:00<20:14, 282kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [04:00<14:59, 381kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [04:00<10:36, 536kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [04:00<07:31, 753kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [04:00<05:25, 1.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [04:02<10:13, 552kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [04:02<07:58, 708kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [04:02<05:41, 987kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [04:02<04:06, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [04:02<02:59, 1.86MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [04:05<53:25, 104kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [04:05<38:08, 146kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [04:05<26:43, 208kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [04:05<18:44, 295kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:07<15:30, 355kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:07<13:32, 407kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:07<10:04, 546kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:07<07:53, 697kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:07<06:28, 848kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:07<05:11, 1.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:08<04:07, 1.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:08<03:23, 1.61MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:08<02:57, 1.85MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:08<02:40, 2.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:08<02:42, 2.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:08<02:29, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:08<02:20, 2.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:08<02:01, 2.68MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:09<02:49, 1.92MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:09<02:13, 2.44MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:09<01:38, 3.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:09<01:15, 4.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:11<14:37, 367kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:11<11:02, 486kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:11<07:53, 678kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:12<05:46, 924kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:12<04:14, 1.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:12<03:07, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:13<05:42, 929kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:13<06:17, 842kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:13<04:39, 1.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:13<03:36, 1.46MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:14<03:00, 1.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:14<02:33, 2.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:14<02:11, 2.40MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:14<01:39, 3.16MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:15<04:07, 1.27MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:15<03:26, 1.52MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:15<02:28, 2.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:17<03:20, 1.55MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:17<03:07, 1.65MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:17<02:16, 2.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:17<01:43, 2.96MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:18<01:22, 3.72MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:19<07:51, 648kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:19<06:29, 785kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:19<04:36, 1.10MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:19<03:19, 1.52MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:21<05:06, 984kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:21<04:48, 1.04MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:21<03:27, 1.45MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:21<02:29, 2.00MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:23<05:22, 922kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:23<04:30, 1.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:23<03:12, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:23<02:19, 2.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:25<28:36, 171kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:25<20:44, 236kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:25<14:28, 335kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:25<10:09, 474kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:27<1:44:44, 46.0kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:27<1:13:58, 65.1kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:27<51:25, 92.9kB/s]  .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:30<38:15, 124kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:30<27:05, 175kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:30<18:52, 249kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:32<14:56, 314kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:32<11:06, 421kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:32<07:48, 596kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:34<06:36, 699kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:34<05:16, 875kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:34<03:42, 1.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:36<04:21, 1.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:36<03:41, 1.23MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:36<02:41, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:36<01:58, 2.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:38<02:54, 1.54MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:38<02:37, 1.70MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:38<01:51, 2.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:40<03:29, 1.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:40<04:03, 1.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:40<03:11, 1.38MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:40<02:23, 1.83MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:40<01:45, 2.48MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:42<02:38, 1.64MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:42<02:29, 1.74MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:42<01:46, 2.42MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:44<03:20, 1.28MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:44<03:33, 1.20MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:44<02:36, 1.63MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:44<01:53, 2.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:44<01:24, 2.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:48<2:59:24, 23.4kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:48<2:06:04, 33.3kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:48<1:27:46, 47.5kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:48<1:01:11, 67.8kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:50<44:59, 91.9kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:50<33:40, 123kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:50<24:56, 166kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:50<19:09, 215kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:50<14:47, 279kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:50<11:54, 346kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:50<08:42, 473kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:50<06:08, 667kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:52<05:13, 777kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:52<03:57, 1.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:52<02:49, 1.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:52<02:04, 1.94MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:54<03:33, 1.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:54<03:59, 999kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:54<03:01, 1.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:54<02:14, 1.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:54<01:41, 2.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:54<01:18, 3.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:56<04:51, 808kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:57<03:44, 1.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:57<02:39, 1.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:58<03:06, 1.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:59<02:44, 1.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:59<01:56, 1.97MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [05:00<02:58, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [05:00<02:38, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [05:01<01:53, 1.98MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [05:01<01:23, 2.68MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [05:02<05:37, 661kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [05:03<04:12, 882kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [05:03<02:57, 1.24MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [05:04<03:53, 938kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [05:04<03:11, 1.14MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [05:05<02:14, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [05:07<03:47, 946kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [05:07<03:09, 1.13MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [05:07<02:13, 1.59MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [05:09<02:47, 1.26MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [05:09<03:16, 1.08MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [05:09<02:32, 1.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [05:09<01:50, 1.89MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [05:11<02:13, 1.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [05:11<02:02, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [05:11<01:26, 2.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:13<03:46, 894kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:13<03:35, 938kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:13<02:36, 1.29MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:13<01:50, 1.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:15<03:24, 970kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:15<02:41, 1.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:15<01:54, 1.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:17<02:21, 1.38MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:17<02:03, 1.57MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:17<01:33, 2.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:21<01:46, 1.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:21<01:36, 1.94MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:21<01:07, 2.70MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:23<05:43, 532kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:23<04:24, 688kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:23<03:03, 974kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:25<04:07, 722kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:25<03:32, 839kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:25<02:31, 1.17MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:27<02:20, 1.24MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:27<02:29, 1.16MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:27<01:51, 1.55MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:27<01:18, 2.17MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:33<11:55, 238kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:33<09:11, 308kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:33<06:43, 421kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:33<04:50, 583kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:33<03:29, 803kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:34<02:43, 1.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:34<01:52, 1.44MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:36<05:13, 516kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:36<03:52, 695kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:38<03:02, 865kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:38<02:16, 1.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:40<01:57, 1.31MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:40<01:30, 1.69MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:42<01:25, 1.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:42<01:11, 2.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:44<01:12, 2.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:44<01:12, 2.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:44<00:51, 2.77MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:46<01:26, 1.63MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:46<01:18, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:46<00:55, 2.49MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:48<02:21, 968kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:48<01:54, 1.20MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:50<01:40, 1.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:50<01:30, 1.46MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:50<01:03, 2.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:52<01:34, 1.36MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:52<01:20, 1.59MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:52<00:56, 2.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:55<02:07, 978kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:55<01:46, 1.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:55<01:13, 1.64MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:56<01:52, 1.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:56<01:29, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:56<01:05, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:56<00:47, 2.46MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:58<01:23, 1.40MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:58<01:12, 1.60MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:58<00:52, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:58<00:38, 2.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [06:00<01:33, 1.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [06:00<01:15, 1.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [06:00<00:53, 2.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [06:02<01:15, 1.43MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [06:02<01:09, 1.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [06:02<00:49, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [06:02<00:35, 2.93MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [06:04<06:59, 248kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [06:04<05:08, 337kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [06:04<03:32, 478kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [06:06<02:55, 570kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [06:06<02:15, 736kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [06:06<01:34, 1.03MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [06:06<01:07, 1.44MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [06:08<02:09, 742kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [06:08<01:44, 915kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [06:08<01:13, 1.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [06:10<01:18, 1.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [06:10<01:08, 1.33MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [06:10<00:48, 1.84MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [06:12<00:53, 1.63MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [06:12<00:47, 1.84MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [06:12<00:33, 2.52MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [06:13<00:25, 3.29MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [06:14<01:00, 1.38MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [06:14<00:54, 1.53MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [06:14<00:39, 2.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [06:14<00:28, 2.82MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [06:16<01:10, 1.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [06:16<00:57, 1.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:16<00:41, 1.88MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:16<00:29, 2.54MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [06:18<01:12, 1.04MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [06:18<01:02, 1.21MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [06:18<00:44, 1.66MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:18<00:31, 2.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:20<00:58, 1.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:20<00:49, 1.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:20<00:35, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:20<00:25, 2.64MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:22<00:47, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:22<00:46, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:22<00:35, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:22<00:26, 2.42MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:22<00:20, 3.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:23<00:16, 3.84MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:24<00:49, 1.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:24<00:46, 1.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:24<00:34, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:24<00:26, 2.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:24<00:20, 2.97MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:25<00:15, 3.83MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:27<02:31, 387kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:27<01:54, 511kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:27<01:19, 717kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:27<00:55, 1.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:29<01:04, 849kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:29<01:05, 833kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:29<00:48, 1.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:29<00:35, 1.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:30<00:25, 2.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:30<00:19, 2.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:31<00:46, 1.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:31<00:36, 1.35MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:31<00:32, 1.54MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:32<09:33, 86.0kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:32<06:15, 123kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:34<04:35, 163kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:34<03:12, 230kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:36<02:10, 312kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:36<01:40, 402kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:36<01:08, 568kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:40<01:03, 576kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:40<00:49, 732kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:40<00:33, 1.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:40<00:23, 1.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:42<00:33, 957kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:42<00:34, 933kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:42<00:25, 1.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:42<00:18, 1.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:42<00:13, 2.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:42<00:09, 2.94MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:44<00:44, 637kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:44<00:45, 620kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:44<00:33, 837kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:44<00:21, 1.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:46<00:31, 766kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:46<00:35, 677kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:46<00:27, 855kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:46<00:17, 1.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:48<00:19, 1.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:48<00:15, 1.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:48<00:10, 1.72MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:48<00:07, 2.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:48<00:05, 2.96MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:50<00:24, 643kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:50<00:21, 748kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:50<00:13, 1.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:50<00:08, 1.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:52<00:11, 1.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:52<00:10, 1.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:52<00:07, 1.51MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:52<00:04, 2.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:52<00:03, 2.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:54<00:05, 1.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:54<00:04, 1.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:54<00:02, 2.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:54<00:01, 2.97MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:56<00:02, 1.51MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:56<00:01, 1.64MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:56<00:00, 2.24MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:56<00:00, 2.91MB/s].vector_cache/glove.6B.zip: 862MB [06:56, 2.07MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 858/400000 [00:00<00:46, 8576.02it/s]  0%|          | 1762/400000 [00:00<00:45, 8709.59it/s]  1%|          | 2611/400000 [00:00<00:45, 8640.07it/s]  1%|          | 3536/400000 [00:00<00:44, 8812.61it/s]  1%|          | 4438/400000 [00:00<00:44, 8873.09it/s]  1%|▏         | 5311/400000 [00:00<00:44, 8829.24it/s]  2%|▏         | 6223/400000 [00:00<00:44, 8913.46it/s]  2%|▏         | 7161/400000 [00:00<00:43, 9046.29it/s]  2%|▏         | 8133/400000 [00:00<00:42, 9238.16it/s]  2%|▏         | 9024/400000 [00:01<00:43, 8972.73it/s]  2%|▏         | 9900/400000 [00:01<00:43, 8884.22it/s]  3%|▎         | 10826/400000 [00:01<00:43, 8993.41it/s]  3%|▎         | 11733/400000 [00:01<00:43, 9015.73it/s]  3%|▎         | 12628/400000 [00:01<00:43, 8990.36it/s]  3%|▎         | 13545/400000 [00:01<00:42, 9042.05it/s]  4%|▎         | 14460/400000 [00:01<00:42, 9072.43it/s]  4%|▍         | 15385/400000 [00:01<00:42, 9123.56it/s]  4%|▍         | 16327/400000 [00:01<00:41, 9203.26it/s]  4%|▍         | 17247/400000 [00:01<00:42, 9079.49it/s]  5%|▍         | 18174/400000 [00:02<00:41, 9134.67it/s]  5%|▍         | 19101/400000 [00:02<00:41, 9173.40it/s]  5%|▌         | 20033/400000 [00:02<00:41, 9215.77it/s]  5%|▌         | 20963/400000 [00:02<00:41, 9237.78it/s]  5%|▌         | 21887/400000 [00:02<00:41, 9123.44it/s]  6%|▌         | 22800/400000 [00:02<00:42, 8913.34it/s]  6%|▌         | 23710/400000 [00:02<00:41, 8968.32it/s]  6%|▌         | 24621/400000 [00:02<00:41, 9008.17it/s]  6%|▋         | 25543/400000 [00:02<00:41, 9068.25it/s]  7%|▋         | 26451/400000 [00:02<00:41, 9021.52it/s]  7%|▋         | 27401/400000 [00:03<00:40, 9158.13it/s]  7%|▋         | 28324/400000 [00:03<00:40, 9177.23it/s]  7%|▋         | 29286/400000 [00:03<00:39, 9304.15it/s]  8%|▊         | 30218/400000 [00:03<00:39, 9298.79it/s]  8%|▊         | 31194/400000 [00:03<00:39, 9431.23it/s]  8%|▊         | 32138/400000 [00:03<00:39, 9333.68it/s]  8%|▊         | 33073/400000 [00:03<00:39, 9286.96it/s]  9%|▊         | 34020/400000 [00:03<00:39, 9340.51it/s]  9%|▊         | 34980/400000 [00:03<00:38, 9415.49it/s]  9%|▉         | 35981/400000 [00:03<00:37, 9584.42it/s]  9%|▉         | 36963/400000 [00:04<00:37, 9652.32it/s]  9%|▉         | 37930/400000 [00:04<00:38, 9526.88it/s] 10%|▉         | 38911/400000 [00:04<00:37, 9609.18it/s] 10%|▉         | 39873/400000 [00:04<00:37, 9585.49it/s] 10%|█         | 40872/400000 [00:04<00:37, 9702.94it/s] 10%|█         | 41844/400000 [00:04<00:36, 9683.96it/s] 11%|█         | 42813/400000 [00:04<00:37, 9560.49it/s] 11%|█         | 43832/400000 [00:04<00:36, 9740.19it/s] 11%|█         | 44808/400000 [00:04<00:36, 9642.18it/s] 11%|█▏        | 45774/400000 [00:04<00:36, 9630.72it/s] 12%|█▏        | 46738/400000 [00:05<00:37, 9410.18it/s] 12%|█▏        | 47681/400000 [00:05<00:37, 9389.78it/s] 12%|█▏        | 48622/400000 [00:05<00:38, 9154.06it/s] 12%|█▏        | 49540/400000 [00:05<00:38, 9121.88it/s] 13%|█▎        | 50521/400000 [00:05<00:37, 9316.16it/s] 13%|█▎        | 51458/400000 [00:05<00:37, 9330.24it/s] 13%|█▎        | 52393/400000 [00:05<00:37, 9290.37it/s] 13%|█▎        | 53337/400000 [00:05<00:37, 9332.47it/s] 14%|█▎        | 54313/400000 [00:05<00:36, 9454.41it/s] 14%|█▍        | 55294/400000 [00:05<00:36, 9558.03it/s] 14%|█▍        | 56270/400000 [00:06<00:35, 9615.75it/s] 14%|█▍        | 57233/400000 [00:06<00:35, 9585.86it/s] 15%|█▍        | 58193/400000 [00:06<00:36, 9379.91it/s] 15%|█▍        | 59133/400000 [00:06<00:36, 9363.49it/s] 15%|█▌        | 60136/400000 [00:06<00:35, 9552.00it/s] 15%|█▌        | 61093/400000 [00:06<00:35, 9541.45it/s] 16%|█▌        | 62049/400000 [00:06<00:35, 9489.34it/s] 16%|█▌        | 63039/400000 [00:06<00:35, 9607.42it/s] 16%|█▌        | 64001/400000 [00:06<00:35, 9583.59it/s] 16%|█▌        | 64967/400000 [00:06<00:34, 9605.26it/s] 16%|█▋        | 65944/400000 [00:07<00:34, 9652.47it/s] 17%|█▋        | 66910/400000 [00:07<00:34, 9619.39it/s] 17%|█▋        | 67873/400000 [00:07<00:34, 9619.22it/s] 17%|█▋        | 68843/400000 [00:07<00:34, 9640.75it/s] 17%|█▋        | 69831/400000 [00:07<00:34, 9710.09it/s] 18%|█▊        | 70803/400000 [00:07<00:34, 9652.65it/s] 18%|█▊        | 71769/400000 [00:07<00:34, 9582.30it/s] 18%|█▊        | 72729/400000 [00:07<00:34, 9587.50it/s] 18%|█▊        | 73688/400000 [00:07<00:34, 9568.43it/s] 19%|█▊        | 74713/400000 [00:07<00:33, 9761.62it/s] 19%|█▉        | 75727/400000 [00:08<00:32, 9871.34it/s] 19%|█▉        | 76716/400000 [00:08<00:33, 9614.11it/s] 19%|█▉        | 77680/400000 [00:08<00:33, 9591.94it/s] 20%|█▉        | 78641/400000 [00:08<00:33, 9554.06it/s] 20%|█▉        | 79683/400000 [00:08<00:32, 9796.45it/s] 20%|██        | 80665/400000 [00:08<00:33, 9606.23it/s] 20%|██        | 81629/400000 [00:08<00:33, 9433.71it/s] 21%|██        | 82640/400000 [00:08<00:32, 9625.16it/s] 21%|██        | 83634/400000 [00:08<00:32, 9715.81it/s] 21%|██        | 84608/400000 [00:09<00:32, 9713.97it/s] 21%|██▏       | 85588/400000 [00:09<00:32, 9736.51it/s] 22%|██▏       | 86570/400000 [00:09<00:32, 9760.19it/s] 22%|██▏       | 87553/400000 [00:09<00:31, 9779.25it/s] 22%|██▏       | 88532/400000 [00:09<00:32, 9718.84it/s] 22%|██▏       | 89505/400000 [00:09<00:32, 9651.05it/s] 23%|██▎       | 90496/400000 [00:09<00:31, 9724.43it/s] 23%|██▎       | 91469/400000 [00:09<00:32, 9511.63it/s] 23%|██▎       | 92431/400000 [00:09<00:32, 9540.69it/s] 23%|██▎       | 93387/400000 [00:09<00:32, 9529.89it/s] 24%|██▎       | 94341/400000 [00:10<00:32, 9526.81it/s] 24%|██▍       | 95298/400000 [00:10<00:31, 9536.84it/s] 24%|██▍       | 96253/400000 [00:10<00:32, 9374.38it/s] 24%|██▍       | 97192/400000 [00:10<00:32, 9298.94it/s] 25%|██▍       | 98123/400000 [00:10<00:32, 9294.83it/s] 25%|██▍       | 99110/400000 [00:10<00:31, 9459.19it/s] 25%|██▌       | 100082/400000 [00:10<00:31, 9535.35it/s] 25%|██▌       | 101037/400000 [00:10<00:31, 9376.38it/s] 26%|██▌       | 102031/400000 [00:10<00:31, 9536.89it/s] 26%|██▌       | 102997/400000 [00:10<00:31, 9569.62it/s] 26%|██▌       | 103992/400000 [00:11<00:30, 9679.35it/s] 26%|██▌       | 104962/400000 [00:11<00:30, 9623.23it/s] 26%|██▋       | 105926/400000 [00:11<00:30, 9497.72it/s] 27%|██▋       | 106877/400000 [00:11<00:31, 9454.61it/s] 27%|██▋       | 107826/400000 [00:11<00:30, 9462.06it/s] 27%|██▋       | 108773/400000 [00:11<00:30, 9444.20it/s] 27%|██▋       | 109718/400000 [00:11<00:30, 9417.78it/s] 28%|██▊       | 110687/400000 [00:11<00:30, 9496.40it/s] 28%|██▊       | 111640/400000 [00:11<00:30, 9504.29it/s] 28%|██▊       | 112591/400000 [00:11<00:30, 9472.37it/s] 28%|██▊       | 113539/400000 [00:12<00:30, 9458.04it/s] 29%|██▊       | 114518/400000 [00:12<00:29, 9552.72it/s] 29%|██▉       | 115481/400000 [00:12<00:29, 9572.93it/s] 29%|██▉       | 116472/400000 [00:12<00:29, 9669.08it/s] 29%|██▉       | 117440/400000 [00:12<00:29, 9666.33it/s] 30%|██▉       | 118407/400000 [00:12<00:29, 9460.64it/s] 30%|██▉       | 119355/400000 [00:12<00:30, 9324.27it/s] 30%|███       | 120289/400000 [00:12<00:30, 9128.22it/s] 30%|███       | 121204/400000 [00:12<00:30, 9102.95it/s] 31%|███       | 122116/400000 [00:12<00:30, 9071.59it/s] 31%|███       | 123046/400000 [00:13<00:30, 9136.74it/s] 31%|███       | 123978/400000 [00:13<00:30, 9189.58it/s] 31%|███       | 124898/400000 [00:13<00:30, 9021.89it/s] 31%|███▏      | 125821/400000 [00:13<00:30, 9083.17it/s] 32%|███▏      | 126735/400000 [00:13<00:30, 9098.14it/s] 32%|███▏      | 127665/400000 [00:13<00:29, 9157.18it/s] 32%|███▏      | 128582/400000 [00:13<00:30, 9010.37it/s] 32%|███▏      | 129484/400000 [00:13<00:30, 8987.39it/s] 33%|███▎      | 130398/400000 [00:13<00:29, 9030.01it/s] 33%|███▎      | 131313/400000 [00:13<00:29, 9063.67it/s] 33%|███▎      | 132258/400000 [00:14<00:29, 9174.82it/s] 33%|███▎      | 133207/400000 [00:14<00:28, 9265.94it/s] 34%|███▎      | 134135/400000 [00:14<00:28, 9182.46it/s] 34%|███▍      | 135066/400000 [00:14<00:28, 9217.68it/s] 34%|███▍      | 136006/400000 [00:14<00:28, 9268.96it/s] 34%|███▍      | 136935/400000 [00:14<00:28, 9274.48it/s] 34%|███▍      | 137871/400000 [00:14<00:28, 9299.70it/s] 35%|███▍      | 138802/400000 [00:14<00:28, 9213.18it/s] 35%|███▍      | 139739/400000 [00:14<00:28, 9258.74it/s] 35%|███▌      | 140700/400000 [00:14<00:27, 9359.65it/s] 35%|███▌      | 141656/400000 [00:15<00:27, 9417.20it/s] 36%|███▌      | 142618/400000 [00:15<00:27, 9475.79it/s] 36%|███▌      | 143566/400000 [00:15<00:27, 9222.53it/s] 36%|███▌      | 144509/400000 [00:15<00:27, 9283.02it/s] 36%|███▋      | 145439/400000 [00:15<00:27, 9238.31it/s] 37%|███▋      | 146364/400000 [00:15<00:27, 9209.92it/s] 37%|███▋      | 147286/400000 [00:15<00:27, 9129.18it/s] 37%|███▋      | 148200/400000 [00:15<00:27, 9105.63it/s] 37%|███▋      | 149143/400000 [00:15<00:27, 9200.19it/s] 38%|███▊      | 150064/400000 [00:16<00:27, 8955.98it/s] 38%|███▊      | 150962/400000 [00:16<00:27, 8925.71it/s] 38%|███▊      | 151869/400000 [00:16<00:27, 8968.22it/s] 38%|███▊      | 152771/400000 [00:16<00:27, 8982.93it/s] 38%|███▊      | 153670/400000 [00:16<00:27, 8956.11it/s] 39%|███▊      | 154592/400000 [00:16<00:27, 9031.05it/s] 39%|███▉      | 155534/400000 [00:16<00:26, 9143.88it/s] 39%|███▉      | 156450/400000 [00:16<00:26, 9117.00it/s] 39%|███▉      | 157363/400000 [00:16<00:26, 9022.10it/s] 40%|███▉      | 158341/400000 [00:16<00:26, 9234.87it/s] 40%|███▉      | 159267/400000 [00:17<00:26, 9190.03it/s] 40%|████      | 160188/400000 [00:17<00:26, 9156.31it/s] 40%|████      | 161116/400000 [00:17<00:25, 9191.63it/s] 41%|████      | 162036/400000 [00:17<00:26, 9014.30it/s] 41%|████      | 162973/400000 [00:17<00:26, 9116.38it/s] 41%|████      | 163886/400000 [00:17<00:25, 9116.53it/s] 41%|████      | 164826/400000 [00:17<00:25, 9198.44it/s] 41%|████▏     | 165761/400000 [00:17<00:25, 9241.97it/s] 42%|████▏     | 166693/400000 [00:17<00:25, 9263.36it/s] 42%|████▏     | 167628/400000 [00:17<00:25, 9287.78it/s] 42%|████▏     | 168558/400000 [00:18<00:24, 9279.44it/s] 42%|████▏     | 169490/400000 [00:18<00:24, 9290.72it/s] 43%|████▎     | 170420/400000 [00:18<00:24, 9221.87it/s] 43%|████▎     | 171343/400000 [00:18<00:25, 9010.30it/s] 43%|████▎     | 172246/400000 [00:18<00:25, 8964.97it/s] 43%|████▎     | 173152/400000 [00:18<00:25, 8992.34it/s] 44%|████▎     | 174079/400000 [00:18<00:24, 9072.19it/s] 44%|████▍     | 175031/400000 [00:18<00:24, 9200.69it/s] 44%|████▍     | 175952/400000 [00:18<00:24, 9147.97it/s] 44%|████▍     | 176940/400000 [00:18<00:23, 9354.92it/s] 44%|████▍     | 177878/400000 [00:19<00:23, 9330.38it/s] 45%|████▍     | 178813/400000 [00:19<00:23, 9284.18it/s] 45%|████▍     | 179767/400000 [00:19<00:23, 9358.68it/s] 45%|████▌     | 180704/400000 [00:19<00:23, 9275.84it/s] 45%|████▌     | 181692/400000 [00:19<00:23, 9449.02it/s] 46%|████▌     | 182652/400000 [00:19<00:22, 9492.77it/s] 46%|████▌     | 183603/400000 [00:19<00:22, 9478.83it/s] 46%|████▌     | 184552/400000 [00:19<00:22, 9396.10it/s] 46%|████▋     | 185493/400000 [00:19<00:22, 9379.75it/s] 47%|████▋     | 186432/400000 [00:19<00:22, 9297.78it/s] 47%|████▋     | 187363/400000 [00:20<00:22, 9268.73it/s] 47%|████▋     | 188296/400000 [00:20<00:22, 9284.94it/s] 47%|████▋     | 189225/400000 [00:20<00:23, 9133.12it/s] 48%|████▊     | 190204/400000 [00:20<00:22, 9320.60it/s] 48%|████▊     | 191166/400000 [00:20<00:22, 9405.40it/s] 48%|████▊     | 192141/400000 [00:20<00:21, 9503.33it/s] 48%|████▊     | 193093/400000 [00:20<00:21, 9461.72it/s] 49%|████▊     | 194040/400000 [00:20<00:22, 9268.84it/s] 49%|████▊     | 194969/400000 [00:20<00:22, 9225.26it/s] 49%|████▉     | 195985/400000 [00:20<00:21, 9484.44it/s] 49%|████▉     | 196936/400000 [00:21<00:21, 9358.60it/s] 49%|████▉     | 197951/400000 [00:21<00:21, 9582.66it/s] 50%|████▉     | 198943/400000 [00:21<00:20, 9679.73it/s] 50%|████▉     | 199915/400000 [00:21<00:20, 9691.48it/s] 50%|█████     | 200891/400000 [00:21<00:20, 9709.64it/s] 50%|█████     | 201897/400000 [00:21<00:20, 9810.88it/s] 51%|█████     | 202899/400000 [00:21<00:19, 9870.64it/s] 51%|█████     | 203887/400000 [00:21<00:19, 9849.21it/s] 51%|█████     | 204873/400000 [00:21<00:20, 9624.00it/s] 51%|█████▏    | 205837/400000 [00:22<00:20, 9521.96it/s] 52%|█████▏    | 206791/400000 [00:22<00:20, 9499.28it/s] 52%|█████▏    | 207746/400000 [00:22<00:20, 9510.43it/s] 52%|█████▏    | 208698/400000 [00:22<00:20, 9268.62it/s] 52%|█████▏    | 209627/400000 [00:22<00:20, 9252.86it/s] 53%|█████▎    | 210556/400000 [00:22<00:20, 9261.25it/s] 53%|█████▎    | 211515/400000 [00:22<00:20, 9355.11it/s] 53%|█████▎    | 212459/400000 [00:22<00:19, 9379.58it/s] 53%|█████▎    | 213452/400000 [00:22<00:19, 9535.23it/s] 54%|█████▎    | 214418/400000 [00:22<00:19, 9570.86it/s] 54%|█████▍    | 215403/400000 [00:23<00:19, 9651.48it/s] 54%|█████▍    | 216395/400000 [00:23<00:18, 9728.46it/s] 54%|█████▍    | 217369/400000 [00:23<00:18, 9681.37it/s] 55%|█████▍    | 218338/400000 [00:23<00:18, 9602.31it/s] 55%|█████▍    | 219329/400000 [00:23<00:18, 9691.20it/s] 55%|█████▌    | 220331/400000 [00:23<00:18, 9779.25it/s] 55%|█████▌    | 221310/400000 [00:23<00:18, 9685.70it/s] 56%|█████▌    | 222280/400000 [00:23<00:18, 9538.61it/s] 56%|█████▌    | 223275/400000 [00:23<00:18, 9657.43it/s] 56%|█████▌    | 224242/400000 [00:23<00:18, 9611.15it/s] 56%|█████▋    | 225204/400000 [00:24<00:18, 9603.56it/s] 57%|█████▋    | 226165/400000 [00:24<00:18, 9368.48it/s] 57%|█████▋    | 227114/400000 [00:24<00:18, 9402.50it/s] 57%|█████▋    | 228075/400000 [00:24<00:18, 9463.60it/s] 57%|█████▋    | 229061/400000 [00:24<00:17, 9578.76it/s] 58%|█████▊    | 230057/400000 [00:24<00:17, 9688.66it/s] 58%|█████▊    | 231027/400000 [00:24<00:17, 9651.57it/s] 58%|█████▊    | 231993/400000 [00:24<00:17, 9579.83it/s] 58%|█████▊    | 232961/400000 [00:24<00:17, 9607.85it/s] 58%|█████▊    | 233923/400000 [00:24<00:17, 9286.93it/s] 59%|█████▊    | 234855/400000 [00:25<00:18, 9010.79it/s] 59%|█████▉    | 235779/400000 [00:25<00:18, 9075.75it/s] 59%|█████▉    | 236704/400000 [00:25<00:17, 9126.62it/s] 59%|█████▉    | 237627/400000 [00:25<00:17, 9156.97it/s] 60%|█████▉    | 238545/400000 [00:25<00:17, 9042.11it/s] 60%|█████▉    | 239562/400000 [00:25<00:17, 9351.68it/s] 60%|██████    | 240600/400000 [00:25<00:16, 9637.73it/s] 60%|██████    | 241569/400000 [00:25<00:16, 9617.04it/s] 61%|██████    | 242535/400000 [00:25<00:16, 9551.56it/s] 61%|██████    | 243495/400000 [00:25<00:16, 9562.98it/s] 61%|██████    | 244453/400000 [00:26<00:16, 9485.04it/s] 61%|██████▏   | 245426/400000 [00:26<00:16, 9555.44it/s] 62%|██████▏   | 246422/400000 [00:26<00:15, 9670.23it/s] 62%|██████▏   | 247391/400000 [00:26<00:15, 9605.26it/s] 62%|██████▏   | 248353/400000 [00:26<00:15, 9540.76it/s] 62%|██████▏   | 249346/400000 [00:26<00:15, 9652.95it/s] 63%|██████▎   | 250313/400000 [00:26<00:15, 9583.14it/s] 63%|██████▎   | 251272/400000 [00:26<00:15, 9505.32it/s] 63%|██████▎   | 252224/400000 [00:26<00:15, 9423.73it/s] 63%|██████▎   | 253167/400000 [00:26<00:15, 9261.03it/s] 64%|██████▎   | 254095/400000 [00:27<00:15, 9219.78it/s] 64%|██████▍   | 255018/400000 [00:27<00:15, 9219.27it/s] 64%|██████▍   | 255941/400000 [00:27<00:15, 9168.75it/s] 64%|██████▍   | 256859/400000 [00:27<00:15, 9145.25it/s] 64%|██████▍   | 257774/400000 [00:27<00:15, 8947.31it/s] 65%|██████▍   | 258670/400000 [00:27<00:15, 8927.14it/s] 65%|██████▍   | 259627/400000 [00:27<00:15, 9109.53it/s] 65%|██████▌   | 260594/400000 [00:27<00:15, 9270.38it/s] 65%|██████▌   | 261523/400000 [00:27<00:15, 9226.69it/s] 66%|██████▌   | 262447/400000 [00:28<00:15, 9131.91it/s] 66%|██████▌   | 263362/400000 [00:28<00:15, 9095.33it/s] 66%|██████▌   | 264273/400000 [00:28<00:14, 9068.76it/s] 66%|██████▋   | 265181/400000 [00:28<00:14, 8994.29it/s] 67%|██████▋   | 266096/400000 [00:28<00:14, 9038.38it/s] 67%|██████▋   | 267001/400000 [00:28<00:14, 8997.99it/s] 67%|██████▋   | 267913/400000 [00:28<00:14, 9031.71it/s] 67%|██████▋   | 268820/400000 [00:28<00:14, 9041.21it/s] 67%|██████▋   | 269725/400000 [00:28<00:14, 8924.25it/s] 68%|██████▊   | 270637/400000 [00:28<00:14, 8981.49it/s] 68%|██████▊   | 271537/400000 [00:29<00:14, 8983.98it/s] 68%|██████▊   | 272436/400000 [00:29<00:14, 8930.58it/s] 68%|██████▊   | 273330/400000 [00:29<00:14, 8873.58it/s] 69%|██████▊   | 274229/400000 [00:29<00:14, 8904.86it/s] 69%|██████▉   | 275172/400000 [00:29<00:13, 9055.33it/s] 69%|██████▉   | 276096/400000 [00:29<00:13, 9107.44it/s] 69%|██████▉   | 277008/400000 [00:29<00:13, 9066.47it/s] 69%|██████▉   | 277940/400000 [00:29<00:13, 9139.28it/s] 70%|██████▉   | 278855/400000 [00:29<00:13, 9088.47it/s] 70%|██████▉   | 279765/400000 [00:29<00:13, 8742.06it/s] 70%|███████   | 280643/400000 [00:30<00:13, 8704.45it/s] 70%|███████   | 281543/400000 [00:30<00:13, 8789.89it/s] 71%|███████   | 282424/400000 [00:30<00:13, 8786.66it/s] 71%|███████   | 283304/400000 [00:30<00:13, 8781.81it/s] 71%|███████   | 284184/400000 [00:30<00:13, 8773.97it/s] 71%|███████▏  | 285075/400000 [00:30<00:13, 8813.14it/s] 71%|███████▏  | 285974/400000 [00:30<00:12, 8863.28it/s] 72%|███████▏  | 286863/400000 [00:30<00:12, 8868.71it/s] 72%|███████▏  | 287790/400000 [00:30<00:12, 8983.34it/s] 72%|███████▏  | 288751/400000 [00:30<00:12, 9160.88it/s] 72%|███████▏  | 289669/400000 [00:31<00:12, 9116.32it/s] 73%|███████▎  | 290582/400000 [00:31<00:12, 9064.56it/s] 73%|███████▎  | 291490/400000 [00:31<00:12, 9018.53it/s] 73%|███████▎  | 292393/400000 [00:31<00:11, 8988.35it/s] 73%|███████▎  | 293296/400000 [00:31<00:11, 8998.98it/s] 74%|███████▎  | 294197/400000 [00:31<00:11, 8946.05it/s] 74%|███████▍  | 295096/400000 [00:31<00:11, 8959.05it/s] 74%|███████▍  | 295999/400000 [00:31<00:11, 8979.55it/s] 74%|███████▍  | 296898/400000 [00:31<00:11, 8961.82it/s] 74%|███████▍  | 297797/400000 [00:31<00:11, 8968.03it/s] 75%|███████▍  | 298694/400000 [00:32<00:11, 8919.44it/s] 75%|███████▍  | 299606/400000 [00:32<00:11, 8976.20it/s] 75%|███████▌  | 300504/400000 [00:32<00:11, 8969.20it/s] 75%|███████▌  | 301402/400000 [00:32<00:11, 8939.23it/s] 76%|███████▌  | 302297/400000 [00:32<00:10, 8913.61it/s] 76%|███████▌  | 303193/400000 [00:32<00:10, 8891.80it/s] 76%|███████▌  | 304092/400000 [00:32<00:10, 8920.44it/s] 76%|███████▋  | 305033/400000 [00:32<00:10, 9061.70it/s] 76%|███████▋  | 305992/400000 [00:32<00:10, 9212.06it/s] 77%|███████▋  | 306915/400000 [00:32<00:10, 9168.15it/s] 77%|███████▋  | 307833/400000 [00:33<00:10, 9166.56it/s] 77%|███████▋  | 308789/400000 [00:33<00:09, 9279.01it/s] 77%|███████▋  | 309759/400000 [00:33<00:09, 9399.57it/s] 78%|███████▊  | 310747/400000 [00:33<00:09, 9538.01it/s] 78%|███████▊  | 311737/400000 [00:33<00:09, 9641.76it/s] 78%|███████▊  | 312703/400000 [00:33<00:09, 9611.23it/s] 78%|███████▊  | 313700/400000 [00:33<00:08, 9714.19it/s] 79%|███████▊  | 314673/400000 [00:33<00:08, 9653.44it/s] 79%|███████▉  | 315662/400000 [00:33<00:08, 9722.83it/s] 79%|███████▉  | 316635/400000 [00:33<00:08, 9697.90it/s] 79%|███████▉  | 317606/400000 [00:34<00:08, 9516.26it/s] 80%|███████▉  | 318559/400000 [00:34<00:08, 9332.65it/s] 80%|███████▉  | 319494/400000 [00:34<00:08, 9202.88it/s] 80%|████████  | 320416/400000 [00:34<00:08, 9156.63it/s] 80%|████████  | 321369/400000 [00:34<00:08, 9265.41it/s] 81%|████████  | 322316/400000 [00:34<00:08, 9324.43it/s] 81%|████████  | 323282/400000 [00:34<00:08, 9421.38it/s] 81%|████████  | 324225/400000 [00:34<00:08, 9343.97it/s] 81%|████████▏ | 325164/400000 [00:34<00:07, 9357.14it/s] 82%|████████▏ | 326122/400000 [00:35<00:07, 9421.77it/s] 82%|████████▏ | 327065/400000 [00:35<00:07, 9216.56it/s] 82%|████████▏ | 327988/400000 [00:35<00:07, 9172.09it/s] 82%|████████▏ | 328907/400000 [00:35<00:07, 9083.66it/s] 82%|████████▏ | 329843/400000 [00:35<00:07, 9162.01it/s] 83%|████████▎ | 330760/400000 [00:35<00:07, 9025.77it/s] 83%|████████▎ | 331664/400000 [00:35<00:07, 8911.45it/s] 83%|████████▎ | 332563/400000 [00:35<00:07, 8932.46it/s] 83%|████████▎ | 333457/400000 [00:35<00:07, 8717.32it/s] 84%|████████▎ | 334343/400000 [00:35<00:07, 8759.28it/s] 84%|████████▍ | 335221/400000 [00:36<00:07, 8685.51it/s] 84%|████████▍ | 336100/400000 [00:36<00:07, 8715.35it/s] 84%|████████▍ | 336989/400000 [00:36<00:07, 8765.37it/s] 84%|████████▍ | 337873/400000 [00:36<00:07, 8786.45it/s] 85%|████████▍ | 338783/400000 [00:36<00:06, 8875.58it/s] 85%|████████▍ | 339710/400000 [00:36<00:06, 8989.26it/s] 85%|████████▌ | 340648/400000 [00:36<00:06, 9102.39it/s] 85%|████████▌ | 341596/400000 [00:36<00:06, 9210.58it/s] 86%|████████▌ | 342518/400000 [00:36<00:06, 9204.23it/s] 86%|████████▌ | 343440/400000 [00:36<00:06, 9199.17it/s] 86%|████████▌ | 344361/400000 [00:37<00:06, 9102.26it/s] 86%|████████▋ | 345272/400000 [00:37<00:06, 9073.79it/s] 87%|████████▋ | 346183/400000 [00:37<00:05, 9083.68it/s] 87%|████████▋ | 347092/400000 [00:37<00:05, 9050.60it/s] 87%|████████▋ | 348022/400000 [00:37<00:05, 9122.52it/s] 87%|████████▋ | 348935/400000 [00:37<00:05, 9003.64it/s] 87%|████████▋ | 349841/400000 [00:37<00:05, 9014.90it/s] 88%|████████▊ | 350794/400000 [00:37<00:05, 9163.48it/s] 88%|████████▊ | 351777/400000 [00:37<00:05, 9351.41it/s] 88%|████████▊ | 352767/400000 [00:37<00:04, 9506.99it/s] 88%|████████▊ | 353773/400000 [00:38<00:04, 9663.93it/s] 89%|████████▊ | 354761/400000 [00:38<00:04, 9726.03it/s] 89%|████████▉ | 355735/400000 [00:38<00:04, 9651.86it/s] 89%|████████▉ | 356728/400000 [00:38<00:04, 9730.51it/s] 89%|████████▉ | 357775/400000 [00:38<00:04, 9939.69it/s] 90%|████████▉ | 358810/400000 [00:38<00:04, 10056.41it/s] 90%|████████▉ | 359818/400000 [00:38<00:04, 9942.39it/s]  90%|█████████ | 360814/400000 [00:38<00:03, 9929.73it/s] 90%|█████████ | 361808/400000 [00:38<00:03, 9916.69it/s] 91%|█████████ | 362801/400000 [00:38<00:03, 9739.96it/s] 91%|█████████ | 363779/400000 [00:39<00:03, 9751.05it/s] 91%|█████████ | 364755/400000 [00:39<00:03, 9603.42it/s] 91%|█████████▏| 365717/400000 [00:39<00:03, 9599.29it/s] 92%|█████████▏| 366678/400000 [00:39<00:03, 9271.49it/s] 92%|█████████▏| 367609/400000 [00:39<00:03, 9254.65it/s] 92%|█████████▏| 368584/400000 [00:39<00:03, 9397.13it/s] 92%|█████████▏| 369526/400000 [00:39<00:03, 9403.07it/s] 93%|█████████▎| 370517/400000 [00:39<00:03, 9549.27it/s] 93%|█████████▎| 371486/400000 [00:39<00:02, 9589.82it/s] 93%|█████████▎| 372447/400000 [00:39<00:02, 9543.92it/s] 93%|█████████▎| 373449/400000 [00:40<00:02, 9681.90it/s] 94%|█████████▎| 374419/400000 [00:40<00:02, 9592.11it/s] 94%|█████████▍| 375380/400000 [00:40<00:02, 9452.09it/s] 94%|█████████▍| 376361/400000 [00:40<00:02, 9556.04it/s] 94%|█████████▍| 377318/400000 [00:40<00:02, 9558.93it/s] 95%|█████████▍| 378345/400000 [00:40<00:02, 9760.86it/s] 95%|█████████▍| 379323/400000 [00:40<00:02, 9588.15it/s] 95%|█████████▌| 380284/400000 [00:40<00:02, 9526.85it/s] 95%|█████████▌| 381239/400000 [00:40<00:01, 9466.49it/s] 96%|█████████▌| 382224/400000 [00:41<00:01, 9575.80it/s] 96%|█████████▌| 383183/400000 [00:41<00:01, 9458.20it/s] 96%|█████████▌| 384130/400000 [00:41<00:01, 9337.53it/s] 96%|█████████▋| 385069/400000 [00:41<00:01, 9350.51it/s] 97%|█████████▋| 386037/400000 [00:41<00:01, 9446.23it/s] 97%|█████████▋| 387026/400000 [00:41<00:01, 9574.01it/s] 97%|█████████▋| 387985/400000 [00:41<00:01, 9512.48it/s] 97%|█████████▋| 388938/400000 [00:41<00:01, 9475.87it/s] 97%|█████████▋| 389931/400000 [00:41<00:01, 9606.39it/s] 98%|█████████▊| 390918/400000 [00:41<00:00, 9681.69it/s] 98%|█████████▊| 391935/400000 [00:42<00:00, 9821.53it/s] 98%|█████████▊| 392919/400000 [00:42<00:00, 9741.10it/s] 98%|█████████▊| 393894/400000 [00:42<00:00, 9724.08it/s] 99%|█████████▊| 394868/400000 [00:42<00:00, 9653.36it/s] 99%|█████████▉| 395834/400000 [00:42<00:00, 9619.41it/s] 99%|█████████▉| 396821/400000 [00:42<00:00, 9691.44it/s] 99%|█████████▉| 397791/400000 [00:42<00:00, 9670.12it/s]100%|█████████▉| 398759/400000 [00:42<00:00, 9579.34it/s]100%|█████████▉| 399718/400000 [00:42<00:00, 9462.44it/s]100%|█████████▉| 399999/400000 [00:42<00:00, 9332.19it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f71a593ac18> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011563800435209479 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.011062349563458293 	 Accuracy: 62

  model saves at 62% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15907 out of table with 15572 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15907 out of table with 15572 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-14 04:25:26.546152: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 04:25:26.550644: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095094999 Hz
2020-05-14 04:25:26.550786: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560387773b90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 04:25:26.550799: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f7152323da0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7126 - accuracy: 0.4970
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5976 - accuracy: 0.5045 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5235 - accuracy: 0.5093
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.5440 - accuracy: 0.5080
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5838 - accuracy: 0.5054
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.4980 - accuracy: 0.5110
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5155 - accuracy: 0.5099
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6091 - accuracy: 0.5038
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.5712 - accuracy: 0.5062
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5869 - accuracy: 0.5052
11000/25000 [============>.................] - ETA: 3s - loss: 7.5886 - accuracy: 0.5051
12000/25000 [=============>................] - ETA: 3s - loss: 7.6078 - accuracy: 0.5038
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6489 - accuracy: 0.5012
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6688 - accuracy: 0.4999
15000/25000 [=================>............] - ETA: 2s - loss: 7.6513 - accuracy: 0.5010
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6925 - accuracy: 0.4983
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7081 - accuracy: 0.4973
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6998 - accuracy: 0.4978
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6868 - accuracy: 0.4987
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6881 - accuracy: 0.4986
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6717 - accuracy: 0.4997
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6659 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6426 - accuracy: 0.5016
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
25000/25000 [==============================] - 7s 274us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f7112339b00> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f711351c0f0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4792 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.4102 - val_crf_viterbi_accuracy: 0.0133

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

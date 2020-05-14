
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f65abf39fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 05:12:30.136401
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 05:12:30.140063
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 05:12:30.142950
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 05:12:30.145837
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f65b7f51400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354343.8438
Epoch 2/10

1/1 [==============================] - 0s 108ms/step - loss: 258503.7500
Epoch 3/10

1/1 [==============================] - 0s 100ms/step - loss: 167779.4219
Epoch 4/10

1/1 [==============================] - 0s 104ms/step - loss: 94825.8516
Epoch 5/10

1/1 [==============================] - 0s 104ms/step - loss: 54544.1797
Epoch 6/10

1/1 [==============================] - 0s 101ms/step - loss: 33347.2500
Epoch 7/10

1/1 [==============================] - 0s 104ms/step - loss: 21798.0176
Epoch 8/10

1/1 [==============================] - 0s 105ms/step - loss: 15149.7129
Epoch 9/10

1/1 [==============================] - 0s 97ms/step - loss: 11118.8154
Epoch 10/10

1/1 [==============================] - 0s 102ms/step - loss: 8556.8848

  #### Inference Need return ypred, ytrue ######################### 
[[-0.8656982  -0.34061468 -0.3609035  -1.0724233   0.5121222  -1.0862973
  -0.08394504  1.1593304  -2.008974    0.21915027 -1.6559094   0.88506174
   1.8576323   1.4048681  -1.9984851  -0.46127954  1.0116975  -0.87423396
  -1.2056991   0.1545738   0.68218464 -0.9649814   1.1582878  -0.44078732
   0.8767093  -0.12869471 -1.6076882   0.52817285  0.64496666 -0.3887956
  -0.34373158  0.23189011  1.7443051  -1.4821048   0.32189634  0.19472912
  -0.5754969  -0.24963027 -0.94364464  0.7731123  -0.4209363   1.6134355
   1.878506    0.64307016 -0.63560534 -0.30449048  0.11992311 -0.9419292
   0.01305699 -0.75888723 -0.2997301   0.5984191  -1.6816759  -0.4080556
  -0.50013065  0.9758101  -0.16838807  0.8075261  -0.731949    0.45578635
  -1.8013204  -1.1837063  -0.79504305 -0.29479656 -0.6529188   1.0518003
   0.52268404  0.0908972   0.10935695  1.56601     0.6569682   0.33776543
  -1.1522274  -1.2360483  -0.03657645 -0.753694   -0.64705884  1.7458842
   0.5385698  -0.3299538  -0.56075037  2.0924087  -1.125332    0.631232
   0.63228875 -0.55841094  0.8991458  -1.016021   -0.16766965 -0.40595773
  -1.2776034   0.37439364 -1.2412596  -0.3176971   0.5110971  -0.7451706
   0.52958477  1.222983   -1.5997301  -0.18702641  0.83591914  0.01900266
   0.19364515 -0.28791392 -1.4864421   2.1239843   0.94740856 -0.41722074
   1.4985702  -0.709762   -0.7515149   0.06110662  1.3797678  -0.14275949
  -0.7820022   1.0021232   1.930906    0.9067467   0.78389317 -0.9035891
   0.18582559  7.143401    7.6740155   7.164341    6.481685    6.1542945
   6.7476215   7.3643513   7.721552    7.1754465   7.0050015   7.1611037
   8.21315     8.550603    7.4249077   6.42465     7.4198823   7.47025
   6.7303658   7.5203004   7.232463    7.8599167   6.4742947   5.0238194
   7.3308043   6.227511    8.890901    6.2817864   8.074282    7.826518
   9.220208    6.746784    7.8946056   7.801248    8.331376    8.359985
   6.263824    6.4655075   6.6378894   6.845978    6.5596647   6.7406797
   5.9297457   5.363227    7.3524494   8.441581    8.620426    8.236946
   6.4040556   7.330056    6.318096    7.531506    6.6771665   6.989807
   6.7925653   7.858162    7.054769    5.2725577   5.5205      9.304099
   0.17354798  2.1891446   0.48165703  0.62352264  0.6009691   0.7970145
   1.3968558   1.9155827   1.4532068   1.4414335   0.24131346  0.25869042
   0.2018708   0.54631406  1.8164072   0.24065495  0.21916091  1.8245984
   0.66329944  2.3880882   0.15465784  1.6469817   2.7816944   2.3399944
   2.0027566   0.3678013   0.81015587  1.2946123   0.2103759   0.4528777
   0.16692358  0.23616242  0.31249583  1.8362913   1.03624     1.9346664
   0.55011606  1.1387101   1.1547539   1.2407727   2.000425    1.4949946
   1.8182406   1.1708001   1.4016814   0.84154123  1.9460242   0.6121014
   2.3495069   1.3218157   1.4368091   0.39891255  1.4322495   0.46817595
   0.8301761   0.9377887   1.1954181   2.9233527   0.6771206   0.15306884
   0.25606704  1.8720125   2.098189    1.1319773   0.12661481  1.0279388
   1.1201229   0.6666013   0.26065117  2.981635    0.5998311   2.1837916
   1.8468947   1.408753    0.9721171   1.1445165   0.40396047  0.38058215
   2.2944164   0.3651805   1.9188422   1.8752613   0.18728006  1.0206645
   1.6662855   1.9869242   0.2678591   2.7105834   0.09050548  0.15786856
   0.6286499   2.5011563   0.51013625  0.540908    0.73526204  0.7781239
   0.5590771   1.9193668   0.45954943  0.27464885  1.7861683   0.3312422
   0.401976    2.1224957   0.71887314  2.197411    0.28105187  0.6483614
   0.71247387  1.2464089   1.8344724   0.48550677  1.891285    0.20006472
   0.2202425   0.8387761   3.3894868   1.6031806   1.3271294   0.7525135
   0.11344028  6.721008    7.319492    8.278512    5.962836    7.905032
   8.026696    6.6109066   8.5172825   7.5120397   6.089726    7.8092628
   6.4864025   8.443009    6.8313093   7.9110703   7.4712954   7.667455
   7.59261     8.066258    6.0823803   8.700166    8.745229    5.8665934
   7.100665    8.10189     6.0095797   5.997837    7.4231706   8.030036
   8.758926    7.048342    7.7748446   7.525751    6.552824    7.1847496
   6.331071    7.3712487   8.440196    6.3045764   6.3908257   6.872851
   7.91476     8.117279    6.6536584   7.963178    6.3162193   5.743176
   6.4358106   7.8365345   7.0754776   6.859908    7.4285784   7.4961314
   8.189009    6.868069    8.080111    7.5275807   7.1708245   9.16522
  -9.060962   -3.6917953   2.3898227 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 05:12:39.078758
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.0786
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 05:12:39.082374
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9063.25
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 05:12:39.085471
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.7291
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 05:12:39.088506
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -810.678
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140074296725576
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140071784034936
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140071784035440
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140071784035944
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140071784036448
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140071784036952

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f65b3ddc240> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.669552
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.628277
grad_step = 000002, loss = 0.598157
grad_step = 000003, loss = 0.564534
grad_step = 000004, loss = 0.525730
grad_step = 000005, loss = 0.489071
grad_step = 000006, loss = 0.465375
grad_step = 000007, loss = 0.441597
grad_step = 000008, loss = 0.404167
grad_step = 000009, loss = 0.373369
grad_step = 000010, loss = 0.359578
grad_step = 000011, loss = 0.345397
grad_step = 000012, loss = 0.324257
grad_step = 000013, loss = 0.306216
grad_step = 000014, loss = 0.294898
grad_step = 000015, loss = 0.281923
grad_step = 000016, loss = 0.268294
grad_step = 000017, loss = 0.258704
grad_step = 000018, loss = 0.248641
grad_step = 000019, loss = 0.236295
grad_step = 000020, loss = 0.224425
grad_step = 000021, loss = 0.214953
grad_step = 000022, loss = 0.205573
grad_step = 000023, loss = 0.194889
grad_step = 000024, loss = 0.184518
grad_step = 000025, loss = 0.174916
grad_step = 000026, loss = 0.165695
grad_step = 000027, loss = 0.157363
grad_step = 000028, loss = 0.150076
grad_step = 000029, loss = 0.142330
grad_step = 000030, loss = 0.133831
grad_step = 000031, loss = 0.126231
grad_step = 000032, loss = 0.119568
grad_step = 000033, loss = 0.112716
grad_step = 000034, loss = 0.105701
grad_step = 000035, loss = 0.098974
grad_step = 000036, loss = 0.092373
grad_step = 000037, loss = 0.086274
grad_step = 000038, loss = 0.080851
grad_step = 000039, loss = 0.075383
grad_step = 000040, loss = 0.070035
grad_step = 000041, loss = 0.065001
grad_step = 000042, loss = 0.059907
grad_step = 000043, loss = 0.055181
grad_step = 000044, loss = 0.050902
grad_step = 000045, loss = 0.046566
grad_step = 000046, loss = 0.042490
grad_step = 000047, loss = 0.038732
grad_step = 000048, loss = 0.035278
grad_step = 000049, loss = 0.032112
grad_step = 000050, loss = 0.028993
grad_step = 000051, loss = 0.026184
grad_step = 000052, loss = 0.023583
grad_step = 000053, loss = 0.021072
grad_step = 000054, loss = 0.018906
grad_step = 000055, loss = 0.016899
grad_step = 000056, loss = 0.015103
grad_step = 000057, loss = 0.013489
grad_step = 000058, loss = 0.012069
grad_step = 000059, loss = 0.010788
grad_step = 000060, loss = 0.009613
grad_step = 000061, loss = 0.008642
grad_step = 000062, loss = 0.007729
grad_step = 000063, loss = 0.006983
grad_step = 000064, loss = 0.006326
grad_step = 000065, loss = 0.005772
grad_step = 000066, loss = 0.005286
grad_step = 000067, loss = 0.004903
grad_step = 000068, loss = 0.004548
grad_step = 000069, loss = 0.004259
grad_step = 000070, loss = 0.004016
grad_step = 000071, loss = 0.003801
grad_step = 000072, loss = 0.003625
grad_step = 000073, loss = 0.003479
grad_step = 000074, loss = 0.003346
grad_step = 000075, loss = 0.003239
grad_step = 000076, loss = 0.003142
grad_step = 000077, loss = 0.003044
grad_step = 000078, loss = 0.002963
grad_step = 000079, loss = 0.002890
grad_step = 000080, loss = 0.002825
grad_step = 000081, loss = 0.002768
grad_step = 000082, loss = 0.002707
grad_step = 000083, loss = 0.002662
grad_step = 000084, loss = 0.002613
grad_step = 000085, loss = 0.002566
grad_step = 000086, loss = 0.002527
grad_step = 000087, loss = 0.002491
grad_step = 000088, loss = 0.002458
grad_step = 000089, loss = 0.002430
grad_step = 000090, loss = 0.002402
grad_step = 000091, loss = 0.002380
grad_step = 000092, loss = 0.002358
grad_step = 000093, loss = 0.002339
grad_step = 000094, loss = 0.002324
grad_step = 000095, loss = 0.002308
grad_step = 000096, loss = 0.002297
grad_step = 000097, loss = 0.002286
grad_step = 000098, loss = 0.002276
grad_step = 000099, loss = 0.002268
grad_step = 000100, loss = 0.002260
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002253
grad_step = 000102, loss = 0.002246
grad_step = 000103, loss = 0.002240
grad_step = 000104, loss = 0.002235
grad_step = 000105, loss = 0.002229
grad_step = 000106, loss = 0.002223
grad_step = 000107, loss = 0.002217
grad_step = 000108, loss = 0.002211
grad_step = 000109, loss = 0.002205
grad_step = 000110, loss = 0.002199
grad_step = 000111, loss = 0.002192
grad_step = 000112, loss = 0.002186
grad_step = 000113, loss = 0.002179
grad_step = 000114, loss = 0.002172
grad_step = 000115, loss = 0.002165
grad_step = 000116, loss = 0.002159
grad_step = 000117, loss = 0.002152
grad_step = 000118, loss = 0.002145
grad_step = 000119, loss = 0.002138
grad_step = 000120, loss = 0.002131
grad_step = 000121, loss = 0.002124
grad_step = 000122, loss = 0.002117
grad_step = 000123, loss = 0.002117
grad_step = 000124, loss = 0.002123
grad_step = 000125, loss = 0.002100
grad_step = 000126, loss = 0.002108
grad_step = 000127, loss = 0.002100
grad_step = 000128, loss = 0.002088
grad_step = 000129, loss = 0.002096
grad_step = 000130, loss = 0.002079
grad_step = 000131, loss = 0.002082
grad_step = 000132, loss = 0.002078
grad_step = 000133, loss = 0.002066
grad_step = 000134, loss = 0.002071
grad_step = 000135, loss = 0.002063
grad_step = 000136, loss = 0.002054
grad_step = 000137, loss = 0.002057
grad_step = 000138, loss = 0.002052
grad_step = 000139, loss = 0.002042
grad_step = 000140, loss = 0.002039
grad_step = 000141, loss = 0.002039
grad_step = 000142, loss = 0.002036
grad_step = 000143, loss = 0.002028
grad_step = 000144, loss = 0.002018
grad_step = 000145, loss = 0.002014
grad_step = 000146, loss = 0.002014
grad_step = 000147, loss = 0.002024
grad_step = 000148, loss = 0.002039
grad_step = 000149, loss = 0.002027
grad_step = 000150, loss = 0.001992
grad_step = 000151, loss = 0.001984
grad_step = 000152, loss = 0.002000
grad_step = 000153, loss = 0.001993
grad_step = 000154, loss = 0.001972
grad_step = 000155, loss = 0.001957
grad_step = 000156, loss = 0.001959
grad_step = 000157, loss = 0.001971
grad_step = 000158, loss = 0.001971
grad_step = 000159, loss = 0.001960
grad_step = 000160, loss = 0.001932
grad_step = 000161, loss = 0.001915
grad_step = 000162, loss = 0.001912
grad_step = 000163, loss = 0.001924
grad_step = 000164, loss = 0.001966
grad_step = 000165, loss = 0.001988
grad_step = 000166, loss = 0.001945
grad_step = 000167, loss = 0.001886
grad_step = 000168, loss = 0.001871
grad_step = 000169, loss = 0.001898
grad_step = 000170, loss = 0.001944
grad_step = 000171, loss = 0.001948
grad_step = 000172, loss = 0.001902
grad_step = 000173, loss = 0.001851
grad_step = 000174, loss = 0.001837
grad_step = 000175, loss = 0.001866
grad_step = 000176, loss = 0.001914
grad_step = 000177, loss = 0.001959
grad_step = 000178, loss = 0.001934
grad_step = 000179, loss = 0.001868
grad_step = 000180, loss = 0.001815
grad_step = 000181, loss = 0.001833
grad_step = 000182, loss = 0.001880
grad_step = 000183, loss = 0.001886
grad_step = 000184, loss = 0.001849
grad_step = 000185, loss = 0.001807
grad_step = 000186, loss = 0.001807
grad_step = 000187, loss = 0.001836
grad_step = 000188, loss = 0.001853
grad_step = 000189, loss = 0.001848
grad_step = 000190, loss = 0.001818
grad_step = 000191, loss = 0.001793
grad_step = 000192, loss = 0.001790
grad_step = 000193, loss = 0.001806
grad_step = 000194, loss = 0.001831
grad_step = 000195, loss = 0.001839
grad_step = 000196, loss = 0.001831
grad_step = 000197, loss = 0.001804
grad_step = 000198, loss = 0.001782
grad_step = 000199, loss = 0.001776
grad_step = 000200, loss = 0.001786
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001796
grad_step = 000202, loss = 0.001798
grad_step = 000203, loss = 0.001795
grad_step = 000204, loss = 0.001790
grad_step = 000205, loss = 0.001782
grad_step = 000206, loss = 0.001775
grad_step = 000207, loss = 0.001771
grad_step = 000208, loss = 0.001770
grad_step = 000209, loss = 0.001775
grad_step = 000210, loss = 0.001789
grad_step = 000211, loss = 0.001813
grad_step = 000212, loss = 0.001856
grad_step = 000213, loss = 0.001883
grad_step = 000214, loss = 0.001904
grad_step = 000215, loss = 0.001863
grad_step = 000216, loss = 0.001821
grad_step = 000217, loss = 0.001783
grad_step = 000218, loss = 0.001758
grad_step = 000219, loss = 0.001760
grad_step = 000220, loss = 0.001787
grad_step = 000221, loss = 0.001805
grad_step = 000222, loss = 0.001782
grad_step = 000223, loss = 0.001748
grad_step = 000224, loss = 0.001741
grad_step = 000225, loss = 0.001758
grad_step = 000226, loss = 0.001760
grad_step = 000227, loss = 0.001743
grad_step = 000228, loss = 0.001728
grad_step = 000229, loss = 0.001736
grad_step = 000230, loss = 0.001755
grad_step = 000231, loss = 0.001794
grad_step = 000232, loss = 0.001873
grad_step = 000233, loss = 0.002072
grad_step = 000234, loss = 0.002051
grad_step = 000235, loss = 0.001936
grad_step = 000236, loss = 0.001810
grad_step = 000237, loss = 0.001788
grad_step = 000238, loss = 0.001770
grad_step = 000239, loss = 0.001742
grad_step = 000240, loss = 0.001781
grad_step = 000241, loss = 0.001818
grad_step = 000242, loss = 0.001757
grad_step = 000243, loss = 0.001702
grad_step = 000244, loss = 0.001729
grad_step = 000245, loss = 0.001745
grad_step = 000246, loss = 0.001721
grad_step = 000247, loss = 0.001719
grad_step = 000248, loss = 0.001728
grad_step = 000249, loss = 0.001736
grad_step = 000250, loss = 0.001684
grad_step = 000251, loss = 0.001697
grad_step = 000252, loss = 0.001750
grad_step = 000253, loss = 0.001767
grad_step = 000254, loss = 0.001698
grad_step = 000255, loss = 0.001691
grad_step = 000256, loss = 0.001698
grad_step = 000257, loss = 0.001688
grad_step = 000258, loss = 0.001664
grad_step = 000259, loss = 0.001675
grad_step = 000260, loss = 0.001695
grad_step = 000261, loss = 0.001674
grad_step = 000262, loss = 0.001647
grad_step = 000263, loss = 0.001636
grad_step = 000264, loss = 0.001664
grad_step = 000265, loss = 0.001684
grad_step = 000266, loss = 0.001665
grad_step = 000267, loss = 0.001647
grad_step = 000268, loss = 0.001658
grad_step = 000269, loss = 0.001691
grad_step = 000270, loss = 0.001725
grad_step = 000271, loss = 0.001766
grad_step = 000272, loss = 0.001777
grad_step = 000273, loss = 0.001760
grad_step = 000274, loss = 0.001683
grad_step = 000275, loss = 0.001604
grad_step = 000276, loss = 0.001578
grad_step = 000277, loss = 0.001595
grad_step = 000278, loss = 0.001618
grad_step = 000279, loss = 0.001623
grad_step = 000280, loss = 0.001621
grad_step = 000281, loss = 0.001603
grad_step = 000282, loss = 0.001574
grad_step = 000283, loss = 0.001543
grad_step = 000284, loss = 0.001532
grad_step = 000285, loss = 0.001541
grad_step = 000286, loss = 0.001558
grad_step = 000287, loss = 0.001564
grad_step = 000288, loss = 0.001553
grad_step = 000289, loss = 0.001550
grad_step = 000290, loss = 0.001584
grad_step = 000291, loss = 0.001668
grad_step = 000292, loss = 0.001760
grad_step = 000293, loss = 0.001714
grad_step = 000294, loss = 0.001577
grad_step = 000295, loss = 0.001498
grad_step = 000296, loss = 0.001558
grad_step = 000297, loss = 0.001629
grad_step = 000298, loss = 0.001543
grad_step = 000299, loss = 0.001471
grad_step = 000300, loss = 0.001504
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001553
grad_step = 000302, loss = 0.001502
grad_step = 000303, loss = 0.001447
grad_step = 000304, loss = 0.001468
grad_step = 000305, loss = 0.001490
grad_step = 000306, loss = 0.001432
grad_step = 000307, loss = 0.001405
grad_step = 000308, loss = 0.001435
grad_step = 000309, loss = 0.001450
grad_step = 000310, loss = 0.001427
grad_step = 000311, loss = 0.001399
grad_step = 000312, loss = 0.001397
grad_step = 000313, loss = 0.001406
grad_step = 000314, loss = 0.001400
grad_step = 000315, loss = 0.001377
grad_step = 000316, loss = 0.001352
grad_step = 000317, loss = 0.001344
grad_step = 000318, loss = 0.001358
grad_step = 000319, loss = 0.001393
grad_step = 000320, loss = 0.001455
grad_step = 000321, loss = 0.001566
grad_step = 000322, loss = 0.001784
grad_step = 000323, loss = 0.002067
grad_step = 000324, loss = 0.002137
grad_step = 000325, loss = 0.001703
grad_step = 000326, loss = 0.001473
grad_step = 000327, loss = 0.001552
grad_step = 000328, loss = 0.001489
grad_step = 000329, loss = 0.001353
grad_step = 000330, loss = 0.001585
grad_step = 000331, loss = 0.001591
grad_step = 000332, loss = 0.001361
grad_step = 000333, loss = 0.001429
grad_step = 000334, loss = 0.001399
grad_step = 000335, loss = 0.001296
grad_step = 000336, loss = 0.001485
grad_step = 000337, loss = 0.001388
grad_step = 000338, loss = 0.001287
grad_step = 000339, loss = 0.001361
grad_step = 000340, loss = 0.001242
grad_step = 000341, loss = 0.001283
grad_step = 000342, loss = 0.001359
grad_step = 000343, loss = 0.001241
grad_step = 000344, loss = 0.001218
grad_step = 000345, loss = 0.001246
grad_step = 000346, loss = 0.001196
grad_step = 000347, loss = 0.001182
grad_step = 000348, loss = 0.001226
grad_step = 000349, loss = 0.001206
grad_step = 000350, loss = 0.001143
grad_step = 000351, loss = 0.001153
grad_step = 000352, loss = 0.001172
grad_step = 000353, loss = 0.001139
grad_step = 000354, loss = 0.001123
grad_step = 000355, loss = 0.001132
grad_step = 000356, loss = 0.001126
grad_step = 000357, loss = 0.001103
grad_step = 000358, loss = 0.001080
grad_step = 000359, loss = 0.001075
grad_step = 000360, loss = 0.001080
grad_step = 000361, loss = 0.001085
grad_step = 000362, loss = 0.001082
grad_step = 000363, loss = 0.001073
grad_step = 000364, loss = 0.001070
grad_step = 000365, loss = 0.001074
grad_step = 000366, loss = 0.001086
grad_step = 000367, loss = 0.001087
grad_step = 000368, loss = 0.001086
grad_step = 000369, loss = 0.001073
grad_step = 000370, loss = 0.001055
grad_step = 000371, loss = 0.001029
grad_step = 000372, loss = 0.001011
grad_step = 000373, loss = 0.000995
grad_step = 000374, loss = 0.000987
grad_step = 000375, loss = 0.000991
grad_step = 000376, loss = 0.001000
grad_step = 000377, loss = 0.001014
grad_step = 000378, loss = 0.001077
grad_step = 000379, loss = 0.001149
grad_step = 000380, loss = 0.001243
grad_step = 000381, loss = 0.001170
grad_step = 000382, loss = 0.001052
grad_step = 000383, loss = 0.000996
grad_step = 000384, loss = 0.001071
grad_step = 000385, loss = 0.001089
grad_step = 000386, loss = 0.000955
grad_step = 000387, loss = 0.000907
grad_step = 000388, loss = 0.000984
grad_step = 000389, loss = 0.000993
grad_step = 000390, loss = 0.000939
grad_step = 000391, loss = 0.000892
grad_step = 000392, loss = 0.000912
grad_step = 000393, loss = 0.000945
grad_step = 000394, loss = 0.000916
grad_step = 000395, loss = 0.000867
grad_step = 000396, loss = 0.000847
grad_step = 000397, loss = 0.000871
grad_step = 000398, loss = 0.000914
grad_step = 000399, loss = 0.000921
grad_step = 000400, loss = 0.000929
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000909
grad_step = 000402, loss = 0.000896
grad_step = 000403, loss = 0.000889
grad_step = 000404, loss = 0.000901
grad_step = 000405, loss = 0.000913
grad_step = 000406, loss = 0.000901
grad_step = 000407, loss = 0.000861
grad_step = 000408, loss = 0.000814
grad_step = 000409, loss = 0.000780
grad_step = 000410, loss = 0.000771
grad_step = 000411, loss = 0.000782
grad_step = 000412, loss = 0.000799
grad_step = 000413, loss = 0.000817
grad_step = 000414, loss = 0.000821
grad_step = 000415, loss = 0.000794
grad_step = 000416, loss = 0.000767
grad_step = 000417, loss = 0.000748
grad_step = 000418, loss = 0.000740
grad_step = 000419, loss = 0.000746
grad_step = 000420, loss = 0.000761
grad_step = 000421, loss = 0.000780
grad_step = 000422, loss = 0.000799
grad_step = 000423, loss = 0.000832
grad_step = 000424, loss = 0.000896
grad_step = 000425, loss = 0.001080
grad_step = 000426, loss = 0.001471
grad_step = 000427, loss = 0.001679
grad_step = 000428, loss = 0.001483
grad_step = 000429, loss = 0.001003
grad_step = 000430, loss = 0.000942
grad_step = 000431, loss = 0.001077
grad_step = 000432, loss = 0.000822
grad_step = 000433, loss = 0.000718
grad_step = 000434, loss = 0.000922
grad_step = 000435, loss = 0.000878
grad_step = 000436, loss = 0.000789
grad_step = 000437, loss = 0.000839
grad_step = 000438, loss = 0.000748
grad_step = 000439, loss = 0.000668
grad_step = 000440, loss = 0.000734
grad_step = 000441, loss = 0.000710
grad_step = 000442, loss = 0.000678
grad_step = 000443, loss = 0.000733
grad_step = 000444, loss = 0.000721
grad_step = 000445, loss = 0.000657
grad_step = 000446, loss = 0.000635
grad_step = 000447, loss = 0.000636
grad_step = 000448, loss = 0.000650
grad_step = 000449, loss = 0.000635
grad_step = 000450, loss = 0.000619
grad_step = 000451, loss = 0.000628
grad_step = 000452, loss = 0.000633
grad_step = 000453, loss = 0.000621
grad_step = 000454, loss = 0.000603
grad_step = 000455, loss = 0.000580
grad_step = 000456, loss = 0.000573
grad_step = 000457, loss = 0.000578
grad_step = 000458, loss = 0.000584
grad_step = 000459, loss = 0.000587
grad_step = 000460, loss = 0.000581
grad_step = 000461, loss = 0.000571
grad_step = 000462, loss = 0.000563
grad_step = 000463, loss = 0.000555
grad_step = 000464, loss = 0.000549
grad_step = 000465, loss = 0.000545
grad_step = 000466, loss = 0.000540
grad_step = 000467, loss = 0.000538
grad_step = 000468, loss = 0.000537
grad_step = 000469, loss = 0.000535
grad_step = 000470, loss = 0.000535
grad_step = 000471, loss = 0.000534
grad_step = 000472, loss = 0.000532
grad_step = 000473, loss = 0.000529
grad_step = 000474, loss = 0.000525
grad_step = 000475, loss = 0.000520
grad_step = 000476, loss = 0.000516
grad_step = 000477, loss = 0.000511
grad_step = 000478, loss = 0.000508
grad_step = 000479, loss = 0.000505
grad_step = 000480, loss = 0.000504
grad_step = 000481, loss = 0.000506
grad_step = 000482, loss = 0.000515
grad_step = 000483, loss = 0.000531
grad_step = 000484, loss = 0.000557
grad_step = 000485, loss = 0.000610
grad_step = 000486, loss = 0.000653
grad_step = 000487, loss = 0.000711
grad_step = 000488, loss = 0.000659
grad_step = 000489, loss = 0.000584
grad_step = 000490, loss = 0.000514
grad_step = 000491, loss = 0.000546
grad_step = 000492, loss = 0.000611
grad_step = 000493, loss = 0.000599
grad_step = 000494, loss = 0.000554
grad_step = 000495, loss = 0.000554
grad_step = 000496, loss = 0.000606
grad_step = 000497, loss = 0.000642
grad_step = 000498, loss = 0.000640
grad_step = 000499, loss = 0.000612
grad_step = 000500, loss = 0.000624
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000667
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

  date_run                              2020-05-14 05:12:58.345702
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.179632
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 05:12:58.351153
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0749544
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 05:12:58.358664
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.112642
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 05:12:58.363661
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.138958
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
0   2020-05-14 05:12:30.136401  ...    mean_absolute_error
1   2020-05-14 05:12:30.140063  ...     mean_squared_error
2   2020-05-14 05:12:30.142950  ...  median_absolute_error
3   2020-05-14 05:12:30.145837  ...               r2_score
4   2020-05-14 05:12:39.078758  ...    mean_absolute_error
5   2020-05-14 05:12:39.082374  ...     mean_squared_error
6   2020-05-14 05:12:39.085471  ...  median_absolute_error
7   2020-05-14 05:12:39.088506  ...               r2_score
8   2020-05-14 05:12:58.345702  ...    mean_absolute_error
9   2020-05-14 05:12:58.351153  ...     mean_squared_error
10  2020-05-14 05:12:58.358664  ...  median_absolute_error
11  2020-05-14 05:12:58.363661  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a8f1cc898> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 23%|██▎       | 2277376/9912422 [00:00<00:00, 22716858.23it/s]9920512it [00:00, 30977649.27it/s]                             
0it [00:00, ?it/s]32768it [00:00, 736014.62it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 480485.22it/s]1654784it [00:00, 11856949.83it/s]                         
0it [00:00, ?it/s]8192it [00:00, 225576.01it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a8f1cca90> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a3e9c80b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a41b7ae10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a8f184e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a41b7ae10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a8f184e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a41b7ae10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a8f184e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a41b7ae10> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6a8f184e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fe1b08e3208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=0971f3e4473016fa7f5d08254618eac91ed09a07683ca018bf27aa7c0ed33e57
  Stored in directory: /tmp/pip-ephem-wheel-cache-7ufa23du/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fe1a6a4e048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 2s
 1736704/17464789 [=>............................] - ETA: 0s
 8470528/17464789 [=============>................] - ETA: 0s
 9781248/17464789 [===============>..............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 05:14:25.589099: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 05:14:25.593609: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095090000 Hz
2020-05-14 05:14:25.593864: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56131c2b0fe0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 05:14:25.593883: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8660 - accuracy: 0.4870
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8123 - accuracy: 0.4905 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6820 - accuracy: 0.4990
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6628 - accuracy: 0.5002
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6574 - accuracy: 0.5006
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6794 - accuracy: 0.4992
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6710 - accuracy: 0.4997
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6858 - accuracy: 0.4988
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6956 - accuracy: 0.4981
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7249 - accuracy: 0.4962
11000/25000 [============>.................] - ETA: 3s - loss: 7.7057 - accuracy: 0.4975
12000/25000 [=============>................] - ETA: 3s - loss: 7.7292 - accuracy: 0.4959
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6949 - accuracy: 0.4982
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6754 - accuracy: 0.4994
15000/25000 [=================>............] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6522 - accuracy: 0.5009
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6581 - accuracy: 0.5006
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6755 - accuracy: 0.4994
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6559 - accuracy: 0.5007
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6528 - accuracy: 0.5009
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6555 - accuracy: 0.5007
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6606 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
25000/25000 [==============================] - 7s 280us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 05:14:39.282202
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 05:14:39.282202  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:57:08, 10.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:18:37, 14.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:28:19, 20.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<8:02:14, 29.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 1.72M/862M [00:01<5:40:49, 42.1kB/s].vector_cache/glove.6B.zip:   0%|          | 2.95M/862M [00:02<3:58:42, 60.0kB/s].vector_cache/glove.6B.zip:   1%|          | 7.04M/862M [00:02<2:46:24, 85.7kB/s].vector_cache/glove.6B.zip:   1%|          | 10.8M/862M [00:02<1:56:05, 122kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 15.3M/862M [00:02<1:20:55, 174kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.6M/862M [00:02<56:28, 249kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 24.9M/862M [00:02<39:21, 355kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.6M/862M [00:02<27:33, 504kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.4M/862M [00:02<19:18, 716kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.2M/862M [00:02<13:31, 1.02MB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.7M/862M [00:02<09:32, 1.43MB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.2M/862M [00:03<13:55, 983kB/s] .vector_cache/glove.6B.zip:   5%|▌         | 45.4M/862M [00:03<09:47, 1.39MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.9M/862M [00:03<06:56, 1.95MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:04<05:33, 2.43MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<03:58, 3.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:06<28:35, 470kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:06<22:09, 606kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:06<15:57, 840kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:06<11:23, 1.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:08<13:14, 1.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:08<11:11, 1.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:08<08:17, 1.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:08<06:00, 2.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:10<11:48, 1.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:10<10:12, 1.30MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:10<07:37, 1.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:12<07:46, 1.70MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:12<06:53, 1.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:12<05:10, 2.54MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:14<06:29, 2.02MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:14<05:59, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:14<04:31, 2.90MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:16<06:01, 2.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:16<05:38, 2.31MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:16<04:14, 3.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:18<06:05, 2.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:18<07:04, 1.84MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:18<05:32, 2.34MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.8M/862M [00:18<04:09, 3.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:20<06:20, 2.04MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:20<05:46, 2.24MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.5M/862M [00:20<04:19, 2.98MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:22<06:03, 2.13MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:22<05:33, 2.31MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:22<04:10, 3.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:24<05:55, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:24<05:28, 2.34MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:24<04:06, 3.11MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:26<05:54, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:26<05:26, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:26<04:05, 3.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<05:50, 2.17MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<05:22, 2.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:28<04:04, 3.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<05:49, 2.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<06:39, 1.89MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:30<05:11, 2.42MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<03:48, 3.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<08:27, 1.48MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:32<07:14, 1.73MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:32<05:23, 2.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<06:39, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:34<05:56, 2.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:34<04:27, 2.78MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:36<06:03, 2.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:36<05:28, 2.26MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:36<04:09, 2.98MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<05:48, 2.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<05:20, 2.30MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:38<04:03, 3.03MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<05:43, 2.14MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<05:16, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:40<03:59, 3.06MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<05:40, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<05:13, 2.33MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:42<03:53, 3.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<02:53, 4.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<3:22:42, 59.8kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<2:31:59, 79.7kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<1:48:37, 112kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:45<1:17:07, 157kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:45<54:10, 223kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:45<38:05, 317kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<33:12, 363kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<25:07, 480kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:46<18:01, 668kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:46<12:49, 936kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<13:04, 916kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:48<10:55, 1.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:48<08:04, 1.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:46, 2.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<25:24, 469kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<19:28, 611kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:50<13:55, 854kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:50<09:58, 1.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<11:55, 993kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<09:55, 1.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:52<07:17, 1.62MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:17, 2.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<09:18, 1.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<07:54, 1.49MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:54<05:49, 2.02MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<06:29, 1.80MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<07:00, 1.67MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:56<05:29, 2.13MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<04:00, 2.91MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<07:55, 1.47MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<06:34, 1.77MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:58<05:10, 2.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<03:48, 3.05MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:00<07:04, 1.64MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:00<06:11, 1.87MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:00<04:38, 2.49MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<03:22, 3.40MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<47:49, 240kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:02<34:38, 332kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:02<24:27, 469kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<19:44, 579kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:04<14:59, 762kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:04<10:45, 1.06MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<10:11, 1.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<08:06, 1.40MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:06<05:55, 1.91MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<04:20, 2.60MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:08<10:17, 1.10MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:08<09:32, 1.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:08<07:15, 1.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<05:12, 2.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<09:44, 1.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<08:01, 1.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:10<05:51, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<04:14, 2.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<29:27, 379kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<21:48, 511kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:12<15:32, 716kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<13:23, 828kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<11:39, 951kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:14<08:40, 1.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<06:11, 1.78MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<09:49, 1.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<08:00, 1.38MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:16<05:52, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<06:39, 1.64MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<06:54, 1.59MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:18<05:21, 2.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<03:50, 2.83MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<1:13:33, 148kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:20<52:35, 207kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:20<36:59, 293kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<28:20, 382kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<20:46, 520kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:22<14:56, 722kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<10:31, 1.02MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<50:33, 212kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<36:27, 295kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:24<25:41, 417kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<20:28, 521kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<15:15, 700kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:26<11:00, 967kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<07:47, 1.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:27<45:26, 233kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<34:06, 311kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:28<24:23, 434kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<18:41, 564kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<14:10, 743kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:30<10:07, 1.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<09:31, 1.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<07:45, 1.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:32<05:40, 1.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<06:25, 1.62MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<05:33, 1.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:34<04:08, 2.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<05:20, 1.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:35<04:48, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:36<03:37, 2.85MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<04:57, 2.07MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:37<04:21, 2.35MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<03:31, 2.91MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<02:33, 3.98MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<40:12, 254kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:39<29:10, 349kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:40<20:37, 493kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:41<16:47, 603kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:41<12:36, 803kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:41<09:13, 1.10MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:42<06:32, 1.54MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:43<42:28, 237kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:43<31:52, 316kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:43<22:43, 442kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:44<16:06, 622kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:45<13:47, 725kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:45<10:42, 933kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:45<07:43, 1.29MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<07:43, 1.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<06:26, 1.54MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:47<04:44, 2.08MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<05:38, 1.75MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<05:55, 1.66MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:49<04:35, 2.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:49<03:24, 2.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<05:09, 1.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:51<04:38, 2.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:51<03:29, 2.79MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<04:43, 2.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:53<04:19, 2.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:53<03:13, 3.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:55<04:31, 2.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:55<04:10, 2.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:55<03:07, 3.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<02:18, 4.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:57<1:22:46, 116kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<59:49, 160kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<42:18, 226kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<29:34, 322kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<9:25:48, 16.8kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<6:36:48, 23.9kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:59<4:37:13, 34.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<3:15:31, 48.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<2:17:45, 68.5kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:01<1:36:24, 97.6kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<1:09:26, 135kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<49:33, 189kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:03<34:49, 268kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<26:29, 351kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<19:29, 477kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:05<13:52, 668kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<09:47, 944kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:07<29:41, 311kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:07<21:44, 425kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:07<15:24, 597kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:09<12:55, 710kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<09:59, 917kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:09<07:12, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:11<07:10, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:11<05:57, 1.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:11<04:20, 2.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<03:10, 2.84MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:13<13:25, 672kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:13<10:19, 874kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:13<07:26, 1.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<07:17, 1.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<06:13, 1.44MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:15<04:37, 1.93MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<03:21, 2.65MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:17<10:17, 864kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:17<08:13, 1.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:17<06:18, 1.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:17<04:34, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:19<05:31, 1.60MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:19<04:40, 1.89MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:19<03:46, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<02:45, 3.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:21<05:18, 1.65MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:21<04:37, 1.89MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:21<03:27, 2.52MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<04:26, 1.95MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:23<04:55, 1.76MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:23<03:52, 2.24MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<02:52, 3.01MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:24<04:36, 1.87MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:25<04:07, 2.09MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:25<03:05, 2.78MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:26<04:07, 2.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:27<03:44, 2.28MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:27<02:49, 3.01MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<03:58, 2.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:28<03:39, 2.32MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:29<02:46, 3.05MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:30<03:55, 2.14MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:30<04:29, 1.87MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:31<03:29, 2.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:31<02:39, 3.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<03:52, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<03:34, 2.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:33<02:42, 3.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<03:59, 2.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:35<04:29, 1.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:35<03:31, 2.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:35<02:37, 3.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:36<04:16, 1.92MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:36<03:51, 2.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:37<02:54, 2.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<03:55, 2.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<03:36, 2.25MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:39<02:43, 2.98MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:39<02:02, 3.97MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<06:55, 1.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<05:36, 1.44MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:40<04:28, 1.80MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<03:12, 2.50MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:42<07:30, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<06:09, 1.30MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:42<04:28, 1.78MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<03:15, 2.43MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:44<07:16, 1.09MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<05:59, 1.32MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:45<04:22, 1.80MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<04:44, 1.66MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<04:56, 1.59MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:47<03:51, 2.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<02:47, 2.80MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:48<06:37, 1.18MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:48<05:26, 1.43MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<04:00, 1.94MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:50<04:35, 1.68MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:50<04:01, 1.91MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:50<02:59, 2.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<02:10, 3.51MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:52<51:56, 147kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:52<37:08, 206kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:52<26:06, 292kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:54<19:58, 380kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:54<14:45, 514kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:54<10:29, 720kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:56<09:05, 827kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:56<07:07, 1.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:56<05:09, 1.45MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:58<05:21, 1.39MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<04:30, 1.65MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:58<03:19, 2.24MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<02:24, 3.06MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<56:29, 131kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:00<40:17, 183kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:00<28:17, 260kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:02<21:25, 341kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:02<15:37, 468kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:02<11:09, 653kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<07:50, 923kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<32:47, 221kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<23:41, 305kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:04<16:42, 431kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:06<13:20, 538kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:06<10:04, 711kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:06<07:10, 994kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:08<06:40, 1.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:08<05:24, 1.31MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:08<03:57, 1.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:10<04:24, 1.59MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:10<03:48, 1.84MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:10<02:50, 2.46MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:12<03:38, 1.91MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:12<03:15, 2.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:12<02:27, 2.83MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:14<03:20, 2.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:14<02:56, 2.35MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:14<02:17, 3.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<01:40, 4.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:16<28:37, 239kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:16<20:43, 329kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:16<14:37, 465kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:18<11:46, 574kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:18<09:37, 702kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:18<07:02, 957kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:18<05:00, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:20<06:17, 1.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:20<05:06, 1.31MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:20<03:44, 1.78MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:22<04:09, 1.59MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:22<03:38, 1.82MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:22<02:41, 2.45MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:24<03:30, 1.86MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:24<03:53, 1.68MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:24<03:00, 2.17MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:24<02:17, 2.85MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:26<03:04, 2.10MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:26<02:52, 2.25MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:26<02:09, 3.00MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<01:36, 4.01MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:28<24:35, 261kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:28<17:54, 358kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:28<12:40, 505kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:28<08:56, 712kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:30<10:02, 633kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:30<07:47, 814kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:30<05:38, 1.12MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:32<05:14, 1.20MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:32<04:21, 1.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:32<03:10, 1.97MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<02:18, 2.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:33<1:02:20, 99.6kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:34<44:15, 140kB/s]   .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:34<30:59, 199kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:35<23:01, 267kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:36<16:46, 366kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:36<11:50, 516kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<09:40, 628kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<07:26, 816kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:38<05:20, 1.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<05:07, 1.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<04:14, 1.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:40<03:07, 1.91MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<03:32, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<03:08, 1.89MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<02:21, 2.51MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:43<02:59, 1.96MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:43<02:44, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<02:04, 2.82MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:45<02:48, 2.07MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:45<02:35, 2.23MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<01:56, 2.97MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<01:25, 4.01MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:47<38:20, 150kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<27:25, 209kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:47<19:16, 296kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:49<14:55, 379kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:50<11:40, 485kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:50<08:28, 667kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:50<05:59, 937kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:51<06:02, 926kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<04:51, 1.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:52<03:31, 1.58MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:52<02:31, 2.18MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:53<21:57, 252kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:53<15:57, 346kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:54<11:15, 488kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<07:54, 691kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:55<50:07, 109kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<36:15, 150kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<25:34, 213kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:56<17:59, 301kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:57<13:45, 392kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:58<10:17, 523kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<07:18, 734kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:58<05:10, 1.03MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:59<08:03, 660kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<06:17, 844kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:00<04:32, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:01<04:16, 1.23MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<03:37, 1.44MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:02<02:41, 1.94MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:03<02:59, 1.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<02:40, 1.94MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:04<02:00, 2.57MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:05<02:33, 2.00MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:05<02:21, 2.16MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:06<01:47, 2.84MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:07<02:24, 2.09MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:07<02:14, 2.25MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<01:41, 2.95MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:09<02:20, 2.12MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:09<02:10, 2.29MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:10<01:38, 3.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:11<02:18, 2.12MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:11<02:03, 2.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:11<01:44, 2.81MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:12<01:16, 3.83MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:13<03:43, 1.30MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:13<03:08, 1.54MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:13<02:18, 2.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<01:41, 2.83MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:15<04:17, 1.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:15<03:30, 1.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:15<02:46, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:15<01:59, 2.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:17<04:10, 1.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:17<03:29, 1.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:17<02:33, 1.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<01:50, 2.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:19<4:36:12, 16.8kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:19<3:13:29, 23.9kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:19<2:15:24, 34.1kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<1:33:48, 48.7kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:21<1:14:33, 61.2kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:21<52:30, 86.8kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:21<36:53, 123kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:21<25:34, 176kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:23<4:46:10, 15.7kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:23<3:20:31, 22.4kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:23<2:19:40, 31.9kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:25<1:38:01, 45.1kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:25<1:08:59, 64.0kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:25<48:07, 91.2kB/s]  .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:27<34:26, 126kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:27<24:32, 177kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:27<17:10, 252kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<11:59, 358kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:29<4:20:03, 16.5kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:29<3:02:14, 23.5kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:29<2:06:54, 33.5kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:31<1:29:04, 47.4kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:31<1:02:42, 67.2kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:31<43:44, 95.7kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:33<31:19, 132kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:33<22:20, 185kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:33<15:38, 263kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:35<11:48, 346kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:35<08:42, 468kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:35<06:09, 658kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:37<05:12, 770kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:37<04:05, 980kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<02:57, 1.35MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<02:58, 1.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<02:30, 1.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:39<01:49, 2.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:41<02:11, 1.77MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:41<01:53, 2.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:41<01:29, 2.58MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:41<01:04, 3.54MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:43<14:57, 254kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:43<10:52, 350kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:43<07:38, 494kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<06:10, 605kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<04:43, 789kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:45<03:23, 1.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:47<03:12, 1.14MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:47<02:34, 1.42MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<02:00, 1.83MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<01:25, 2.53MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:49<15:18, 235kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:49<11:05, 324kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:49<07:48, 457kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<06:13, 567kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:51<04:45, 742kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:51<03:23, 1.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:53<03:09, 1.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<02:29, 1.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<01:57, 1.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:53<01:24, 2.42MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:54<02:29, 1.37MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<02:06, 1.60MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:55<01:32, 2.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:55<01:07, 2.97MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:56<25:33, 130kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<18:13, 182kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:57<12:44, 258kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:58<09:34, 340kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:59<07:03, 461kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:59<04:57, 650kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:59<03:29, 915kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:00<29:04, 110kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:01<20:39, 154kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:01<14:24, 219kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:01<10:01, 312kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:02<1:07:56, 45.9kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:02<47:48, 65.2kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<33:24, 92.8kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:03<23:11, 132kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:04<17:56, 170kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:05<13:13, 231kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<09:21, 324kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<06:34, 459kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:06<05:14, 569kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:06<03:59, 745kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<02:52, 1.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<02:01, 1.44MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:08<04:47, 608kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:08<03:42, 784kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:08<02:44, 1.06MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:09<01:56, 1.47MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:11<02:41, 1.06MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:11<04:22, 651kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<03:44, 761kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<02:44, 1.03MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<01:59, 1.41MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<01:56, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<01:30, 1.83MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:13<01:05, 2.50MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<01:36, 1.68MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<01:41, 1.60MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<01:17, 2.07MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:15<00:58, 2.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<01:18, 2.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<01:13, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<00:54, 2.85MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:17<00:40, 3.86MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<09:18, 276kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<06:47, 377kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<04:46, 532kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<03:20, 752kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:21<07:04, 354kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:21<05:12, 479kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<03:40, 671kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:23<03:06, 783kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<02:26, 995kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<01:44, 1.38MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:23<01:14, 1.92MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:25<18:15, 130kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<13:00, 181kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:25<09:04, 257kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:27<06:47, 338kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<04:59, 459kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<03:30, 644kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 729M/862M [05:29<02:56, 757kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<02:17, 967kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<01:38, 1.33MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:30<01:38, 1.31MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:31<01:22, 1.56MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:31<01:00, 2.10MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:32<01:11, 1.76MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:33<01:03, 1.98MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<00:51, 2.44MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:33<00:36, 3.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:34<01:41, 1.20MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:34<01:23, 1.45MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<01:04, 1.86MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:35<00:46, 2.56MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:36<01:30, 1.29MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:37<01:16, 1.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:37<00:55, 2.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:37<00:39, 2.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:38<14:17, 132kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:38<10:10, 184kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<07:04, 262kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:40<05:17, 343kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:40<03:51, 468kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<02:48, 643kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:41<01:55, 909kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:42<03:33, 491kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:42<02:40, 651kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<01:53, 908kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:44<01:41, 995kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:44<01:19, 1.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:44<01:01, 1.63MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:45<00:42, 2.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:46<02:36, 618kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:46<01:59, 803kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:46<01:25, 1.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:47<00:59, 1.56MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:48<10:43, 144kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:48<07:37, 202kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<05:23, 283kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<03:41, 403kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:50<03:41, 400kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:50<02:41, 546kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<01:57, 742kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<01:21, 1.05MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:52<01:57, 718kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:52<01:31, 920kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:52<01:04, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:54<01:02, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:54<00:52, 1.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<00:38, 2.06MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:56<00:43, 1.74MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:56<00:38, 1.95MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:28, 2.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:56<00:20, 3.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:58<09:02, 133kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:58<06:25, 185kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<04:25, 264kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:58<03:01, 374kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:00<11:34, 97.7kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:00<08:10, 138kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<05:36, 196kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:02<04:04, 260kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<03:04, 343kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<02:10, 480kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<01:31, 672kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<01:15, 787kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:59, 1.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<00:41, 1.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:41, 1.35MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:34, 1.59MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<00:24, 2.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:06<00:17, 2.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:08<06:11, 138kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:08<04:30, 189kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<03:09, 266kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:09<02:04, 379kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:10<46:05, 17.1kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:10<32:05, 24.3kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<21:44, 34.8kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:10<14:31, 49.6kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:12<13:23, 53.7kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:12<09:21, 76.1kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<06:19, 108kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:14<04:21, 149kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<03:05, 209kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<02:05, 296kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:16<01:30, 384kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<01:06, 518kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:45, 727kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:30, 1.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:18<03:41, 139kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:18<02:36, 194kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<01:46, 275kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<01:08, 391kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<03:24, 130kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<02:23, 183kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<01:40, 258kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:20<01:03, 366kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:22<00:57, 394kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:22<00:41, 535kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:22<00:29, 733kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<00:19, 1.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:24<00:19, 943kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:24<00:15, 1.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:10, 1.60MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:26<00:09, 1.46MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:26<00:09, 1.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:26<00:07, 1.84MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:26<00:04, 2.55MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:28<00:22, 456kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:28<00:16, 606kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:09, 850kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:28<00:05, 1.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:30<00:25, 236kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:30<00:17, 325kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:30<00:09, 458kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:32<00:03, 567kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:32<00:02, 744kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:32<00:00, 1.03MB/s].vector_cache/glove.6B.zip: 862MB [06:32, 2.20MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 836/400000 [00:00<00:47, 8358.09it/s]  0%|          | 1712/400000 [00:00<00:47, 8472.90it/s]  1%|          | 2597/400000 [00:00<00:46, 8580.56it/s]  1%|          | 3459/400000 [00:00<00:46, 8590.85it/s]  1%|          | 4314/400000 [00:00<00:46, 8578.22it/s]  1%|▏         | 5187/400000 [00:00<00:45, 8622.34it/s]  2%|▏         | 6058/400000 [00:00<00:45, 8645.97it/s]  2%|▏         | 6936/400000 [00:00<00:45, 8684.34it/s]  2%|▏         | 7809/400000 [00:00<00:45, 8696.65it/s]  2%|▏         | 8658/400000 [00:01<00:45, 8633.24it/s]  2%|▏         | 9498/400000 [00:01<00:46, 8451.68it/s]  3%|▎         | 10377/400000 [00:01<00:45, 8549.97it/s]  3%|▎         | 11223/400000 [00:01<00:46, 8441.23it/s]  3%|▎         | 12101/400000 [00:01<00:45, 8539.85it/s]  3%|▎         | 12964/400000 [00:01<00:45, 8564.73it/s]  3%|▎         | 13842/400000 [00:01<00:44, 8626.24it/s]  4%|▎         | 14703/400000 [00:01<00:44, 8617.13it/s]  4%|▍         | 15564/400000 [00:01<00:44, 8593.65it/s]  4%|▍         | 16423/400000 [00:01<00:44, 8548.18it/s]  4%|▍         | 17283/400000 [00:02<00:44, 8563.56it/s]  5%|▍         | 18156/400000 [00:02<00:44, 8610.30it/s]  5%|▍         | 19017/400000 [00:02<00:44, 8590.06it/s]  5%|▍         | 19876/400000 [00:02<00:45, 8355.14it/s]  5%|▌         | 20713/400000 [00:02<00:45, 8316.01it/s]  5%|▌         | 21567/400000 [00:02<00:45, 8380.97it/s]  6%|▌         | 22419/400000 [00:02<00:44, 8419.97it/s]  6%|▌         | 23299/400000 [00:02<00:44, 8528.41it/s]  6%|▌         | 24176/400000 [00:02<00:43, 8597.29it/s]  6%|▋         | 25055/400000 [00:02<00:43, 8652.21it/s]  6%|▋         | 25921/400000 [00:03<00:43, 8626.03it/s]  7%|▋         | 26785/400000 [00:03<00:43, 8584.60it/s]  7%|▋         | 27645/400000 [00:03<00:43, 8588.19it/s]  7%|▋         | 28532/400000 [00:03<00:42, 8668.45it/s]  7%|▋         | 29413/400000 [00:03<00:42, 8710.02it/s]  8%|▊         | 30285/400000 [00:03<00:43, 8534.37it/s]  8%|▊         | 31163/400000 [00:03<00:42, 8606.50it/s]  8%|▊         | 32025/400000 [00:03<00:43, 8442.16it/s]  8%|▊         | 32871/400000 [00:03<00:43, 8355.70it/s]  8%|▊         | 33723/400000 [00:03<00:43, 8403.16it/s]  9%|▊         | 34569/400000 [00:04<00:43, 8417.30it/s]  9%|▉         | 35429/400000 [00:04<00:43, 8471.25it/s]  9%|▉         | 36277/400000 [00:04<00:44, 8159.65it/s]  9%|▉         | 37135/400000 [00:04<00:43, 8280.93it/s]  9%|▉         | 37995/400000 [00:04<00:43, 8372.51it/s] 10%|▉         | 38847/400000 [00:04<00:42, 8415.78it/s] 10%|▉         | 39706/400000 [00:04<00:42, 8466.20it/s] 10%|█         | 40586/400000 [00:04<00:41, 8561.27it/s] 10%|█         | 41444/400000 [00:04<00:41, 8563.04it/s] 11%|█         | 42306/400000 [00:04<00:41, 8578.70it/s] 11%|█         | 43165/400000 [00:05<00:42, 8451.97it/s] 11%|█         | 44011/400000 [00:05<00:42, 8416.32it/s] 11%|█         | 44860/400000 [00:05<00:42, 8438.03it/s] 11%|█▏        | 45742/400000 [00:05<00:41, 8547.01it/s] 12%|█▏        | 46608/400000 [00:05<00:41, 8578.86it/s] 12%|█▏        | 47480/400000 [00:05<00:40, 8615.68it/s] 12%|█▏        | 48363/400000 [00:05<00:40, 8676.48it/s] 12%|█▏        | 49232/400000 [00:05<00:40, 8564.14it/s] 13%|█▎        | 50089/400000 [00:05<00:40, 8551.42it/s] 13%|█▎        | 50959/400000 [00:05<00:40, 8594.12it/s] 13%|█▎        | 51822/400000 [00:06<00:40, 8602.68it/s] 13%|█▎        | 52683/400000 [00:06<00:41, 8441.06it/s] 13%|█▎        | 53533/400000 [00:06<00:40, 8457.26it/s] 14%|█▎        | 54392/400000 [00:06<00:40, 8496.08it/s] 14%|█▍        | 55243/400000 [00:06<00:40, 8478.46it/s] 14%|█▍        | 56092/400000 [00:06<00:40, 8470.73it/s] 14%|█▍        | 56955/400000 [00:06<00:40, 8517.24it/s] 14%|█▍        | 57826/400000 [00:06<00:39, 8573.10it/s] 15%|█▍        | 58684/400000 [00:06<00:40, 8526.73it/s] 15%|█▍        | 59545/400000 [00:06<00:39, 8548.88it/s] 15%|█▌        | 60401/400000 [00:07<00:39, 8544.82it/s] 15%|█▌        | 61280/400000 [00:07<00:39, 8615.90it/s] 16%|█▌        | 62146/400000 [00:07<00:39, 8627.35it/s] 16%|█▌        | 63009/400000 [00:07<00:39, 8522.11it/s] 16%|█▌        | 63886/400000 [00:07<00:39, 8592.15it/s] 16%|█▌        | 64746/400000 [00:07<00:39, 8507.78it/s] 16%|█▋        | 65598/400000 [00:07<00:39, 8504.43it/s] 17%|█▋        | 66468/400000 [00:07<00:38, 8559.78it/s] 17%|█▋        | 67336/400000 [00:07<00:38, 8594.64it/s] 17%|█▋        | 68216/400000 [00:07<00:38, 8654.76it/s] 17%|█▋        | 69082/400000 [00:08<00:38, 8602.19it/s] 17%|█▋        | 69958/400000 [00:08<00:38, 8648.67it/s] 18%|█▊        | 70840/400000 [00:08<00:37, 8697.02it/s] 18%|█▊        | 71710/400000 [00:08<00:38, 8545.10it/s] 18%|█▊        | 72566/400000 [00:08<00:38, 8524.26it/s] 18%|█▊        | 73427/400000 [00:08<00:38, 8549.80it/s] 19%|█▊        | 74287/400000 [00:08<00:38, 8563.68it/s] 19%|█▉        | 75155/400000 [00:08<00:37, 8595.51it/s] 19%|█▉        | 76024/400000 [00:08<00:37, 8622.48it/s] 19%|█▉        | 76906/400000 [00:08<00:37, 8679.44it/s] 19%|█▉        | 77775/400000 [00:09<00:37, 8479.86it/s] 20%|█▉        | 78625/400000 [00:09<00:39, 8153.30it/s] 20%|█▉        | 79500/400000 [00:09<00:38, 8321.79it/s] 20%|██        | 80380/400000 [00:09<00:37, 8458.84it/s] 20%|██        | 81254/400000 [00:09<00:37, 8540.10it/s] 21%|██        | 82111/400000 [00:09<00:37, 8451.26it/s] 21%|██        | 82968/400000 [00:09<00:37, 8486.37it/s] 21%|██        | 83818/400000 [00:09<00:37, 8335.85it/s] 21%|██        | 84675/400000 [00:09<00:37, 8402.61it/s] 21%|██▏       | 85544/400000 [00:10<00:37, 8486.29it/s] 22%|██▏       | 86396/400000 [00:10<00:36, 8495.38it/s] 22%|██▏       | 87270/400000 [00:10<00:36, 8566.97it/s] 22%|██▏       | 88142/400000 [00:10<00:36, 8610.61it/s] 22%|██▏       | 89004/400000 [00:10<00:36, 8577.79it/s] 22%|██▏       | 89863/400000 [00:10<00:36, 8558.33it/s] 23%|██▎       | 90720/400000 [00:10<00:36, 8554.46it/s] 23%|██▎       | 91589/400000 [00:10<00:35, 8592.46it/s] 23%|██▎       | 92457/400000 [00:10<00:35, 8617.66it/s] 23%|██▎       | 93327/400000 [00:10<00:35, 8640.45it/s] 24%|██▎       | 94192/400000 [00:11<00:35, 8584.86it/s] 24%|██▍       | 95051/400000 [00:11<00:36, 8470.01it/s] 24%|██▍       | 95915/400000 [00:11<00:35, 8520.20it/s] 24%|██▍       | 96782/400000 [00:11<00:35, 8563.68it/s] 24%|██▍       | 97654/400000 [00:11<00:35, 8608.33it/s] 25%|██▍       | 98516/400000 [00:11<00:35, 8599.69it/s] 25%|██▍       | 99381/400000 [00:11<00:34, 8612.97it/s] 25%|██▌       | 100247/400000 [00:11<00:34, 8626.08it/s] 25%|██▌       | 101116/400000 [00:11<00:34, 8643.05it/s] 25%|██▌       | 101988/400000 [00:11<00:34, 8664.31it/s] 26%|██▌       | 102855/400000 [00:12<00:34, 8651.63it/s] 26%|██▌       | 103721/400000 [00:12<00:34, 8614.89it/s] 26%|██▌       | 104600/400000 [00:12<00:34, 8665.85it/s] 26%|██▋       | 105467/400000 [00:12<00:34, 8617.84it/s] 27%|██▋       | 106329/400000 [00:12<00:34, 8580.40it/s] 27%|██▋       | 107188/400000 [00:12<00:34, 8538.56it/s] 27%|██▋       | 108043/400000 [00:12<00:34, 8471.16it/s] 27%|██▋       | 108926/400000 [00:12<00:33, 8573.39it/s] 27%|██▋       | 109794/400000 [00:12<00:33, 8604.26it/s] 28%|██▊       | 110664/400000 [00:12<00:33, 8632.46it/s] 28%|██▊       | 111543/400000 [00:13<00:33, 8678.45it/s] 28%|██▊       | 112412/400000 [00:13<00:33, 8644.85it/s] 28%|██▊       | 113277/400000 [00:13<00:33, 8576.53it/s] 29%|██▊       | 114135/400000 [00:13<00:33, 8546.72it/s] 29%|██▉       | 115015/400000 [00:13<00:33, 8618.41it/s] 29%|██▉       | 115892/400000 [00:13<00:32, 8661.60it/s] 29%|██▉       | 116759/400000 [00:13<00:32, 8615.07it/s] 29%|██▉       | 117632/400000 [00:13<00:32, 8648.37it/s] 30%|██▉       | 118503/400000 [00:13<00:32, 8666.62it/s] 30%|██▉       | 119374/400000 [00:13<00:32, 8678.27it/s] 30%|███       | 120242/400000 [00:14<00:32, 8672.72it/s] 30%|███       | 121110/400000 [00:14<00:32, 8534.92it/s] 30%|███       | 121991/400000 [00:14<00:32, 8612.79it/s] 31%|███       | 122864/400000 [00:14<00:32, 8646.64it/s] 31%|███       | 123749/400000 [00:14<00:31, 8704.74it/s] 31%|███       | 124620/400000 [00:14<00:31, 8692.94it/s] 31%|███▏      | 125490/400000 [00:14<00:31, 8615.41it/s] 32%|███▏      | 126370/400000 [00:14<00:31, 8667.56it/s] 32%|███▏      | 127252/400000 [00:14<00:31, 8710.33it/s] 32%|███▏      | 128124/400000 [00:14<00:31, 8666.74it/s] 32%|███▏      | 128991/400000 [00:15<00:31, 8608.92it/s] 32%|███▏      | 129853/400000 [00:15<00:32, 8425.18it/s] 33%|███▎      | 130706/400000 [00:15<00:31, 8456.14it/s] 33%|███▎      | 131580/400000 [00:15<00:31, 8538.96it/s] 33%|███▎      | 132442/400000 [00:15<00:31, 8560.38it/s] 33%|███▎      | 133325/400000 [00:15<00:30, 8636.81it/s] 34%|███▎      | 134196/400000 [00:15<00:30, 8657.92it/s] 34%|███▍      | 135072/400000 [00:15<00:30, 8685.48it/s] 34%|███▍      | 135956/400000 [00:15<00:30, 8729.45it/s] 34%|███▍      | 136830/400000 [00:15<00:30, 8645.27it/s] 34%|███▍      | 137695/400000 [00:16<00:30, 8482.41it/s] 35%|███▍      | 138545/400000 [00:16<00:30, 8485.17it/s] 35%|███▍      | 139429/400000 [00:16<00:30, 8586.58it/s] 35%|███▌      | 140297/400000 [00:16<00:30, 8613.22it/s] 35%|███▌      | 141187/400000 [00:16<00:29, 8696.36it/s] 36%|███▌      | 142058/400000 [00:16<00:29, 8627.08it/s] 36%|███▌      | 142925/400000 [00:16<00:29, 8638.24it/s] 36%|███▌      | 143801/400000 [00:16<00:29, 8671.69it/s] 36%|███▌      | 144669/400000 [00:16<00:29, 8604.99it/s] 36%|███▋      | 145543/400000 [00:16<00:29, 8641.92it/s] 37%|███▋      | 146408/400000 [00:17<00:29, 8563.77it/s] 37%|███▋      | 147280/400000 [00:17<00:29, 8608.70it/s] 37%|███▋      | 148161/400000 [00:17<00:29, 8667.62it/s] 37%|███▋      | 149030/400000 [00:17<00:28, 8672.07it/s] 37%|███▋      | 149917/400000 [00:17<00:28, 8728.92it/s] 38%|███▊      | 150797/400000 [00:17<00:28, 8749.09it/s] 38%|███▊      | 151673/400000 [00:17<00:28, 8699.07it/s] 38%|███▊      | 152544/400000 [00:17<00:28, 8691.85it/s] 38%|███▊      | 153414/400000 [00:17<00:28, 8640.23it/s] 39%|███▊      | 154285/400000 [00:18<00:28, 8660.48it/s] 39%|███▉      | 155162/400000 [00:18<00:28, 8692.46it/s] 39%|███▉      | 156032/400000 [00:18<00:28, 8617.08it/s] 39%|███▉      | 156902/400000 [00:18<00:28, 8639.52it/s] 39%|███▉      | 157781/400000 [00:18<00:27, 8683.11it/s] 40%|███▉      | 158650/400000 [00:18<00:28, 8503.57it/s] 40%|███▉      | 159502/400000 [00:18<00:28, 8398.74it/s] 40%|████      | 160354/400000 [00:18<00:28, 8431.98it/s] 40%|████      | 161220/400000 [00:18<00:28, 8497.07it/s] 41%|████      | 162117/400000 [00:18<00:27, 8631.91it/s] 41%|████      | 163009/400000 [00:19<00:27, 8714.12it/s] 41%|████      | 163882/400000 [00:19<00:27, 8684.79it/s] 41%|████      | 164770/400000 [00:19<00:26, 8739.74it/s] 41%|████▏     | 165645/400000 [00:19<00:26, 8703.40it/s] 42%|████▏     | 166516/400000 [00:19<00:26, 8675.10it/s] 42%|████▏     | 167384/400000 [00:19<00:27, 8603.94it/s] 42%|████▏     | 168257/400000 [00:19<00:26, 8640.97it/s] 42%|████▏     | 169122/400000 [00:19<00:27, 8543.21it/s] 42%|████▏     | 169998/400000 [00:19<00:26, 8605.31it/s] 43%|████▎     | 170859/400000 [00:19<00:26, 8605.33it/s] 43%|████▎     | 171720/400000 [00:20<00:26, 8507.70it/s] 43%|████▎     | 172575/400000 [00:20<00:26, 8515.45it/s] 43%|████▎     | 173427/400000 [00:20<00:27, 8302.33it/s] 44%|████▎     | 174292/400000 [00:20<00:26, 8403.62it/s] 44%|████▍     | 175168/400000 [00:20<00:26, 8506.75it/s] 44%|████▍     | 176046/400000 [00:20<00:26, 8585.00it/s] 44%|████▍     | 176906/400000 [00:20<00:26, 8532.50it/s] 44%|████▍     | 177791/400000 [00:20<00:25, 8622.76it/s] 45%|████▍     | 178662/400000 [00:20<00:25, 8645.97it/s] 45%|████▍     | 179542/400000 [00:20<00:25, 8690.01it/s] 45%|████▌     | 180412/400000 [00:21<00:25, 8563.46it/s] 45%|████▌     | 181270/400000 [00:21<00:25, 8539.32it/s] 46%|████▌     | 182137/400000 [00:21<00:25, 8576.11it/s] 46%|████▌     | 182996/400000 [00:21<00:25, 8463.45it/s] 46%|████▌     | 183843/400000 [00:21<00:25, 8424.67it/s] 46%|████▌     | 184736/400000 [00:21<00:25, 8568.08it/s] 46%|████▋     | 185627/400000 [00:21<00:24, 8667.30it/s] 47%|████▋     | 186514/400000 [00:21<00:24, 8724.64it/s] 47%|████▋     | 187388/400000 [00:21<00:24, 8715.29it/s] 47%|████▋     | 188261/400000 [00:21<00:25, 8301.14it/s] 47%|████▋     | 189132/400000 [00:22<00:25, 8417.24it/s] 48%|████▊     | 190005/400000 [00:22<00:24, 8506.30it/s] 48%|████▊     | 190882/400000 [00:22<00:24, 8581.80it/s] 48%|████▊     | 191743/400000 [00:22<00:24, 8549.28it/s] 48%|████▊     | 192627/400000 [00:22<00:24, 8629.58it/s] 48%|████▊     | 193492/400000 [00:22<00:23, 8633.82it/s] 49%|████▊     | 194369/400000 [00:22<00:23, 8672.60it/s] 49%|████▉     | 195244/400000 [00:22<00:23, 8695.66it/s] 49%|████▉     | 196115/400000 [00:22<00:23, 8670.28it/s] 49%|████▉     | 196991/400000 [00:22<00:23, 8696.69it/s] 49%|████▉     | 197861/400000 [00:23<00:23, 8689.42it/s] 50%|████▉     | 198731/400000 [00:23<00:23, 8679.83it/s] 50%|████▉     | 199600/400000 [00:23<00:23, 8520.32it/s] 50%|█████     | 200453/400000 [00:23<00:23, 8446.02it/s] 50%|█████     | 201299/400000 [00:23<00:23, 8400.42it/s] 51%|█████     | 202140/400000 [00:23<00:23, 8350.35it/s] 51%|█████     | 202994/400000 [00:23<00:23, 8405.72it/s] 51%|█████     | 203835/400000 [00:23<00:23, 8281.16it/s] 51%|█████     | 204664/400000 [00:23<00:23, 8263.87it/s] 51%|█████▏    | 205517/400000 [00:23<00:23, 8339.50it/s] 52%|█████▏    | 206384/400000 [00:24<00:22, 8435.14it/s] 52%|█████▏    | 207266/400000 [00:24<00:22, 8544.97it/s] 52%|█████▏    | 208141/400000 [00:24<00:22, 8602.98it/s] 52%|█████▏    | 209002/400000 [00:24<00:22, 8590.93it/s] 52%|█████▏    | 209885/400000 [00:24<00:21, 8659.67it/s] 53%|█████▎    | 210769/400000 [00:24<00:21, 8712.11it/s] 53%|█████▎    | 211641/400000 [00:24<00:21, 8679.84it/s] 53%|█████▎    | 212513/400000 [00:24<00:21, 8689.86it/s] 53%|█████▎    | 213383/400000 [00:24<00:21, 8635.73it/s] 54%|█████▎    | 214264/400000 [00:25<00:21, 8685.93it/s] 54%|█████▍    | 215133/400000 [00:25<00:21, 8632.35it/s] 54%|█████▍    | 215999/400000 [00:25<00:21, 8639.62it/s] 54%|█████▍    | 216864/400000 [00:25<00:21, 8633.59it/s] 54%|█████▍    | 217728/400000 [00:25<00:21, 8607.64it/s] 55%|█████▍    | 218589/400000 [00:25<00:21, 8515.21it/s] 55%|█████▍    | 219442/400000 [00:25<00:21, 8518.44it/s] 55%|█████▌    | 220324/400000 [00:25<00:20, 8606.44it/s] 55%|█████▌    | 221208/400000 [00:25<00:20, 8672.61it/s] 56%|█████▌    | 222076/400000 [00:25<00:20, 8659.75it/s] 56%|█████▌    | 222957/400000 [00:26<00:20, 8701.73it/s] 56%|█████▌    | 223842/400000 [00:26<00:20, 8743.06it/s] 56%|█████▌    | 224717/400000 [00:26<00:20, 8589.72it/s] 56%|█████▋    | 225577/400000 [00:26<00:20, 8418.23it/s] 57%|█████▋    | 226454/400000 [00:26<00:20, 8518.80it/s] 57%|█████▋    | 227338/400000 [00:26<00:20, 8612.49it/s] 57%|█████▋    | 228227/400000 [00:26<00:19, 8693.83it/s] 57%|█████▋    | 229113/400000 [00:26<00:19, 8740.20it/s] 57%|█████▋    | 229996/400000 [00:26<00:19, 8765.43it/s] 58%|█████▊    | 230874/400000 [00:26<00:19, 8693.06it/s] 58%|█████▊    | 231749/400000 [00:27<00:19, 8709.67it/s] 58%|█████▊    | 232631/400000 [00:27<00:19, 8741.15it/s] 58%|█████▊    | 233506/400000 [00:27<00:19, 8687.05it/s] 59%|█████▊    | 234381/400000 [00:27<00:19, 8704.11it/s] 59%|█████▉    | 235256/400000 [00:27<00:18, 8717.70it/s] 59%|█████▉    | 236132/400000 [00:27<00:18, 8728.08it/s] 59%|█████▉    | 237005/400000 [00:27<00:18, 8590.74it/s] 59%|█████▉    | 237865/400000 [00:27<00:18, 8592.04it/s] 60%|█████▉    | 238726/400000 [00:27<00:18, 8597.33it/s] 60%|█████▉    | 239587/400000 [00:27<00:18, 8528.80it/s] 60%|██████    | 240441/400000 [00:28<00:19, 8284.92it/s] 60%|██████    | 241287/400000 [00:28<00:19, 8333.88it/s] 61%|██████    | 242122/400000 [00:28<00:19, 8154.42it/s] 61%|██████    | 242969/400000 [00:28<00:19, 8244.74it/s] 61%|██████    | 243800/400000 [00:28<00:18, 8262.51it/s] 61%|██████    | 244675/400000 [00:28<00:18, 8401.70it/s] 61%|██████▏   | 245556/400000 [00:28<00:18, 8519.42it/s] 62%|██████▏   | 246410/400000 [00:28<00:18, 8500.67it/s] 62%|██████▏   | 247261/400000 [00:28<00:18, 8468.07it/s] 62%|██████▏   | 248138/400000 [00:28<00:17, 8554.43it/s] 62%|██████▏   | 249007/400000 [00:29<00:17, 8558.56it/s] 62%|██████▏   | 249894/400000 [00:29<00:17, 8647.14it/s] 63%|██████▎   | 250779/400000 [00:29<00:17, 8706.28it/s] 63%|██████▎   | 251661/400000 [00:29<00:16, 8738.57it/s] 63%|██████▎   | 252541/400000 [00:29<00:16, 8755.06it/s] 63%|██████▎   | 253417/400000 [00:29<00:16, 8738.17it/s] 64%|██████▎   | 254292/400000 [00:29<00:16, 8732.17it/s] 64%|██████▍   | 255166/400000 [00:29<00:16, 8531.04it/s] 64%|██████▍   | 256021/400000 [00:29<00:17, 8429.57it/s] 64%|██████▍   | 256866/400000 [00:29<00:17, 8412.09it/s] 64%|██████▍   | 257708/400000 [00:30<00:16, 8379.79it/s] 65%|██████▍   | 258571/400000 [00:30<00:16, 8451.72it/s] 65%|██████▍   | 259447/400000 [00:30<00:16, 8540.58it/s] 65%|██████▌   | 260321/400000 [00:30<00:16, 8599.24it/s] 65%|██████▌   | 261182/400000 [00:30<00:16, 8559.32it/s] 66%|██████▌   | 262039/400000 [00:30<00:16, 8553.56it/s] 66%|██████▌   | 262911/400000 [00:30<00:15, 8601.07it/s] 66%|██████▌   | 263773/400000 [00:30<00:15, 8604.19it/s] 66%|██████▌   | 264634/400000 [00:30<00:15, 8564.30it/s] 66%|██████▋   | 265495/400000 [00:30<00:15, 8575.06it/s] 67%|██████▋   | 266353/400000 [00:31<00:15, 8464.72it/s] 67%|██████▋   | 267200/400000 [00:31<00:15, 8422.87it/s] 67%|██████▋   | 268051/400000 [00:31<00:15, 8447.84it/s] 67%|██████▋   | 268897/400000 [00:31<00:15, 8404.37it/s] 67%|██████▋   | 269779/400000 [00:31<00:15, 8524.24it/s] 68%|██████▊   | 270635/400000 [00:31<00:15, 8532.42it/s] 68%|██████▊   | 271532/400000 [00:31<00:14, 8657.55it/s] 68%|██████▊   | 272432/400000 [00:31<00:14, 8756.48it/s] 68%|██████▊   | 273321/400000 [00:31<00:14, 8795.81it/s] 69%|██████▊   | 274214/400000 [00:31<00:14, 8833.00it/s] 69%|██████▉   | 275098/400000 [00:32<00:14, 8636.40it/s] 69%|██████▉   | 275973/400000 [00:32<00:14, 8664.52it/s] 69%|██████▉   | 276857/400000 [00:32<00:14, 8715.29it/s] 69%|██████▉   | 277730/400000 [00:32<00:14, 8711.41it/s] 70%|██████▉   | 278605/400000 [00:32<00:13, 8720.79it/s] 70%|██████▉   | 279478/400000 [00:32<00:13, 8677.12it/s] 70%|███████   | 280347/400000 [00:32<00:13, 8674.11it/s] 70%|███████   | 281227/400000 [00:32<00:13, 8710.47it/s] 71%|███████   | 282107/400000 [00:32<00:13, 8736.62it/s] 71%|███████   | 282981/400000 [00:32<00:13, 8720.37it/s] 71%|███████   | 283854/400000 [00:33<00:13, 8563.09it/s] 71%|███████   | 284714/400000 [00:33<00:13, 8573.82it/s] 71%|███████▏  | 285579/400000 [00:33<00:13, 8596.09it/s] 72%|███████▏  | 286447/400000 [00:33<00:13, 8620.63it/s] 72%|███████▏  | 287310/400000 [00:33<00:13, 8317.29it/s] 72%|███████▏  | 288145/400000 [00:33<00:13, 8154.98it/s] 72%|███████▏  | 288985/400000 [00:33<00:13, 8226.28it/s] 72%|███████▏  | 289826/400000 [00:33<00:13, 8279.12it/s] 73%|███████▎  | 290689/400000 [00:33<00:13, 8378.77it/s] 73%|███████▎  | 291533/400000 [00:34<00:12, 8396.19it/s] 73%|███████▎  | 292374/400000 [00:34<00:13, 8264.84it/s] 73%|███████▎  | 293242/400000 [00:34<00:12, 8383.14it/s] 74%|███████▎  | 294082/400000 [00:34<00:12, 8202.40it/s] 74%|███████▎  | 294954/400000 [00:34<00:12, 8350.08it/s] 74%|███████▍  | 295824/400000 [00:34<00:12, 8450.06it/s] 74%|███████▍  | 296682/400000 [00:34<00:12, 8486.37it/s] 74%|███████▍  | 297532/400000 [00:34<00:12, 8472.15it/s] 75%|███████▍  | 298392/400000 [00:34<00:11, 8509.44it/s] 75%|███████▍  | 299273/400000 [00:34<00:11, 8594.69it/s] 75%|███████▌  | 300149/400000 [00:35<00:11, 8642.91it/s] 75%|███████▌  | 301014/400000 [00:35<00:11, 8642.88it/s] 75%|███████▌  | 301896/400000 [00:35<00:11, 8694.37it/s] 76%|███████▌  | 302766/400000 [00:35<00:11, 8694.39it/s] 76%|███████▌  | 303636/400000 [00:35<00:11, 8687.69it/s] 76%|███████▌  | 304514/400000 [00:35<00:10, 8713.69it/s] 76%|███████▋  | 305386/400000 [00:35<00:10, 8715.30it/s] 77%|███████▋  | 306262/400000 [00:35<00:10, 8726.04it/s] 77%|███████▋  | 307135/400000 [00:35<00:10, 8714.99it/s] 77%|███████▋  | 308007/400000 [00:35<00:10, 8627.76it/s] 77%|███████▋  | 308871/400000 [00:36<00:10, 8622.14it/s] 77%|███████▋  | 309734/400000 [00:36<00:10, 8619.85it/s] 78%|███████▊  | 310597/400000 [00:36<00:10, 8500.86it/s] 78%|███████▊  | 311448/400000 [00:36<00:10, 8051.48it/s] 78%|███████▊  | 312298/400000 [00:36<00:10, 8180.34it/s] 78%|███████▊  | 313131/400000 [00:36<00:10, 8224.22it/s] 78%|███████▊  | 313966/400000 [00:36<00:10, 8259.57it/s] 79%|███████▊  | 314829/400000 [00:36<00:10, 8365.90it/s] 79%|███████▉  | 315690/400000 [00:36<00:09, 8436.77it/s] 79%|███████▉  | 316556/400000 [00:36<00:09, 8499.95it/s] 79%|███████▉  | 317428/400000 [00:37<00:09, 8563.92it/s] 80%|███████▉  | 318289/400000 [00:37<00:09, 8575.87it/s] 80%|███████▉  | 319165/400000 [00:37<00:09, 8629.96it/s] 80%|████████  | 320036/400000 [00:37<00:09, 8652.67it/s] 80%|████████  | 320902/400000 [00:37<00:09, 8626.86it/s] 80%|████████  | 321774/400000 [00:37<00:09, 8654.29it/s] 81%|████████  | 322647/400000 [00:37<00:08, 8671.90it/s] 81%|████████  | 323519/400000 [00:37<00:08, 8684.70it/s] 81%|████████  | 324388/400000 [00:37<00:09, 8252.68it/s] 81%|████████▏ | 325239/400000 [00:37<00:08, 8325.74it/s] 82%|████████▏ | 326110/400000 [00:38<00:08, 8435.74it/s] 82%|████████▏ | 326957/400000 [00:38<00:08, 8226.59it/s] 82%|████████▏ | 327783/400000 [00:38<00:09, 7942.36it/s] 82%|████████▏ | 328588/400000 [00:38<00:08, 7972.91it/s] 82%|████████▏ | 329389/400000 [00:38<00:09, 7841.05it/s] 83%|████████▎ | 330187/400000 [00:38<00:08, 7881.33it/s] 83%|████████▎ | 331049/400000 [00:38<00:08, 8087.97it/s] 83%|████████▎ | 331861/400000 [00:38<00:08, 8073.82it/s] 83%|████████▎ | 332671/400000 [00:38<00:08, 8059.65it/s] 83%|████████▎ | 333490/400000 [00:39<00:08, 8097.34it/s] 84%|████████▎ | 334301/400000 [00:39<00:08, 7781.80it/s] 84%|████████▍ | 335133/400000 [00:39<00:08, 7934.23it/s] 84%|████████▍ | 335985/400000 [00:39<00:07, 8099.57it/s] 84%|████████▍ | 336798/400000 [00:39<00:07, 8008.03it/s] 84%|████████▍ | 337602/400000 [00:39<00:07, 7846.38it/s] 85%|████████▍ | 338413/400000 [00:39<00:07, 7923.52it/s] 85%|████████▍ | 339208/400000 [00:39<00:07, 7634.25it/s] 85%|████████▌ | 340059/400000 [00:39<00:07, 7877.04it/s] 85%|████████▌ | 340879/400000 [00:39<00:07, 7968.98it/s] 85%|████████▌ | 341680/400000 [00:40<00:07, 7840.85it/s] 86%|████████▌ | 342467/400000 [00:40<00:07, 7742.46it/s] 86%|████████▌ | 343319/400000 [00:40<00:07, 7958.02it/s] 86%|████████▌ | 344174/400000 [00:40<00:06, 8124.87it/s] 86%|████████▌ | 344990/400000 [00:40<00:06, 8045.58it/s] 86%|████████▋ | 345797/400000 [00:40<00:07, 7711.89it/s] 87%|████████▋ | 346590/400000 [00:40<00:06, 7774.82it/s] 87%|████████▋ | 347371/400000 [00:40<00:07, 7445.30it/s] 87%|████████▋ | 348239/400000 [00:40<00:06, 7776.11it/s] 87%|████████▋ | 349103/400000 [00:40<00:06, 8015.87it/s] 87%|████████▋ | 349912/400000 [00:41<00:06, 7499.17it/s] 88%|████████▊ | 350703/400000 [00:41<00:06, 7615.45it/s] 88%|████████▊ | 351474/400000 [00:41<00:06, 7373.44it/s] 88%|████████▊ | 352220/400000 [00:41<00:06, 7106.63it/s] 88%|████████▊ | 352968/400000 [00:41<00:06, 7213.00it/s] 88%|████████▊ | 353699/400000 [00:41<00:06, 7227.72it/s] 89%|████████▊ | 354480/400000 [00:41<00:06, 7392.33it/s] 89%|████████▉ | 355246/400000 [00:41<00:05, 7470.60it/s] 89%|████████▉ | 356090/400000 [00:41<00:05, 7735.37it/s] 89%|████████▉ | 356898/400000 [00:42<00:05, 7835.29it/s] 89%|████████▉ | 357685/400000 [00:42<00:05, 7288.27it/s] 90%|████████▉ | 358475/400000 [00:42<00:05, 7460.42it/s] 90%|████████▉ | 359322/400000 [00:42<00:05, 7736.65it/s] 90%|█████████ | 360104/400000 [00:42<00:05, 7732.68it/s] 90%|█████████ | 360924/400000 [00:42<00:04, 7866.27it/s] 90%|█████████ | 361774/400000 [00:42<00:04, 8045.16it/s] 91%|█████████ | 362593/400000 [00:42<00:04, 8085.98it/s] 91%|█████████ | 363405/400000 [00:42<00:04, 7854.02it/s] 91%|█████████ | 364199/400000 [00:42<00:04, 7877.51it/s] 91%|█████████▏| 365071/400000 [00:43<00:04, 8110.69it/s] 91%|█████████▏| 365911/400000 [00:43<00:04, 8193.85it/s] 92%|█████████▏| 366774/400000 [00:43<00:03, 8318.53it/s] 92%|█████████▏| 367621/400000 [00:43<00:03, 8361.53it/s] 92%|█████████▏| 368459/400000 [00:43<00:03, 8300.33it/s] 92%|█████████▏| 369291/400000 [00:43<00:03, 8239.41it/s] 93%|█████████▎| 370116/400000 [00:43<00:03, 7896.07it/s] 93%|█████████▎| 370910/400000 [00:43<00:03, 7724.31it/s] 93%|█████████▎| 371686/400000 [00:43<00:03, 7130.11it/s] 93%|█████████▎| 372466/400000 [00:44<00:03, 7316.60it/s] 93%|█████████▎| 373207/400000 [00:44<00:03, 7257.34it/s] 93%|█████████▎| 373989/400000 [00:44<00:03, 7415.33it/s] 94%|█████████▎| 374821/400000 [00:44<00:03, 7664.30it/s] 94%|█████████▍| 375687/400000 [00:44<00:03, 7937.67it/s] 94%|█████████▍| 376545/400000 [00:44<00:02, 8118.39it/s] 94%|█████████▍| 377398/400000 [00:44<00:02, 8235.05it/s] 95%|█████████▍| 378235/400000 [00:44<00:02, 8263.46it/s] 95%|█████████▍| 379065/400000 [00:44<00:02, 7993.53it/s] 95%|█████████▍| 379881/400000 [00:44<00:02, 8042.11it/s] 95%|█████████▌| 380689/400000 [00:45<00:02, 7882.30it/s] 95%|█████████▌| 381480/400000 [00:45<00:02, 7803.25it/s] 96%|█████████▌| 382263/400000 [00:45<00:02, 7737.30it/s] 96%|█████████▌| 383072/400000 [00:45<00:02, 7837.71it/s] 96%|█████████▌| 383859/400000 [00:45<00:02, 7846.69it/s] 96%|█████████▌| 384719/400000 [00:45<00:01, 8057.44it/s] 96%|█████████▋| 385527/400000 [00:45<00:01, 7781.40it/s] 97%|█████████▋| 386309/400000 [00:45<00:01, 7460.69it/s] 97%|█████████▋| 387117/400000 [00:45<00:01, 7635.60it/s] 97%|█████████▋| 387886/400000 [00:46<00:01, 7647.64it/s] 97%|█████████▋| 388678/400000 [00:46<00:01, 7725.96it/s] 97%|█████████▋| 389466/400000 [00:46<00:01, 7770.35it/s] 98%|█████████▊| 390290/400000 [00:46<00:01, 7903.51it/s] 98%|█████████▊| 391105/400000 [00:46<00:01, 7975.60it/s] 98%|█████████▊| 391975/400000 [00:46<00:00, 8179.62it/s] 98%|█████████▊| 392816/400000 [00:46<00:00, 8247.12it/s] 98%|█████████▊| 393643/400000 [00:46<00:00, 8135.43it/s] 99%|█████████▊| 394469/400000 [00:46<00:00, 8171.95it/s] 99%|█████████▉| 395288/400000 [00:46<00:00, 7896.13it/s] 99%|█████████▉| 396133/400000 [00:47<00:00, 8054.41it/s] 99%|█████████▉| 396993/400000 [00:47<00:00, 8208.62it/s] 99%|█████████▉| 397878/400000 [00:47<00:00, 8390.02it/s]100%|█████████▉| 398743/400000 [00:47<00:00, 8463.43it/s]100%|█████████▉| 399598/400000 [00:47<00:00, 8467.95it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8425.94it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f082b14ac18> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011543291832756075 	 Accuracy: 54
Train Epoch: 1 	 Loss: 0.01126972907362973 	 Accuracy: 52

  model saves at 52% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15863 out of table with 15833 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15863 out of table with 15833 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-14 05:23:56.806931: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 05:23:56.810795: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095090000 Hz
2020-05-14 05:23:56.810941: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c2a510c2a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 05:23:56.810954: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f07de688160> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.3906 - accuracy: 0.5180
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4826 - accuracy: 0.5120 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6002 - accuracy: 0.5043
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5708 - accuracy: 0.5063
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6206 - accuracy: 0.5030
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5772 - accuracy: 0.5058
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5199 - accuracy: 0.5096
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5880 - accuracy: 0.5051
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6053 - accuracy: 0.5040
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6314 - accuracy: 0.5023
11000/25000 [============>.................] - ETA: 3s - loss: 7.6415 - accuracy: 0.5016
12000/25000 [=============>................] - ETA: 3s - loss: 7.6590 - accuracy: 0.5005
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6242 - accuracy: 0.5028
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6436 - accuracy: 0.5015
15000/25000 [=================>............] - ETA: 2s - loss: 7.6513 - accuracy: 0.5010
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6551 - accuracy: 0.5008
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6585 - accuracy: 0.5005
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6394 - accuracy: 0.5018
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6528 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6484 - accuracy: 0.5012
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6520 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
25000/25000 [==============================] - 7s 297us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f0797b4fda0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f07a837e128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.6888 - crf_viterbi_accuracy: 0.1067 - val_loss: 1.6655 - val_crf_viterbi_accuracy: 0.1200

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

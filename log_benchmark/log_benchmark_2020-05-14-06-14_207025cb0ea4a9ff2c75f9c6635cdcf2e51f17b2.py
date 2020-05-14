
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fd52b05ef98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 06:14:24.980932
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 06:14:24.985412
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 06:14:24.989333
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 06:14:24.993211
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fd536e284a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355117.8750
Epoch 2/10

1/1 [==============================] - 0s 105ms/step - loss: 251565.2656
Epoch 3/10

1/1 [==============================] - 0s 102ms/step - loss: 157377.4219
Epoch 4/10

1/1 [==============================] - 0s 103ms/step - loss: 86794.2734
Epoch 5/10

1/1 [==============================] - 0s 101ms/step - loss: 46565.6914
Epoch 6/10

1/1 [==============================] - 0s 101ms/step - loss: 26036.8965
Epoch 7/10

1/1 [==============================] - 0s 103ms/step - loss: 15472.1797
Epoch 8/10

1/1 [==============================] - 0s 109ms/step - loss: 9959.8623
Epoch 9/10

1/1 [==============================] - 0s 105ms/step - loss: 6920.8721
Epoch 10/10

1/1 [==============================] - 0s 104ms/step - loss: 5178.7817

  #### Inference Need return ypred, ytrue ######################### 
[[  0.9696882    0.5099982   -0.4495637    1.3678095   -0.02066809
   -1.7368366    1.0222514   -1.7298577    1.6992407    0.11138624
    1.1805947   -1.2663928   -0.868034    -0.84833384  -1.1984432
    0.2206878    1.4065938   -1.090604    -1.3505056   -1.0047562
    0.19675648  -0.85113865  -1.1704559   -0.32906428  -1.6155019
   -1.7964998    1.0611683   -0.3325419   -0.09009796   0.37846428
    0.1167469   -0.30957335   0.5703931    1.9380788    1.22295
   -0.9936274    0.60006654   0.3220386   -0.41070333   0.6481957
   -1.0221179   -0.86417264   1.8007164   -0.97526485  -0.14137784
   -1.51617     -0.01910535   1.7244623    1.3406498    1.0046792
   -1.2276785    0.26187897   0.9008666   -0.13554361   0.9135816
    0.20586716  -0.5927589   -0.57341915  -0.8862524   -1.6795777
    0.47683066   9.18834      7.5438395    8.669246     9.287164
    8.899149     8.40032      9.398826    11.150917    10.500247
   10.459236     9.315413    10.26266     10.208714    10.034402
   10.094105     8.9144745    9.025121    11.019404     9.740769
   10.635798     7.651289    10.438119    10.076966    10.125413
   10.172227    10.920748     9.145823     9.449882     8.875457
    7.8211813   10.362959     9.363053     9.17398      8.793019
    8.808857     8.362746     8.581834     8.824718    10.084456
    7.606159    10.639422    10.026788    11.023663    10.974856
    8.742874    10.759473    10.33323      8.209815     9.49038
   10.966437     7.094109     9.118573    11.10118      8.802382
    9.717052     9.248443    11.13578     11.235427     9.93922
    1.0083429   -0.63139      0.23383258   1.6248158   -0.34455827
   -0.23533171   0.05166906   0.83800864   2.0234747   -0.50227094
   -1.5438864   -0.2835393   -3.1051931    0.05455178   0.37194854
    1.6881254    2.2570705   -1.5016102   -0.67608833  -2.9939876
    1.9992926   -0.8994178   -0.32075924   0.68351156  -0.16337466
   -1.7799344    0.8728483   -0.6774398    0.49379435   0.47481608
   -0.32615006  -0.21955758  -1.011266     0.43427673  -0.7807963
    0.16756018   0.11724606   0.19959402  -0.667251     0.35494488
   -0.11752582   1.4464135    1.1142917    0.9814624   -1.5319966
   -0.62039495  -1.0415163    1.346199    -1.6518831   -1.3929641
   -1.2385365    0.38072377   0.99958134   2.1994014    0.22545773
   -1.2145778   -0.62923354  -0.757995     1.0436666   -1.0826337
    1.2855463    0.98743546   1.1187596    0.50144273   2.3610191
    0.95671      1.5653894    1.3356614    1.6196502    0.605301
    0.16209364   0.2651552    0.7751711    1.6176629    2.5157928
    1.1801066    0.5015608    0.8924833    1.0252382    3.2519436
    1.7650802    0.82067966   1.9640594    2.7890606    3.7074285
    1.866438     0.3107798    0.32161498   2.8421645    0.5673845
    0.21153778   1.2363065    1.2367065    2.0855777    0.15819049
    1.4420724    0.27600646   2.343318     1.4378399    0.7425287
    1.2639809    0.7206392    0.26899874   0.46714938   1.1445842
    0.27247834   0.7946476    0.48805273   0.8580763    0.4859513
    2.46615      0.44409657   1.9713461    1.9228169    2.8348708
    0.84368837   0.9293289    0.5097202    0.998088     1.3478501
    0.11178654  10.441399    11.801768     9.569278     8.575121
   11.310746     8.952032    10.736122    10.47905     11.403899
   12.16055      7.2527876    9.966646     8.584134    10.317153
    8.411837     9.171019    10.792755    10.426058     9.512741
    9.569178     7.9436345    9.5869       9.129591     9.997763
   11.211731     9.301798     9.746872     8.536023     9.273956
    8.237613     9.511142     9.540731    10.520072     7.5154204
   10.005889    10.270242    10.802746    10.946369     9.448683
   10.186379    10.042717     9.401737     9.478598     8.751446
   10.17688     11.356877     9.670277     8.963519     8.731653
    9.2149315   10.450594    10.38008      9.170107    10.485322
    9.984003     9.666353     9.528283     8.509435    10.617001
    0.9724111    2.8397064    0.21262813   1.3360636    1.8181809
    1.2566267    2.0631015    1.2938449    1.8488898    0.6199804
    3.0761971    0.2669549    0.62337804   0.89363766   1.5241591
    0.16425443   1.0574498    2.7773328    0.19569635   0.8439548
    1.9820212    0.964442     0.3581801    0.8047333    0.05089617
    3.2129216    0.30714428   0.62781      1.1297691    1.1284693
    2.2635393    1.4438641    0.39883113   1.5792398    1.4029593
    0.8915464    0.40976864   1.4563062    0.28910285   2.9314404
    1.6847372    0.31512254   0.37005544   0.07243609   1.7876995
    1.6917131    1.2384458    0.8311516    2.0305133    0.20845187
    1.5140755    1.2178586    0.24888545   0.6428001    0.34530342
    0.6148275    1.2757138    0.647996     0.83931017   1.3779428
   -6.536357     6.901878   -11.7636385 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 06:14:35.280813
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.1347
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 06:14:35.285796
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8515.88
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 06:14:35.289859
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.7672
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 06:14:35.294546
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -761.657
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140553184650800
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140552091640440
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140552091640944
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140552091641448
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140552091641952
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140552091642456

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fd532ca7f28> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.790373
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.751493
grad_step = 000002, loss = 0.722954
grad_step = 000003, loss = 0.692637
grad_step = 000004, loss = 0.658395
grad_step = 000005, loss = 0.620092
grad_step = 000006, loss = 0.580613
grad_step = 000007, loss = 0.544317
grad_step = 000008, loss = 0.509472
grad_step = 000009, loss = 0.474678
grad_step = 000010, loss = 0.447857
grad_step = 000011, loss = 0.434584
grad_step = 000012, loss = 0.417924
grad_step = 000013, loss = 0.397243
grad_step = 000014, loss = 0.379869
grad_step = 000015, loss = 0.367758
grad_step = 000016, loss = 0.357927
grad_step = 000017, loss = 0.346825
grad_step = 000018, loss = 0.333031
grad_step = 000019, loss = 0.317046
grad_step = 000020, loss = 0.300584
grad_step = 000021, loss = 0.285735
grad_step = 000022, loss = 0.273858
grad_step = 000023, loss = 0.263545
grad_step = 000024, loss = 0.252000
grad_step = 000025, loss = 0.239184
grad_step = 000026, loss = 0.226736
grad_step = 000027, loss = 0.215995
grad_step = 000028, loss = 0.206363
grad_step = 000029, loss = 0.196627
grad_step = 000030, loss = 0.186456
grad_step = 000031, loss = 0.176063
grad_step = 000032, loss = 0.165916
grad_step = 000033, loss = 0.156639
grad_step = 000034, loss = 0.148229
grad_step = 000035, loss = 0.140246
grad_step = 000036, loss = 0.132114
grad_step = 000037, loss = 0.123831
grad_step = 000038, loss = 0.115993
grad_step = 000039, loss = 0.108719
grad_step = 000040, loss = 0.101779
grad_step = 000041, loss = 0.095014
grad_step = 000042, loss = 0.088482
grad_step = 000043, loss = 0.082355
grad_step = 000044, loss = 0.076665
grad_step = 000045, loss = 0.071140
grad_step = 000046, loss = 0.065719
grad_step = 000047, loss = 0.060551
grad_step = 000048, loss = 0.055755
grad_step = 000049, loss = 0.051296
grad_step = 000050, loss = 0.047020
grad_step = 000051, loss = 0.042894
grad_step = 000052, loss = 0.039078
grad_step = 000053, loss = 0.035572
grad_step = 000054, loss = 0.032271
grad_step = 000055, loss = 0.029082
grad_step = 000056, loss = 0.026064
grad_step = 000057, loss = 0.023379
grad_step = 000058, loss = 0.020973
grad_step = 000059, loss = 0.018737
grad_step = 000060, loss = 0.016604
grad_step = 000061, loss = 0.014654
grad_step = 000062, loss = 0.012955
grad_step = 000063, loss = 0.011461
grad_step = 000064, loss = 0.010099
grad_step = 000065, loss = 0.008863
grad_step = 000066, loss = 0.007799
grad_step = 000067, loss = 0.006889
grad_step = 000068, loss = 0.006101
grad_step = 000069, loss = 0.005413
grad_step = 000070, loss = 0.004838
grad_step = 000071, loss = 0.004364
grad_step = 000072, loss = 0.003973
grad_step = 000073, loss = 0.003643
grad_step = 000074, loss = 0.003377
grad_step = 000075, loss = 0.003165
grad_step = 000076, loss = 0.003001
grad_step = 000077, loss = 0.002872
grad_step = 000078, loss = 0.002773
grad_step = 000079, loss = 0.002694
grad_step = 000080, loss = 0.002634
grad_step = 000081, loss = 0.002592
grad_step = 000082, loss = 0.002557
grad_step = 000083, loss = 0.002523
grad_step = 000084, loss = 0.002491
grad_step = 000085, loss = 0.002470
grad_step = 000086, loss = 0.002453
grad_step = 000087, loss = 0.002433
grad_step = 000088, loss = 0.002416
grad_step = 000089, loss = 0.002415
grad_step = 000090, loss = 0.002451
grad_step = 000091, loss = 0.002539
grad_step = 000092, loss = 0.002672
grad_step = 000093, loss = 0.002657
grad_step = 000094, loss = 0.002423
grad_step = 000095, loss = 0.002203
grad_step = 000096, loss = 0.002262
grad_step = 000097, loss = 0.002386
grad_step = 000098, loss = 0.002268
grad_step = 000099, loss = 0.002097
grad_step = 000100, loss = 0.002136
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002207
grad_step = 000102, loss = 0.002111
grad_step = 000103, loss = 0.002012
grad_step = 000104, loss = 0.002067
grad_step = 000105, loss = 0.002095
grad_step = 000106, loss = 0.002003
grad_step = 000107, loss = 0.001966
grad_step = 000108, loss = 0.002020
grad_step = 000109, loss = 0.002009
grad_step = 000110, loss = 0.001944
grad_step = 000111, loss = 0.001941
grad_step = 000112, loss = 0.001977
grad_step = 000113, loss = 0.001956
grad_step = 000114, loss = 0.001913
grad_step = 000115, loss = 0.001921
grad_step = 000116, loss = 0.001943
grad_step = 000117, loss = 0.001920
grad_step = 000118, loss = 0.001892
grad_step = 000119, loss = 0.001899
grad_step = 000120, loss = 0.001911
grad_step = 000121, loss = 0.001896
grad_step = 000122, loss = 0.001873
grad_step = 000123, loss = 0.001875
grad_step = 000124, loss = 0.001883
grad_step = 000125, loss = 0.001876
grad_step = 000126, loss = 0.001857
grad_step = 000127, loss = 0.001851
grad_step = 000128, loss = 0.001855
grad_step = 000129, loss = 0.001856
grad_step = 000130, loss = 0.001844
grad_step = 000131, loss = 0.001833
grad_step = 000132, loss = 0.001829
grad_step = 000133, loss = 0.001831
grad_step = 000134, loss = 0.001830
grad_step = 000135, loss = 0.001823
grad_step = 000136, loss = 0.001815
grad_step = 000137, loss = 0.001809
grad_step = 000138, loss = 0.001806
grad_step = 000139, loss = 0.001805
grad_step = 000140, loss = 0.001804
grad_step = 000141, loss = 0.001802
grad_step = 000142, loss = 0.001798
grad_step = 000143, loss = 0.001792
grad_step = 000144, loss = 0.001787
grad_step = 000145, loss = 0.001782
grad_step = 000146, loss = 0.001777
grad_step = 000147, loss = 0.001774
grad_step = 000148, loss = 0.001770
grad_step = 000149, loss = 0.001767
grad_step = 000150, loss = 0.001764
grad_step = 000151, loss = 0.001761
grad_step = 000152, loss = 0.001758
grad_step = 000153, loss = 0.001755
grad_step = 000154, loss = 0.001754
grad_step = 000155, loss = 0.001756
grad_step = 000156, loss = 0.001767
grad_step = 000157, loss = 0.001805
grad_step = 000158, loss = 0.001920
grad_step = 000159, loss = 0.002177
grad_step = 000160, loss = 0.002696
grad_step = 000161, loss = 0.002891
grad_step = 000162, loss = 0.002432
grad_step = 000163, loss = 0.001753
grad_step = 000164, loss = 0.002075
grad_step = 000165, loss = 0.002503
grad_step = 000166, loss = 0.001975
grad_step = 000167, loss = 0.001786
grad_step = 000168, loss = 0.002215
grad_step = 000169, loss = 0.002026
grad_step = 000170, loss = 0.001773
grad_step = 000171, loss = 0.001982
grad_step = 000172, loss = 0.001965
grad_step = 000173, loss = 0.001807
grad_step = 000174, loss = 0.001845
grad_step = 000175, loss = 0.001883
grad_step = 000176, loss = 0.001817
grad_step = 000177, loss = 0.001773
grad_step = 000178, loss = 0.001827
grad_step = 000179, loss = 0.001814
grad_step = 000180, loss = 0.001723
grad_step = 000181, loss = 0.001807
grad_step = 000182, loss = 0.001793
grad_step = 000183, loss = 0.001700
grad_step = 000184, loss = 0.001794
grad_step = 000185, loss = 0.001768
grad_step = 000186, loss = 0.001692
grad_step = 000187, loss = 0.001776
grad_step = 000188, loss = 0.001734
grad_step = 000189, loss = 0.001701
grad_step = 000190, loss = 0.001739
grad_step = 000191, loss = 0.001710
grad_step = 000192, loss = 0.001708
grad_step = 000193, loss = 0.001701
grad_step = 000194, loss = 0.001693
grad_step = 000195, loss = 0.001707
grad_step = 000196, loss = 0.001674
grad_step = 000197, loss = 0.001683
grad_step = 000198, loss = 0.001694
grad_step = 000199, loss = 0.001664
grad_step = 000200, loss = 0.001674
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001676
grad_step = 000202, loss = 0.001663
grad_step = 000203, loss = 0.001665
grad_step = 000204, loss = 0.001658
grad_step = 000205, loss = 0.001659
grad_step = 000206, loss = 0.001660
grad_step = 000207, loss = 0.001646
grad_step = 000208, loss = 0.001650
grad_step = 000209, loss = 0.001652
grad_step = 000210, loss = 0.001642
grad_step = 000211, loss = 0.001642
grad_step = 000212, loss = 0.001639
grad_step = 000213, loss = 0.001636
grad_step = 000214, loss = 0.001636
grad_step = 000215, loss = 0.001631
grad_step = 000216, loss = 0.001628
grad_step = 000217, loss = 0.001630
grad_step = 000218, loss = 0.001625
grad_step = 000219, loss = 0.001623
grad_step = 000220, loss = 0.001622
grad_step = 000221, loss = 0.001619
grad_step = 000222, loss = 0.001618
grad_step = 000223, loss = 0.001617
grad_step = 000224, loss = 0.001613
grad_step = 000225, loss = 0.001611
grad_step = 000226, loss = 0.001610
grad_step = 000227, loss = 0.001608
grad_step = 000228, loss = 0.001607
grad_step = 000229, loss = 0.001605
grad_step = 000230, loss = 0.001602
grad_step = 000231, loss = 0.001600
grad_step = 000232, loss = 0.001599
grad_step = 000233, loss = 0.001597
grad_step = 000234, loss = 0.001595
grad_step = 000235, loss = 0.001594
grad_step = 000236, loss = 0.001592
grad_step = 000237, loss = 0.001590
grad_step = 000238, loss = 0.001588
grad_step = 000239, loss = 0.001586
grad_step = 000240, loss = 0.001584
grad_step = 000241, loss = 0.001583
grad_step = 000242, loss = 0.001581
grad_step = 000243, loss = 0.001580
grad_step = 000244, loss = 0.001578
grad_step = 000245, loss = 0.001577
grad_step = 000246, loss = 0.001577
grad_step = 000247, loss = 0.001581
grad_step = 000248, loss = 0.001590
grad_step = 000249, loss = 0.001612
grad_step = 000250, loss = 0.001660
grad_step = 000251, loss = 0.001766
grad_step = 000252, loss = 0.001934
grad_step = 000253, loss = 0.002238
grad_step = 000254, loss = 0.002397
grad_step = 000255, loss = 0.002359
grad_step = 000256, loss = 0.001862
grad_step = 000257, loss = 0.001564
grad_step = 000258, loss = 0.001725
grad_step = 000259, loss = 0.001978
grad_step = 000260, loss = 0.001923
grad_step = 000261, loss = 0.001621
grad_step = 000262, loss = 0.001593
grad_step = 000263, loss = 0.001799
grad_step = 000264, loss = 0.001797
grad_step = 000265, loss = 0.001610
grad_step = 000266, loss = 0.001560
grad_step = 000267, loss = 0.001688
grad_step = 000268, loss = 0.001719
grad_step = 000269, loss = 0.001589
grad_step = 000270, loss = 0.001547
grad_step = 000271, loss = 0.001632
grad_step = 000272, loss = 0.001653
grad_step = 000273, loss = 0.001575
grad_step = 000274, loss = 0.001536
grad_step = 000275, loss = 0.001586
grad_step = 000276, loss = 0.001613
grad_step = 000277, loss = 0.001562
grad_step = 000278, loss = 0.001528
grad_step = 000279, loss = 0.001555
grad_step = 000280, loss = 0.001578
grad_step = 000281, loss = 0.001552
grad_step = 000282, loss = 0.001522
grad_step = 000283, loss = 0.001532
grad_step = 000284, loss = 0.001553
grad_step = 000285, loss = 0.001542
grad_step = 000286, loss = 0.001519
grad_step = 000287, loss = 0.001516
grad_step = 000288, loss = 0.001529
grad_step = 000289, loss = 0.001533
grad_step = 000290, loss = 0.001519
grad_step = 000291, loss = 0.001506
grad_step = 000292, loss = 0.001508
grad_step = 000293, loss = 0.001516
grad_step = 000294, loss = 0.001515
grad_step = 000295, loss = 0.001504
grad_step = 000296, loss = 0.001496
grad_step = 000297, loss = 0.001497
grad_step = 000298, loss = 0.001502
grad_step = 000299, loss = 0.001501
grad_step = 000300, loss = 0.001494
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001487
grad_step = 000302, loss = 0.001485
grad_step = 000303, loss = 0.001486
grad_step = 000304, loss = 0.001487
grad_step = 000305, loss = 0.001486
grad_step = 000306, loss = 0.001482
grad_step = 000307, loss = 0.001477
grad_step = 000308, loss = 0.001474
grad_step = 000309, loss = 0.001471
grad_step = 000310, loss = 0.001470
grad_step = 000311, loss = 0.001469
grad_step = 000312, loss = 0.001469
grad_step = 000313, loss = 0.001469
grad_step = 000314, loss = 0.001472
grad_step = 000315, loss = 0.001475
grad_step = 000316, loss = 0.001478
grad_step = 000317, loss = 0.001480
grad_step = 000318, loss = 0.001480
grad_step = 000319, loss = 0.001477
grad_step = 000320, loss = 0.001474
grad_step = 000321, loss = 0.001469
grad_step = 000322, loss = 0.001464
grad_step = 000323, loss = 0.001457
grad_step = 000324, loss = 0.001451
grad_step = 000325, loss = 0.001446
grad_step = 000326, loss = 0.001442
grad_step = 000327, loss = 0.001440
grad_step = 000328, loss = 0.001438
grad_step = 000329, loss = 0.001436
grad_step = 000330, loss = 0.001435
grad_step = 000331, loss = 0.001434
grad_step = 000332, loss = 0.001434
grad_step = 000333, loss = 0.001436
grad_step = 000334, loss = 0.001443
grad_step = 000335, loss = 0.001460
grad_step = 000336, loss = 0.001495
grad_step = 000337, loss = 0.001559
grad_step = 000338, loss = 0.001656
grad_step = 000339, loss = 0.001757
grad_step = 000340, loss = 0.001841
grad_step = 000341, loss = 0.001903
grad_step = 000342, loss = 0.001863
grad_step = 000343, loss = 0.001796
grad_step = 000344, loss = 0.001538
grad_step = 000345, loss = 0.001412
grad_step = 000346, loss = 0.001502
grad_step = 000347, loss = 0.001609
grad_step = 000348, loss = 0.001609
grad_step = 000349, loss = 0.001507
grad_step = 000350, loss = 0.001466
grad_step = 000351, loss = 0.001468
grad_step = 000352, loss = 0.001454
grad_step = 000353, loss = 0.001468
grad_step = 000354, loss = 0.001501
grad_step = 000355, loss = 0.001477
grad_step = 000356, loss = 0.001416
grad_step = 000357, loss = 0.001396
grad_step = 000358, loss = 0.001437
grad_step = 000359, loss = 0.001466
grad_step = 000360, loss = 0.001432
grad_step = 000361, loss = 0.001395
grad_step = 000362, loss = 0.001400
grad_step = 000363, loss = 0.001415
grad_step = 000364, loss = 0.001411
grad_step = 000365, loss = 0.001396
grad_step = 000366, loss = 0.001397
grad_step = 000367, loss = 0.001402
grad_step = 000368, loss = 0.001391
grad_step = 000369, loss = 0.001377
grad_step = 000370, loss = 0.001378
grad_step = 000371, loss = 0.001389
grad_step = 000372, loss = 0.001391
grad_step = 000373, loss = 0.001377
grad_step = 000374, loss = 0.001367
grad_step = 000375, loss = 0.001368
grad_step = 000376, loss = 0.001373
grad_step = 000377, loss = 0.001372
grad_step = 000378, loss = 0.001366
grad_step = 000379, loss = 0.001362
grad_step = 000380, loss = 0.001362
grad_step = 000381, loss = 0.001362
grad_step = 000382, loss = 0.001359
grad_step = 000383, loss = 0.001353
grad_step = 000384, loss = 0.001350
grad_step = 000385, loss = 0.001351
grad_step = 000386, loss = 0.001352
grad_step = 000387, loss = 0.001352
grad_step = 000388, loss = 0.001348
grad_step = 000389, loss = 0.001345
grad_step = 000390, loss = 0.001342
grad_step = 000391, loss = 0.001341
grad_step = 000392, loss = 0.001340
grad_step = 000393, loss = 0.001338
grad_step = 000394, loss = 0.001335
grad_step = 000395, loss = 0.001333
grad_step = 000396, loss = 0.001331
grad_step = 000397, loss = 0.001330
grad_step = 000398, loss = 0.001329
grad_step = 000399, loss = 0.001328
grad_step = 000400, loss = 0.001327
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001325
grad_step = 000402, loss = 0.001324
grad_step = 000403, loss = 0.001323
grad_step = 000404, loss = 0.001325
grad_step = 000405, loss = 0.001328
grad_step = 000406, loss = 0.001335
grad_step = 000407, loss = 0.001347
grad_step = 000408, loss = 0.001369
grad_step = 000409, loss = 0.001406
grad_step = 000410, loss = 0.001466
grad_step = 000411, loss = 0.001551
grad_step = 000412, loss = 0.001655
grad_step = 000413, loss = 0.001764
grad_step = 000414, loss = 0.001815
grad_step = 000415, loss = 0.001837
grad_step = 000416, loss = 0.001687
grad_step = 000417, loss = 0.001522
grad_step = 000418, loss = 0.001342
grad_step = 000419, loss = 0.001322
grad_step = 000420, loss = 0.001439
grad_step = 000421, loss = 0.001517
grad_step = 000422, loss = 0.001481
grad_step = 000423, loss = 0.001346
grad_step = 000424, loss = 0.001298
grad_step = 000425, loss = 0.001354
grad_step = 000426, loss = 0.001392
grad_step = 000427, loss = 0.001370
grad_step = 000428, loss = 0.001322
grad_step = 000429, loss = 0.001317
grad_step = 000430, loss = 0.001338
grad_step = 000431, loss = 0.001325
grad_step = 000432, loss = 0.001297
grad_step = 000433, loss = 0.001291
grad_step = 000434, loss = 0.001311
grad_step = 000435, loss = 0.001321
grad_step = 000436, loss = 0.001296
grad_step = 000437, loss = 0.001267
grad_step = 000438, loss = 0.001265
grad_step = 000439, loss = 0.001284
grad_step = 000440, loss = 0.001294
grad_step = 000441, loss = 0.001280
grad_step = 000442, loss = 0.001261
grad_step = 000443, loss = 0.001255
grad_step = 000444, loss = 0.001260
grad_step = 000445, loss = 0.001264
grad_step = 000446, loss = 0.001259
grad_step = 000447, loss = 0.001252
grad_step = 000448, loss = 0.001249
grad_step = 000449, loss = 0.001251
grad_step = 000450, loss = 0.001250
grad_step = 000451, loss = 0.001245
grad_step = 000452, loss = 0.001237
grad_step = 000453, loss = 0.001232
grad_step = 000454, loss = 0.001231
grad_step = 000455, loss = 0.001233
grad_step = 000456, loss = 0.001235
grad_step = 000457, loss = 0.001234
grad_step = 000458, loss = 0.001230
grad_step = 000459, loss = 0.001227
grad_step = 000460, loss = 0.001227
grad_step = 000461, loss = 0.001229
grad_step = 000462, loss = 0.001234
grad_step = 000463, loss = 0.001240
grad_step = 000464, loss = 0.001246
grad_step = 000465, loss = 0.001253
grad_step = 000466, loss = 0.001260
grad_step = 000467, loss = 0.001270
grad_step = 000468, loss = 0.001277
grad_step = 000469, loss = 0.001279
grad_step = 000470, loss = 0.001273
grad_step = 000471, loss = 0.001260
grad_step = 000472, loss = 0.001241
grad_step = 000473, loss = 0.001223
grad_step = 000474, loss = 0.001208
grad_step = 000475, loss = 0.001199
grad_step = 000476, loss = 0.001195
grad_step = 000477, loss = 0.001196
grad_step = 000478, loss = 0.001199
grad_step = 000479, loss = 0.001202
grad_step = 000480, loss = 0.001204
grad_step = 000481, loss = 0.001204
grad_step = 000482, loss = 0.001203
grad_step = 000483, loss = 0.001199
grad_step = 000484, loss = 0.001195
grad_step = 000485, loss = 0.001190
grad_step = 000486, loss = 0.001185
grad_step = 000487, loss = 0.001181
grad_step = 000488, loss = 0.001178
grad_step = 000489, loss = 0.001176
grad_step = 000490, loss = 0.001176
grad_step = 000491, loss = 0.001176
grad_step = 000492, loss = 0.001179
grad_step = 000493, loss = 0.001182
grad_step = 000494, loss = 0.001188
grad_step = 000495, loss = 0.001193
grad_step = 000496, loss = 0.001200
grad_step = 000497, loss = 0.001203
grad_step = 000498, loss = 0.001204
grad_step = 000499, loss = 0.001199
grad_step = 000500, loss = 0.001191
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001182
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

  date_run                              2020-05-14 06:14:58.749522
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.279046
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 06:14:58.756583
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.213906
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 06:14:58.764892
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.139115
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 06:14:58.771346
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -2.25037
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
0   2020-05-14 06:14:24.980932  ...    mean_absolute_error
1   2020-05-14 06:14:24.985412  ...     mean_squared_error
2   2020-05-14 06:14:24.989333  ...  median_absolute_error
3   2020-05-14 06:14:24.993211  ...               r2_score
4   2020-05-14 06:14:35.280813  ...    mean_absolute_error
5   2020-05-14 06:14:35.285796  ...     mean_squared_error
6   2020-05-14 06:14:35.289859  ...  median_absolute_error
7   2020-05-14 06:14:35.294546  ...               r2_score
8   2020-05-14 06:14:58.749522  ...    mean_absolute_error
9   2020-05-14 06:14:58.756583  ...     mean_squared_error
10  2020-05-14 06:14:58.764892  ...  median_absolute_error
11  2020-05-14 06:14:58.771346  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1025c7efd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:19, 123721.24it/s] 80%|███████▉  | 7897088/9912422 [00:00<00:11, 176623.95it/s]9920512it [00:00, 40061313.46it/s]                           
0it [00:00, ?it/s]32768it [00:00, 540947.03it/s]
0it [00:00, ?it/s]  2%|▏         | 40960/1648877 [00:00<00:03, 406912.09it/s]1654784it [00:00, 11578121.71it/s]                         
0it [00:00, ?it/s]8192it [00:00, 208272.40it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0fd8680e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0fd7cb10b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0fd54414e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0fd7c060b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0fd8680e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0fd542cbe0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0fd54414e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0fd7bc56d8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0fd8680e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0fd7a7e550> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f73eee3f1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=3109d16bc66c7cc5dc2f97f50be6b8f074a1dac22fe91c452e912afe5514ca81
  Stored in directory: /tmp/pip-ephem-wheel-cache-_z8o84gd/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f73e4faa080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2875392/17464789 [===>..........................] - ETA: 0s
11313152/17464789 [==================>...........] - ETA: 0s
16384000/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 06:16:27.848175: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 06:16:27.853254: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-14 06:16:27.853425: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558a699dd9d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 06:16:27.853444: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.8200 - accuracy: 0.4900
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7586 - accuracy: 0.4940
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7535 - accuracy: 0.4943 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7318 - accuracy: 0.4958
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6758 - accuracy: 0.4994
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7024 - accuracy: 0.4977
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7104 - accuracy: 0.4971
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6896 - accuracy: 0.4985
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
11000/25000 [============>.................] - ETA: 4s - loss: 7.6206 - accuracy: 0.5030
12000/25000 [=============>................] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6643 - accuracy: 0.5002
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6579 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 3s - loss: 7.6400 - accuracy: 0.5017
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6484 - accuracy: 0.5012
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6648 - accuracy: 0.5001
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6598 - accuracy: 0.5004
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6763 - accuracy: 0.4994
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6820 - accuracy: 0.4990
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7046 - accuracy: 0.4975
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6889 - accuracy: 0.4985
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6646 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6615 - accuracy: 0.5003
25000/25000 [==============================] - 10s 396us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 06:16:45.435152
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 06:16:45.435152  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:19:40, 11.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:10:41, 15.8kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:40:54, 22.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:29:12, 32.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.63M/862M [00:01<5:13:36, 45.6kB/s].vector_cache/glove.6B.zip:   1%|          | 6.73M/862M [00:01<3:38:52, 65.1kB/s].vector_cache/glove.6B.zip:   1%|          | 10.0M/862M [00:01<2:32:45, 93.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.1M/862M [00:01<1:46:40, 133kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 16.9M/862M [00:01<1:14:27, 189kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:01<51:57, 270kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 24.9M/862M [00:01<36:19, 384kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.0M/862M [00:01<25:24, 547kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.7M/862M [00:02<17:49, 776kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.7M/862M [00:02<12:30, 1.10MB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.5M/862M [00:02<08:49, 1.55MB/s].vector_cache/glove.6B.zip:   5%|▍         | 43.0M/862M [00:02<06:19, 2.16MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.7M/862M [00:02<04:31, 3.01MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.0M/862M [00:02<03:14, 4.16MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:03<03:52, 3.48MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:03<02:49, 4.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<14:21, 935kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<12:28, 1.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:05<09:11, 1.46MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.0M/862M [00:05<06:45, 1.98MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:07<08:39, 1.54MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<07:53, 1.69MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:07<05:54, 2.26MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:07<04:21, 3.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:09<11:04, 1.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<09:36, 1.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.3M/862M [00:09<07:09, 1.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:11<07:36, 1.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:11<06:34, 2.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:11<05:07, 2.57MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:11<03:45, 3.50MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:13<10:02, 1.31MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:13<08:12, 1.60MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:13<06:16, 2.09MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:13<04:32, 2.88MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:15<14:07, 926kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:15<11:16, 1.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<08:14, 1.58MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<08:47, 1.48MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:17<07:25, 1.75MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:17<05:59, 2.17MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.7M/862M [00:17<04:27, 2.91MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:17<03:21, 3.85MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<1:15:13, 172kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:19<54:57, 235kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:19<38:57, 332kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.0M/862M [00:19<27:30, 469kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<23:12, 555kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:21<17:47, 723kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:21<12:46, 1.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:21<09:04, 1.41MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<4:20:42, 49.1kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:23<3:03:48, 69.6kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:23<2:08:45, 99.2kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<1:32:51, 137kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:25<1:06:22, 192kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:25<46:42, 272kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<35:33, 356kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<26:02, 486kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<18:38, 679kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<13:13, 955kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<15:40, 804kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<12:19, 1.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<08:53, 1.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<09:10, 1.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<07:47, 1.61MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<05:43, 2.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<07:28, 1.67MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<07:56, 1.57MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<06:13, 2.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<05:17, 2.35MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:13, 1.99MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<05:41, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<04:15, 2.90MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<03:09, 3.91MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<31:39, 389kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<23:27, 525kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<16:39, 738kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<14:34, 841kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<11:31, 1.06MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<08:23, 1.46MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<09:28, 1.29MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<11:54, 1.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<09:40, 1.26MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<06:59, 1.74MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<05:03, 2.39MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<11:03:10, 18.3kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<7:45:17, 26.0kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<5:25:20, 37.2kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<3:47:13, 53.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<3:49:50, 52.4kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<2:42:07, 74.3kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<1:53:33, 106kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<1:22:04, 146kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<58:43, 204kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<41:20, 289kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<31:38, 377kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<23:26, 508kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<16:41, 712kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<14:26, 821kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<11:08, 1.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<08:27, 1.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:01, 1.95MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<48:19, 244kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<35:05, 335kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<24:46, 474kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<20:03, 584kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<15:18, 764kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:55<10:57, 1.07MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<10:23, 1.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<08:32, 1.36MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<06:17, 1.85MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<07:05, 1.63MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:59<06:12, 1.86MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<04:36, 2.50MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:56, 1.94MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<05:24, 2.12MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<04:02, 2.84MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:30, 2.07MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<04:52, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<04:03, 2.81MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<02:57, 3.85MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<44:33, 255kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<32:24, 351kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<22:56, 494kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<16:26, 687kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<2:08:13, 88.1kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<1:30:41, 125kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:07<1:03:55, 176kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<44:40, 251kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<1:13:19, 153kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<52:30, 214kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:09<36:55, 303kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<28:24, 393kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<21:05, 529kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<15:01, 740kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<13:05, 848kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<10:08, 1.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<07:39, 1.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:27, 2.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<45:04, 245kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<32:44, 336kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<23:07, 475kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<18:41, 586kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<14:14, 768kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:17<10:14, 1.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<09:44, 1.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<07:47, 1.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:05, 1.79MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:22, 2.48MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<11:33, 936kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<09:12, 1.17MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<06:43, 1.60MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<07:15, 1.48MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<06:14, 1.72MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<04:37, 2.32MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<05:43, 1.86MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<05:10, 2.06MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<03:54, 2.73MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<05:12, 2.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<04:48, 2.20MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<03:37, 2.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<05:00, 2.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<04:40, 2.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<03:30, 2.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<04:54, 2.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<04:36, 2.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<03:29, 2.98MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<05:55, 1.76MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<13:20, 779kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<11:47, 882kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:33<08:48, 1.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<06:28, 1.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<06:03, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<04:31, 2.28MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<05:18, 1.93MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<06:02, 1.70MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:37<04:41, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:37<03:33, 2.87MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<04:51, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<04:31, 2.25MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<03:24, 2.99MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:40<04:43, 2.14MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<04:25, 2.29MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<03:17, 3.06MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<02:27, 4.10MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:42<1:07:25, 149kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<48:02, 209kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<34:11, 294kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<23:55, 418kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<54:11, 184kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:44<38:59, 256kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<27:29, 362kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<21:30, 461kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<16:07, 615kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<11:29, 861kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<10:20, 953kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<08:19, 1.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<06:03, 1.62MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<06:31, 1.50MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<05:35, 1.75MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<04:10, 2.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<05:11, 1.87MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:52<04:31, 2.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:52<03:34, 2.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<02:36, 3.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<40:11, 240kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<29:10, 330kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<20:34, 467kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<14:29, 661kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<2:30:50, 63.5kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<1:46:33, 89.8kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<1:14:38, 128kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<54:16, 175kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<39:00, 244kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<27:26, 345kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<19:16, 490kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<9:27:41, 16.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<6:38:08, 23.7kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<4:38:08, 33.8kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<3:14:06, 48.3kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<3:17:09, 47.5kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<2:18:56, 67.4kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:02<1:37:13, 96.1kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<1:09:58, 133kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<49:58, 186kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<35:08, 264kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<26:39, 346kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<19:39, 470kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<13:55, 661kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<09:50, 932kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:08<4:36:18, 33.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 313M/862M [02:08<3:14:17, 47.2kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<2:15:51, 67.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:10<1:36:51, 93.9kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<1:08:45, 132kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<48:12, 188kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:12<35:44, 253kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<25:58, 347kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<18:22, 490kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<14:54, 601kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<11:24, 785kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<08:11, 1.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<07:48, 1.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<06:24, 1.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<04:42, 1.88MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<05:21, 1.65MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:18<04:42, 1.87MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:18<04:48, 1.83MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<04:35, 1.90MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:20<04:09, 2.10MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<03:05, 2.82MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<02:17, 3.79MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<38:02, 228kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<27:32, 315kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<19:25, 445kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<15:35, 553kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:24<11:48, 729kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<08:27, 1.02MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<07:54, 1.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<06:27, 1.32MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<04:42, 1.81MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<05:17, 1.60MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<04:37, 1.83MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:28<03:27, 2.44MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:29<04:19, 1.94MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:29<04:46, 1.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<03:46, 2.22MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<02:42, 3.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<17:58, 464kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<13:31, 617kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:32<09:37, 863kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<06:50, 1.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<16:52, 490kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<12:45, 648kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<09:08, 902kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<08:11, 1.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<06:35, 1.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<04:47, 1.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<05:15, 1.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<04:24, 1.85MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<03:31, 2.30MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<02:32, 3.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<33:57, 237kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<24:37, 327kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<17:22, 463kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<14:02, 569kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<10:42, 746kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<07:41, 1.04MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<07:12, 1.10MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<05:55, 1.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:19, 1.83MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:07, 2.52MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<1:08:01, 116kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<48:26, 162kB/s]  .vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<33:58, 230kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<23:47, 328kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<1:02:35, 124kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<44:38, 174kB/s]  .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<31:19, 248kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<23:38, 327kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<17:23, 444kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<12:20, 623kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<10:23, 736kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<08:05, 946kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:49, 1.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<05:50, 1.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<04:55, 1.54MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:38, 2.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<04:17, 1.75MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<03:49, 1.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<02:52, 2.60MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<03:44, 1.99MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<03:26, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<02:36, 2.84MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:59<03:32, 2.08MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<03:17, 2.25MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<02:29, 2.95MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:01<03:27, 2.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<03:03, 2.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<02:22, 3.07MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<01:45, 4.14MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<08:09, 889kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<06:29, 1.12MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:43, 1.53MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<04:58, 1.44MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<04:15, 1.68MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:10, 2.25MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:07<03:52, 1.84MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:07<03:28, 2.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<02:34, 2.74MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<01:53, 3.71MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:09<1:24:06, 83.7kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:09<59:35, 118kB/s]   .vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<41:45, 168kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:11<30:42, 227kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:11<22:14, 313kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<15:40, 443kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:13<12:32, 550kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<09:30, 725kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<06:49, 1.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:15<06:21, 1.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:15<05:10, 1.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<03:47, 1.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:17<04:14, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:17<03:40, 1.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<02:44, 2.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<03:29, 1.92MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:19<03:10, 2.11MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<02:21, 2.82MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<03:12, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:21<02:57, 2.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<02:12, 2.98MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<01:38, 4.01MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<39:41, 165kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:23<28:27, 230kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<19:59, 326kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<15:27, 420kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:25<11:30, 563kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<08:11, 787kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<07:13, 889kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<05:44, 1.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<04:09, 1.54MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:58, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<1:16:39, 82.8kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<54:17, 117kB/s]   .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<37:59, 166kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:30<27:56, 225kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<20:12, 310kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:31<14:15, 438kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<11:23, 546kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<08:37, 720kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<06:10, 1.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<05:44, 1.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<04:40, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:25, 1.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<04:13, 1.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<05:35, 1.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<04:34, 1.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<03:20, 1.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<03:31, 1.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<02:47, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<02:00, 2.97MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<05:46, 1.03MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<05:17, 1.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:41<03:57, 1.50MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<02:54, 2.03MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<03:28, 1.69MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:43<03:07, 1.88MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<02:18, 2.52MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<01:41, 3.44MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<35:43, 162kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:45<25:37, 226kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<17:59, 320kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<13:50, 414kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<10:17, 556kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:47<07:19, 778kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<06:25, 881kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<05:06, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:49<03:43, 1.51MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<03:53, 1.44MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<03:20, 1.67MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:51<02:26, 2.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<01:47, 3.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<24:26, 226kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<17:41, 312kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<12:26, 441kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<09:56, 548kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<07:32, 722kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<05:24, 1.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:56<05:01, 1.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:56<04:05, 1.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<03:00, 1.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<03:20, 1.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<02:54, 1.83MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:09, 2.44MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:00<02:44, 1.91MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 547M/862M [04:00<02:22, 2.21MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<01:54, 2.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<01:22, 3.76MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:02<20:22, 254kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<14:50, 349kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<10:28, 491kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<08:28, 603kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:04<06:29, 787kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<04:39, 1.09MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:17, 1.53MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<35:15, 143kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<25:05, 201kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<17:49, 282kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<12:24, 401kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:08<27:16, 183kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:08<19:35, 254kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<13:46, 359kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:10<10:43, 458kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:10<08:01, 610kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<05:43, 852kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<05:06, 947kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<04:05, 1.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:59, 1.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:14<03:11, 1.49MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:14<02:39, 1.79MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<02:07, 2.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:30, 3.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:16<09:24, 500kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:16<07:04, 664kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<05:02, 925kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:18<04:35, 1.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<03:42, 1.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<02:42, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<02:57, 1.55MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:20<02:33, 1.78MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<01:54, 2.38MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<02:22, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:22<02:09, 2.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<01:35, 2.81MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<01:10, 3.78MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<52:56, 83.7kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<37:29, 118kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<26:11, 168kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<19:12, 227kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<13:54, 313kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<09:47, 442kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<07:48, 550kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<05:50, 734kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<04:15, 1.00MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<02:59, 1.41MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:30<07:52, 536kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:30<05:58, 706kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<04:14, 987kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:00, 1.38MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<36:57, 112kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:32<26:16, 158kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<18:23, 224kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<13:42, 298kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<09:56, 410kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:34<07:10, 567kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<05:00, 803kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<08:33, 470kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:36<06:24, 626kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:36<04:33, 876kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<03:12, 1.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:37<40:54, 96.5kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:38<29:00, 136kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:38<20:17, 193kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:39<14:58, 259kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<10:53, 356kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:40<07:38, 503kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<05:21, 711kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:41<48:33, 78.5kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<34:21, 111kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:42<23:59, 158kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<17:30, 214kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<12:38, 296kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<08:52, 418kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<06:12, 592kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:45<35:07, 105kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:45<24:56, 147kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<17:25, 209kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<12:07, 297kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<34:56, 103kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<24:48, 145kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<17:19, 206kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:49<12:50, 275kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:49<09:21, 377kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<06:35, 531kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:51<05:23, 644kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:51<04:08, 836kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<02:57, 1.16MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:53<02:50, 1.19MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<02:21, 1.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:43, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<01:52, 1.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:56<02:46, 1.20MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:56<02:17, 1.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<01:40, 1.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:57<01:53, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:58<01:40, 1.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:58<01:14, 2.60MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<00:54, 3.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:59<16:12, 196kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:00<11:39, 272kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:00<08:10, 384kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<06:23, 486kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:02<04:47, 647kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<03:24, 901kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<03:04, 989kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<02:28, 1.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<01:47, 1.68MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<01:56, 1.53MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<01:40, 1.77MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<01:13, 2.40MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<00:53, 3.26MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<21:58, 132kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<15:39, 185kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:08<10:56, 262kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:09<08:13, 345kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:09<06:03, 467kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<04:16, 656kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:11<03:36, 768kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<02:49, 978kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<02:01, 1.35MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:13<02:01, 1.33MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<01:39, 1.62MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<01:15, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<00:54, 2.92MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:15<02:28, 1.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<02:00, 1.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:27, 1.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<01:36, 1.59MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<01:24, 1.82MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:02, 2.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<01:18, 1.91MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<01:10, 2.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<00:53, 2.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<01:10, 2.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<01:05, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<00:48, 2.93MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:23<01:06, 2.11MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:23<01:01, 2.27MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<00:46, 3.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:25<01:04, 2.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:25<00:59, 2.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<00:45, 3.00MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:27<01:02, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<00:57, 2.30MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<00:42, 3.06MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:29<01:00, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<00:56, 2.29MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<00:42, 3.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:31<00:58, 2.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:31<00:54, 2.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<00:40, 3.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:33<00:56, 2.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:33<00:52, 2.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<00:39, 3.01MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:35<00:54, 2.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:35<00:50, 2.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<00:37, 3.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:37<00:52, 2.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:37<00:48, 2.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<00:36, 3.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:39<00:50, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:39<00:46, 2.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:35, 3.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:41<00:48, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:41<00:42, 2.43MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:41<00:36, 2.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:25, 3.92MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:43<03:10, 527kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:43<02:23, 696kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<01:41, 968kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:45<01:31, 1.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:45<01:14, 1.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:53, 1.76MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:47<00:58, 1.58MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:47<00:50, 1.81MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<00:36, 2.44MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:26, 3.32MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<09:51, 149kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:49<07:01, 207kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<04:52, 294kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:50<03:38, 382kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:51<02:41, 515kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:51<01:53, 721kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:52<01:35, 830kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:53<01:15, 1.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<00:53, 1.45MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:54<00:54, 1.40MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:54<00:44, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<00:34, 2.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:24, 2.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:56<01:28, 810kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:57<01:16, 924kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:57<00:57, 1.23MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:40, 1.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:58<00:50, 1.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:59<00:42, 1.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<00:30, 2.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:00<00:35, 1.78MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:01<00:31, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<00:23, 2.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<00:29, 2.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<00:25, 2.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:03<00:21, 2.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:14, 3.75MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:04<01:00, 909kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:04<00:47, 1.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:05<00:33, 1.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:06<00:35, 1.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<00:29, 1.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:07<00:21, 2.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<00:25, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<00:22, 2.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:09<00:16, 2.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:20, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:19, 2.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:11<00:14, 2.89MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:12<00:18, 2.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:12<00:16, 2.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:12, 3.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:14<00:16, 2.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:14<00:14, 2.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:10, 3.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:16<00:14, 2.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:16<00:12, 2.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:09, 3.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:18<00:12, 2.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:18<00:11, 2.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:08, 3.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:20<00:10, 2.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:20<00:09, 2.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:06, 3.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:22<00:08, 2.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:22<00:07, 2.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:05, 3.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:24<00:06, 2.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:24<00:05, 2.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:03, 3.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:04, 2.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:04, 2.29MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:02, 3.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:28<00:02, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:28<00:02, 2.30MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 3.01MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:00, 2.14MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:00, 2.29MB/s].vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 730/400000 [00:00<00:54, 7294.04it/s]  0%|          | 1470/400000 [00:00<00:54, 7325.50it/s]  1%|          | 2205/400000 [00:00<00:54, 7332.70it/s]  1%|          | 2942/400000 [00:00<00:54, 7343.37it/s]  1%|          | 3700/400000 [00:00<00:53, 7411.56it/s]  1%|          | 4448/400000 [00:00<00:53, 7430.73it/s]  1%|▏         | 5187/400000 [00:00<00:53, 7416.39it/s]  1%|▏         | 5927/400000 [00:00<00:53, 7411.00it/s]  2%|▏         | 6627/400000 [00:00<00:54, 7218.12it/s]  2%|▏         | 7363/400000 [00:01<00:54, 7258.03it/s]  2%|▏         | 8110/400000 [00:01<00:53, 7319.55it/s]  2%|▏         | 8857/400000 [00:01<00:53, 7363.10it/s]  2%|▏         | 9596/400000 [00:01<00:52, 7367.87it/s]  3%|▎         | 10336/400000 [00:01<00:52, 7375.37it/s]  3%|▎         | 11073/400000 [00:01<00:52, 7371.27it/s]  3%|▎         | 11816/400000 [00:01<00:52, 7388.12it/s]  3%|▎         | 12558/400000 [00:01<00:52, 7395.03it/s]  3%|▎         | 13320/400000 [00:01<00:51, 7459.83it/s]  4%|▎         | 14076/400000 [00:01<00:51, 7487.53it/s]  4%|▎         | 14829/400000 [00:02<00:51, 7497.90it/s]  4%|▍         | 15584/400000 [00:02<00:51, 7513.42it/s]  4%|▍         | 16336/400000 [00:02<00:51, 7514.84it/s]  4%|▍         | 17088/400000 [00:02<00:51, 7474.78it/s]  4%|▍         | 17836/400000 [00:02<00:51, 7446.26it/s]  5%|▍         | 18581/400000 [00:02<00:51, 7375.28it/s]  5%|▍         | 19319/400000 [00:02<00:51, 7360.66it/s]  5%|▌         | 20056/400000 [00:02<00:51, 7352.16it/s]  5%|▌         | 20802/400000 [00:02<00:51, 7382.67it/s]  5%|▌         | 21541/400000 [00:02<00:52, 7255.74it/s]  6%|▌         | 22268/400000 [00:03<00:52, 7172.93it/s]  6%|▌         | 23005/400000 [00:03<00:52, 7229.83it/s]  6%|▌         | 23729/400000 [00:03<00:53, 6986.79it/s]  6%|▌         | 24445/400000 [00:03<00:53, 7037.36it/s]  6%|▋         | 25178/400000 [00:03<00:52, 7122.22it/s]  6%|▋         | 25912/400000 [00:03<00:52, 7186.18it/s]  7%|▋         | 26635/400000 [00:03<00:51, 7197.79it/s]  7%|▋         | 27365/400000 [00:03<00:51, 7216.58it/s]  7%|▋         | 28088/400000 [00:03<00:53, 6988.88it/s]  7%|▋         | 28858/400000 [00:03<00:51, 7187.84it/s]  7%|▋         | 29595/400000 [00:04<00:51, 7240.11it/s]  8%|▊         | 30337/400000 [00:04<00:50, 7292.88it/s]  8%|▊         | 31082/400000 [00:04<00:50, 7338.90it/s]  8%|▊         | 31831/400000 [00:04<00:49, 7380.20it/s]  8%|▊         | 32570/400000 [00:04<00:50, 7344.16it/s]  8%|▊         | 33316/400000 [00:04<00:49, 7375.83it/s]  9%|▊         | 34055/400000 [00:04<00:49, 7358.11it/s]  9%|▊         | 34803/400000 [00:04<00:49, 7392.95it/s]  9%|▉         | 35543/400000 [00:04<00:50, 7287.86it/s]  9%|▉         | 36282/400000 [00:04<00:49, 7313.09it/s]  9%|▉         | 37025/400000 [00:05<00:49, 7345.48it/s]  9%|▉         | 37781/400000 [00:05<00:48, 7406.57it/s] 10%|▉         | 38523/400000 [00:05<00:48, 7408.24it/s] 10%|▉         | 39274/400000 [00:05<00:48, 7435.68it/s] 10%|█         | 40030/400000 [00:05<00:48, 7469.70it/s] 10%|█         | 40778/400000 [00:05<00:48, 7351.39it/s] 10%|█         | 41538/400000 [00:05<00:48, 7421.88it/s] 11%|█         | 42281/400000 [00:05<00:50, 7115.34it/s] 11%|█         | 43015/400000 [00:05<00:49, 7180.74it/s] 11%|█         | 43736/400000 [00:05<00:50, 7107.97it/s] 11%|█         | 44487/400000 [00:06<00:49, 7221.91it/s] 11%|█▏        | 45230/400000 [00:06<00:48, 7281.20it/s] 11%|█▏        | 45965/400000 [00:06<00:48, 7301.68it/s] 12%|█▏        | 46697/400000 [00:06<00:49, 7101.34it/s] 12%|█▏        | 47427/400000 [00:06<00:49, 7157.54it/s] 12%|█▏        | 48145/400000 [00:06<00:49, 7150.35it/s] 12%|█▏        | 48875/400000 [00:06<00:48, 7192.77it/s] 12%|█▏        | 49603/400000 [00:06<00:48, 7216.49it/s] 13%|█▎        | 50338/400000 [00:06<00:48, 7253.91it/s] 13%|█▎        | 51089/400000 [00:06<00:47, 7328.57it/s] 13%|█▎        | 51836/400000 [00:07<00:47, 7368.35it/s] 13%|█▎        | 52579/400000 [00:07<00:47, 7384.19it/s] 13%|█▎        | 53318/400000 [00:07<00:46, 7377.33it/s] 14%|█▎        | 54056/400000 [00:07<00:46, 7374.04it/s] 14%|█▎        | 54795/400000 [00:07<00:46, 7377.33it/s] 14%|█▍        | 55542/400000 [00:07<00:46, 7402.80it/s] 14%|█▍        | 56289/400000 [00:07<00:46, 7422.22it/s] 14%|█▍        | 57041/400000 [00:07<00:46, 7450.81it/s] 14%|█▍        | 57804/400000 [00:07<00:45, 7501.45it/s] 15%|█▍        | 58555/400000 [00:07<00:45, 7471.86it/s] 15%|█▍        | 59303/400000 [00:08<00:45, 7441.69it/s] 15%|█▌        | 60048/400000 [00:08<00:45, 7443.04it/s] 15%|█▌        | 60793/400000 [00:08<00:45, 7433.21it/s] 15%|█▌        | 61537/400000 [00:08<00:45, 7363.49it/s] 16%|█▌        | 62274/400000 [00:08<00:45, 7349.52it/s] 16%|█▌        | 63033/400000 [00:08<00:45, 7417.99it/s] 16%|█▌        | 63776/400000 [00:08<00:45, 7366.29it/s] 16%|█▌        | 64532/400000 [00:08<00:45, 7421.40it/s] 16%|█▋        | 65275/400000 [00:08<00:45, 7419.01it/s] 17%|█▋        | 66018/400000 [00:09<00:45, 7416.21it/s] 17%|█▋        | 66760/400000 [00:09<00:44, 7414.04it/s] 17%|█▋        | 67502/400000 [00:09<00:45, 7383.62it/s] 17%|█▋        | 68246/400000 [00:09<00:44, 7397.48it/s] 17%|█▋        | 68986/400000 [00:09<00:45, 7266.67it/s] 17%|█▋        | 69745/400000 [00:09<00:44, 7358.59it/s] 18%|█▊        | 70503/400000 [00:09<00:44, 7423.06it/s] 18%|█▊        | 71246/400000 [00:09<00:44, 7418.00it/s] 18%|█▊        | 71989/400000 [00:09<00:44, 7420.70it/s] 18%|█▊        | 72732/400000 [00:09<00:44, 7419.53it/s] 18%|█▊        | 73475/400000 [00:10<00:44, 7378.37it/s] 19%|█▊        | 74230/400000 [00:10<00:43, 7426.71it/s] 19%|█▊        | 74973/400000 [00:10<00:43, 7397.32it/s] 19%|█▉        | 75728/400000 [00:10<00:43, 7439.74it/s] 19%|█▉        | 76473/400000 [00:10<00:43, 7435.02it/s] 19%|█▉        | 77238/400000 [00:10<00:43, 7497.53it/s] 19%|█▉        | 77993/400000 [00:10<00:42, 7511.21it/s] 20%|█▉        | 78745/400000 [00:10<00:43, 7461.88it/s] 20%|█▉        | 79492/400000 [00:10<00:43, 7290.12it/s] 20%|██        | 80251/400000 [00:10<00:43, 7375.59it/s] 20%|██        | 81016/400000 [00:11<00:42, 7454.85it/s] 20%|██        | 81775/400000 [00:11<00:42, 7493.37it/s] 21%|██        | 82525/400000 [00:11<00:42, 7466.34it/s] 21%|██        | 83273/400000 [00:11<00:42, 7435.11it/s] 21%|██        | 84017/400000 [00:11<00:42, 7413.08it/s] 21%|██        | 84759/400000 [00:11<00:42, 7334.19it/s] 21%|██▏       | 85493/400000 [00:11<00:42, 7335.72it/s] 22%|██▏       | 86229/400000 [00:11<00:42, 7342.15it/s] 22%|██▏       | 86987/400000 [00:11<00:42, 7411.26it/s] 22%|██▏       | 87761/400000 [00:11<00:41, 7505.14it/s] 22%|██▏       | 88536/400000 [00:12<00:41, 7575.06it/s] 22%|██▏       | 89300/400000 [00:12<00:40, 7591.76it/s] 23%|██▎       | 90060/400000 [00:12<00:41, 7544.94it/s] 23%|██▎       | 90815/400000 [00:12<00:41, 7487.52it/s] 23%|██▎       | 91565/400000 [00:12<00:41, 7441.05it/s] 23%|██▎       | 92310/400000 [00:12<00:41, 7422.08it/s] 23%|██▎       | 93062/400000 [00:12<00:41, 7448.97it/s] 23%|██▎       | 93809/400000 [00:12<00:41, 7454.25it/s] 24%|██▎       | 94564/400000 [00:12<00:40, 7481.37it/s] 24%|██▍       | 95322/400000 [00:12<00:40, 7510.05it/s] 24%|██▍       | 96096/400000 [00:13<00:40, 7576.58it/s] 24%|██▍       | 96856/400000 [00:13<00:39, 7582.72it/s] 24%|██▍       | 97615/400000 [00:13<00:40, 7511.96it/s] 25%|██▍       | 98367/400000 [00:13<00:40, 7446.26it/s] 25%|██▍       | 99112/400000 [00:13<00:40, 7394.27it/s] 25%|██▍       | 99863/400000 [00:13<00:40, 7427.92it/s] 25%|██▌       | 100607/400000 [00:13<00:40, 7373.90it/s] 25%|██▌       | 101345/400000 [00:13<00:41, 7273.27it/s] 26%|██▌       | 102079/400000 [00:13<00:40, 7290.94it/s] 26%|██▌       | 102839/400000 [00:13<00:40, 7380.37it/s] 26%|██▌       | 103586/400000 [00:14<00:40, 7404.76it/s] 26%|██▌       | 104344/400000 [00:14<00:39, 7454.93it/s] 26%|██▋       | 105090/400000 [00:14<00:39, 7420.03it/s] 26%|██▋       | 105833/400000 [00:14<00:39, 7409.44it/s] 27%|██▋       | 106575/400000 [00:14<00:39, 7348.67it/s] 27%|██▋       | 107311/400000 [00:14<00:39, 7320.05it/s] 27%|██▋       | 108044/400000 [00:14<00:40, 7196.86it/s] 27%|██▋       | 108775/400000 [00:14<00:40, 7227.90it/s] 27%|██▋       | 109516/400000 [00:14<00:39, 7279.39it/s] 28%|██▊       | 110255/400000 [00:14<00:39, 7311.75it/s] 28%|██▊       | 110989/400000 [00:15<00:39, 7318.47it/s] 28%|██▊       | 111722/400000 [00:15<00:39, 7306.33it/s] 28%|██▊       | 112454/400000 [00:15<00:39, 7308.64it/s] 28%|██▊       | 113212/400000 [00:15<00:38, 7385.79it/s] 28%|██▊       | 113971/400000 [00:15<00:38, 7443.19it/s] 29%|██▊       | 114720/400000 [00:15<00:38, 7455.06it/s] 29%|██▉       | 115466/400000 [00:15<00:38, 7420.74it/s] 29%|██▉       | 116209/400000 [00:15<00:38, 7324.14it/s] 29%|██▉       | 116946/400000 [00:15<00:38, 7336.82it/s] 29%|██▉       | 117680/400000 [00:15<00:39, 7145.93it/s] 30%|██▉       | 118396/400000 [00:16<00:39, 7067.59it/s] 30%|██▉       | 119104/400000 [00:16<00:40, 6960.68it/s] 30%|██▉       | 119802/400000 [00:16<00:40, 6878.79it/s] 30%|███       | 120491/400000 [00:16<00:40, 6825.85it/s] 30%|███       | 121175/400000 [00:16<00:41, 6711.55it/s] 30%|███       | 121848/400000 [00:16<00:41, 6668.72it/s] 31%|███       | 122586/400000 [00:16<00:40, 6865.53it/s] 31%|███       | 123299/400000 [00:16<00:39, 6940.35it/s] 31%|███       | 124026/400000 [00:16<00:39, 7035.35it/s] 31%|███       | 124766/400000 [00:17<00:38, 7140.71it/s] 31%|███▏      | 125499/400000 [00:17<00:38, 7194.90it/s] 32%|███▏      | 126244/400000 [00:17<00:37, 7269.17it/s] 32%|███▏      | 126985/400000 [00:17<00:37, 7309.67it/s] 32%|███▏      | 127717/400000 [00:17<00:37, 7303.73it/s] 32%|███▏      | 128470/400000 [00:17<00:36, 7366.99it/s] 32%|███▏      | 129223/400000 [00:17<00:36, 7413.37it/s] 32%|███▏      | 129981/400000 [00:17<00:36, 7461.16it/s] 33%|███▎      | 130728/400000 [00:17<00:36, 7415.72it/s] 33%|███▎      | 131470/400000 [00:17<00:36, 7351.15it/s] 33%|███▎      | 132216/400000 [00:18<00:36, 7381.87it/s] 33%|███▎      | 132974/400000 [00:18<00:35, 7439.21it/s] 33%|███▎      | 133726/400000 [00:18<00:35, 7461.74it/s] 34%|███▎      | 134473/400000 [00:18<00:35, 7396.68it/s] 34%|███▍      | 135220/400000 [00:18<00:35, 7418.19it/s] 34%|███▍      | 135976/400000 [00:18<00:35, 7456.86it/s] 34%|███▍      | 136729/400000 [00:18<00:35, 7478.27it/s] 34%|███▍      | 137477/400000 [00:18<00:35, 7429.67it/s] 35%|███▍      | 138221/400000 [00:18<00:35, 7306.32it/s] 35%|███▍      | 138960/400000 [00:18<00:35, 7329.07it/s] 35%|███▍      | 139694/400000 [00:19<00:35, 7330.59it/s] 35%|███▌      | 140428/400000 [00:19<00:35, 7324.09it/s] 35%|███▌      | 141161/400000 [00:19<00:35, 7313.11it/s] 35%|███▌      | 141900/400000 [00:19<00:35, 7334.09it/s] 36%|███▌      | 142638/400000 [00:19<00:35, 7347.12it/s] 36%|███▌      | 143373/400000 [00:19<00:35, 7281.73it/s] 36%|███▌      | 144119/400000 [00:19<00:34, 7333.08it/s] 36%|███▌      | 144870/400000 [00:19<00:34, 7384.19it/s] 36%|███▋      | 145609/400000 [00:19<00:34, 7347.37it/s] 37%|███▋      | 146353/400000 [00:19<00:34, 7373.56it/s] 37%|███▋      | 147095/400000 [00:20<00:34, 7385.42it/s] 37%|███▋      | 147840/400000 [00:20<00:34, 7403.68it/s] 37%|███▋      | 148591/400000 [00:20<00:33, 7432.81it/s] 37%|███▋      | 149336/400000 [00:20<00:33, 7437.16it/s] 38%|███▊      | 150080/400000 [00:20<00:33, 7428.14it/s] 38%|███▊      | 150846/400000 [00:20<00:33, 7495.73it/s] 38%|███▊      | 151597/400000 [00:20<00:33, 7498.57it/s] 38%|███▊      | 152350/400000 [00:20<00:32, 7506.42it/s] 38%|███▊      | 153101/400000 [00:20<00:32, 7501.90it/s] 38%|███▊      | 153852/400000 [00:20<00:33, 7413.90it/s] 39%|███▊      | 154627/400000 [00:21<00:32, 7511.31it/s] 39%|███▉      | 155379/400000 [00:21<00:32, 7488.58it/s] 39%|███▉      | 156131/400000 [00:21<00:32, 7494.99it/s] 39%|███▉      | 156881/400000 [00:21<00:32, 7453.91it/s] 39%|███▉      | 157627/400000 [00:21<00:32, 7438.73it/s] 40%|███▉      | 158372/400000 [00:21<00:32, 7421.87it/s] 40%|███▉      | 159115/400000 [00:21<00:32, 7406.04it/s] 40%|███▉      | 159866/400000 [00:21<00:32, 7435.55it/s] 40%|████      | 160610/400000 [00:21<00:32, 7429.30it/s] 40%|████      | 161372/400000 [00:21<00:31, 7482.70it/s] 41%|████      | 162121/400000 [00:22<00:31, 7484.53it/s] 41%|████      | 162883/400000 [00:22<00:31, 7523.66it/s] 41%|████      | 163636/400000 [00:22<00:31, 7506.66it/s] 41%|████      | 164387/400000 [00:22<00:31, 7475.14it/s] 41%|████▏     | 165135/400000 [00:22<00:31, 7391.48it/s] 41%|████▏     | 165894/400000 [00:22<00:31, 7448.29it/s] 42%|████▏     | 166656/400000 [00:22<00:31, 7496.70it/s] 42%|████▏     | 167406/400000 [00:22<00:31, 7467.56it/s] 42%|████▏     | 168153/400000 [00:22<00:31, 7434.61it/s] 42%|████▏     | 168907/400000 [00:22<00:30, 7464.10it/s] 42%|████▏     | 169663/400000 [00:23<00:30, 7490.29it/s] 43%|████▎     | 170422/400000 [00:23<00:30, 7519.76it/s] 43%|████▎     | 171175/400000 [00:23<00:30, 7486.53it/s] 43%|████▎     | 171924/400000 [00:23<00:30, 7431.37it/s] 43%|████▎     | 172668/400000 [00:23<00:31, 7330.14it/s] 43%|████▎     | 173412/400000 [00:23<00:30, 7360.92it/s] 44%|████▎     | 174149/400000 [00:23<00:30, 7330.42it/s] 44%|████▎     | 174893/400000 [00:23<00:30, 7360.19it/s] 44%|████▍     | 175630/400000 [00:23<00:30, 7331.12it/s] 44%|████▍     | 176377/400000 [00:23<00:30, 7370.56it/s] 44%|████▍     | 177115/400000 [00:24<00:30, 7371.77it/s] 44%|████▍     | 177862/400000 [00:24<00:30, 7399.23it/s] 45%|████▍     | 178615/400000 [00:24<00:29, 7437.91it/s] 45%|████▍     | 179359/400000 [00:24<00:29, 7437.56it/s] 45%|████▌     | 180103/400000 [00:24<00:29, 7403.33it/s] 45%|████▌     | 180844/400000 [00:24<00:30, 7285.98it/s] 45%|████▌     | 181574/400000 [00:24<00:30, 7223.15it/s] 46%|████▌     | 182320/400000 [00:24<00:29, 7289.61it/s] 46%|████▌     | 183050/400000 [00:24<00:29, 7267.65it/s] 46%|████▌     | 183792/400000 [00:24<00:29, 7310.57it/s] 46%|████▌     | 184530/400000 [00:25<00:29, 7330.98it/s] 46%|████▋     | 185271/400000 [00:25<00:29, 7352.46it/s] 47%|████▋     | 186021/400000 [00:25<00:28, 7393.65it/s] 47%|████▋     | 186761/400000 [00:25<00:28, 7359.54it/s] 47%|████▋     | 187503/400000 [00:25<00:28, 7376.66it/s] 47%|████▋     | 188241/400000 [00:25<00:28, 7356.94it/s] 47%|████▋     | 188977/400000 [00:25<00:28, 7339.44it/s] 47%|████▋     | 189712/400000 [00:25<00:28, 7266.34it/s] 48%|████▊     | 190439/400000 [00:25<00:28, 7250.72it/s] 48%|████▊     | 191169/400000 [00:25<00:28, 7263.50it/s] 48%|████▊     | 191904/400000 [00:26<00:28, 7288.59it/s] 48%|████▊     | 192637/400000 [00:26<00:28, 7299.21it/s] 48%|████▊     | 193384/400000 [00:26<00:28, 7349.55it/s] 49%|████▊     | 194120/400000 [00:26<00:28, 7323.88it/s] 49%|████▊     | 194853/400000 [00:26<00:28, 7221.24it/s] 49%|████▉     | 195591/400000 [00:26<00:28, 7267.18it/s] 49%|████▉     | 196324/400000 [00:26<00:27, 7284.48it/s] 49%|████▉     | 197079/400000 [00:26<00:27, 7359.53it/s] 49%|████▉     | 197816/400000 [00:26<00:27, 7361.27it/s] 50%|████▉     | 198576/400000 [00:26<00:27, 7428.89it/s] 50%|████▉     | 199340/400000 [00:27<00:26, 7489.78it/s] 50%|█████     | 200090/400000 [00:27<00:26, 7459.44it/s] 50%|█████     | 200850/400000 [00:27<00:26, 7498.48it/s] 50%|█████     | 201601/400000 [00:27<00:26, 7464.91it/s] 51%|█████     | 202348/400000 [00:27<00:26, 7464.69it/s] 51%|█████     | 203111/400000 [00:27<00:26, 7513.35it/s] 51%|█████     | 203863/400000 [00:27<00:26, 7502.19it/s] 51%|█████     | 204614/400000 [00:27<00:26, 7492.10it/s] 51%|█████▏    | 205364/400000 [00:27<00:26, 7475.37it/s] 52%|█████▏    | 206112/400000 [00:27<00:25, 7473.93it/s] 52%|█████▏    | 206873/400000 [00:28<00:25, 7512.98it/s] 52%|█████▏    | 207625/400000 [00:28<00:25, 7508.76it/s] 52%|█████▏    | 208384/400000 [00:28<00:25, 7530.91it/s] 52%|█████▏    | 209138/400000 [00:28<00:25, 7513.96it/s] 52%|█████▏    | 209890/400000 [00:28<00:25, 7448.08it/s] 53%|█████▎    | 210654/400000 [00:28<00:25, 7502.43it/s] 53%|█████▎    | 211405/400000 [00:28<00:25, 7441.66it/s] 53%|█████▎    | 212157/400000 [00:28<00:25, 7464.01it/s] 53%|█████▎    | 212904/400000 [00:28<00:25, 7389.15it/s] 53%|█████▎    | 213648/400000 [00:29<00:25, 7402.23it/s] 54%|█████▎    | 214389/400000 [00:29<00:25, 7396.64it/s] 54%|█████▍    | 215135/400000 [00:29<00:24, 7413.86it/s] 54%|█████▍    | 215877/400000 [00:29<00:24, 7400.45it/s] 54%|█████▍    | 216618/400000 [00:29<00:24, 7390.11it/s] 54%|█████▍    | 217358/400000 [00:29<00:25, 7300.10it/s] 55%|█████▍    | 218089/400000 [00:29<00:25, 7249.65it/s] 55%|█████▍    | 218817/400000 [00:29<00:24, 7257.56it/s] 55%|█████▍    | 219564/400000 [00:29<00:24, 7318.16it/s] 55%|█████▌    | 220297/400000 [00:29<00:24, 7294.41it/s] 55%|█████▌    | 221030/400000 [00:30<00:24, 7304.69it/s] 55%|█████▌    | 221772/400000 [00:30<00:24, 7338.15it/s] 56%|█████▌    | 222520/400000 [00:30<00:24, 7379.81it/s] 56%|█████▌    | 223260/400000 [00:30<00:23, 7384.99it/s] 56%|█████▌    | 224007/400000 [00:30<00:23, 7409.35it/s] 56%|█████▌    | 224749/400000 [00:30<00:23, 7380.24it/s] 56%|█████▋    | 225488/400000 [00:30<00:23, 7382.55it/s] 57%|█████▋    | 226227/400000 [00:30<00:23, 7293.62it/s] 57%|█████▋    | 226965/400000 [00:30<00:23, 7318.98it/s] 57%|█████▋    | 227698/400000 [00:30<00:23, 7282.57it/s] 57%|█████▋    | 228427/400000 [00:31<00:23, 7267.14it/s] 57%|█████▋    | 229158/400000 [00:31<00:23, 7277.71it/s] 57%|█████▋    | 229887/400000 [00:31<00:23, 7279.44it/s] 58%|█████▊    | 230616/400000 [00:31<00:23, 7185.36it/s] 58%|█████▊    | 231356/400000 [00:31<00:23, 7245.99it/s] 58%|█████▊    | 232081/400000 [00:31<00:23, 7235.70it/s] 58%|█████▊    | 232820/400000 [00:31<00:22, 7280.32it/s] 58%|█████▊    | 233550/400000 [00:31<00:22, 7284.09it/s] 59%|█████▊    | 234285/400000 [00:31<00:22, 7301.23it/s] 59%|█████▉    | 235023/400000 [00:31<00:22, 7324.24it/s] 59%|█████▉    | 235760/400000 [00:32<00:22, 7335.36it/s] 59%|█████▉    | 236516/400000 [00:32<00:22, 7398.53it/s] 59%|█████▉    | 237257/400000 [00:32<00:22, 7370.27it/s] 59%|█████▉    | 237995/400000 [00:32<00:22, 7350.53it/s] 60%|█████▉    | 238731/400000 [00:32<00:22, 7328.26it/s] 60%|█████▉    | 239464/400000 [00:32<00:21, 7321.38it/s] 60%|██████    | 240217/400000 [00:32<00:21, 7381.96it/s] 60%|██████    | 240956/400000 [00:32<00:21, 7370.82it/s] 60%|██████    | 241694/400000 [00:32<00:21, 7362.35it/s] 61%|██████    | 242431/400000 [00:32<00:21, 7284.24it/s] 61%|██████    | 243175/400000 [00:33<00:21, 7329.71it/s] 61%|██████    | 243925/400000 [00:33<00:21, 7378.31it/s] 61%|██████    | 244667/400000 [00:33<00:21, 7388.02it/s] 61%|██████▏   | 245406/400000 [00:33<00:20, 7385.95it/s] 62%|██████▏   | 246145/400000 [00:33<00:20, 7381.73it/s] 62%|██████▏   | 246899/400000 [00:33<00:20, 7426.98it/s] 62%|██████▏   | 247659/400000 [00:33<00:20, 7476.14it/s] 62%|██████▏   | 248407/400000 [00:33<00:20, 7354.13it/s] 62%|██████▏   | 249145/400000 [00:33<00:20, 7360.62it/s] 62%|██████▏   | 249882/400000 [00:33<00:21, 7049.16it/s] 63%|██████▎   | 250618/400000 [00:34<00:20, 7139.60it/s] 63%|██████▎   | 251367/400000 [00:34<00:20, 7239.65it/s] 63%|██████▎   | 252127/400000 [00:34<00:20, 7342.12it/s] 63%|██████▎   | 252872/400000 [00:34<00:19, 7373.49it/s] 63%|██████▎   | 253613/400000 [00:34<00:19, 7383.50it/s] 64%|██████▎   | 254354/400000 [00:34<00:19, 7389.99it/s] 64%|██████▍   | 255104/400000 [00:34<00:19, 7421.56it/s] 64%|██████▍   | 255847/400000 [00:34<00:19, 7361.29it/s] 64%|██████▍   | 256593/400000 [00:34<00:19, 7388.92it/s] 64%|██████▍   | 257333/400000 [00:34<00:19, 7352.53it/s] 65%|██████▍   | 258070/400000 [00:35<00:19, 7355.92it/s] 65%|██████▍   | 258815/400000 [00:35<00:19, 7382.71it/s] 65%|██████▍   | 259554/400000 [00:35<00:19, 7335.95it/s] 65%|██████▌   | 260310/400000 [00:35<00:18, 7401.70it/s] 65%|██████▌   | 261054/400000 [00:35<00:18, 7411.35it/s] 65%|██████▌   | 261801/400000 [00:35<00:18, 7427.91it/s] 66%|██████▌   | 262545/400000 [00:35<00:18, 7430.83it/s] 66%|██████▌   | 263289/400000 [00:35<00:18, 7432.72it/s] 66%|██████▌   | 264033/400000 [00:35<00:18, 7414.73it/s] 66%|██████▌   | 264775/400000 [00:35<00:18, 7368.45it/s] 66%|██████▋   | 265512/400000 [00:36<00:18, 7354.09it/s] 67%|██████▋   | 266255/400000 [00:36<00:18, 7376.40it/s] 67%|██████▋   | 267012/400000 [00:36<00:17, 7432.79it/s] 67%|██████▋   | 267756/400000 [00:36<00:17, 7387.29it/s] 67%|██████▋   | 268495/400000 [00:36<00:17, 7343.17it/s] 67%|██████▋   | 269230/400000 [00:36<00:17, 7339.72it/s] 67%|██████▋   | 269972/400000 [00:36<00:17, 7363.56it/s] 68%|██████▊   | 270722/400000 [00:36<00:17, 7403.80it/s] 68%|██████▊   | 271497/400000 [00:36<00:17, 7503.04it/s] 68%|██████▊   | 272253/400000 [00:36<00:16, 7517.11it/s] 68%|██████▊   | 273009/400000 [00:37<00:16, 7527.94it/s] 68%|██████▊   | 273766/400000 [00:37<00:16, 7538.73it/s] 69%|██████▊   | 274521/400000 [00:37<00:16, 7508.57it/s] 69%|██████▉   | 275272/400000 [00:37<00:16, 7367.67it/s] 69%|██████▉   | 276012/400000 [00:37<00:16, 7374.47it/s] 69%|██████▉   | 276750/400000 [00:37<00:17, 7141.39it/s] 69%|██████▉   | 277468/400000 [00:37<00:17, 7151.28it/s] 70%|██████▉   | 278196/400000 [00:37<00:16, 7188.77it/s] 70%|██████▉   | 278938/400000 [00:37<00:16, 7255.79it/s] 70%|██████▉   | 279687/400000 [00:37<00:16, 7323.73it/s] 70%|███████   | 280449/400000 [00:38<00:16, 7408.49it/s] 70%|███████   | 281195/400000 [00:38<00:16, 7421.77it/s] 70%|███████   | 281954/400000 [00:38<00:15, 7469.63it/s] 71%|███████   | 282702/400000 [00:38<00:15, 7426.40it/s] 71%|███████   | 283451/400000 [00:38<00:15, 7443.29it/s] 71%|███████   | 284196/400000 [00:38<00:15, 7445.26it/s] 71%|███████   | 284941/400000 [00:38<00:15, 7428.92it/s] 71%|███████▏  | 285685/400000 [00:38<00:15, 7187.31it/s] 72%|███████▏  | 286406/400000 [00:38<00:15, 7185.51it/s] 72%|███████▏  | 287133/400000 [00:39<00:15, 7210.26it/s] 72%|███████▏  | 287867/400000 [00:39<00:15, 7248.64it/s] 72%|███████▏  | 288611/400000 [00:39<00:15, 7303.06it/s] 72%|███████▏  | 289361/400000 [00:39<00:15, 7360.56it/s] 73%|███████▎  | 290098/400000 [00:39<00:15, 7229.18it/s] 73%|███████▎  | 290850/400000 [00:39<00:14, 7311.83it/s] 73%|███████▎  | 291584/400000 [00:39<00:14, 7316.87it/s] 73%|███████▎  | 292330/400000 [00:39<00:14, 7357.19it/s] 73%|███████▎  | 293080/400000 [00:39<00:14, 7399.02it/s] 73%|███████▎  | 293821/400000 [00:39<00:14, 7387.46it/s] 74%|███████▎  | 294561/400000 [00:40<00:14, 7340.64it/s] 74%|███████▍  | 295296/400000 [00:40<00:14, 7281.85it/s] 74%|███████▍  | 296058/400000 [00:40<00:14, 7379.95it/s] 74%|███████▍  | 296826/400000 [00:40<00:13, 7465.21it/s] 74%|███████▍  | 297584/400000 [00:40<00:13, 7498.77it/s] 75%|███████▍  | 298343/400000 [00:40<00:13, 7523.64it/s] 75%|███████▍  | 299096/400000 [00:40<00:13, 7523.12it/s] 75%|███████▍  | 299849/400000 [00:40<00:13, 7493.86it/s] 75%|███████▌  | 300599/400000 [00:40<00:13, 7484.26it/s] 75%|███████▌  | 301348/400000 [00:40<00:13, 7421.88it/s] 76%|███████▌  | 302091/400000 [00:41<00:13, 7372.85it/s] 76%|███████▌  | 302839/400000 [00:41<00:13, 7403.74it/s] 76%|███████▌  | 303591/400000 [00:41<00:12, 7437.52it/s] 76%|███████▌  | 304370/400000 [00:41<00:12, 7537.23it/s] 76%|███████▋  | 305125/400000 [00:41<00:13, 7243.46it/s] 76%|███████▋  | 305880/400000 [00:41<00:12, 7331.04it/s] 77%|███████▋  | 306616/400000 [00:41<00:12, 7304.04it/s] 77%|███████▋  | 307348/400000 [00:41<00:12, 7224.67it/s] 77%|███████▋  | 308083/400000 [00:41<00:12, 7260.16it/s] 77%|███████▋  | 308814/400000 [00:41<00:12, 7274.84it/s] 77%|███████▋  | 309543/400000 [00:42<00:12, 7277.45it/s] 78%|███████▊  | 310277/400000 [00:42<00:12, 7295.75it/s] 78%|███████▊  | 311022/400000 [00:42<00:12, 7339.43it/s] 78%|███████▊  | 311767/400000 [00:42<00:11, 7369.27it/s] 78%|███████▊  | 312514/400000 [00:42<00:11, 7396.85it/s] 78%|███████▊  | 313273/400000 [00:42<00:11, 7453.68it/s] 79%|███████▊  | 314034/400000 [00:42<00:11, 7499.11it/s] 79%|███████▊  | 314785/400000 [00:42<00:11, 7465.93it/s] 79%|███████▉  | 315532/400000 [00:42<00:11, 7463.01it/s] 79%|███████▉  | 316279/400000 [00:42<00:11, 7414.89it/s] 79%|███████▉  | 317036/400000 [00:43<00:11, 7458.40it/s] 79%|███████▉  | 317783/400000 [00:43<00:11, 7171.42it/s] 80%|███████▉  | 318524/400000 [00:43<00:11, 7240.78it/s] 80%|███████▉  | 319250/400000 [00:43<00:11, 7244.19it/s] 80%|███████▉  | 319995/400000 [00:43<00:10, 7304.36it/s] 80%|████████  | 320737/400000 [00:43<00:10, 7337.54it/s] 80%|████████  | 321492/400000 [00:43<00:10, 7397.95it/s] 81%|████████  | 322233/400000 [00:43<00:10, 7373.99it/s] 81%|████████  | 322975/400000 [00:43<00:10, 7386.93it/s] 81%|████████  | 323715/400000 [00:43<00:10, 7370.52it/s] 81%|████████  | 324468/400000 [00:44<00:10, 7417.37it/s] 81%|████████▏ | 325216/400000 [00:44<00:10, 7434.59it/s] 81%|████████▏ | 325960/400000 [00:44<00:10, 7352.83it/s] 82%|████████▏ | 326696/400000 [00:44<00:10, 7291.79it/s] 82%|████████▏ | 327440/400000 [00:44<00:09, 7333.63it/s] 82%|████████▏ | 328189/400000 [00:44<00:09, 7378.45it/s] 82%|████████▏ | 328945/400000 [00:44<00:09, 7429.91it/s] 82%|████████▏ | 329692/400000 [00:44<00:09, 7440.31it/s] 83%|████████▎ | 330442/400000 [00:44<00:09, 7457.97it/s] 83%|████████▎ | 331188/400000 [00:44<00:09, 7404.88it/s] 83%|████████▎ | 331929/400000 [00:45<00:09, 7325.53it/s] 83%|████████▎ | 332662/400000 [00:45<00:09, 7167.51it/s] 83%|████████▎ | 333406/400000 [00:45<00:09, 7246.72it/s] 84%|████████▎ | 334132/400000 [00:45<00:09, 7103.49it/s] 84%|████████▎ | 334881/400000 [00:45<00:09, 7213.17it/s] 84%|████████▍ | 335628/400000 [00:45<00:08, 7285.53it/s] 84%|████████▍ | 336358/400000 [00:45<00:08, 7226.63it/s] 84%|████████▍ | 337105/400000 [00:45<00:08, 7297.54it/s] 84%|████████▍ | 337836/400000 [00:45<00:08, 7275.41it/s] 85%|████████▍ | 338593/400000 [00:46<00:08, 7358.57it/s] 85%|████████▍ | 339330/400000 [00:46<00:08, 7191.31it/s] 85%|████████▌ | 340070/400000 [00:46<00:08, 7250.25it/s] 85%|████████▌ | 340797/400000 [00:46<00:08, 7250.97it/s] 85%|████████▌ | 341527/400000 [00:46<00:08, 7263.26it/s] 86%|████████▌ | 342254/400000 [00:46<00:08, 7158.29it/s] 86%|████████▌ | 342999/400000 [00:46<00:07, 7240.96it/s] 86%|████████▌ | 343740/400000 [00:46<00:07, 7290.68it/s] 86%|████████▌ | 344483/400000 [00:46<00:07, 7329.90it/s] 86%|████████▋ | 345245/400000 [00:46<00:07, 7412.90it/s] 86%|████████▋ | 345990/400000 [00:47<00:07, 7421.68it/s] 87%|████████▋ | 346759/400000 [00:47<00:07, 7499.09it/s] 87%|████████▋ | 347515/400000 [00:47<00:06, 7515.73it/s] 87%|████████▋ | 348268/400000 [00:47<00:06, 7517.16it/s] 87%|████████▋ | 349020/400000 [00:47<00:06, 7487.81it/s] 87%|████████▋ | 349769/400000 [00:47<00:06, 7470.93it/s] 88%|████████▊ | 350517/400000 [00:47<00:06, 7429.34it/s] 88%|████████▊ | 351261/400000 [00:47<00:06, 7342.30it/s] 88%|████████▊ | 352006/400000 [00:47<00:06, 7371.71it/s] 88%|████████▊ | 352744/400000 [00:47<00:06, 7363.97it/s] 88%|████████▊ | 353481/400000 [00:48<00:06, 7289.85it/s] 89%|████████▊ | 354222/400000 [00:48<00:06, 7324.28it/s] 89%|████████▊ | 354966/400000 [00:48<00:06, 7357.77it/s] 89%|████████▉ | 355711/400000 [00:48<00:05, 7383.60it/s] 89%|████████▉ | 356455/400000 [00:48<00:05, 7397.93it/s] 89%|████████▉ | 357195/400000 [00:48<00:05, 7398.21it/s] 89%|████████▉ | 357937/400000 [00:48<00:05, 7404.73it/s] 90%|████████▉ | 358678/400000 [00:48<00:05, 7361.98it/s] 90%|████████▉ | 359415/400000 [00:48<00:05, 7330.32it/s] 90%|█████████ | 360152/400000 [00:48<00:05, 7341.81it/s] 90%|█████████ | 360887/400000 [00:49<00:05, 7338.35it/s] 90%|█████████ | 361623/400000 [00:49<00:05, 7344.05it/s] 91%|█████████ | 362361/400000 [00:49<00:05, 7353.80it/s] 91%|█████████ | 363100/400000 [00:49<00:05, 7361.52it/s] 91%|█████████ | 363848/400000 [00:49<00:04, 7394.18it/s] 91%|█████████ | 364589/400000 [00:49<00:04, 7389.30it/s] 91%|█████████▏| 365339/400000 [00:49<00:04, 7422.03it/s] 92%|█████████▏| 366082/400000 [00:49<00:04, 7385.37it/s] 92%|█████████▏| 366821/400000 [00:49<00:04, 7300.99it/s] 92%|█████████▏| 367581/400000 [00:49<00:04, 7385.13it/s] 92%|█████████▏| 368321/400000 [00:50<00:04, 7387.02it/s] 92%|█████████▏| 369061/400000 [00:50<00:04, 7364.62it/s] 92%|█████████▏| 369798/400000 [00:50<00:04, 7321.67it/s] 93%|█████████▎| 370531/400000 [00:50<00:04, 7262.58it/s] 93%|█████████▎| 371277/400000 [00:50<00:03, 7318.95it/s] 93%|█████████▎| 372046/400000 [00:50<00:03, 7425.36it/s] 93%|█████████▎| 372806/400000 [00:50<00:03, 7475.82it/s] 93%|█████████▎| 373555/400000 [00:50<00:03, 7473.94it/s] 94%|█████████▎| 374303/400000 [00:50<00:03, 7370.70it/s] 94%|█████████▍| 375041/400000 [00:50<00:03, 7361.83it/s] 94%|█████████▍| 375778/400000 [00:51<00:03, 7309.92it/s] 94%|█████████▍| 376522/400000 [00:51<00:03, 7347.69it/s] 94%|█████████▍| 377269/400000 [00:51<00:03, 7382.26it/s] 95%|█████████▍| 378008/400000 [00:51<00:02, 7354.90it/s] 95%|█████████▍| 378764/400000 [00:51<00:02, 7414.90it/s] 95%|█████████▍| 379516/400000 [00:51<00:02, 7443.28it/s] 95%|█████████▌| 380261/400000 [00:51<00:02, 7438.10it/s] 95%|█████████▌| 381005/400000 [00:51<00:02, 7339.00it/s] 95%|█████████▌| 381743/400000 [00:51<00:02, 7349.42it/s] 96%|█████████▌| 382488/400000 [00:51<00:02, 7376.56it/s] 96%|█████████▌| 383227/400000 [00:52<00:02, 7378.75it/s] 96%|█████████▌| 383968/400000 [00:52<00:02, 7387.83it/s] 96%|█████████▌| 384723/400000 [00:52<00:02, 7435.35it/s] 96%|█████████▋| 385467/400000 [00:52<00:01, 7399.23it/s] 97%|█████████▋| 386219/400000 [00:52<00:01, 7432.66it/s] 97%|█████████▋| 386976/400000 [00:52<00:01, 7471.52it/s] 97%|█████████▋| 387729/400000 [00:52<00:01, 7487.52it/s] 97%|█████████▋| 388478/400000 [00:52<00:01, 7479.99it/s] 97%|█████████▋| 389233/400000 [00:52<00:01, 7497.97it/s] 98%|█████████▊| 390005/400000 [00:52<00:01, 7561.41it/s] 98%|█████████▊| 390762/400000 [00:53<00:01, 7466.41it/s] 98%|█████████▊| 391510/400000 [00:53<00:01, 7451.91it/s] 98%|█████████▊| 392256/400000 [00:53<00:01, 7387.32it/s] 98%|█████████▊| 392996/400000 [00:53<00:00, 7358.63it/s] 98%|█████████▊| 393740/400000 [00:53<00:00, 7380.34it/s] 99%|█████████▊| 394481/400000 [00:53<00:00, 7387.83it/s] 99%|█████████▉| 395231/400000 [00:53<00:00, 7418.42it/s] 99%|█████████▉| 395994/400000 [00:53<00:00, 7479.87it/s] 99%|█████████▉| 396743/400000 [00:53<00:00, 7409.04it/s] 99%|█████████▉| 397496/400000 [00:53<00:00, 7444.19it/s]100%|█████████▉| 398260/400000 [00:54<00:00, 7500.79it/s]100%|█████████▉| 399011/400000 [00:54<00:00, 7479.94it/s]100%|█████████▉| 399768/400000 [00:54<00:00, 7504.93it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7364.33it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f3abefc0a58> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01102830642282707 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.010883713246986618 	 Accuracy: 61

  model saves at 61% accuracy 

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
2020-05-14 06:26:09.073909: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 06:26:09.078202: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-14 06:26:09.078351: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a8ef33b5b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 06:26:09.078368: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f3a64b21fd0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.9580 - accuracy: 0.4810
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6666 - accuracy: 0.5000
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.7535 - accuracy: 0.4943 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7893 - accuracy: 0.4920
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7556 - accuracy: 0.4942
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.7177 - accuracy: 0.4967
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6622 - accuracy: 0.5003
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6954 - accuracy: 0.4981
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7177 - accuracy: 0.4967
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7065 - accuracy: 0.4974
11000/25000 [============>.................] - ETA: 4s - loss: 7.6820 - accuracy: 0.4990
12000/25000 [=============>................] - ETA: 4s - loss: 7.6538 - accuracy: 0.5008
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6548 - accuracy: 0.5008
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6568 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 3s - loss: 7.6533 - accuracy: 0.5009
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6599 - accuracy: 0.5004
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6648 - accuracy: 0.5001
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6871 - accuracy: 0.4987
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6820 - accuracy: 0.4990
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6758 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6929 - accuracy: 0.4983
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6875 - accuracy: 0.4986
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6813 - accuracy: 0.4990
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6711 - accuracy: 0.4997
25000/25000 [==============================] - 10s 399us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f3a1ff746d8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f3a64a80390> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4704 - crf_viterbi_accuracy: 0.1333 - val_loss: 1.3577 - val_crf_viterbi_accuracy: 0.0933

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

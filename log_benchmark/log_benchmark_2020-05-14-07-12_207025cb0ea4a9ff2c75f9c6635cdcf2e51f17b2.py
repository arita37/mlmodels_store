
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fbc74c01f60> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 07:12:50.960932
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 07:12:50.964834
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 07:12:50.967907
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 07:12:50.971058
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fbc809cb438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354771.9375
Epoch 2/10

1/1 [==============================] - 0s 109ms/step - loss: 272512.8438
Epoch 3/10

1/1 [==============================] - 0s 108ms/step - loss: 192459.4531
Epoch 4/10

1/1 [==============================] - 0s 98ms/step - loss: 118883.5312
Epoch 5/10

1/1 [==============================] - 0s 102ms/step - loss: 71142.3906
Epoch 6/10

1/1 [==============================] - 0s 96ms/step - loss: 43059.8945
Epoch 7/10

1/1 [==============================] - 0s 110ms/step - loss: 27486.0996
Epoch 8/10

1/1 [==============================] - 0s 101ms/step - loss: 18743.6113
Epoch 9/10

1/1 [==============================] - 0s 115ms/step - loss: 13424.5967
Epoch 10/10

1/1 [==============================] - 0s 99ms/step - loss: 10029.2344

  #### Inference Need return ypred, ytrue ######################### 
[[-7.65469968e-02  4.55682516e+00  5.33534765e+00  5.43529177e+00
   7.77476358e+00  5.69773579e+00  5.70399761e+00  5.53261566e+00
   5.88962555e+00  6.98802853e+00  6.38339138e+00  7.16243505e+00
   6.75415707e+00  6.91005421e+00  6.98459291e+00  6.75974989e+00
   6.37139654e+00  4.52534008e+00  7.81457329e+00  5.58724689e+00
   6.63226080e+00  6.77710152e+00  7.68788576e+00  5.48207140e+00
   6.71711063e+00  7.54148388e+00  6.02214622e+00  5.57978201e+00
   5.90234232e+00  5.49395037e+00  6.53395987e+00  4.89559937e+00
   7.01060152e+00  6.62972212e+00  5.43342733e+00  7.54247046e+00
   6.14525080e+00  5.79161358e+00  5.61126995e+00  6.54026461e+00
   6.77200937e+00  7.06404686e+00  6.15108585e+00  6.53411674e+00
   6.01481533e+00  6.88595533e+00  7.26806688e+00  7.52304840e+00
   6.72493649e+00  5.71169615e+00  7.21963978e+00  5.27420330e+00
   6.15777731e+00  6.86242914e+00  5.85837841e+00  7.68902540e+00
   6.11273098e+00  4.63981199e+00  6.08907700e+00  6.37726593e+00
   1.13215709e+00 -1.29356325e+00 -3.13315541e-01 -1.20714080e+00
  -9.64128375e-01  1.54849458e+00  1.28928766e-01  5.47314882e-01
  -1.25100970e+00 -6.07499719e-01  1.07744741e+00  1.30066824e+00
  -6.93577528e-01 -6.89121485e-01 -1.30021679e+00  8.95646811e-02
   7.42537856e-01  3.91998619e-01 -4.58188146e-01  2.97381252e-01
  -3.40111047e-01 -3.86227906e-01  1.54849982e+00 -8.81711006e-01
   8.13496709e-01  7.01314807e-01  9.13856626e-01 -5.38043320e-01
  -3.85482967e-01 -6.84303463e-01  2.95054317e-02  7.07046330e-01
  -1.20985150e+00 -1.18496805e-01  2.23797187e-02  5.78687787e-02
   2.39045292e-01 -7.37897873e-01  4.19760764e-01  5.87988734e-01
   4.78624046e-01 -4.34233189e-01  8.96831155e-01 -5.58192551e-01
  -8.88494551e-01 -2.87634641e-01  2.59511888e-01 -8.33776593e-03
   9.61204767e-01  7.57226497e-02 -1.41571864e-01 -1.53955591e+00
  -7.10816383e-02 -1.04144943e+00  6.05269015e-01 -2.08725989e-01
  -4.32684630e-01  7.39751458e-01  2.87669092e-01  1.30895448e+00
  -5.71228802e-01  1.91259384e+00 -3.49488437e-01  2.64821053e-02
   1.87439704e+00 -9.64445591e-01  5.09806037e-01 -7.03117549e-01
   9.01525438e-01 -1.09052151e-01  5.94733655e-01  1.19757247e+00
  -1.32011855e+00  2.14502871e-01 -5.11799693e-01 -4.40740287e-02
  -1.35891080e+00  6.16458297e-01  1.43563941e-01  2.58252084e-01
  -5.09562671e-01 -7.42285967e-01 -2.08798647e-01 -4.20212358e-01
  -1.05865109e+00 -6.46149516e-01  2.55858600e-02 -1.58355677e+00
  -1.28454447e+00  1.86550438e-01 -1.10366929e+00 -8.18206072e-01
   3.01450729e-01  1.51072037e+00  5.53212821e-01  9.52447414e-01
   9.48069334e-01  2.37247765e-01 -8.54782283e-01  6.78129196e-02
   7.59436250e-01  4.97774482e-02  7.17663050e-01  1.47332668e-01
  -3.88427675e-01  1.48709679e+00  5.37976861e-01 -9.17570829e-01
  -4.49602932e-01 -1.06801152e+00  1.33200073e+00 -1.00184429e+00
  -3.16914558e-01  1.56388283e+00 -2.51197845e-01 -5.18663049e-01
   3.85359764e-01 -4.03805733e-01  2.48392344e-01  6.57698870e-01
   2.87988186e-02  7.11108971e+00  6.23863316e+00  7.93649721e+00
   5.83091068e+00  6.83572769e+00  6.27675724e+00  6.79232407e+00
   5.49230766e+00  5.35979986e+00  7.47400665e+00  6.56893063e+00
   6.07243061e+00  5.21685886e+00  7.72992849e+00  7.95047998e+00
   6.65794802e+00  6.89968204e+00  7.00695229e+00  8.22468376e+00
   6.16086292e+00  6.41361523e+00  6.70951557e+00  6.88736725e+00
   7.27422190e+00  6.67354298e+00  6.65502834e+00  7.66044378e+00
   6.35866785e+00  6.51657248e+00  6.43127108e+00  6.35601854e+00
   6.88011122e+00  6.43589020e+00  7.44955730e+00  6.82542276e+00
   7.42240906e+00  6.55369854e+00  7.02117014e+00  7.46087790e+00
   6.82567072e+00  8.16575050e+00  6.36767912e+00  7.65771675e+00
   7.21126699e+00  7.65177011e+00  8.07814407e+00  5.44923830e+00
   7.15535927e+00  6.90565968e+00  7.68105936e+00  6.50203037e+00
   6.10835791e+00  6.49352837e+00  8.07925987e+00  6.46311235e+00
   7.45710421e+00  7.39250851e+00  8.19017315e+00  6.06374836e+00
   2.58931684e+00  1.74104798e+00  7.15626955e-01  1.88887119e+00
   1.30192971e+00  6.50841713e-01  6.42269731e-01  4.06634331e-01
   7.00324774e-01  4.62922454e-01  8.21335614e-01  2.53039169e+00
   4.31106389e-01  6.93609655e-01  1.81260681e+00  5.17210424e-01
   1.85409880e+00  1.79829991e+00  1.19945383e+00  3.55668306e-01
   1.62376881e-01  4.22313452e-01  5.68857133e-01  5.56838334e-01
   2.83480477e+00  9.74736333e-01  1.77761316e-01  1.34432101e+00
   1.85725963e+00  2.25110674e+00  1.73072422e+00  5.22592008e-01
   1.95335126e+00  8.55321467e-01  1.94249439e+00  1.93045568e+00
   1.20349646e+00  2.00349498e+00  3.37941408e-01  2.05120587e+00
   4.10553813e-01  1.44669724e+00  1.66907597e+00  1.03823602e+00
   2.64275551e-01  9.72952962e-01  5.74401319e-01  9.55685556e-01
   7.31389284e-01  1.03020716e+00  1.15437758e+00  1.26174426e+00
   1.80395067e-01  2.81250477e-01  2.76785970e-01  5.91284811e-01
   1.97621417e+00  1.98964667e+00  1.09535837e+00  1.28301764e+00
   3.23365211e-01  5.61028540e-01  7.90991068e-01  1.56681156e+00
   4.70197737e-01  3.86019111e-01  5.16714454e-01  2.49765491e+00
   1.66467905e+00  8.15479934e-01  6.14810050e-01  1.31268597e+00
   4.97952700e-01  4.13290858e-01  2.02316427e+00  2.31699228e-01
   1.27828574e+00  7.84274638e-01  1.39851332e+00  8.38394165e-01
   3.66627932e-01  5.52302182e-01  3.95227075e-01  1.17988300e+00
   4.83138680e-01  3.97815645e-01  1.12899375e+00  6.55935645e-01
   6.26540422e-01  7.60673046e-01  2.36466265e+00  5.62263131e-01
   4.34571087e-01  4.22529995e-01  3.09191346e-01  6.52867436e-01
   1.59679437e+00  2.43344188e-01  7.58766711e-01  1.34295511e+00
   2.28380728e+00  5.63590646e-01  7.50430942e-01  2.37740755e+00
   1.67753077e+00  2.66195357e-01  2.74979591e-01  1.10421157e+00
   1.26202857e+00  2.94031322e-01  1.62549269e+00  1.27381611e+00
   7.42350578e-01  7.59703934e-01  1.18993962e+00  8.67869973e-01
   8.03422987e-01  2.21420956e+00  6.97906494e-01  8.56447995e-01
   1.00736427e+01 -6.27558661e+00 -5.92094231e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 07:13:00.033633
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    95.075
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 07:13:00.037297
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9060.63
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 07:13:00.040399
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.3911
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 07:13:00.043647
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -810.443
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140447030260008
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140445820404232
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140445820404736
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140445820405240
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140445820405744
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140445820406248

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fbc605dce10> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.518762
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.498959
grad_step = 000002, loss = 0.485064
grad_step = 000003, loss = 0.470423
grad_step = 000004, loss = 0.455381
grad_step = 000005, loss = 0.441027
grad_step = 000006, loss = 0.431597
grad_step = 000007, loss = 0.428087
grad_step = 000008, loss = 0.418240
grad_step = 000009, loss = 0.407145
grad_step = 000010, loss = 0.398175
grad_step = 000011, loss = 0.390950
grad_step = 000012, loss = 0.383858
grad_step = 000013, loss = 0.376090
grad_step = 000014, loss = 0.367680
grad_step = 000015, loss = 0.358642
grad_step = 000016, loss = 0.349667
grad_step = 000017, loss = 0.340383
grad_step = 000018, loss = 0.331015
grad_step = 000019, loss = 0.321177
grad_step = 000020, loss = 0.310654
grad_step = 000021, loss = 0.300547
grad_step = 000022, loss = 0.292276
grad_step = 000023, loss = 0.284544
grad_step = 000024, loss = 0.275703
grad_step = 000025, loss = 0.267650
grad_step = 000026, loss = 0.260242
grad_step = 000027, loss = 0.252651
grad_step = 000028, loss = 0.244369
grad_step = 000029, loss = 0.235428
grad_step = 000030, loss = 0.226531
grad_step = 000031, loss = 0.218494
grad_step = 000032, loss = 0.211318
grad_step = 000033, loss = 0.204108
grad_step = 000034, loss = 0.196426
grad_step = 000035, loss = 0.188942
grad_step = 000036, loss = 0.182123
grad_step = 000037, loss = 0.175741
grad_step = 000038, loss = 0.169219
grad_step = 000039, loss = 0.162532
grad_step = 000040, loss = 0.155869
grad_step = 000041, loss = 0.149275
grad_step = 000042, loss = 0.142852
grad_step = 000043, loss = 0.136803
grad_step = 000044, loss = 0.130955
grad_step = 000045, loss = 0.125122
grad_step = 000046, loss = 0.119453
grad_step = 000047, loss = 0.114111
grad_step = 000048, loss = 0.108940
grad_step = 000049, loss = 0.103861
grad_step = 000050, loss = 0.098911
grad_step = 000051, loss = 0.094016
grad_step = 000052, loss = 0.089256
grad_step = 000053, loss = 0.084781
grad_step = 000054, loss = 0.080536
grad_step = 000055, loss = 0.076424
grad_step = 000056, loss = 0.072445
grad_step = 000057, loss = 0.068599
grad_step = 000058, loss = 0.064925
grad_step = 000059, loss = 0.061414
grad_step = 000060, loss = 0.058016
grad_step = 000061, loss = 0.054756
grad_step = 000062, loss = 0.051633
grad_step = 000063, loss = 0.048673
grad_step = 000064, loss = 0.045891
grad_step = 000065, loss = 0.043232
grad_step = 000066, loss = 0.040693
grad_step = 000067, loss = 0.038260
grad_step = 000068, loss = 0.035975
grad_step = 000069, loss = 0.033851
grad_step = 000070, loss = 0.031899
grad_step = 000071, loss = 0.029918
grad_step = 000072, loss = 0.027983
grad_step = 000073, loss = 0.026267
grad_step = 000074, loss = 0.024716
grad_step = 000075, loss = 0.023176
grad_step = 000076, loss = 0.021692
grad_step = 000077, loss = 0.020403
grad_step = 000078, loss = 0.019170
grad_step = 000079, loss = 0.017939
grad_step = 000080, loss = 0.016853
grad_step = 000081, loss = 0.015881
grad_step = 000082, loss = 0.014905
grad_step = 000083, loss = 0.013979
grad_step = 000084, loss = 0.013180
grad_step = 000085, loss = 0.012427
grad_step = 000086, loss = 0.011681
grad_step = 000087, loss = 0.010999
grad_step = 000088, loss = 0.010400
grad_step = 000089, loss = 0.009821
grad_step = 000090, loss = 0.009254
grad_step = 000091, loss = 0.008745
grad_step = 000092, loss = 0.008294
grad_step = 000093, loss = 0.007859
grad_step = 000094, loss = 0.007433
grad_step = 000095, loss = 0.007040
grad_step = 000096, loss = 0.006694
grad_step = 000097, loss = 0.006370
grad_step = 000098, loss = 0.006054
grad_step = 000099, loss = 0.005753
grad_step = 000100, loss = 0.005480
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.005236
grad_step = 000102, loss = 0.005007
grad_step = 000103, loss = 0.004787
grad_step = 000104, loss = 0.004575
grad_step = 000105, loss = 0.004379
grad_step = 000106, loss = 0.004200
grad_step = 000107, loss = 0.004038
grad_step = 000108, loss = 0.003888
grad_step = 000109, loss = 0.003748
grad_step = 000110, loss = 0.003614
grad_step = 000111, loss = 0.003488
grad_step = 000112, loss = 0.003369
grad_step = 000113, loss = 0.003258
grad_step = 000114, loss = 0.003156
grad_step = 000115, loss = 0.003062
grad_step = 000116, loss = 0.002975
grad_step = 000117, loss = 0.002896
grad_step = 000118, loss = 0.002822
grad_step = 000119, loss = 0.002755
grad_step = 000120, loss = 0.002694
grad_step = 000121, loss = 0.002640
grad_step = 000122, loss = 0.002595
grad_step = 000123, loss = 0.002565
grad_step = 000124, loss = 0.002557
grad_step = 000125, loss = 0.002586
grad_step = 000126, loss = 0.002622
grad_step = 000127, loss = 0.002593
grad_step = 000128, loss = 0.002441
grad_step = 000129, loss = 0.002310
grad_step = 000130, loss = 0.002324
grad_step = 000131, loss = 0.002386
grad_step = 000132, loss = 0.002342
grad_step = 000133, loss = 0.002230
grad_step = 000134, loss = 0.002205
grad_step = 000135, loss = 0.002248
grad_step = 000136, loss = 0.002232
grad_step = 000137, loss = 0.002162
grad_step = 000138, loss = 0.002133
grad_step = 000139, loss = 0.002159
grad_step = 000140, loss = 0.002159
grad_step = 000141, loss = 0.002109
grad_step = 000142, loss = 0.002082
grad_step = 000143, loss = 0.002098
grad_step = 000144, loss = 0.002105
grad_step = 000145, loss = 0.002078
grad_step = 000146, loss = 0.002047
grad_step = 000147, loss = 0.002047
grad_step = 000148, loss = 0.002059
grad_step = 000149, loss = 0.002051
grad_step = 000150, loss = 0.002027
grad_step = 000151, loss = 0.002012
grad_step = 000152, loss = 0.002016
grad_step = 000153, loss = 0.002021
grad_step = 000154, loss = 0.002014
grad_step = 000155, loss = 0.001998
grad_step = 000156, loss = 0.001986
grad_step = 000157, loss = 0.001984
grad_step = 000158, loss = 0.001987
grad_step = 000159, loss = 0.001987
grad_step = 000160, loss = 0.001982
grad_step = 000161, loss = 0.001972
grad_step = 000162, loss = 0.001963
grad_step = 000163, loss = 0.001956
grad_step = 000164, loss = 0.001951
grad_step = 000165, loss = 0.001949
grad_step = 000166, loss = 0.001949
grad_step = 000167, loss = 0.001951
grad_step = 000168, loss = 0.001953
grad_step = 000169, loss = 0.001958
grad_step = 000170, loss = 0.001970
grad_step = 000171, loss = 0.001993
grad_step = 000172, loss = 0.002019
grad_step = 000173, loss = 0.002053
grad_step = 000174, loss = 0.002063
grad_step = 000175, loss = 0.002051
grad_step = 000176, loss = 0.001997
grad_step = 000177, loss = 0.001937
grad_step = 000178, loss = 0.001902
grad_step = 000179, loss = 0.001907
grad_step = 000180, loss = 0.001937
grad_step = 000181, loss = 0.001965
grad_step = 000182, loss = 0.001976
grad_step = 000183, loss = 0.001964
grad_step = 000184, loss = 0.001943
grad_step = 000185, loss = 0.001912
grad_step = 000186, loss = 0.001887
grad_step = 000187, loss = 0.001873
grad_step = 000188, loss = 0.001875
grad_step = 000189, loss = 0.001890
grad_step = 000190, loss = 0.001909
grad_step = 000191, loss = 0.001920
grad_step = 000192, loss = 0.001923
grad_step = 000193, loss = 0.001921
grad_step = 000194, loss = 0.001913
grad_step = 000195, loss = 0.001902
grad_step = 000196, loss = 0.001884
grad_step = 000197, loss = 0.001864
grad_step = 000198, loss = 0.001849
grad_step = 000199, loss = 0.001839
grad_step = 000200, loss = 0.001836
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001836
grad_step = 000202, loss = 0.001834
grad_step = 000203, loss = 0.001830
grad_step = 000204, loss = 0.001824
grad_step = 000205, loss = 0.001819
grad_step = 000206, loss = 0.001817
grad_step = 000207, loss = 0.001819
grad_step = 000208, loss = 0.001829
grad_step = 000209, loss = 0.001865
grad_step = 000210, loss = 0.001960
grad_step = 000211, loss = 0.002153
grad_step = 000212, loss = 0.002421
grad_step = 000213, loss = 0.002502
grad_step = 000214, loss = 0.002370
grad_step = 000215, loss = 0.002011
grad_step = 000216, loss = 0.001875
grad_step = 000217, loss = 0.002214
grad_step = 000218, loss = 0.002245
grad_step = 000219, loss = 0.001819
grad_step = 000220, loss = 0.002021
grad_step = 000221, loss = 0.002147
grad_step = 000222, loss = 0.001927
grad_step = 000223, loss = 0.002005
grad_step = 000224, loss = 0.001964
grad_step = 000225, loss = 0.001885
grad_step = 000226, loss = 0.002054
grad_step = 000227, loss = 0.001876
grad_step = 000228, loss = 0.001827
grad_step = 000229, loss = 0.001982
grad_step = 000230, loss = 0.001835
grad_step = 000231, loss = 0.001873
grad_step = 000232, loss = 0.001849
grad_step = 000233, loss = 0.001818
grad_step = 000234, loss = 0.001871
grad_step = 000235, loss = 0.001794
grad_step = 000236, loss = 0.001789
grad_step = 000237, loss = 0.001840
grad_step = 000238, loss = 0.001774
grad_step = 000239, loss = 0.001794
grad_step = 000240, loss = 0.001787
grad_step = 000241, loss = 0.001773
grad_step = 000242, loss = 0.001799
grad_step = 000243, loss = 0.001755
grad_step = 000244, loss = 0.001760
grad_step = 000245, loss = 0.001778
grad_step = 000246, loss = 0.001756
grad_step = 000247, loss = 0.001751
grad_step = 000248, loss = 0.001753
grad_step = 000249, loss = 0.001739
grad_step = 000250, loss = 0.001756
grad_step = 000251, loss = 0.001741
grad_step = 000252, loss = 0.001727
grad_step = 000253, loss = 0.001738
grad_step = 000254, loss = 0.001731
grad_step = 000255, loss = 0.001729
grad_step = 000256, loss = 0.001729
grad_step = 000257, loss = 0.001718
grad_step = 000258, loss = 0.001715
grad_step = 000259, loss = 0.001721
grad_step = 000260, loss = 0.001714
grad_step = 000261, loss = 0.001711
grad_step = 000262, loss = 0.001710
grad_step = 000263, loss = 0.001702
grad_step = 000264, loss = 0.001701
grad_step = 000265, loss = 0.001702
grad_step = 000266, loss = 0.001700
grad_step = 000267, loss = 0.001696
grad_step = 000268, loss = 0.001695
grad_step = 000269, loss = 0.001691
grad_step = 000270, loss = 0.001686
grad_step = 000271, loss = 0.001685
grad_step = 000272, loss = 0.001684
grad_step = 000273, loss = 0.001682
grad_step = 000274, loss = 0.001679
grad_step = 000275, loss = 0.001679
grad_step = 000276, loss = 0.001678
grad_step = 000277, loss = 0.001675
grad_step = 000278, loss = 0.001673
grad_step = 000279, loss = 0.001673
grad_step = 000280, loss = 0.001673
grad_step = 000281, loss = 0.001674
grad_step = 000282, loss = 0.001677
grad_step = 000283, loss = 0.001686
grad_step = 000284, loss = 0.001703
grad_step = 000285, loss = 0.001739
grad_step = 000286, loss = 0.001800
grad_step = 000287, loss = 0.001909
grad_step = 000288, loss = 0.002030
grad_step = 000289, loss = 0.002107
grad_step = 000290, loss = 0.002005
grad_step = 000291, loss = 0.001805
grad_step = 000292, loss = 0.001679
grad_step = 000293, loss = 0.001703
grad_step = 000294, loss = 0.001788
grad_step = 000295, loss = 0.001830
grad_step = 000296, loss = 0.001790
grad_step = 000297, loss = 0.001693
grad_step = 000298, loss = 0.001647
grad_step = 000299, loss = 0.001696
grad_step = 000300, loss = 0.001758
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001730
grad_step = 000302, loss = 0.001649
grad_step = 000303, loss = 0.001633
grad_step = 000304, loss = 0.001675
grad_step = 000305, loss = 0.001692
grad_step = 000306, loss = 0.001672
grad_step = 000307, loss = 0.001649
grad_step = 000308, loss = 0.001635
grad_step = 000309, loss = 0.001629
grad_step = 000310, loss = 0.001640
grad_step = 000311, loss = 0.001655
grad_step = 000312, loss = 0.001645
grad_step = 000313, loss = 0.001620
grad_step = 000314, loss = 0.001605
grad_step = 000315, loss = 0.001614
grad_step = 000316, loss = 0.001629
grad_step = 000317, loss = 0.001627
grad_step = 000318, loss = 0.001613
grad_step = 000319, loss = 0.001602
grad_step = 000320, loss = 0.001599
grad_step = 000321, loss = 0.001600
grad_step = 000322, loss = 0.001600
grad_step = 000323, loss = 0.001600
grad_step = 000324, loss = 0.001599
grad_step = 000325, loss = 0.001598
grad_step = 000326, loss = 0.001594
grad_step = 000327, loss = 0.001588
grad_step = 000328, loss = 0.001582
grad_step = 000329, loss = 0.001578
grad_step = 000330, loss = 0.001577
grad_step = 000331, loss = 0.001579
grad_step = 000332, loss = 0.001581
grad_step = 000333, loss = 0.001582
grad_step = 000334, loss = 0.001583
grad_step = 000335, loss = 0.001583
grad_step = 000336, loss = 0.001583
grad_step = 000337, loss = 0.001584
grad_step = 000338, loss = 0.001586
grad_step = 000339, loss = 0.001592
grad_step = 000340, loss = 0.001605
grad_step = 000341, loss = 0.001631
grad_step = 000342, loss = 0.001683
grad_step = 000343, loss = 0.001769
grad_step = 000344, loss = 0.001893
grad_step = 000345, loss = 0.001993
grad_step = 000346, loss = 0.001999
grad_step = 000347, loss = 0.001874
grad_step = 000348, loss = 0.001714
grad_step = 000349, loss = 0.001641
grad_step = 000350, loss = 0.001656
grad_step = 000351, loss = 0.001661
grad_step = 000352, loss = 0.001644
grad_step = 000353, loss = 0.001659
grad_step = 000354, loss = 0.001686
grad_step = 000355, loss = 0.001624
grad_step = 000356, loss = 0.001546
grad_step = 000357, loss = 0.001565
grad_step = 000358, loss = 0.001628
grad_step = 000359, loss = 0.001616
grad_step = 000360, loss = 0.001554
grad_step = 000361, loss = 0.001551
grad_step = 000362, loss = 0.001574
grad_step = 000363, loss = 0.001555
grad_step = 000364, loss = 0.001537
grad_step = 000365, loss = 0.001557
grad_step = 000366, loss = 0.001569
grad_step = 000367, loss = 0.001542
grad_step = 000368, loss = 0.001518
grad_step = 000369, loss = 0.001525
grad_step = 000370, loss = 0.001537
grad_step = 000371, loss = 0.001527
grad_step = 000372, loss = 0.001519
grad_step = 000373, loss = 0.001526
grad_step = 000374, loss = 0.001530
grad_step = 000375, loss = 0.001518
grad_step = 000376, loss = 0.001505
grad_step = 000377, loss = 0.001505
grad_step = 000378, loss = 0.001507
grad_step = 000379, loss = 0.001503
grad_step = 000380, loss = 0.001496
grad_step = 000381, loss = 0.001493
grad_step = 000382, loss = 0.001497
grad_step = 000383, loss = 0.001501
grad_step = 000384, loss = 0.001500
grad_step = 000385, loss = 0.001498
grad_step = 000386, loss = 0.001502
grad_step = 000387, loss = 0.001511
grad_step = 000388, loss = 0.001527
grad_step = 000389, loss = 0.001552
grad_step = 000390, loss = 0.001593
grad_step = 000391, loss = 0.001663
grad_step = 000392, loss = 0.001775
grad_step = 000393, loss = 0.001913
grad_step = 000394, loss = 0.002023
grad_step = 000395, loss = 0.001993
grad_step = 000396, loss = 0.001789
grad_step = 000397, loss = 0.001551
grad_step = 000398, loss = 0.001488
grad_step = 000399, loss = 0.001599
grad_step = 000400, loss = 0.001697
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001648
grad_step = 000402, loss = 0.001536
grad_step = 000403, loss = 0.001507
grad_step = 000404, loss = 0.001543
grad_step = 000405, loss = 0.001563
grad_step = 000406, loss = 0.001538
grad_step = 000407, loss = 0.001511
grad_step = 000408, loss = 0.001514
grad_step = 000409, loss = 0.001506
grad_step = 000410, loss = 0.001488
grad_step = 000411, loss = 0.001484
grad_step = 000412, loss = 0.001497
grad_step = 000413, loss = 0.001502
grad_step = 000414, loss = 0.001469
grad_step = 000415, loss = 0.001448
grad_step = 000416, loss = 0.001464
grad_step = 000417, loss = 0.001482
grad_step = 000418, loss = 0.001472
grad_step = 000419, loss = 0.001443
grad_step = 000420, loss = 0.001435
grad_step = 000421, loss = 0.001451
grad_step = 000422, loss = 0.001458
grad_step = 000423, loss = 0.001448
grad_step = 000424, loss = 0.001433
grad_step = 000425, loss = 0.001431
grad_step = 000426, loss = 0.001438
grad_step = 000427, loss = 0.001436
grad_step = 000428, loss = 0.001428
grad_step = 000429, loss = 0.001420
grad_step = 000430, loss = 0.001421
grad_step = 000431, loss = 0.001427
grad_step = 000432, loss = 0.001427
grad_step = 000433, loss = 0.001421
grad_step = 000434, loss = 0.001414
grad_step = 000435, loss = 0.001412
grad_step = 000436, loss = 0.001413
grad_step = 000437, loss = 0.001413
grad_step = 000438, loss = 0.001410
grad_step = 000439, loss = 0.001405
grad_step = 000440, loss = 0.001401
grad_step = 000441, loss = 0.001400
grad_step = 000442, loss = 0.001400
grad_step = 000443, loss = 0.001400
grad_step = 000444, loss = 0.001399
grad_step = 000445, loss = 0.001396
grad_step = 000446, loss = 0.001393
grad_step = 000447, loss = 0.001392
grad_step = 000448, loss = 0.001393
grad_step = 000449, loss = 0.001397
grad_step = 000450, loss = 0.001408
grad_step = 000451, loss = 0.001432
grad_step = 000452, loss = 0.001484
grad_step = 000453, loss = 0.001598
grad_step = 000454, loss = 0.001804
grad_step = 000455, loss = 0.002158
grad_step = 000456, loss = 0.002398
grad_step = 000457, loss = 0.002293
grad_step = 000458, loss = 0.001790
grad_step = 000459, loss = 0.001530
grad_step = 000460, loss = 0.001713
grad_step = 000461, loss = 0.001821
grad_step = 000462, loss = 0.001600
grad_step = 000463, loss = 0.001563
grad_step = 000464, loss = 0.001686
grad_step = 000465, loss = 0.001597
grad_step = 000466, loss = 0.001504
grad_step = 000467, loss = 0.001645
grad_step = 000468, loss = 0.001549
grad_step = 000469, loss = 0.001419
grad_step = 000470, loss = 0.001555
grad_step = 000471, loss = 0.001530
grad_step = 000472, loss = 0.001414
grad_step = 000473, loss = 0.001433
grad_step = 000474, loss = 0.001512
grad_step = 000475, loss = 0.001423
grad_step = 000476, loss = 0.001396
grad_step = 000477, loss = 0.001467
grad_step = 000478, loss = 0.001419
grad_step = 000479, loss = 0.001386
grad_step = 000480, loss = 0.001422
grad_step = 000481, loss = 0.001418
grad_step = 000482, loss = 0.001369
grad_step = 000483, loss = 0.001404
grad_step = 000484, loss = 0.001398
grad_step = 000485, loss = 0.001363
grad_step = 000486, loss = 0.001385
grad_step = 000487, loss = 0.001386
grad_step = 000488, loss = 0.001359
grad_step = 000489, loss = 0.001363
grad_step = 000490, loss = 0.001376
grad_step = 000491, loss = 0.001356
grad_step = 000492, loss = 0.001350
grad_step = 000493, loss = 0.001364
grad_step = 000494, loss = 0.001351
grad_step = 000495, loss = 0.001343
grad_step = 000496, loss = 0.001352
grad_step = 000497, loss = 0.001346
grad_step = 000498, loss = 0.001337
grad_step = 000499, loss = 0.001339
grad_step = 000500, loss = 0.001342
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001334
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

  date_run                              2020-05-14 07:13:19.528491
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.205192
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 07:13:19.538898
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.095262
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 07:13:19.545752
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.133617
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 07:13:19.552665
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -0.44754
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
0   2020-05-14 07:12:50.960932  ...    mean_absolute_error
1   2020-05-14 07:12:50.964834  ...     mean_squared_error
2   2020-05-14 07:12:50.967907  ...  median_absolute_error
3   2020-05-14 07:12:50.971058  ...               r2_score
4   2020-05-14 07:13:00.033633  ...    mean_absolute_error
5   2020-05-14 07:13:00.037297  ...     mean_squared_error
6   2020-05-14 07:13:00.040399  ...  median_absolute_error
7   2020-05-14 07:13:00.043647  ...               r2_score
8   2020-05-14 07:13:19.528491  ...    mean_absolute_error
9   2020-05-14 07:13:19.538898  ...     mean_squared_error
10  2020-05-14 07:13:19.545752  ...  median_absolute_error
11  2020-05-14 07:13:19.552665  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f773d2b5fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 157508.35it/s] 27%|██▋       | 2646016/9912422 [00:00<00:32, 224434.91it/s] 80%|████████  | 7938048/9912422 [00:00<00:06, 320039.48it/s]9920512it [00:00, 29848605.92it/s]                           
0it [00:00, ?it/s]32768it [00:00, 570056.67it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 464029.20it/s]1654784it [00:00, 11511067.39it/s]                         
0it [00:00, ?it/s]8192it [00:00, 191744.96it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f76efcb6e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f76ef2e70b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f76efcb6e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f76ef23d0f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f76eca784e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f76eca63c50> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f76efcb6e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f76ef1fb710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f76eca784e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f76ef2e7128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f089159b208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=78f567c72f51ea3da32d648a0925684f822962f9efd287bcf75e5368cdd4e2d8
  Stored in directory: /tmp/pip-ephem-wheel-cache-xc4b9orf/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f0829396748> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2113536/17464789 [==>...........................] - ETA: 0s
10633216/17464789 [=================>............] - ETA: 0s
16900096/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 07:14:46.256869: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 07:14:46.260633: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095090000 Hz
2020-05-14 07:14:46.260784: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e032cd5310 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 07:14:46.260797: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7433 - accuracy: 0.4950
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7510 - accuracy: 0.4945 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7586 - accuracy: 0.4940
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8276 - accuracy: 0.4895
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.8200 - accuracy: 0.4900
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7458 - accuracy: 0.4948
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6995 - accuracy: 0.4979
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6839 - accuracy: 0.4989
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6785 - accuracy: 0.4992
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6712 - accuracy: 0.4997
11000/25000 [============>.................] - ETA: 3s - loss: 7.7001 - accuracy: 0.4978
12000/25000 [=============>................] - ETA: 3s - loss: 7.7292 - accuracy: 0.4959
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7256 - accuracy: 0.4962
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7061 - accuracy: 0.4974
15000/25000 [=================>............] - ETA: 2s - loss: 7.6963 - accuracy: 0.4981
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7069 - accuracy: 0.4974
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6991 - accuracy: 0.4979
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6896 - accuracy: 0.4985
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6884 - accuracy: 0.4986
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6958 - accuracy: 0.4981
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6893 - accuracy: 0.4985
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6638 - accuracy: 0.5002
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6560 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6570 - accuracy: 0.5006
25000/25000 [==============================] - 7s 285us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 07:15:00.000995
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 07:15:00.000995  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<42:45:57, 5.60kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<30:10:09, 7.94kB/s].vector_cache/glove.6B.zip:   0%|          | 205k/862M [00:01<21:10:39, 11.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 844k/862M [00:01<14:49:42, 16.1kB/s].vector_cache/glove.6B.zip:   0%|          | 3.38M/862M [00:02<10:21:11, 23.0kB/s].vector_cache/glove.6B.zip:   1%|          | 7.16M/862M [00:02<7:13:01, 32.9kB/s] .vector_cache/glove.6B.zip:   1%|          | 10.7M/862M [00:02<5:02:00, 47.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.7M/862M [00:02<3:30:29, 67.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.0M/862M [00:02<2:26:41, 95.8kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.0M/862M [00:03<1:43:22, 135kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.2M/862M [00:03<1:12:08, 193kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.1M/862M [00:03<50:16, 275kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 34.7M/862M [00:03<35:09, 392kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.0M/862M [00:03<24:35, 558kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.6M/862M [00:03<17:12, 793kB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.6M/862M [00:03<12:05, 1.12MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:03<08:31, 1.59MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:06<14:58, 902kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:06<18:52, 715kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.8M/862M [00:06<16:19, 826kB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.0M/862M [00:06<13:19, 1.01MB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.6M/862M [00:06<10:03, 1.34MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.7M/862M [00:06<07:23, 1.82MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:06<05:28, 2.45MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:08<11:15, 1.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:08<09:22, 1.43MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:08<07:10, 1.87MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.2M/862M [00:08<05:17, 2.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:08<04:01, 3.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:10<16:17, 820kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:10<14:04, 948kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:10<10:47, 1.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:10<08:06, 1.65MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.5M/862M [00:10<05:59, 2.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:10<04:30, 2.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:12<30:45, 432kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:12<24:16, 547kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:12<17:53, 742kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.8M/862M [00:12<12:51, 1.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:12<09:14, 1.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:14<14:12, 930kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:14<12:16, 1.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:14<09:12, 1.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:14<06:46, 1.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:14<05:04, 2.59MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:16<10:41, 1.23MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:16<10:10, 1.29MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:16<07:48, 1.68MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.2M/862M [00:16<05:48, 2.26MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:16<04:16, 3.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:18<19:55, 657kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:18<15:47, 828kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:18<11:27, 1.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:18<08:15, 1.58MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:20<10:51, 1.20MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:20<08:59, 1.45MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:20<06:34, 1.97MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:20<04:59, 2.60MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:20<03:45, 3.45MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:22<12:18:43, 17.5kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:22<8:39:32, 24.9kB/s] .vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:22<6:03:57, 35.5kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:22<4:14:27, 50.7kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:24<3:02:04, 70.7kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:24<2:09:12, 99.6kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:24<1:30:42, 142kB/s] .vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:24<1:03:36, 202kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:26<50:22, 254kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:26<37:02, 346kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:26<26:17, 486kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.2M/862M [00:26<18:47, 679kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:26<13:27, 947kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:28<18:20, 695kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:28<15:43, 810kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:28<11:41, 1.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:28<08:34, 1.48MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<06:09, 2.06MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<27:30, 460kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:30<20:50, 607kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:30<14:51, 851kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:30<10:45, 1.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<11:40, 1.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:32<09:40, 1.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:32<13:28, 934kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:32<09:46, 1.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<06:59, 1.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:34<12:23, 1.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:34<09:07, 1.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<06:36, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:36<08:44, 1.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:36<08:24, 1.48MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:36<06:19, 1.96MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:36<04:44, 2.62MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:38<06:32, 1.89MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:38<05:47, 2.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:38<04:29, 2.75MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<03:22, 3.66MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<06:42, 1.84MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:40<06:04, 2.02MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:40<04:34, 2.69MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:42<05:50, 2.10MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:42<06:37, 1.85MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:42<05:15, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:44<05:38, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:44<05:12, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:44<03:57, 3.07MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:45<05:35, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:46<06:24, 1.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:46<05:01, 2.41MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<03:41, 3.26MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:47<07:20, 1.64MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:48<06:23, 1.88MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:48<04:42, 2.55MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:48<03:27, 3.46MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:49<53:11, 225kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:50<38:27, 311kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:50<27:10, 439kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:50<19:03, 624kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:52<1:02:14, 191kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:52<48:25, 245kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:52<35:05, 339kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:52<24:45, 479kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:54<19:56, 593kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:54<15:09, 779kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<10:53, 1.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:56<10:21, 1.13MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:56<09:39, 1.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:56<07:16, 1.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:56<05:23, 2.17MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:58<06:35, 1.77MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:58<05:48, 2.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:58<04:21, 2.67MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:00<05:26, 2.13MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:00<08:43, 1.33MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:00<07:18, 1.59MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:00<05:22, 2.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:02<06:22, 1.81MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:02<05:40, 2.03MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<04:16, 2.69MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<03:07, 3.67MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:04<44:43, 256kB/s] .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:04<33:39, 340kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<24:07, 474kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:06<18:39, 610kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:06<14:15, 799kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<10:14, 1.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:08<10:16, 1.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:08<11:52, 953kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:09<09:28, 1.19MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<06:54, 1.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:10<07:23, 1.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:10<06:19, 1.78MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<04:42, 2.38MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:12<05:54, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<05:16, 2.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<03:58, 2.81MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:14<05:24, 2.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:14<06:03, 1.83MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<04:43, 2.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<03:27, 3.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:16<07:14, 1.53MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:16<06:10, 1.79MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<04:33, 2.42MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<05:48, 1.89MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<06:19, 1.74MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:18<04:53, 2.24MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:18<03:34, 3.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:20<07:29, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:20<06:22, 1.71MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:20<04:43, 2.30MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:22<05:50, 1.85MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:22<05:11, 2.09MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<03:53, 2.77MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:24<05:12, 2.07MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:24<05:55, 1.82MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<04:39, 2.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:24<03:21, 3.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:26<28:27, 376kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<21:01, 508kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<14:58, 713kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:28<12:55, 822kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<10:07, 1.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<07:19, 1.45MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:30<07:34, 1.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<06:23, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<04:43, 2.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:32<05:46, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:32<05:07, 2.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<03:50, 2.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:34<05:08, 2.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:34<04:39, 2.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<03:28, 2.99MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:36<04:54, 2.11MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:36<05:33, 1.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<04:19, 2.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:36<03:08, 3.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:38<12:09, 847kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:38<09:32, 1.08MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<06:55, 1.48MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:40<07:13, 1.41MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:40<06:06, 1.67MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:40<04:31, 2.25MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:42<05:32, 1.83MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:42<05:56, 1.71MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:42<04:40, 2.17MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:44<04:55, 2.05MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:44<04:30, 2.23MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:44<03:24, 2.95MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:46<04:42, 2.13MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:46<04:18, 2.32MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:46<03:15, 3.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:48<04:38, 2.14MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:48<06:21, 1.57MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:48<06:29, 1.53MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:48<05:27, 1.82MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<04:09, 2.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:48<03:02, 3.24MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<09:25, 1.05MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<07:50, 1.26MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<05:46, 1.71MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:52<05:59, 1.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:52<05:00, 1.96MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:52<04:02, 2.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:52<02:54, 3.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:54<20:29, 475kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:54<16:13, 600kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:54<12:31, 778kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:54<09:06, 1.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:54<06:28, 1.50MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:55<13:07, 737kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:56<10:37, 910kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:56<07:57, 1.21MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:56<05:48, 1.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:57<06:14, 1.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:58<05:38, 1.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:58<04:39, 2.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:58<03:30, 2.72MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:58<02:36, 3.67MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:59<13:20, 714kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:00<10:35, 900kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:00<07:59, 1.19MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:00<05:49, 1.63MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:01<06:10, 1.53MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<05:23, 1.76MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<04:16, 2.21MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:02<03:11, 2.96MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:03<04:41, 2.00MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:03<04:47, 1.96MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:04<03:56, 2.38MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:04<03:06, 3.02MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:04<02:15, 4.13MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:06<1:22:17, 113kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:06<1:00:51, 153kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:06<43:21, 215kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:06<30:25, 305kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:08<23:45, 389kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:08<17:47, 519kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:08<12:39, 728kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<10:41, 859kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<08:17, 1.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 313M/862M [02:10<06:19, 1.45MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:10<04:37, 1.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:10<03:25, 2.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:12<11:46, 773kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:12<09:56, 916kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:12<07:50, 1.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:12<05:45, 1.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:14<05:41, 1.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:14<04:55, 1.83MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:14<03:46, 2.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<02:43, 3.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:16<1:00:53, 147kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:16<43:22, 207kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<30:51, 290kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:16<21:42, 411kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:18<17:56, 496kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:18<13:26, 662kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<09:52, 900kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:18<07:03, 1.26MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:20<07:45, 1.14MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<06:16, 1.40MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<12:25, 710kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:20<08:54, 988kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<07:54, 1.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<06:12, 1.41MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:22<04:47, 1.82MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<04:47, 1.81MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<05:16, 1.65MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<04:31, 1.92MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:24<03:23, 2.56MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:26<04:04, 2.12MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:26<03:44, 2.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:26<02:49, 3.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:28<03:58, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:28<03:39, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:28<03:00, 2.84MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<02:11, 3.87MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:30<07:22, 1.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<05:57, 1.42MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<04:37, 1.83MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<03:18, 2.54MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:32<09:30, 885kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:32<08:25, 998kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:32<06:16, 1.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:32<04:31, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:33<05:56, 1.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<05:01, 1.66MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:34<03:43, 2.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:35<04:31, 1.83MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<04:50, 1.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:36<03:44, 2.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:36<02:50, 2.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:37<03:52, 2.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<03:35, 2.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:38<02:43, 3.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:39<03:46, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:39<04:20, 1.87MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:40<03:27, 2.34MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:40<02:30, 3.22MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:41<07:09, 1.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:41<05:51, 1.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:42<04:17, 1.87MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:43<04:50, 1.65MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<04:12, 1.90MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:44<03:05, 2.58MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:44<02:16, 3.49MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:46<1:47:35, 73.7kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:46<1:23:07, 95.4kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:46<59:50, 132kB/s]   .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<42:11, 188kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:46<29:34, 267kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:48<23:07, 340kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:48<16:58, 463kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:48<12:02, 650kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:50<10:34, 737kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<10:34, 736kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<08:11, 950kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:51<05:54, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:52<05:52, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:52<04:55, 1.57MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:52<03:38, 2.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:54<04:36, 1.66MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:54<04:51, 1.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:55<03:48, 2.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:55<02:46, 2.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:57<05:34, 1.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:57<05:39, 1.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:57<04:18, 1.76MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:57<03:15, 2.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:57<02:22, 3.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:59<13:09, 571kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:59<10:11, 738kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:59<07:18, 1.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:59<05:13, 1.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:01<07:57, 936kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:01<06:30, 1.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:01<04:47, 1.55MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:01<03:24, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:03<09:47, 753kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:03<08:41, 848kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:03<06:32, 1.13MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:03<04:48, 1.52MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:05<04:45, 1.53MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:05<04:14, 1.72MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:05<03:18, 2.20MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:05<02:27, 2.95MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:07<03:44, 1.93MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:07<03:15, 2.22MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:07<02:41, 2.68MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:07<02:01, 3.54MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:09<03:26, 2.08MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:09<03:09, 2.27MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:09<02:21, 3.02MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<01:47, 3.98MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:11<06:28, 1.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:12<10:18, 687kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:12<08:59, 789kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:12<06:37, 1.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:12<04:44, 1.49MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:13<05:18, 1.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:14<04:28, 1.57MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:14<03:17, 2.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:15<03:50, 1.81MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:16<03:23, 2.05MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:16<02:31, 2.74MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:17<03:23, 2.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:18<03:47, 1.82MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:18<02:57, 2.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:18<02:07, 3.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:19<07:40, 888kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:20<06:04, 1.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:20<04:23, 1.55MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:21<04:38, 1.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:21<04:37, 1.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:22<03:34, 1.88MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:22<02:34, 2.60MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:23<1:19:39, 83.8kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:23<56:23, 118kB/s]   .vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:24<39:29, 168kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:25<29:03, 227kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:25<20:59, 315kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:26<14:48, 444kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:27<11:51, 552kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:27<08:58, 729kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:28<06:25, 1.01MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:29<05:59, 1.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:29<05:31, 1.17MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:29<04:11, 1.54MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:30<02:59, 2.14MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:31<48:50, 131kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:31<34:48, 184kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:31<24:25, 261kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:33<18:30, 342kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:33<13:35, 466kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:33<09:36, 655kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:35<08:11, 765kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:35<06:21, 983kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:35<04:35, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:37<04:40, 1.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:37<03:53, 1.59MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:37<02:51, 2.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:39<03:26, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:39<03:02, 2.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:39<02:15, 2.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:41<03:00, 2.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:41<02:43, 2.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:41<02:03, 2.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:43<02:50, 2.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:43<02:36, 2.29MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:43<01:58, 3.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:45<02:46, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:45<03:13, 1.84MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:45<02:36, 2.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:45<02:06, 2.80MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:45<01:30, 3.86MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:48<06:21, 917kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:48<09:14, 630kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:48<07:26, 782kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:48<05:26, 1.07MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:48<03:57, 1.46MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:50<04:02, 1.42MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:50<03:26, 1.67MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:50<02:32, 2.26MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:52<03:02, 1.87MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:52<03:18, 1.72MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:52<02:35, 2.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:54<02:42, 2.07MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:54<02:28, 2.26MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:54<01:52, 2.97MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:56<02:45, 2.01MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:56<03:11, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:56<02:29, 2.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:56<01:49, 3.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:58<03:12, 1.71MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:58<02:50, 1.93MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:58<02:06, 2.58MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:00<02:39, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:00<03:01, 1.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [04:00<02:25, 2.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:00<01:45, 3.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:02<03:29, 1.53MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:02<02:54, 1.83MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:02<02:09, 2.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:02<01:34, 3.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:04<20:58, 251kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:04<15:12, 346kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:04<10:42, 489kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:06<08:39, 601kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:06<07:07, 729kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:06<05:14, 989kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:06<03:41, 1.39MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:08<08:48, 583kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:08<06:42, 765kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:08<04:46, 1.07MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:08<03:23, 1.49MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:10<22:28, 225kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:10<16:13, 312kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:10<11:25, 441kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:10<07:59, 625kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:12<42:39, 117kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:12<30:20, 164kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:12<21:14, 234kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:14<15:55, 309kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:14<11:38, 422kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:14<08:14, 594kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:16<06:52, 707kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:16<05:49, 834kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:16<04:17, 1.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:16<03:01, 1.58MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:18<09:08, 524kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:18<06:53, 693kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:18<04:54, 968kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:20<04:31, 1.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:20<04:07, 1.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:20<03:05, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:20<02:16, 2.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:22<02:46, 1.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:22<02:27, 1.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:22<02:02, 2.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:22<01:31, 3.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:24<02:13, 2.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:24<02:11, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:24<01:42, 2.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:24<01:15, 3.59MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:26<02:45, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:26<02:31, 1.79MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:26<01:53, 2.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:26<01:23, 3.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:28<02:49, 1.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:28<02:33, 1.73MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:28<02:01, 2.19MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:28<01:31, 2.90MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:29<02:04, 2.10MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:30<01:57, 2.24MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:30<01:37, 2.69MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:30<01:11, 3.65MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:31<02:33, 1.68MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:32<02:28, 1.73MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:32<01:59, 2.15MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:32<01:28, 2.91MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:34<02:18, 1.84MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:34<02:40, 1.59MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:34<02:07, 1.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:34<01:31, 2.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:36<02:53, 1.44MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:36<02:28, 1.68MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:36<01:52, 2.20MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:36<01:25, 2.91MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:36<01:06, 3.73MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:38<03:36, 1.14MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:38<03:55, 1.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 616M/862M [04:38<03:43, 1.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:38<03:15, 1.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:38<02:27, 1.66MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:38<01:46, 2.28MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:40<02:37, 1.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:40<02:13, 1.81MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:40<01:53, 2.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:40<01:29, 2.68MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:40<01:07, 3.52MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:42<02:08, 1.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:42<02:11, 1.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:42<01:46, 2.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:42<01:20, 2.93MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:42<01:00, 3.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:43<03:45, 1.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:44<03:19, 1.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:44<02:47, 1.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:44<02:08, 1.81MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:44<01:33, 2.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:45<02:11, 1.75MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:46<02:12, 1.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:46<02:05, 1.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:46<01:44, 2.19MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:46<01:18, 2.88MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:46<01:00, 3.73MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:47<02:24, 1.56MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:48<02:18, 1.63MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:48<01:51, 2.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:48<01:24, 2.64MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:48<01:03, 3.52MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:49<02:44, 1.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:50<02:32, 1.45MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:50<01:59, 1.85MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:50<01:29, 2.45MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:50<01:06, 3.26MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:51<02:53, 1.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:51<02:46, 1.30MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:52<02:36, 1.38MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:52<02:09, 1.67MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:52<01:40, 2.14MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:52<01:14, 2.89MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:53<02:08, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:54<02:03, 1.71MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:54<01:39, 2.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:54<01:16, 2.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:54<00:59, 3.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:54<00:47, 4.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:55<03:45, 927kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:55<03:30, 992kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:56<02:38, 1.31MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:56<01:57, 1.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:56<01:25, 2.41MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:57<03:00, 1.13MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:57<02:35, 1.31MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:58<01:54, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:58<01:23, 2.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:59<02:26, 1.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:59<02:12, 1.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:00<01:39, 2.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:00<01:11, 2.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:01<05:35, 584kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:01<04:22, 745kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:02<03:08, 1.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:02<02:15, 1.43MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:03<02:46, 1.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:03<02:23, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:04<01:46, 1.78MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:05<01:49, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:05<01:41, 1.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:06<01:16, 2.44MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:06<00:55, 3.31MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:07<02:31, 1.21MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:07<02:06, 1.45MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:08<01:34, 1.94MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:08<01:07, 2.66MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:09<04:02, 740kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:09<03:12, 930kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:09<02:18, 1.28MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:10<01:38, 1.78MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:11<03:44, 780kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:11<03:20, 874kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:12<02:29, 1.17MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:12<01:47, 1.61MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:13<01:54, 1.49MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:13<01:37, 1.74MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:13<01:12, 2.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:15<01:28, 1.88MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:15<01:19, 2.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:15<00:59, 2.79MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:17<01:19, 2.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:17<01:12, 2.25MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:17<00:53, 3.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:19<01:14, 2.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:19<01:24, 1.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:19<01:06, 2.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:19<00:48, 3.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:21<01:31, 1.68MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:21<01:20, 1.92MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:21<00:59, 2.57MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:23<01:16, 1.97MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:23<01:08, 2.18MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:23<00:51, 2.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:25<01:10, 2.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:25<01:19, 1.83MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:25<01:03, 2.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:25<00:44, 3.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:27<04:50, 491kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:27<03:34, 660kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:27<02:33, 920kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:27<01:47, 1.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:29<06:48, 338kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:29<04:59, 461kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:29<03:30, 647kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:31<02:56, 758kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:31<02:31, 886kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:31<01:51, 1.20MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:31<01:18, 1.67MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:33<01:48, 1.20MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:33<01:29, 1.45MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:33<01:04, 1.98MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:35<01:13, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:35<01:04, 1.95MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:35<00:47, 2.62MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:37<01:01, 1.99MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:37<01:07, 1.79MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:37<00:53, 2.28MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:37<00:37, 3.13MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:39<01:48, 1.08MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:39<01:25, 1.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:39<01:01, 1.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:39<00:44, 2.56MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:41<02:05, 908kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:41<01:39, 1.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:41<01:10, 1.57MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:43<01:14, 1.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:43<01:03, 1.71MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:43<00:46, 2.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:45<00:57, 1.82MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:45<00:52, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:45<00:39, 2.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:47<00:48, 2.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:47<00:55, 1.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:47<00:44, 2.26MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:47<00:31, 3.11MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:49<04:13, 383kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:49<03:07, 516kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:49<02:13, 718kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:49<01:34, 1.00MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:51<01:35, 977kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:51<01:22, 1.13MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:51<01:04, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:51<00:46, 1.96MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:53<00:52, 1.70MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:53<00:47, 1.87MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:53<00:37, 2.36MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:53<00:27, 3.18MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:55<00:46, 1.81MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:55<00:41, 2.04MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:55<00:30, 2.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:55<00:21, 3.71MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:57<04:36, 291kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:57<03:46, 355kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:57<02:46, 481kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:57<01:55, 677kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:59<01:36, 794kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:59<01:14, 1.02MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:59<00:52, 1.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:01<00:52, 1.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:01<00:51, 1.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:01<00:38, 1.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:01<00:26, 2.53MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:03<1:05:30, 17.3kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:03<45:43, 24.7kB/s]  .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:03<31:18, 35.2kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:05<21:26, 49.7kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:05<15:10, 70.0kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:05<10:32, 99.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:05<07:07, 142kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:07<05:30, 181kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:07<03:56, 252kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:07<02:42, 356kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:09<02:02, 455kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:09<01:36, 573kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:09<01:09, 785kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:11<00:54, 949kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:11<00:43, 1.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:11<00:30, 1.62MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:13<00:35, 1.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:13<01:04, 741kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:14<00:54, 861kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:14<00:40, 1.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:14<00:27, 1.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:15<00:35, 1.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:15<00:29, 1.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:16<00:20, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:17<00:22, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:17<00:23, 1.63MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:17<00:18, 2.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:18<00:12, 2.89MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:19<01:27, 402kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:19<01:04, 542kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:19<00:43, 760kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:21<00:35, 866kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:21<00:27, 1.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:21<00:19, 1.51MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:23<00:18, 1.43MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:23<00:18, 1.44MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:23<00:13, 1.86MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:25<00:13, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:26<00:18, 1.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:26<00:15, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:26<00:10, 1.99MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:27<00:10, 1.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:28<00:09, 1.98MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:28<00:06, 2.65MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:30<00:08, 1.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:30<00:18, 804kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:30<00:15, 926kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:30<00:11, 1.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:31<00:06, 1.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:32<00:09, 1.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:32<00:07, 1.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:32<00:05, 1.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:32<00:03, 2.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:34<00:03, 1.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:34<00:03, 1.99MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:34<00:02, 2.38MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:34<00:01, 3.22MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:36<00:01, 1.90MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:36<00:00, 2.06MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:36<00:00, 2.72MB/s].vector_cache/glove.6B.zip: 862MB [06:36, 2.17MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 858/400000 [00:00<00:46, 8574.22it/s]  0%|          | 1729/400000 [00:00<00:46, 8613.65it/s]  1%|          | 2602/400000 [00:00<00:45, 8648.23it/s]  1%|          | 3406/400000 [00:00<00:46, 8455.11it/s]  1%|          | 4220/400000 [00:00<00:47, 8354.92it/s]  1%|▏         | 5075/400000 [00:00<00:46, 8410.87it/s]  1%|▏         | 5918/400000 [00:00<00:46, 8416.27it/s]  2%|▏         | 6776/400000 [00:00<00:46, 8463.69it/s]  2%|▏         | 7626/400000 [00:00<00:46, 8473.73it/s]  2%|▏         | 8481/400000 [00:01<00:46, 8494.13it/s]  2%|▏         | 9307/400000 [00:01<00:46, 8395.20it/s]  3%|▎         | 10186/400000 [00:01<00:45, 8507.94it/s]  3%|▎         | 11063/400000 [00:01<00:45, 8583.96it/s]  3%|▎         | 11914/400000 [00:01<00:45, 8557.85it/s]  3%|▎         | 12765/400000 [00:01<00:46, 8305.44it/s]  3%|▎         | 13595/400000 [00:01<00:46, 8302.86it/s]  4%|▎         | 14456/400000 [00:01<00:45, 8390.97it/s]  4%|▍         | 15305/400000 [00:01<00:45, 8417.87it/s]  4%|▍         | 16147/400000 [00:01<00:46, 8300.40it/s]  4%|▍         | 17021/400000 [00:02<00:45, 8426.08it/s]  4%|▍         | 17897/400000 [00:02<00:44, 8521.13it/s]  5%|▍         | 18755/400000 [00:02<00:44, 8535.94it/s]  5%|▍         | 19612/400000 [00:02<00:44, 8544.82it/s]  5%|▌         | 20468/400000 [00:02<00:44, 8543.38it/s]  5%|▌         | 21323/400000 [00:02<00:44, 8540.49it/s]  6%|▌         | 22190/400000 [00:02<00:44, 8577.01it/s]  6%|▌         | 23048/400000 [00:02<00:44, 8537.92it/s]  6%|▌         | 23915/400000 [00:02<00:43, 8576.79it/s]  6%|▌         | 24781/400000 [00:02<00:43, 8598.97it/s]  6%|▋         | 25647/400000 [00:03<00:43, 8615.45it/s]  7%|▋         | 26528/400000 [00:03<00:43, 8668.20it/s]  7%|▋         | 27395/400000 [00:03<00:42, 8666.03it/s]  7%|▋         | 28262/400000 [00:03<00:43, 8636.08it/s]  7%|▋         | 29126/400000 [00:03<00:43, 8593.88it/s]  7%|▋         | 29986/400000 [00:03<00:43, 8587.27it/s]  8%|▊         | 30855/400000 [00:03<00:42, 8615.80it/s]  8%|▊         | 31717/400000 [00:03<00:42, 8568.80it/s]  8%|▊         | 32574/400000 [00:03<00:42, 8559.41it/s]  8%|▊         | 33432/400000 [00:03<00:42, 8565.09it/s]  9%|▊         | 34310/400000 [00:04<00:42, 8626.79it/s]  9%|▉         | 35173/400000 [00:04<00:42, 8489.91it/s]  9%|▉         | 36046/400000 [00:04<00:42, 8560.47it/s]  9%|▉         | 36916/400000 [00:04<00:42, 8601.76it/s]  9%|▉         | 37794/400000 [00:04<00:41, 8653.99it/s] 10%|▉         | 38660/400000 [00:04<00:41, 8654.41it/s] 10%|▉         | 39526/400000 [00:04<00:41, 8626.29it/s] 10%|█         | 40404/400000 [00:04<00:41, 8669.32it/s] 10%|█         | 41276/400000 [00:04<00:41, 8682.75it/s] 11%|█         | 42145/400000 [00:04<00:42, 8427.94it/s] 11%|█         | 42990/400000 [00:05<00:42, 8315.74it/s] 11%|█         | 43824/400000 [00:05<00:43, 8140.08it/s] 11%|█         | 44640/400000 [00:05<00:44, 8011.12it/s] 11%|█▏        | 45492/400000 [00:05<00:43, 8156.63it/s] 12%|█▏        | 46310/400000 [00:05<00:43, 8116.75it/s] 12%|█▏        | 47124/400000 [00:05<00:44, 7963.61it/s] 12%|█▏        | 47971/400000 [00:05<00:43, 8107.87it/s] 12%|█▏        | 48815/400000 [00:05<00:42, 8203.27it/s] 12%|█▏        | 49684/400000 [00:05<00:41, 8341.31it/s] 13%|█▎        | 50548/400000 [00:05<00:41, 8427.90it/s] 13%|█▎        | 51416/400000 [00:06<00:41, 8501.81it/s] 13%|█▎        | 52293/400000 [00:06<00:40, 8579.71it/s] 13%|█▎        | 53181/400000 [00:06<00:40, 8666.21it/s] 14%|█▎        | 54049/400000 [00:06<00:41, 8325.63it/s] 14%|█▎        | 54906/400000 [00:06<00:41, 8397.07it/s] 14%|█▍        | 55779/400000 [00:06<00:40, 8493.43it/s] 14%|█▍        | 56631/400000 [00:06<00:40, 8455.67it/s] 14%|█▍        | 57496/400000 [00:06<00:40, 8510.45it/s] 15%|█▍        | 58363/400000 [00:06<00:39, 8555.45it/s] 15%|█▍        | 59220/400000 [00:06<00:40, 8451.01it/s] 15%|█▌        | 60092/400000 [00:07<00:39, 8529.33it/s] 15%|█▌        | 60946/400000 [00:07<00:39, 8529.22it/s] 15%|█▌        | 61800/400000 [00:07<00:40, 8376.08it/s] 16%|█▌        | 62668/400000 [00:07<00:39, 8463.15it/s] 16%|█▌        | 63516/400000 [00:07<00:41, 8086.33it/s] 16%|█▌        | 64329/400000 [00:07<00:41, 8010.63it/s] 16%|█▋        | 65193/400000 [00:07<00:40, 8187.63it/s] 17%|█▋        | 66045/400000 [00:07<00:40, 8281.26it/s] 17%|█▋        | 66904/400000 [00:07<00:39, 8370.47it/s] 17%|█▋        | 67743/400000 [00:08<00:41, 8097.24it/s] 17%|█▋        | 68585/400000 [00:08<00:40, 8190.84it/s] 17%|█▋        | 69469/400000 [00:08<00:39, 8374.62it/s] 18%|█▊        | 70341/400000 [00:08<00:38, 8473.98it/s] 18%|█▊        | 71222/400000 [00:08<00:38, 8571.57it/s] 18%|█▊        | 72100/400000 [00:08<00:37, 8632.55it/s] 18%|█▊        | 72965/400000 [00:08<00:38, 8528.73it/s] 18%|█▊        | 73820/400000 [00:08<00:38, 8477.00it/s] 19%|█▊        | 74683/400000 [00:08<00:38, 8519.66it/s] 19%|█▉        | 75536/400000 [00:08<00:39, 8235.47it/s] 19%|█▉        | 76377/400000 [00:09<00:39, 8286.95it/s] 19%|█▉        | 77223/400000 [00:09<00:38, 8335.81it/s] 20%|█▉        | 78092/400000 [00:09<00:38, 8436.37it/s] 20%|█▉        | 78971/400000 [00:09<00:37, 8538.82it/s] 20%|█▉        | 79837/400000 [00:09<00:37, 8573.66it/s] 20%|██        | 80706/400000 [00:09<00:37, 8606.96it/s] 20%|██        | 81568/400000 [00:09<00:37, 8593.93it/s] 21%|██        | 82433/400000 [00:09<00:36, 8607.77it/s] 21%|██        | 83299/400000 [00:09<00:36, 8620.55it/s] 21%|██        | 84164/400000 [00:09<00:36, 8626.70it/s] 21%|██▏       | 85036/400000 [00:10<00:36, 8653.99it/s] 21%|██▏       | 85902/400000 [00:10<00:36, 8634.14it/s] 22%|██▏       | 86766/400000 [00:10<00:36, 8633.31it/s] 22%|██▏       | 87630/400000 [00:10<00:36, 8605.09it/s] 22%|██▏       | 88491/400000 [00:10<00:36, 8581.45it/s] 22%|██▏       | 89370/400000 [00:10<00:35, 8641.25it/s] 23%|██▎       | 90235/400000 [00:10<00:36, 8588.83it/s] 23%|██▎       | 91095/400000 [00:10<00:35, 8586.71it/s] 23%|██▎       | 91963/400000 [00:10<00:35, 8613.21it/s] 23%|██▎       | 92828/400000 [00:10<00:35, 8621.88it/s] 23%|██▎       | 93698/400000 [00:11<00:35, 8642.28it/s] 24%|██▎       | 94571/400000 [00:11<00:35, 8665.77it/s] 24%|██▍       | 95438/400000 [00:11<00:35, 8652.47it/s] 24%|██▍       | 96304/400000 [00:11<00:35, 8587.27it/s] 24%|██▍       | 97173/400000 [00:11<00:35, 8615.07it/s] 25%|██▍       | 98035/400000 [00:11<00:35, 8522.03it/s] 25%|██▍       | 98888/400000 [00:11<00:35, 8373.53it/s] 25%|██▍       | 99727/400000 [00:11<00:36, 8276.17it/s] 25%|██▌       | 100566/400000 [00:11<00:36, 8308.03it/s] 25%|██▌       | 101432/400000 [00:11<00:35, 8408.42it/s] 26%|██▌       | 102301/400000 [00:12<00:35, 8490.55it/s] 26%|██▌       | 103160/400000 [00:12<00:34, 8518.20it/s] 26%|██▌       | 104013/400000 [00:12<00:34, 8509.15it/s] 26%|██▌       | 104874/400000 [00:12<00:34, 8538.37it/s] 26%|██▋       | 105742/400000 [00:12<00:34, 8578.66it/s] 27%|██▋       | 106601/400000 [00:12<00:34, 8494.51it/s] 27%|██▋       | 107464/400000 [00:12<00:34, 8532.17it/s] 27%|██▋       | 108348/400000 [00:12<00:33, 8620.86it/s] 27%|██▋       | 109218/400000 [00:12<00:33, 8643.91it/s] 28%|██▊       | 110083/400000 [00:12<00:33, 8633.65it/s] 28%|██▊       | 110947/400000 [00:13<00:34, 8483.92it/s] 28%|██▊       | 111799/400000 [00:13<00:33, 8493.73it/s] 28%|██▊       | 112649/400000 [00:13<00:33, 8471.83it/s] 28%|██▊       | 113517/400000 [00:13<00:33, 8532.03it/s] 29%|██▊       | 114371/400000 [00:13<00:34, 8384.21it/s] 29%|██▉       | 115224/400000 [00:13<00:33, 8425.21it/s] 29%|██▉       | 116085/400000 [00:13<00:33, 8479.50it/s] 29%|██▉       | 116964/400000 [00:13<00:33, 8569.52it/s] 29%|██▉       | 117855/400000 [00:13<00:32, 8666.10it/s] 30%|██▉       | 118723/400000 [00:13<00:32, 8670.10it/s] 30%|██▉       | 119597/400000 [00:14<00:32, 8689.46it/s] 30%|███       | 120467/400000 [00:14<00:32, 8687.16it/s] 30%|███       | 121345/400000 [00:14<00:31, 8714.19it/s] 31%|███       | 122217/400000 [00:14<00:31, 8706.96it/s] 31%|███       | 123093/400000 [00:14<00:31, 8722.25it/s] 31%|███       | 123966/400000 [00:14<00:31, 8670.86it/s] 31%|███       | 124834/400000 [00:14<00:31, 8656.23it/s] 31%|███▏      | 125700/400000 [00:14<00:31, 8590.73it/s] 32%|███▏      | 126560/400000 [00:14<00:31, 8582.61it/s] 32%|███▏      | 127419/400000 [00:14<00:31, 8561.78it/s] 32%|███▏      | 128288/400000 [00:15<00:31, 8597.65it/s] 32%|███▏      | 129148/400000 [00:15<00:31, 8589.18it/s] 33%|███▎      | 130007/400000 [00:15<00:31, 8559.41it/s] 33%|███▎      | 130864/400000 [00:15<00:32, 8304.31it/s] 33%|███▎      | 131702/400000 [00:15<00:32, 8325.27it/s] 33%|███▎      | 132554/400000 [00:15<00:31, 8381.81it/s] 33%|███▎      | 133422/400000 [00:15<00:31, 8467.74it/s] 34%|███▎      | 134299/400000 [00:15<00:31, 8555.37it/s] 34%|███▍      | 135168/400000 [00:15<00:30, 8594.69it/s] 34%|███▍      | 136032/400000 [00:16<00:30, 8607.78it/s] 34%|███▍      | 136894/400000 [00:16<00:30, 8581.08it/s] 34%|███▍      | 137774/400000 [00:16<00:30, 8644.45it/s] 35%|███▍      | 138660/400000 [00:16<00:30, 8707.61it/s] 35%|███▍      | 139542/400000 [00:16<00:29, 8740.67it/s] 35%|███▌      | 140431/400000 [00:16<00:29, 8783.30it/s] 35%|███▌      | 141310/400000 [00:16<00:31, 8322.08it/s] 36%|███▌      | 142148/400000 [00:16<00:31, 8295.22it/s] 36%|███▌      | 142982/400000 [00:16<00:31, 8280.66it/s] 36%|███▌      | 143813/400000 [00:16<00:31, 8097.69it/s] 36%|███▌      | 144685/400000 [00:17<00:30, 8273.49it/s] 36%|███▋      | 145538/400000 [00:17<00:30, 8348.13it/s] 37%|███▋      | 146409/400000 [00:17<00:30, 8451.72it/s] 37%|███▋      | 147300/400000 [00:17<00:29, 8581.42it/s] 37%|███▋      | 148160/400000 [00:17<00:29, 8583.48it/s] 37%|███▋      | 149020/400000 [00:17<00:29, 8456.35it/s] 37%|███▋      | 149887/400000 [00:17<00:29, 8519.23it/s] 38%|███▊      | 150745/400000 [00:17<00:29, 8534.78it/s] 38%|███▊      | 151628/400000 [00:17<00:28, 8620.88it/s] 38%|███▊      | 152507/400000 [00:17<00:28, 8670.11it/s] 38%|███▊      | 153375/400000 [00:18<00:28, 8567.22it/s] 39%|███▊      | 154245/400000 [00:18<00:28, 8604.52it/s] 39%|███▉      | 155127/400000 [00:18<00:28, 8667.06it/s] 39%|███▉      | 156017/400000 [00:18<00:27, 8732.97it/s] 39%|███▉      | 156892/400000 [00:18<00:27, 8737.48it/s] 39%|███▉      | 157779/400000 [00:18<00:27, 8775.86it/s] 40%|███▉      | 158666/400000 [00:18<00:27, 8802.50it/s] 40%|███▉      | 159547/400000 [00:18<00:27, 8802.72it/s] 40%|████      | 160434/400000 [00:18<00:27, 8820.44it/s] 40%|████      | 161317/400000 [00:18<00:27, 8802.07it/s] 41%|████      | 162200/400000 [00:19<00:26, 8809.82it/s] 41%|████      | 163082/400000 [00:19<00:27, 8750.01it/s] 41%|████      | 163962/400000 [00:19<00:26, 8764.54it/s] 41%|████      | 164839/400000 [00:19<00:26, 8760.01it/s] 41%|████▏     | 165716/400000 [00:19<00:26, 8716.54it/s] 42%|████▏     | 166588/400000 [00:19<00:26, 8694.24it/s] 42%|████▏     | 167458/400000 [00:19<00:27, 8537.61it/s] 42%|████▏     | 168332/400000 [00:19<00:26, 8595.33it/s] 42%|████▏     | 169205/400000 [00:19<00:26, 8634.53it/s] 43%|████▎     | 170083/400000 [00:19<00:26, 8675.07it/s] 43%|████▎     | 170951/400000 [00:20<00:26, 8640.18it/s] 43%|████▎     | 171829/400000 [00:20<00:26, 8680.91it/s] 43%|████▎     | 172717/400000 [00:20<00:26, 8737.41it/s] 43%|████▎     | 173591/400000 [00:20<00:25, 8733.49it/s] 44%|████▎     | 174470/400000 [00:20<00:25, 8749.59it/s] 44%|████▍     | 175346/400000 [00:20<00:25, 8658.09it/s] 44%|████▍     | 176213/400000 [00:20<00:26, 8598.90it/s] 44%|████▍     | 177074/400000 [00:20<00:26, 8391.55it/s] 44%|████▍     | 177946/400000 [00:20<00:26, 8485.77it/s] 45%|████▍     | 178812/400000 [00:20<00:25, 8535.30it/s] 45%|████▍     | 179679/400000 [00:21<00:25, 8573.93it/s] 45%|████▌     | 180544/400000 [00:21<00:25, 8596.55it/s] 45%|████▌     | 181405/400000 [00:21<00:25, 8596.92it/s] 46%|████▌     | 182280/400000 [00:21<00:25, 8641.20it/s] 46%|████▌     | 183145/400000 [00:21<00:25, 8627.09it/s] 46%|████▌     | 184016/400000 [00:21<00:24, 8650.43it/s] 46%|████▌     | 184882/400000 [00:21<00:24, 8641.26it/s] 46%|████▋     | 185747/400000 [00:21<00:24, 8634.40it/s] 47%|████▋     | 186611/400000 [00:21<00:25, 8496.98it/s] 47%|████▋     | 187479/400000 [00:21<00:24, 8543.01it/s] 47%|████▋     | 188334/400000 [00:22<00:25, 8421.51it/s] 47%|████▋     | 189177/400000 [00:22<00:25, 8348.69it/s] 48%|████▊     | 190013/400000 [00:22<00:25, 8173.50it/s] 48%|████▊     | 190832/400000 [00:22<00:25, 8144.87it/s] 48%|████▊     | 191687/400000 [00:22<00:25, 8260.01it/s] 48%|████▊     | 192544/400000 [00:22<00:24, 8348.35it/s] 48%|████▊     | 193380/400000 [00:22<00:25, 8241.32it/s] 49%|████▊     | 194206/400000 [00:22<00:25, 8198.66it/s] 49%|████▉     | 195027/400000 [00:22<00:25, 8193.30it/s] 49%|████▉     | 195847/400000 [00:23<00:25, 8108.97it/s] 49%|████▉     | 196692/400000 [00:23<00:24, 8206.15it/s] 49%|████▉     | 197564/400000 [00:23<00:24, 8352.53it/s] 50%|████▉     | 198404/400000 [00:23<00:24, 8363.90it/s] 50%|████▉     | 199286/400000 [00:23<00:23, 8493.15it/s] 50%|█████     | 200137/400000 [00:23<00:23, 8460.07it/s] 50%|█████     | 201021/400000 [00:23<00:23, 8569.84it/s] 50%|█████     | 201891/400000 [00:23<00:23, 8605.98it/s] 51%|█████     | 202753/400000 [00:23<00:23, 8522.52it/s] 51%|█████     | 203620/400000 [00:23<00:22, 8564.57it/s] 51%|█████     | 204477/400000 [00:24<00:23, 8494.59it/s] 51%|█████▏    | 205362/400000 [00:24<00:22, 8596.29it/s] 52%|█████▏    | 206253/400000 [00:24<00:22, 8686.29it/s] 52%|█████▏    | 207131/400000 [00:24<00:22, 8713.39it/s] 52%|█████▏    | 208012/400000 [00:24<00:21, 8740.85it/s] 52%|█████▏    | 208906/400000 [00:24<00:21, 8798.85it/s] 52%|█████▏    | 209787/400000 [00:24<00:21, 8765.15it/s] 53%|█████▎    | 210664/400000 [00:24<00:21, 8744.24it/s] 53%|█████▎    | 211539/400000 [00:24<00:21, 8690.55it/s] 53%|█████▎    | 212415/400000 [00:24<00:21, 8709.21it/s] 53%|█████▎    | 213294/400000 [00:25<00:21, 8731.85it/s] 54%|█████▎    | 214176/400000 [00:25<00:21, 8757.78it/s] 54%|█████▍    | 215058/400000 [00:25<00:21, 8773.75it/s] 54%|█████▍    | 215937/400000 [00:25<00:20, 8778.16it/s] 54%|█████▍    | 216824/400000 [00:25<00:20, 8803.79it/s] 54%|█████▍    | 217711/400000 [00:25<00:20, 8822.59it/s] 55%|█████▍    | 218594/400000 [00:25<00:20, 8817.40it/s] 55%|█████▍    | 219478/400000 [00:25<00:20, 8821.71it/s] 55%|█████▌    | 220364/400000 [00:25<00:20, 8831.90it/s] 55%|█████▌    | 221248/400000 [00:25<00:20, 8818.14it/s] 56%|█████▌    | 222146/400000 [00:26<00:20, 8863.13it/s] 56%|█████▌    | 223048/400000 [00:26<00:19, 8907.23it/s] 56%|█████▌    | 223947/400000 [00:26<00:19, 8931.60it/s] 56%|█████▌    | 224841/400000 [00:26<00:19, 8825.03it/s] 56%|█████▋    | 225724/400000 [00:26<00:19, 8779.42it/s] 57%|█████▋    | 226603/400000 [00:26<00:19, 8763.17it/s] 57%|█████▋    | 227480/400000 [00:26<00:19, 8664.92it/s] 57%|█████▋    | 228347/400000 [00:26<00:19, 8584.24it/s] 57%|█████▋    | 229220/400000 [00:26<00:19, 8624.73it/s] 58%|█████▊    | 230083/400000 [00:26<00:20, 8471.42it/s] 58%|█████▊    | 230932/400000 [00:27<00:20, 8395.42it/s] 58%|█████▊    | 231800/400000 [00:27<00:19, 8477.58it/s] 58%|█████▊    | 232687/400000 [00:27<00:19, 8590.74it/s] 58%|█████▊    | 233568/400000 [00:27<00:19, 8654.89it/s] 59%|█████▊    | 234435/400000 [00:27<00:19, 8532.27it/s] 59%|█████▉    | 235293/400000 [00:27<00:19, 8544.22it/s] 59%|█████▉    | 236175/400000 [00:27<00:19, 8621.98it/s] 59%|█████▉    | 237061/400000 [00:27<00:18, 8689.83it/s] 59%|█████▉    | 237943/400000 [00:27<00:18, 8726.70it/s] 60%|█████▉    | 238817/400000 [00:27<00:18, 8706.92it/s] 60%|█████▉    | 239689/400000 [00:28<00:18, 8704.96it/s] 60%|██████    | 240597/400000 [00:28<00:18, 8811.04it/s] 60%|██████    | 241479/400000 [00:28<00:18, 8743.85it/s] 61%|██████    | 242354/400000 [00:28<00:18, 8661.81it/s] 61%|██████    | 243221/400000 [00:28<00:18, 8470.46it/s] 61%|██████    | 244070/400000 [00:28<00:18, 8321.25it/s] 61%|██████    | 244955/400000 [00:28<00:18, 8472.69it/s] 61%|██████▏   | 245833/400000 [00:28<00:18, 8562.06it/s] 62%|██████▏   | 246721/400000 [00:28<00:17, 8653.20it/s] 62%|██████▏   | 247595/400000 [00:28<00:17, 8678.47it/s] 62%|██████▏   | 248464/400000 [00:29<00:17, 8593.09it/s] 62%|██████▏   | 249341/400000 [00:29<00:17, 8644.59it/s] 63%|██████▎   | 250229/400000 [00:29<00:17, 8711.37it/s] 63%|██████▎   | 251122/400000 [00:29<00:16, 8775.68it/s] 63%|██████▎   | 252001/400000 [00:29<00:17, 8615.86it/s] 63%|██████▎   | 252864/400000 [00:29<00:17, 8549.08it/s] 63%|██████▎   | 253728/400000 [00:29<00:17, 8575.48it/s] 64%|██████▎   | 254587/400000 [00:29<00:17, 8357.18it/s] 64%|██████▍   | 255425/400000 [00:29<00:17, 8322.81it/s] 64%|██████▍   | 256286/400000 [00:29<00:17, 8405.33it/s] 64%|██████▍   | 257128/400000 [00:30<00:17, 8235.74it/s] 64%|██████▍   | 257954/400000 [00:30<00:17, 8216.14it/s] 65%|██████▍   | 258843/400000 [00:30<00:16, 8398.62it/s] 65%|██████▍   | 259685/400000 [00:30<00:16, 8396.57it/s] 65%|██████▌   | 260568/400000 [00:30<00:16, 8519.40it/s] 65%|██████▌   | 261432/400000 [00:30<00:16, 8554.46it/s] 66%|██████▌   | 262302/400000 [00:30<00:16, 8595.83it/s] 66%|██████▌   | 263166/400000 [00:30<00:15, 8607.34it/s] 66%|██████▌   | 264029/400000 [00:30<00:15, 8613.57it/s] 66%|██████▌   | 264891/400000 [00:31<00:15, 8579.30it/s] 66%|██████▋   | 265750/400000 [00:31<00:15, 8578.33it/s] 67%|██████▋   | 266623/400000 [00:31<00:15, 8623.01it/s] 67%|██████▋   | 267506/400000 [00:31<00:15, 8683.82it/s] 67%|██████▋   | 268375/400000 [00:31<00:15, 8547.12it/s] 67%|██████▋   | 269238/400000 [00:31<00:15, 8569.54it/s] 68%|██████▊   | 270128/400000 [00:31<00:14, 8663.89it/s] 68%|██████▊   | 271040/400000 [00:31<00:14, 8795.02it/s] 68%|██████▊   | 271939/400000 [00:31<00:14, 8850.38it/s] 68%|██████▊   | 272825/400000 [00:31<00:14, 8831.56it/s] 68%|██████▊   | 273735/400000 [00:32<00:14, 8908.39it/s] 69%|██████▊   | 274627/400000 [00:32<00:14, 8831.60it/s] 69%|██████▉   | 275524/400000 [00:32<00:14, 8869.74it/s] 69%|██████▉   | 276412/400000 [00:32<00:14, 8809.90it/s] 69%|██████▉   | 277294/400000 [00:32<00:14, 8519.02it/s] 70%|██████▉   | 278197/400000 [00:32<00:14, 8664.92it/s] 70%|██████▉   | 279102/400000 [00:32<00:13, 8775.90it/s] 70%|██████▉   | 279997/400000 [00:32<00:13, 8824.66it/s] 70%|███████   | 280899/400000 [00:32<00:13, 8881.07it/s] 70%|███████   | 281815/400000 [00:32<00:13, 8962.63it/s] 71%|███████   | 282721/400000 [00:33<00:13, 8991.56it/s] 71%|███████   | 283621/400000 [00:33<00:13, 8895.87it/s] 71%|███████   | 284512/400000 [00:33<00:12, 8889.45it/s] 71%|███████▏  | 285424/400000 [00:33<00:12, 8949.28it/s] 72%|███████▏  | 286346/400000 [00:33<00:12, 9026.84it/s] 72%|███████▏  | 287260/400000 [00:33<00:12, 9058.12it/s] 72%|███████▏  | 288174/400000 [00:33<00:12, 9081.67it/s] 72%|███████▏  | 289085/400000 [00:33<00:12, 9088.17it/s] 72%|███████▏  | 289996/400000 [00:33<00:12, 9092.22it/s] 73%|███████▎  | 290927/400000 [00:33<00:11, 9156.17it/s] 73%|███████▎  | 291843/400000 [00:34<00:11, 9029.79it/s] 73%|███████▎  | 292747/400000 [00:34<00:12, 8896.75it/s] 73%|███████▎  | 293642/400000 [00:34<00:11, 8910.61it/s] 74%|███████▎  | 294571/400000 [00:34<00:11, 9018.33it/s] 74%|███████▍  | 295474/400000 [00:34<00:11, 8995.97it/s] 74%|███████▍  | 296381/400000 [00:34<00:11, 9015.69it/s] 74%|███████▍  | 297283/400000 [00:34<00:11, 8963.94it/s] 75%|███████▍  | 298180/400000 [00:34<00:11, 8785.01it/s] 75%|███████▍  | 299060/400000 [00:34<00:11, 8736.80it/s] 75%|███████▍  | 299935/400000 [00:34<00:11, 8716.13it/s] 75%|███████▌  | 300808/400000 [00:35<00:11, 8606.32it/s] 75%|███████▌  | 301672/400000 [00:35<00:11, 8613.51it/s] 76%|███████▌  | 302555/400000 [00:35<00:11, 8675.38it/s] 76%|███████▌  | 303433/400000 [00:35<00:11, 8705.00it/s] 76%|███████▌  | 304304/400000 [00:35<00:11, 8565.75it/s] 76%|███████▋  | 305162/400000 [00:35<00:11, 8317.95it/s] 77%|███████▋  | 306052/400000 [00:35<00:11, 8484.41it/s] 77%|███████▋  | 306925/400000 [00:35<00:10, 8554.09it/s] 77%|███████▋  | 307822/400000 [00:35<00:10, 8673.88it/s] 77%|███████▋  | 308729/400000 [00:35<00:10, 8788.80it/s] 77%|███████▋  | 309625/400000 [00:36<00:10, 8837.09it/s] 78%|███████▊  | 310510/400000 [00:36<00:10, 8804.30it/s] 78%|███████▊  | 311392/400000 [00:36<00:10, 8460.92it/s] 78%|███████▊  | 312251/400000 [00:36<00:10, 8495.15it/s] 78%|███████▊  | 313103/400000 [00:36<00:10, 8450.95it/s] 78%|███████▊  | 313950/400000 [00:36<00:10, 8303.81it/s] 79%|███████▊  | 314783/400000 [00:36<00:10, 8230.24it/s] 79%|███████▉  | 315608/400000 [00:36<00:10, 8196.20it/s] 79%|███████▉  | 316429/400000 [00:36<00:10, 8014.04it/s] 79%|███████▉  | 317236/400000 [00:37<00:10, 8028.88it/s] 80%|███████▉  | 318041/400000 [00:37<00:10, 7763.05it/s] 80%|███████▉  | 318878/400000 [00:37<00:10, 7935.15it/s] 80%|███████▉  | 319745/400000 [00:37<00:09, 8141.20it/s] 80%|████████  | 320563/400000 [00:37<00:09, 8121.24it/s] 80%|████████  | 321424/400000 [00:37<00:09, 8261.57it/s] 81%|████████  | 322281/400000 [00:37<00:09, 8349.06it/s] 81%|████████  | 323139/400000 [00:37<00:09, 8416.49it/s] 81%|████████  | 323983/400000 [00:37<00:09, 8322.66it/s] 81%|████████  | 324817/400000 [00:37<00:09, 7740.34it/s] 81%|████████▏ | 325672/400000 [00:38<00:09, 7965.36it/s] 82%|████████▏ | 326521/400000 [00:38<00:09, 8114.63it/s] 82%|████████▏ | 327377/400000 [00:38<00:08, 8242.99it/s] 82%|████████▏ | 328207/400000 [00:38<00:08, 8210.72it/s] 82%|████████▏ | 329032/400000 [00:38<00:08, 8171.19it/s] 82%|████████▏ | 329909/400000 [00:38<00:08, 8341.95it/s] 83%|████████▎ | 330785/400000 [00:38<00:08, 8462.74it/s] 83%|████████▎ | 331670/400000 [00:38<00:07, 8574.03it/s] 83%|████████▎ | 332547/400000 [00:38<00:07, 8631.52it/s] 83%|████████▎ | 333442/400000 [00:38<00:07, 8723.75it/s] 84%|████████▎ | 334321/400000 [00:39<00:07, 8742.96it/s] 84%|████████▍ | 335197/400000 [00:39<00:07, 8744.85it/s] 84%|████████▍ | 336076/400000 [00:39<00:07, 8757.64it/s] 84%|████████▍ | 336953/400000 [00:39<00:07, 8753.72it/s] 84%|████████▍ | 337864/400000 [00:39<00:07, 8856.21it/s] 85%|████████▍ | 338763/400000 [00:39<00:06, 8894.73it/s] 85%|████████▍ | 339675/400000 [00:39<00:06, 8960.55it/s] 85%|████████▌ | 340573/400000 [00:39<00:06, 8965.85it/s] 85%|████████▌ | 341477/400000 [00:39<00:06, 8987.48it/s] 86%|████████▌ | 342376/400000 [00:39<00:06, 8834.37it/s] 86%|████████▌ | 343261/400000 [00:40<00:06, 8752.91it/s] 86%|████████▌ | 344153/400000 [00:40<00:06, 8800.05it/s] 86%|████████▋ | 345034/400000 [00:40<00:06, 8714.75it/s] 86%|████████▋ | 345929/400000 [00:40<00:06, 8781.59it/s] 87%|████████▋ | 346824/400000 [00:40<00:06, 8829.44it/s] 87%|████████▋ | 347719/400000 [00:40<00:05, 8864.66it/s] 87%|████████▋ | 348606/400000 [00:40<00:05, 8840.79it/s] 87%|████████▋ | 349498/400000 [00:40<00:05, 8863.81it/s] 88%|████████▊ | 350385/400000 [00:40<00:05, 8857.20it/s] 88%|████████▊ | 351271/400000 [00:40<00:05, 8845.91it/s] 88%|████████▊ | 352156/400000 [00:41<00:05, 8822.53it/s] 88%|████████▊ | 353039/400000 [00:41<00:05, 8691.48it/s] 88%|████████▊ | 353909/400000 [00:41<00:05, 8690.35it/s] 89%|████████▊ | 354788/400000 [00:41<00:05, 8719.74it/s] 89%|████████▉ | 355681/400000 [00:41<00:05, 8781.70it/s] 89%|████████▉ | 356560/400000 [00:41<00:05, 8650.29it/s] 89%|████████▉ | 357426/400000 [00:41<00:04, 8641.73it/s] 90%|████████▉ | 358291/400000 [00:41<00:04, 8455.79it/s] 90%|████████▉ | 359138/400000 [00:41<00:04, 8452.83it/s] 90%|█████████ | 360005/400000 [00:42<00:04, 8514.58it/s] 90%|█████████ | 360877/400000 [00:42<00:04, 8572.55it/s] 90%|█████████ | 361735/400000 [00:42<00:04, 8225.84it/s] 91%|█████████ | 362605/400000 [00:42<00:04, 8360.81it/s] 91%|█████████ | 363470/400000 [00:42<00:04, 8443.28it/s] 91%|█████████ | 364317/400000 [00:42<00:04, 8437.84it/s] 91%|█████████▏| 365163/400000 [00:42<00:04, 8255.24it/s] 91%|█████████▏| 365991/400000 [00:42<00:04, 8205.03it/s] 92%|█████████▏| 366848/400000 [00:42<00:03, 8310.70it/s] 92%|█████████▏| 367689/400000 [00:42<00:03, 8339.39it/s] 92%|█████████▏| 368553/400000 [00:43<00:03, 8425.98it/s] 92%|█████████▏| 369432/400000 [00:43<00:03, 8530.98it/s] 93%|█████████▎| 370305/400000 [00:43<00:03, 8589.67it/s] 93%|█████████▎| 371172/400000 [00:43<00:03, 8612.27it/s] 93%|█████████▎| 372041/400000 [00:43<00:03, 8633.72it/s] 93%|█████████▎| 372913/400000 [00:43<00:03, 8654.95it/s] 93%|█████████▎| 373779/400000 [00:43<00:03, 8519.45it/s] 94%|█████████▎| 374654/400000 [00:43<00:02, 8586.99it/s] 94%|█████████▍| 375528/400000 [00:43<00:02, 8629.63it/s] 94%|█████████▍| 376404/400000 [00:43<00:02, 8665.21it/s] 94%|█████████▍| 377271/400000 [00:44<00:02, 8632.82it/s] 95%|█████████▍| 378139/400000 [00:44<00:02, 8644.38it/s] 95%|█████████▍| 379004/400000 [00:44<00:02, 8310.65it/s] 95%|█████████▍| 379838/400000 [00:44<00:02, 8315.07it/s] 95%|█████████▌| 380672/400000 [00:44<00:02, 8270.49it/s] 95%|█████████▌| 381501/400000 [00:44<00:02, 8214.21it/s] 96%|█████████▌| 382324/400000 [00:44<00:02, 8202.07it/s] 96%|█████████▌| 383145/400000 [00:44<00:02, 7966.51it/s] 96%|█████████▌| 383944/400000 [00:44<00:02, 7889.43it/s] 96%|█████████▌| 384799/400000 [00:44<00:01, 8074.79it/s] 96%|█████████▋| 385655/400000 [00:45<00:01, 8213.89it/s] 97%|█████████▋| 386534/400000 [00:45<00:01, 8378.32it/s] 97%|█████████▋| 387375/400000 [00:45<00:01, 8100.45it/s] 97%|█████████▋| 388189/400000 [00:45<00:01, 7906.97it/s] 97%|█████████▋| 389061/400000 [00:45<00:01, 8132.47it/s] 97%|█████████▋| 389930/400000 [00:45<00:01, 8290.30it/s] 98%|█████████▊| 390795/400000 [00:45<00:01, 8394.67it/s] 98%|█████████▊| 391638/400000 [00:45<00:00, 8380.74it/s] 98%|█████████▊| 392479/400000 [00:45<00:00, 8300.40it/s] 98%|█████████▊| 393365/400000 [00:45<00:00, 8459.45it/s] 99%|█████████▊| 394213/400000 [00:46<00:00, 8436.78it/s] 99%|█████████▉| 395075/400000 [00:46<00:00, 8489.25it/s] 99%|█████████▉| 395925/400000 [00:46<00:00, 8449.25it/s] 99%|█████████▉| 396790/400000 [00:46<00:00, 8506.03it/s] 99%|█████████▉| 397642/400000 [00:46<00:00, 8449.62it/s]100%|█████████▉| 398508/400000 [00:46<00:00, 8511.09it/s]100%|█████████▉| 399414/400000 [00:46<00:00, 8667.82it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8553.72it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f6b0ab51c18> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011052355575288655 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.01085954027431067 	 Accuracy: 70

  model saves at 70% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 16017 out of table with 15889 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 16017 out of table with 15889 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-14 07:24:14.986617: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 07:24:14.991075: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095090000 Hz
2020-05-14 07:24:14.991358: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55662dd598c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 07:24:14.991373: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f6ab753bdd8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5593 - accuracy: 0.5070
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5440 - accuracy: 0.5080 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5184 - accuracy: 0.5097
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6130 - accuracy: 0.5035
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6360 - accuracy: 0.5020
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6743 - accuracy: 0.4995
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6885 - accuracy: 0.4986
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7011 - accuracy: 0.4978
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7126 - accuracy: 0.4970
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7157 - accuracy: 0.4968
11000/25000 [============>.................] - ETA: 3s - loss: 7.7029 - accuracy: 0.4976
12000/25000 [=============>................] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6572 - accuracy: 0.5006
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6524 - accuracy: 0.5009
15000/25000 [=================>............] - ETA: 2s - loss: 7.6482 - accuracy: 0.5012
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6455 - accuracy: 0.5014
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6504 - accuracy: 0.5011
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6573 - accuracy: 0.5006
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6569 - accuracy: 0.5006
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6429 - accuracy: 0.5016
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6199 - accuracy: 0.5030
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6339 - accuracy: 0.5021
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6486 - accuracy: 0.5012
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
25000/25000 [==============================] - 7s 292us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f6a6a9b2898> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f6a7878f128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.5085 - crf_viterbi_accuracy: 0.2267 - val_loss: 1.4804 - val_crf_viterbi_accuracy: 0.2267

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

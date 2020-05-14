
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f4d64916fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 08:13:29.099509
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 08:13:29.104535
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 08:13:29.108218
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 08:13:29.112164
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f4d7092e438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352000.8438
Epoch 2/10

1/1 [==============================] - 0s 119ms/step - loss: 234042.0156
Epoch 3/10

1/1 [==============================] - 0s 100ms/step - loss: 127425.6328
Epoch 4/10

1/1 [==============================] - 0s 102ms/step - loss: 70998.7578
Epoch 5/10

1/1 [==============================] - 0s 117ms/step - loss: 41314.6055
Epoch 6/10

1/1 [==============================] - 0s 133ms/step - loss: 25628.1055
Epoch 7/10

1/1 [==============================] - 0s 105ms/step - loss: 16854.1504
Epoch 8/10

1/1 [==============================] - 0s 110ms/step - loss: 11511.9824
Epoch 9/10

1/1 [==============================] - 0s 113ms/step - loss: 8116.7861
Epoch 10/10

1/1 [==============================] - 0s 106ms/step - loss: 6080.7783

  #### Inference Need return ypred, ytrue ######################### 
[[  2.0987482    0.4182169   -1.5339944   -0.6288729    0.6289928
   -2.351269    -1.7194598   -0.7872349    0.28688195   0.19309664
   -0.8665633   -1.4163264    0.095916    -0.88176036   0.51740116
   -1.8481863    1.0446125   -0.81172764  -0.8673422    0.68702954
    1.6449009   -0.39930966   1.2531947   -0.35352576   0.76437247
    0.22733778   0.1343469   -0.5433583    0.83283365  -0.952955
   -0.95142895  -1.9668648   -0.19854352   0.71537006  -1.1970836
   -1.2366884    0.3211904   -0.279771     0.357818     0.16107358
   -0.95983917  -0.6744012    0.4912776   -0.7798095   -2.3421133
   -1.0022819   -0.17928931  -0.30586794   0.5021197    0.8025073
    1.3812568   -1.3745506    1.2274276    0.23996349  -0.19414225
    1.2602013    0.3772192    1.4123695   -0.39576188   0.1580585
   -0.08553818   8.373667     8.1293955    9.838375     7.7268357
    9.515726     7.7917814    7.3087163    9.928914     8.463488
    9.295911     9.597387     9.387318     8.313586     7.648624
    9.518691     9.286318     7.381106     8.522159     7.817571
    6.705568     8.936218     9.215058     9.663233     7.8453417
    8.073946     9.762387     9.226624    10.043824     8.729133
    7.1914077    9.200645     8.835575     8.110549     7.663785
    7.842463     8.791623     9.661565     8.637322     7.8474174
    7.4997406    7.8985662    8.921638     7.6647553    7.7692475
    9.389862     7.5437317    8.875806     8.799037     7.71344
    8.261489     8.947654     8.244033     8.612255     8.233498
    8.730189     8.577628     7.52389      7.4361315    8.957033
   -0.34870672   0.04300861   1.2107484   -0.49056163  -0.17272934
    0.39019355   0.07290634   0.22326005   1.7665799   -0.85502815
   -1.1378685    0.43836313  -0.9019897    0.40655613  -0.358957
    0.8146913   -0.72462904   0.42671123  -0.73692656   0.6238586
   -0.46494955  -0.6187508   -0.80911666   0.36537808   1.2562817
   -0.9733528    1.3114476   -0.94254684  -1.5513785    1.7526281
   -0.12743196  -0.7386013   -1.2898195   -0.28434166  -1.0976052
    1.1094604    0.4905104   -0.02541055   0.11290611  -2.2300234
   -0.5969274    1.7324053   -0.85100335  -0.03012574  -0.5559441
   -0.11255597  -1.0278027    0.2788941    0.72300816  -0.437862
   -0.5261732    2.167378     0.56788164  -0.37244532  -1.1288064
    0.71482503  -2.0504885    0.18991178  -0.86887366  -1.4807931
    0.46575242   1.5057938    2.3750677    1.8441899    2.0429187
    1.5560988    1.4979779    2.8954072    1.2173058    0.17994642
    0.29180276   2.2925882    0.9629214    1.484806     0.25630987
    1.1544493    2.1596515    2.3043242    1.0950296    0.16204047
    0.19856703   2.0903378    0.37226665   0.43982446   1.2924902
    1.9439518    1.0444918    0.42378294   0.7899915    1.6822742
    2.0957875    1.8000226    0.41323924   0.8677473    0.56494313
    0.12916613   1.2848803    1.1540923    0.9138327    2.3802166
    2.1388412    0.6474272    1.5249798    0.65016085   1.356067
    1.9903408    1.9583578    0.89517057   0.13082123   1.5649463
    0.66591686   0.14619088   0.7365379    0.5130309    1.8911886
    0.27913898   2.3121297    1.8869376    0.8694935    0.830058
    0.158656     9.498928     7.9068747    9.1976385    9.332022
    9.065866     7.3544993    8.273362     9.321573     8.213668
    8.487183     9.793125     9.319371     8.733442     8.841092
    7.8593287    8.929078     9.8210745    8.57135      8.652445
    8.156069     9.644642     9.191091     8.918036     7.195287
   10.110661    10.617583     9.223159    10.3800335    8.715698
    9.544211     8.644574     9.215015    10.43381      8.459813
    9.241749     9.334869     8.020385     8.834156     8.423197
    8.882114     8.890309     8.58691      8.945376     9.476712
    8.11123      7.742393     8.630045     9.0342455    8.592319
    8.512206     8.413806     7.806933     8.213006     9.919113
    9.131859     8.897639     8.603246     9.472418     7.199857
    2.6913295    0.5249141    0.75993454   0.69991386   1.0879115
    1.9373232    0.6938506    0.9747434    0.5080811    1.021269
    1.4327325    1.673484     0.72572994   0.2584049    1.2057215
    1.9173898    0.59768826   1.1281168    1.7675099    1.1645391
    0.7927412    0.41911364   3.1354885    0.13800287   2.243435
    1.8923323    1.0470767    0.23913628   0.5239853    2.5730464
    0.11294258   2.6630387    0.5482164    0.38131618   1.5046617
    1.7401471    0.7674534    0.6297985    1.5715207    0.4776706
    0.24265158   0.92081577   1.389852     0.84448165   1.4326029
    0.49242806   0.3514669    0.4651419    0.6254198    1.5715227
    0.48817086   0.64712346   0.31817693   2.5608187    0.34016085
    0.21651232   1.019938     0.90325624   0.91041195   1.850868
   -7.607977    14.165194   -10.174521  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 08:13:39.091504
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.4844
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 08:13:39.096105
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8761.79
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 08:13:39.100062
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.3338
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 08:13:39.103908
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -783.681
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139970019782728
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139968809812600
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139968809813104
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139968809813608
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139968809814112
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139968809814616

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f4d6c7afef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.510711
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.492230
grad_step = 000002, loss = 0.475439
grad_step = 000003, loss = 0.458603
grad_step = 000004, loss = 0.441104
grad_step = 000005, loss = 0.421640
grad_step = 000006, loss = 0.401300
grad_step = 000007, loss = 0.379153
grad_step = 000008, loss = 0.361468
grad_step = 000009, loss = 0.347966
grad_step = 000010, loss = 0.336337
grad_step = 000011, loss = 0.325127
grad_step = 000012, loss = 0.313128
grad_step = 000013, loss = 0.302737
grad_step = 000014, loss = 0.293488
grad_step = 000015, loss = 0.283646
grad_step = 000016, loss = 0.272685
grad_step = 000017, loss = 0.261369
grad_step = 000018, loss = 0.250818
grad_step = 000019, loss = 0.241032
grad_step = 000020, loss = 0.231185
grad_step = 000021, loss = 0.221483
grad_step = 000022, loss = 0.212546
grad_step = 000023, loss = 0.203979
grad_step = 000024, loss = 0.195354
grad_step = 000025, loss = 0.186700
grad_step = 000026, loss = 0.178167
grad_step = 000027, loss = 0.169890
grad_step = 000028, loss = 0.162014
grad_step = 000029, loss = 0.154761
grad_step = 000030, loss = 0.147858
grad_step = 000031, loss = 0.140752
grad_step = 000032, loss = 0.133697
grad_step = 000033, loss = 0.127156
grad_step = 000034, loss = 0.120946
grad_step = 000035, loss = 0.114741
grad_step = 000036, loss = 0.108758
grad_step = 000037, loss = 0.103320
grad_step = 000038, loss = 0.098144
grad_step = 000039, loss = 0.092903
grad_step = 000040, loss = 0.087832
grad_step = 000041, loss = 0.083073
grad_step = 000042, loss = 0.078486
grad_step = 000043, loss = 0.074044
grad_step = 000044, loss = 0.069839
grad_step = 000045, loss = 0.065910
grad_step = 000046, loss = 0.062145
grad_step = 000047, loss = 0.058435
grad_step = 000048, loss = 0.054946
grad_step = 000049, loss = 0.051699
grad_step = 000050, loss = 0.048516
grad_step = 000051, loss = 0.045483
grad_step = 000052, loss = 0.042682
grad_step = 000053, loss = 0.039463
grad_step = 000054, loss = 0.036238
grad_step = 000055, loss = 0.033411
grad_step = 000056, loss = 0.031150
grad_step = 000057, loss = 0.029239
grad_step = 000058, loss = 0.027418
grad_step = 000059, loss = 0.025552
grad_step = 000060, loss = 0.023581
grad_step = 000061, loss = 0.021658
grad_step = 000062, loss = 0.019889
grad_step = 000063, loss = 0.018282
grad_step = 000064, loss = 0.016861
grad_step = 000065, loss = 0.015574
grad_step = 000066, loss = 0.014348
grad_step = 000067, loss = 0.013177
grad_step = 000068, loss = 0.012112
grad_step = 000069, loss = 0.011103
grad_step = 000070, loss = 0.010157
grad_step = 000071, loss = 0.009279
grad_step = 000072, loss = 0.008494
grad_step = 000073, loss = 0.007789
grad_step = 000074, loss = 0.007159
grad_step = 000075, loss = 0.006594
grad_step = 000076, loss = 0.006077
grad_step = 000077, loss = 0.005601
grad_step = 000078, loss = 0.005147
grad_step = 000079, loss = 0.004725
grad_step = 000080, loss = 0.004360
grad_step = 000081, loss = 0.004052
grad_step = 000082, loss = 0.003796
grad_step = 000083, loss = 0.003566
grad_step = 000084, loss = 0.003351
grad_step = 000085, loss = 0.003148
grad_step = 000086, loss = 0.002960
grad_step = 000087, loss = 0.002803
grad_step = 000088, loss = 0.002677
grad_step = 000089, loss = 0.002574
grad_step = 000090, loss = 0.002489
grad_step = 000091, loss = 0.002412
grad_step = 000092, loss = 0.002339
grad_step = 000093, loss = 0.002277
grad_step = 000094, loss = 0.002229
grad_step = 000095, loss = 0.002191
grad_step = 000096, loss = 0.002154
grad_step = 000097, loss = 0.002115
grad_step = 000098, loss = 0.002082
grad_step = 000099, loss = 0.002061
grad_step = 000100, loss = 0.002051
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002047
grad_step = 000102, loss = 0.002052
grad_step = 000103, loss = 0.002051
grad_step = 000104, loss = 0.002030
grad_step = 000105, loss = 0.001990
grad_step = 000106, loss = 0.001968
grad_step = 000107, loss = 0.001968
grad_step = 000108, loss = 0.001978
grad_step = 000109, loss = 0.001988
grad_step = 000110, loss = 0.001977
grad_step = 000111, loss = 0.001949
grad_step = 000112, loss = 0.001920
grad_step = 000113, loss = 0.001905
grad_step = 000114, loss = 0.001903
grad_step = 000115, loss = 0.001908
grad_step = 000116, loss = 0.001922
grad_step = 000117, loss = 0.001941
grad_step = 000118, loss = 0.001953
grad_step = 000119, loss = 0.001923
grad_step = 000120, loss = 0.001882
grad_step = 000121, loss = 0.001855
grad_step = 000122, loss = 0.001848
grad_step = 000123, loss = 0.001853
grad_step = 000124, loss = 0.001864
grad_step = 000125, loss = 0.001875
grad_step = 000126, loss = 0.001864
grad_step = 000127, loss = 0.001836
grad_step = 000128, loss = 0.001805
grad_step = 000129, loss = 0.001791
grad_step = 000130, loss = 0.001794
grad_step = 000131, loss = 0.001802
grad_step = 000132, loss = 0.001805
grad_step = 000133, loss = 0.001806
grad_step = 000134, loss = 0.001828
grad_step = 000135, loss = 0.001891
grad_step = 000136, loss = 0.001992
grad_step = 000137, loss = 0.001939
grad_step = 000138, loss = 0.001816
grad_step = 000139, loss = 0.001779
grad_step = 000140, loss = 0.001836
grad_step = 000141, loss = 0.001835
grad_step = 000142, loss = 0.001756
grad_step = 000143, loss = 0.001799
grad_step = 000144, loss = 0.001848
grad_step = 000145, loss = 0.001750
grad_step = 000146, loss = 0.001738
grad_step = 000147, loss = 0.001803
grad_step = 000148, loss = 0.001758
grad_step = 000149, loss = 0.001714
grad_step = 000150, loss = 0.001746
grad_step = 000151, loss = 0.001762
grad_step = 000152, loss = 0.001726
grad_step = 000153, loss = 0.001711
grad_step = 000154, loss = 0.001736
grad_step = 000155, loss = 0.001741
grad_step = 000156, loss = 0.001714
grad_step = 000157, loss = 0.001714
grad_step = 000158, loss = 0.001750
grad_step = 000159, loss = 0.001792
grad_step = 000160, loss = 0.001841
grad_step = 000161, loss = 0.001907
grad_step = 000162, loss = 0.001960
grad_step = 000163, loss = 0.001872
grad_step = 000164, loss = 0.001726
grad_step = 000165, loss = 0.001674
grad_step = 000166, loss = 0.001753
grad_step = 000167, loss = 0.001820
grad_step = 000168, loss = 0.001760
grad_step = 000169, loss = 0.001670
grad_step = 000170, loss = 0.001671
grad_step = 000171, loss = 0.001731
grad_step = 000172, loss = 0.001748
grad_step = 000173, loss = 0.001698
grad_step = 000174, loss = 0.001663
grad_step = 000175, loss = 0.001670
grad_step = 000176, loss = 0.001685
grad_step = 000177, loss = 0.001669
grad_step = 000178, loss = 0.001649
grad_step = 000179, loss = 0.001658
grad_step = 000180, loss = 0.001681
grad_step = 000181, loss = 0.001693
grad_step = 000182, loss = 0.001666
grad_step = 000183, loss = 0.001648
grad_step = 000184, loss = 0.001649
grad_step = 000185, loss = 0.001652
grad_step = 000186, loss = 0.001637
grad_step = 000187, loss = 0.001616
grad_step = 000188, loss = 0.001609
grad_step = 000189, loss = 0.001616
grad_step = 000190, loss = 0.001621
grad_step = 000191, loss = 0.001616
grad_step = 000192, loss = 0.001611
grad_step = 000193, loss = 0.001620
grad_step = 000194, loss = 0.001649
grad_step = 000195, loss = 0.001697
grad_step = 000196, loss = 0.001779
grad_step = 000197, loss = 0.001853
grad_step = 000198, loss = 0.001910
grad_step = 000199, loss = 0.001827
grad_step = 000200, loss = 0.001681
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001603
grad_step = 000202, loss = 0.001651
grad_step = 000203, loss = 0.001712
grad_step = 000204, loss = 0.001671
grad_step = 000205, loss = 0.001613
grad_step = 000206, loss = 0.001615
grad_step = 000207, loss = 0.001630
grad_step = 000208, loss = 0.001622
grad_step = 000209, loss = 0.001604
grad_step = 000210, loss = 0.001595
grad_step = 000211, loss = 0.001592
grad_step = 000212, loss = 0.001579
grad_step = 000213, loss = 0.001576
grad_step = 000214, loss = 0.001582
grad_step = 000215, loss = 0.001573
grad_step = 000216, loss = 0.001558
grad_step = 000217, loss = 0.001549
grad_step = 000218, loss = 0.001551
grad_step = 000219, loss = 0.001554
grad_step = 000220, loss = 0.001545
grad_step = 000221, loss = 0.001536
grad_step = 000222, loss = 0.001530
grad_step = 000223, loss = 0.001528
grad_step = 000224, loss = 0.001524
grad_step = 000225, loss = 0.001517
grad_step = 000226, loss = 0.001512
grad_step = 000227, loss = 0.001511
grad_step = 000228, loss = 0.001513
grad_step = 000229, loss = 0.001515
grad_step = 000230, loss = 0.001511
grad_step = 000231, loss = 0.001502
grad_step = 000232, loss = 0.001496
grad_step = 000233, loss = 0.001496
grad_step = 000234, loss = 0.001504
grad_step = 000235, loss = 0.001524
grad_step = 000236, loss = 0.001572
grad_step = 000237, loss = 0.001638
grad_step = 000238, loss = 0.001743
grad_step = 000239, loss = 0.001752
grad_step = 000240, loss = 0.001702
grad_step = 000241, loss = 0.001560
grad_step = 000242, loss = 0.001493
grad_step = 000243, loss = 0.001527
grad_step = 000244, loss = 0.001526
grad_step = 000245, loss = 0.001494
grad_step = 000246, loss = 0.001497
grad_step = 000247, loss = 0.001526
grad_step = 000248, loss = 0.001518
grad_step = 000249, loss = 0.001444
grad_step = 000250, loss = 0.001423
grad_step = 000251, loss = 0.001462
grad_step = 000252, loss = 0.001473
grad_step = 000253, loss = 0.001454
grad_step = 000254, loss = 0.001436
grad_step = 000255, loss = 0.001445
grad_step = 000256, loss = 0.001451
grad_step = 000257, loss = 0.001433
grad_step = 000258, loss = 0.001403
grad_step = 000259, loss = 0.001385
grad_step = 000260, loss = 0.001390
grad_step = 000261, loss = 0.001402
grad_step = 000262, loss = 0.001398
grad_step = 000263, loss = 0.001385
grad_step = 000264, loss = 0.001376
grad_step = 000265, loss = 0.001378
grad_step = 000266, loss = 0.001387
grad_step = 000267, loss = 0.001396
grad_step = 000268, loss = 0.001406
grad_step = 000269, loss = 0.001410
grad_step = 000270, loss = 0.001425
grad_step = 000271, loss = 0.001454
grad_step = 000272, loss = 0.001501
grad_step = 000273, loss = 0.001564
grad_step = 000274, loss = 0.001585
grad_step = 000275, loss = 0.001561
grad_step = 000276, loss = 0.001443
grad_step = 000277, loss = 0.001349
grad_step = 000278, loss = 0.001337
grad_step = 000279, loss = 0.001386
grad_step = 000280, loss = 0.001424
grad_step = 000281, loss = 0.001404
grad_step = 000282, loss = 0.001352
grad_step = 000283, loss = 0.001328
grad_step = 000284, loss = 0.001343
grad_step = 000285, loss = 0.001369
grad_step = 000286, loss = 0.001366
grad_step = 000287, loss = 0.001334
grad_step = 000288, loss = 0.001308
grad_step = 000289, loss = 0.001315
grad_step = 000290, loss = 0.001338
grad_step = 000291, loss = 0.001342
grad_step = 000292, loss = 0.001334
grad_step = 000293, loss = 0.001321
grad_step = 000294, loss = 0.001314
grad_step = 000295, loss = 0.001314
grad_step = 000296, loss = 0.001324
grad_step = 000297, loss = 0.001331
grad_step = 000298, loss = 0.001328
grad_step = 000299, loss = 0.001315
grad_step = 000300, loss = 0.001303
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001296
grad_step = 000302, loss = 0.001295
grad_step = 000303, loss = 0.001296
grad_step = 000304, loss = 0.001299
grad_step = 000305, loss = 0.001298
grad_step = 000306, loss = 0.001296
grad_step = 000307, loss = 0.001290
grad_step = 000308, loss = 0.001284
grad_step = 000309, loss = 0.001280
grad_step = 000310, loss = 0.001279
grad_step = 000311, loss = 0.001279
grad_step = 000312, loss = 0.001283
grad_step = 000313, loss = 0.001290
grad_step = 000314, loss = 0.001310
grad_step = 000315, loss = 0.001332
grad_step = 000316, loss = 0.001384
grad_step = 000317, loss = 0.001413
grad_step = 000318, loss = 0.001473
grad_step = 000319, loss = 0.001420
grad_step = 000320, loss = 0.001356
grad_step = 000321, loss = 0.001285
grad_step = 000322, loss = 0.001294
grad_step = 000323, loss = 0.001341
grad_step = 000324, loss = 0.001319
grad_step = 000325, loss = 0.001269
grad_step = 000326, loss = 0.001255
grad_step = 000327, loss = 0.001285
grad_step = 000328, loss = 0.001305
grad_step = 000329, loss = 0.001276
grad_step = 000330, loss = 0.001250
grad_step = 000331, loss = 0.001256
grad_step = 000332, loss = 0.001279
grad_step = 000333, loss = 0.001300
grad_step = 000334, loss = 0.001288
grad_step = 000335, loss = 0.001273
grad_step = 000336, loss = 0.001258
grad_step = 000337, loss = 0.001255
grad_step = 000338, loss = 0.001260
grad_step = 000339, loss = 0.001265
grad_step = 000340, loss = 0.001274
grad_step = 000341, loss = 0.001277
grad_step = 000342, loss = 0.001277
grad_step = 000343, loss = 0.001269
grad_step = 000344, loss = 0.001263
grad_step = 000345, loss = 0.001262
grad_step = 000346, loss = 0.001268
grad_step = 000347, loss = 0.001280
grad_step = 000348, loss = 0.001292
grad_step = 000349, loss = 0.001307
grad_step = 000350, loss = 0.001309
grad_step = 000351, loss = 0.001312
grad_step = 000352, loss = 0.001297
grad_step = 000353, loss = 0.001281
grad_step = 000354, loss = 0.001264
grad_step = 000355, loss = 0.001256
grad_step = 000356, loss = 0.001261
grad_step = 000357, loss = 0.001262
grad_step = 000358, loss = 0.001265
grad_step = 000359, loss = 0.001253
grad_step = 000360, loss = 0.001238
grad_step = 000361, loss = 0.001213
grad_step = 000362, loss = 0.001198
grad_step = 000363, loss = 0.001197
grad_step = 000364, loss = 0.001205
grad_step = 000365, loss = 0.001213
grad_step = 000366, loss = 0.001209
grad_step = 000367, loss = 0.001199
grad_step = 000368, loss = 0.001188
grad_step = 000369, loss = 0.001184
grad_step = 000370, loss = 0.001188
grad_step = 000371, loss = 0.001194
grad_step = 000372, loss = 0.001198
grad_step = 000373, loss = 0.001194
grad_step = 000374, loss = 0.001191
grad_step = 000375, loss = 0.001186
grad_step = 000376, loss = 0.001186
grad_step = 000377, loss = 0.001191
grad_step = 000378, loss = 0.001202
grad_step = 000379, loss = 0.001220
grad_step = 000380, loss = 0.001257
grad_step = 000381, loss = 0.001316
grad_step = 000382, loss = 0.001419
grad_step = 000383, loss = 0.001554
grad_step = 000384, loss = 0.001687
grad_step = 000385, loss = 0.001743
grad_step = 000386, loss = 0.001530
grad_step = 000387, loss = 0.001298
grad_step = 000388, loss = 0.001246
grad_step = 000389, loss = 0.001390
grad_step = 000390, loss = 0.001445
grad_step = 000391, loss = 0.001343
grad_step = 000392, loss = 0.001261
grad_step = 000393, loss = 0.001225
grad_step = 000394, loss = 0.001238
grad_step = 000395, loss = 0.001320
grad_step = 000396, loss = 0.001313
grad_step = 000397, loss = 0.001181
grad_step = 000398, loss = 0.001181
grad_step = 000399, loss = 0.001235
grad_step = 000400, loss = 0.001197
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001170
grad_step = 000402, loss = 0.001192
grad_step = 000403, loss = 0.001187
grad_step = 000404, loss = 0.001174
grad_step = 000405, loss = 0.001181
grad_step = 000406, loss = 0.001161
grad_step = 000407, loss = 0.001138
grad_step = 000408, loss = 0.001148
grad_step = 000409, loss = 0.001153
grad_step = 000410, loss = 0.001134
grad_step = 000411, loss = 0.001134
grad_step = 000412, loss = 0.001147
grad_step = 000413, loss = 0.001142
grad_step = 000414, loss = 0.001136
grad_step = 000415, loss = 0.001143
grad_step = 000416, loss = 0.001144
grad_step = 000417, loss = 0.001135
grad_step = 000418, loss = 0.001143
grad_step = 000419, loss = 0.001145
grad_step = 000420, loss = 0.001138
grad_step = 000421, loss = 0.001131
grad_step = 000422, loss = 0.001132
grad_step = 000423, loss = 0.001120
grad_step = 000424, loss = 0.001110
grad_step = 000425, loss = 0.001102
grad_step = 000426, loss = 0.001093
grad_step = 000427, loss = 0.001089
grad_step = 000428, loss = 0.001092
grad_step = 000429, loss = 0.001095
grad_step = 000430, loss = 0.001093
grad_step = 000431, loss = 0.001096
grad_step = 000432, loss = 0.001101
grad_step = 000433, loss = 0.001107
grad_step = 000434, loss = 0.001109
grad_step = 000435, loss = 0.001122
grad_step = 000436, loss = 0.001128
grad_step = 000437, loss = 0.001141
grad_step = 000438, loss = 0.001136
grad_step = 000439, loss = 0.001132
grad_step = 000440, loss = 0.001104
grad_step = 000441, loss = 0.001080
grad_step = 000442, loss = 0.001065
grad_step = 000443, loss = 0.001064
grad_step = 000444, loss = 0.001070
grad_step = 000445, loss = 0.001075
grad_step = 000446, loss = 0.001075
grad_step = 000447, loss = 0.001067
grad_step = 000448, loss = 0.001059
grad_step = 000449, loss = 0.001053
grad_step = 000450, loss = 0.001049
grad_step = 000451, loss = 0.001047
grad_step = 000452, loss = 0.001048
grad_step = 000453, loss = 0.001051
grad_step = 000454, loss = 0.001055
grad_step = 000455, loss = 0.001062
grad_step = 000456, loss = 0.001068
grad_step = 000457, loss = 0.001083
grad_step = 000458, loss = 0.001093
grad_step = 000459, loss = 0.001119
grad_step = 000460, loss = 0.001127
grad_step = 000461, loss = 0.001147
grad_step = 000462, loss = 0.001126
grad_step = 000463, loss = 0.001100
grad_step = 000464, loss = 0.001053
grad_step = 000465, loss = 0.001024
grad_step = 000466, loss = 0.001024
grad_step = 000467, loss = 0.001039
grad_step = 000468, loss = 0.001051
grad_step = 000469, loss = 0.001044
grad_step = 000470, loss = 0.001029
grad_step = 000471, loss = 0.001015
grad_step = 000472, loss = 0.001011
grad_step = 000473, loss = 0.001017
grad_step = 000474, loss = 0.001025
grad_step = 000475, loss = 0.001032
grad_step = 000476, loss = 0.001028
grad_step = 000477, loss = 0.001024
grad_step = 000478, loss = 0.001015
grad_step = 000479, loss = 0.001016
grad_step = 000480, loss = 0.001019
grad_step = 000481, loss = 0.001031
grad_step = 000482, loss = 0.001044
grad_step = 000483, loss = 0.001067
grad_step = 000484, loss = 0.001081
grad_step = 000485, loss = 0.001097
grad_step = 000486, loss = 0.001076
grad_step = 000487, loss = 0.001045
grad_step = 000488, loss = 0.001001
grad_step = 000489, loss = 0.000974
grad_step = 000490, loss = 0.000971
grad_step = 000491, loss = 0.000985
grad_step = 000492, loss = 0.001002
grad_step = 000493, loss = 0.001008
grad_step = 000494, loss = 0.001005
grad_step = 000495, loss = 0.000991
grad_step = 000496, loss = 0.000975
grad_step = 000497, loss = 0.000965
grad_step = 000498, loss = 0.000964
grad_step = 000499, loss = 0.000976
grad_step = 000500, loss = 0.001004
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001065
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

  date_run                              2020-05-14 08:14:04.188737
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.201384
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 08:14:04.195040
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.10685
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 08:14:04.202237
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.110203
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 08:14:04.208061
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -0.62363
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
0   2020-05-14 08:13:29.099509  ...    mean_absolute_error
1   2020-05-14 08:13:29.104535  ...     mean_squared_error
2   2020-05-14 08:13:29.108218  ...  median_absolute_error
3   2020-05-14 08:13:29.112164  ...               r2_score
4   2020-05-14 08:13:39.091504  ...    mean_absolute_error
5   2020-05-14 08:13:39.096105  ...     mean_squared_error
6   2020-05-14 08:13:39.100062  ...  median_absolute_error
7   2020-05-14 08:13:39.103908  ...               r2_score
8   2020-05-14 08:14:04.188737  ...    mean_absolute_error
9   2020-05-14 08:14:04.195040  ...     mean_squared_error
10  2020-05-14 08:14:04.202237  ...  median_absolute_error
11  2020-05-14 08:14:04.208061  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ffa7f672be0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▉      | 3899392/9912422 [00:00<00:00, 38233355.61it/s]9920512it [00:00, 35244428.83it/s]                             
0it [00:00, ?it/s]32768it [00:00, 619314.77it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 462877.94it/s]1654784it [00:00, 11767532.45it/s]                         
0it [00:00, ?it/s]8192it [00:00, 206404.46it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ffa3202ce80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ffa3165a0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ffa3202ce80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ffa7f67dba8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ffa2edee4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ffa7f67dba8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ffa3202ce80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ffa7f67dba8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ffa2edee4e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ffa7f635f28> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fb86f6401d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=f84936657f84e914a36331edcf3e52fd14de29ae4f2d83c434a3f5c6d9174230
  Stored in directory: /tmp/pip-ephem-wheel-cache-t_pw8uq2/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fb80832aba8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3579904/17464789 [=====>........................] - ETA: 0s
 8380416/17464789 [=============>................] - ETA: 0s
11780096/17464789 [===================>..........] - ETA: 0s
15073280/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 08:15:33.044297: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 08:15:33.050332: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-14 08:15:33.050487: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e8a70d4710 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 08:15:33.050503: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.9273 - accuracy: 0.4830
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7970 - accuracy: 0.4915
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.7688 - accuracy: 0.4933 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7931 - accuracy: 0.4918
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6206 - accuracy: 0.5030
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.6794 - accuracy: 0.4992
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6820 - accuracy: 0.4990
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.7069 - accuracy: 0.4974
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6939 - accuracy: 0.4982
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6927 - accuracy: 0.4983
11000/25000 [============>.................] - ETA: 4s - loss: 7.6945 - accuracy: 0.4982
12000/25000 [=============>................] - ETA: 4s - loss: 7.6858 - accuracy: 0.4988
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6749 - accuracy: 0.4995
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6995 - accuracy: 0.4979
15000/25000 [=================>............] - ETA: 3s - loss: 7.6830 - accuracy: 0.4989
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6628 - accuracy: 0.5002
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6576 - accuracy: 0.5006
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6504 - accuracy: 0.5011
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6553 - accuracy: 0.5007
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6574 - accuracy: 0.5006
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6615 - accuracy: 0.5003
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6736 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6700 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6749 - accuracy: 0.4995
25000/25000 [==============================] - 10s 411us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 08:15:51.087811
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 08:15:51.087811  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:26:35, 13.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:08:50, 18.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:15:31, 25.9kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:29:24, 36.9kB/s].vector_cache/glove.6B.zip:   0%|          | 1.98M/862M [00:01<4:32:40, 52.6kB/s].vector_cache/glove.6B.zip:   1%|          | 5.34M/862M [00:01<3:10:15, 75.1kB/s].vector_cache/glove.6B.zip:   1%|          | 9.58M/862M [00:01<2:12:37, 107kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 13.8M/862M [00:01<1:32:28, 153kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.4M/862M [00:01<1:04:34, 218kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.5M/862M [00:01<45:05, 311kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 25.2M/862M [00:01<31:32, 442kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.3M/862M [00:01<22:07, 628kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.8M/862M [00:01<15:36, 887kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.1M/862M [00:02<11:04, 1.25MB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.4M/862M [00:02<07:54, 1.74MB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.1M/862M [00:02<05:40, 2.42MB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.5M/862M [00:02<04:04, 3.35MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.8M/862M [00:02<02:56, 4.63MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.9M/862M [00:02<02:08, 6.31MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<01:50, 7.33MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<01:23, 9.69MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:05<10:16, 1.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<14:32, 923kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<12:05, 1.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<08:55, 1.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:05<06:39, 2.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:05<04:57, 2.69MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:07<23:35, 566kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<17:53, 746kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<12:50, 1.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.7M/862M [00:08<12:05, 1.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:09<11:19, 1.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:09<08:31, 1.56MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:09<06:18, 2.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<07:32, 1.75MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:11<06:39, 1.99MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:11<05:21, 2.46MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.4M/862M [00:11<03:57, 3.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<07:10, 1.83MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.3M/862M [00:13<06:44, 1.95MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.5M/862M [00:13<05:08, 2.55MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<06:08, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:15<07:31, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:15<05:58, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.8M/862M [00:15<04:37, 2.82MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:15<03:33, 3.66MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:15<02:49, 4.60MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<58:48, 221kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:17<44:04, 295kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:17<31:28, 413kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:17<22:09, 585kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<20:36, 628kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:19<15:51, 816kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:19<11:26, 1.13MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:22<13:21, 964kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:22<22:13, 579kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:22<18:35, 692kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:22<13:40, 941kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.2M/862M [00:22<09:46, 1.31MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:23<08:40, 1.48MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:23<06:21, 2.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.2M/862M [00:23<04:47, 2.67MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<03:36, 3.53MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:25<35:14, 362kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:25<28:56, 440kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:25<22:56, 555kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:26<16:48, 757kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:26<12:01, 1.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<12:17, 1.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<14:18, 885kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:28<11:23, 1.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:28<08:17, 1.52MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<08:30, 1.48MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<07:42, 1.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:30<06:17, 2.00MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:30<04:59, 2.52MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<03:52, 3.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<03:03, 4.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<09:36, 1.30MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<09:18, 1.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:32<07:53, 1.59MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:32<06:01, 2.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<04:31, 2.76MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<03:28, 3.59MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<13:16, 939kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:34<12:49, 972kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:34<10:48, 1.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:34<08:23, 1.48MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:34<06:16, 1.98MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<04:39, 2.66MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:36<08:16, 1.50MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:36<08:03, 1.54MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:36<06:07, 2.02MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<04:29, 2.75MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<07:35, 1.62MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<06:44, 1.83MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:38<05:09, 2.39MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:38<03:54, 3.15MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<03:05, 3.98MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<10:08, 1.21MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<09:42, 1.26MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:40<07:27, 1.64MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:20, 2.28MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<14:13, 857kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<11:14, 1.08MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:42<08:08, 1.49MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<08:25, 1.44MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<08:22, 1.45MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:44<06:28, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<04:38, 2.60MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:46<26:26, 456kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:46<19:40, 612kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:46<14:24, 835kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:46<10:22, 1.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:36, 1.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<10:59, 1.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:48<11:09, 1.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:48<09:08, 1.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:48<06:59, 1.71MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:03, 2.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<08:04, 1.47MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<06:58, 1.71MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:50<05:30, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<04:01, 2.94MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<07:37, 1.55MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<07:05, 1.67MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:52<05:47, 2.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:52<04:17, 2.75MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:52<05:53, 2.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<04:11, 2.80MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<14:33, 805kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<20:10, 580kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<17:32, 668kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<13:57, 838kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<10:51, 1.08MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:56<08:07, 1.44MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:56<05:59, 1.95MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<04:27, 2.61MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<37:19, 312kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<37:31, 310kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<29:00, 401kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:59<20:55, 556kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:59<14:52, 780kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:00<13:08, 881kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<10:47, 1.07MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:01<07:51, 1.47MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:01<05:43, 2.01MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<08:55, 1.29MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<07:48, 1.47MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:03<05:51, 1.96MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<04:13, 2.71MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<32:21, 353kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<24:07, 474kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:05<17:13, 662kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<12:10, 934kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<22:10, 513kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:07<17:05, 665kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:07<12:19, 920kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<08:43, 1.29MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<9:03:52, 20.8kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:10<6:31:53, 28.8kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:10<4:38:06, 40.6kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:10<3:16:27, 57.5kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:10<2:18:21, 81.6kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:10<1:37:02, 116kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<1:09:38, 161kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<49:22, 227kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:12<34:48, 322kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:12<24:38, 454kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<17:35, 635kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<25:41, 434kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<19:50, 562kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:14<15:15, 731kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:14<11:08, 1.00MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:14<08:06, 1.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<06:01, 1.84MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:16<10:51, 1.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:16<13:07, 845kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:16<10:36, 1.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:16<07:44, 1.43MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:16<05:35, 1.98MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:18<09:43, 1.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:18<08:08, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:18<05:59, 1.84MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<04:21, 2.52MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:20<09:16, 1.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:20<08:50, 1.24MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:20<06:40, 1.64MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:20<04:54, 2.22MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:22<06:55, 1.57MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:22<07:11, 1.51MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:22<05:37, 1.93MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<04:04, 2.66MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:24<07:50, 1.38MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:24<06:49, 1.58MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:24<05:19, 2.03MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:24<04:00, 2.69MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<03:04, 3.51MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:26<09:35, 1.12MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:27<10:34, 1.02MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:27<08:25, 1.28MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:27<06:05, 1.76MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:28<06:53, 1.55MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:29<05:46, 1.85MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:29<04:43, 2.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:29<03:32, 3.00MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:29<02:41, 3.93MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:31<15:27, 686kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:31<18:31, 572kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:31<15:03, 704kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:31<11:01, 961kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:32<07:57, 1.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:32<05:45, 1.83MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:33<17:26, 604kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:33<14:01, 751kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:33<10:23, 1.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:33<07:33, 1.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:33<05:32, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:35<08:29, 1.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:35<08:04, 1.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:35<06:11, 1.69MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:35<04:29, 2.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:37<06:59, 1.49MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:37<06:10, 1.68MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:37<04:59, 2.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:37<03:43, 2.78MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<02:52, 3.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:39<07:04, 1.46MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:39<06:59, 1.48MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:39<05:27, 1.89MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:39<04:03, 2.54MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:41<05:14, 1.95MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:41<05:05, 2.01MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:41<03:54, 2.62MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<02:52, 3.54MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:43<09:39, 1.06MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:43<07:59, 1.28MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:43<05:54, 1.72MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:43<04:18, 2.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:45<06:51, 1.48MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:45<06:01, 1.68MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:45<04:27, 2.26MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:45<03:16, 3.07MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:47<09:44, 1.03MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:47<07:59, 1.26MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:47<05:50, 1.71MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:47<04:14, 2.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:49<10:43, 931kB/s] .vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:49<09:27, 1.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:49<07:41, 1.30MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:49<05:37, 1.77MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<04:06, 2.41MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:51<08:20, 1.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:51<07:16, 1.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:51<05:22, 1.84MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:51<03:57, 2.50MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:53<06:26, 1.53MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:53<05:44, 1.71MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:53<04:28, 2.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:53<03:22, 2.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<02:36, 3.75MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:55<07:26, 1.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:55<07:18, 1.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:55<05:33, 1.75MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:55<04:09, 2.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:57<05:03, 1.92MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:57<04:47, 2.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:57<03:39, 2.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:59<04:29, 2.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:59<04:19, 2.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:59<03:17, 2.92MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<02:25, 3.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:01<48:38, 197kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:01<38:45, 247kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [02:01<28:13, 339kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:02<19:57, 478kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:02<14:02, 677kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:03<27:17, 348kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:03<19:59, 474kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:03<14:05, 671kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:05<12:28, 754kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:05<11:49, 796kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:05<09:02, 1.04MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<06:28, 1.45MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:07<07:21, 1.27MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:07<06:17, 1.49MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:07<05:06, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:07<03:47, 2.46MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:10<05:44, 1.62MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:10<13:21, 695kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:10<11:22, 815kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:10<08:24, 1.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:10<06:03, 1.52MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:12<06:36, 1.39MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:12<05:33, 1.66MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:12<04:29, 2.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:12<03:19, 2.76MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:14<04:45, 1.92MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:14<04:16, 2.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:14<03:13, 2.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:14<02:22, 3.83MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:16<1:08:46, 132kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:16<49:02, 185kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:16<34:27, 262kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:18<26:09, 344kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:18<19:12, 468kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:18<13:36, 659kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:18<09:36, 929kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:20<28:55, 309kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:20<21:08, 422kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:20<14:58, 594kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:20<10:36, 837kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:22<13:34, 653kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:22<10:41, 829kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:22<07:43, 1.14MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:22<05:30, 1.60MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:25<10:21, 848kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:25<14:27, 607kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:25<12:10, 721kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:25<09:01, 971kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:25<06:23, 1.36MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:27<11:47, 740kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:27<09:10, 949kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:27<06:38, 1.31MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:29<06:33, 1.32MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:29<06:26, 1.34MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:29<04:54, 1.76MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:29<03:38, 2.36MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:31<04:39, 1.84MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:31<04:09, 2.06MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:31<03:09, 2.71MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:31<02:21, 3.62MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:33<05:51, 1.45MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:33<05:47, 1.47MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:33<04:40, 1.82MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:33<03:33, 2.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:33<02:38, 3.20MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:35<05:38, 1.50MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:35<05:16, 1.60MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:35<04:25, 1.91MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:35<03:19, 2.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:36<03:58, 2.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:37<03:40, 2.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:37<02:47, 2.99MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:38<03:52, 2.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:39<03:34, 2.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:39<02:44, 3.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:39<02:05, 3.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:40<04:38, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:41<04:52, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:41<03:47, 2.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:41<02:51, 2.87MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:42<03:58, 2.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:42<03:59, 2.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:43<03:26, 2.37MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:43<02:37, 3.10MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:43<01:58, 4.10MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:44<06:20, 1.28MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:44<05:48, 1.39MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:45<04:42, 1.72MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:45<03:38, 2.22MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:45<02:39, 3.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:46<06:38, 1.21MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:46<05:26, 1.47MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:47<04:23, 1.82MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:47<03:11, 2.50MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:48<04:46, 1.67MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:48<04:07, 1.93MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:49<03:18, 2.40MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:49<02:24, 3.27MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:50<05:15, 1.50MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:50<05:13, 1.51MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:50<04:28, 1.76MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:51<03:19, 2.37MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:51<02:24, 3.24MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:53<46:02, 170kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:53<40:10, 195kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:53<30:30, 256kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:54<22:33, 346kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:54<16:26, 475kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:54<11:48, 660kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:54<08:23, 925kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:55<08:35, 902kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:55<06:34, 1.18MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:55<04:52, 1.58MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:55<03:32, 2.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:57<05:44, 1.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:57<05:09, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:57<03:50, 1.99MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:57<02:47, 2.73MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [03:00<07:45, 982kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [03:00<09:37, 791kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [03:00<07:46, 978kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [03:00<05:39, 1.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:00<04:04, 1.85MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:02<06:41, 1.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:02<05:29, 1.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:02<03:59, 1.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:02<02:53, 2.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:04<13:01, 574kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:04<09:53, 755kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:04<07:04, 1.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:05<06:38, 1.12MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:06<05:24, 1.37MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:06<03:56, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:06<02:50, 2.58MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:07<50:27, 145kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:08<36:49, 199kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:08<26:03, 281kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:08<18:20, 398kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:09<14:47, 491kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:10<11:05, 655kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:10<07:55, 913kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:11<07:12, 999kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:11<05:45, 1.25MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:12<04:10, 1.72MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:13<04:36, 1.55MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:13<03:58, 1.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:14<02:56, 2.42MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:14<02:08, 3.31MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:15<57:58, 122kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:16<42:00, 168kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:16<29:42, 237kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:16<20:50, 337kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:17<16:46, 417kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:17<12:53, 543kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:18<09:14, 754kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:18<06:33, 1.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:19<07:48, 886kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:19<06:21, 1.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:20<04:39, 1.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:21<04:36, 1.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:21<03:56, 1.74MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:22<02:55, 2.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:23<03:37, 1.88MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:23<04:42, 1.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:23<04:01, 1.69MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:24<03:04, 2.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:24<02:13, 3.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:26<07:26, 902kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:26<07:58, 843kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:26<06:13, 1.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:26<04:43, 1.42MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:26<03:25, 1.95MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:28<04:26, 1.50MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:28<04:41, 1.42MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:28<04:28, 1.48MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:28<03:45, 1.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:28<03:02, 2.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:28<02:28, 2.68MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:28<01:57, 3.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:29<01:35, 4.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:30<05:08, 1.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:30<04:38, 1.42MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:30<03:43, 1.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:30<02:46, 2.36MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:30<02:05, 3.11MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:30<01:39, 3.95MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:33<1:03:22, 103kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:33<50:48, 128kB/s]  .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:33<37:04, 176kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:33<26:14, 248kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:33<18:28, 351kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:35<14:59, 430kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:36<16:53, 381kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:36<13:21, 482kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:36<09:44, 661kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:36<06:59, 918kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:36<05:00, 1.28MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:37<06:36, 965kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:38<05:12, 1.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:38<03:50, 1.66MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:38<02:46, 2.28MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:40<06:11, 1.02MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:40<07:51, 803kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:40<06:31, 967kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:40<04:58, 1.27MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:40<03:39, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:40<02:43, 2.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:40<02:04, 3.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:42<43:07, 145kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:42<31:34, 197kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:42<22:27, 277kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:42<15:50, 392kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:42<11:13, 551kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:43<08:02, 768kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:44<18:40, 330kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:44<15:27, 399kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:44<12:21, 499kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:44<09:54, 622kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:44<08:20, 738kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:45<06:52, 896kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:45<05:17, 1.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:45<03:53, 1.57MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:45<02:52, 2.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:46<04:15, 1.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:46<03:18, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:46<02:26, 2.49MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:46<01:53, 3.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:46<01:32, 3.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:48<34:24, 175kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:48<25:19, 238kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:48<17:59, 335kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:48<12:42, 472kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:48<08:56, 667kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:50<20:04, 297kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:50<15:05, 395kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:50<10:58, 542kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:50<07:50, 758kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:50<05:35, 1.06MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:52<06:34, 896kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:52<05:41, 1.04MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:52<04:30, 1.30MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:52<03:18, 1.77MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:52<02:22, 2.45MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:54<15:31, 375kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:54<11:41, 498kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:54<08:31, 682kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:54<06:05, 950kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:54<04:22, 1.32MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:56<06:19, 911kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:56<05:31, 1.04MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:56<04:09, 1.38MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:56<03:00, 1.90MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:58<03:53, 1.46MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:58<03:39, 1.55MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:58<02:47, 2.03MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:58<02:03, 2.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [04:00<03:31, 1.60MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [04:00<03:21, 1.67MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [04:00<02:32, 2.21MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [04:00<01:52, 2.96MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [04:02<03:16, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [04:02<03:11, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [04:02<02:31, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [04:02<02:00, 2.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [04:02<01:37, 3.41MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:02<01:17, 4.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:04<03:51, 1.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:04<03:21, 1.63MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:04<02:28, 2.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:04<01:48, 3.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:06<07:45, 697kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:06<06:03, 892kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [04:06<04:22, 1.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:06<03:11, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:06<02:25, 2.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:08<04:53, 1.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:08<04:36, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:08<03:27, 1.54MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:08<02:35, 2.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:08<01:57, 2.71MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:08<01:33, 3.40MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:10<06:48, 774kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:10<05:57, 885kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:10<04:24, 1.19MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:10<03:13, 1.63MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:10<02:22, 2.20MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:12<04:17, 1.21MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:12<03:56, 1.32MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:12<03:02, 1.70MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:12<02:15, 2.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:12<01:39, 3.11MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:12<01:13, 4.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:15<08:34, 595kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:15<10:09, 501kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:15<08:22, 608kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:15<06:05, 834kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:15<04:22, 1.16MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:17<04:06, 1.22MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:17<03:20, 1.50MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:17<02:31, 1.98MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:17<01:50, 2.70MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:19<03:11, 1.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:19<03:14, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:19<02:31, 1.96MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:19<01:48, 2.71MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:21<4:42:40, 17.3kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:21<3:18:07, 24.6kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:21<2:18:05, 35.2kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:23<1:37:03, 49.6kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:23<1:08:20, 70.4kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:23<47:41, 100kB/s]   .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:25<34:15, 139kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:25<24:55, 190kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:25<17:36, 269kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:25<12:22, 381kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:27<09:54, 472kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:27<07:20, 637kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:27<05:22, 869kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:27<03:48, 1.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:29<04:23, 1.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:29<04:13, 1.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:29<03:10, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:29<02:21, 1.94MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:31<02:38, 1.72MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:31<02:00, 2.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:31<01:26, 3.11MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:33<04:00, 1.11MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:33<03:16, 1.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:33<02:22, 1.87MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:35<02:41, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:35<02:46, 1.58MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:35<02:10, 2.02MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:36<02:12, 1.97MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:37<01:54, 2.26MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:37<01:28, 2.94MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:37<01:04, 4.00MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:38<17:51, 239kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:39<12:55, 330kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:39<09:05, 466kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:40<07:17, 576kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:41<05:31, 758kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:41<03:57, 1.05MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:41<02:47, 1.48MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:42<32:36, 127kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:42<23:13, 178kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:43<16:17, 252kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:43<11:20, 358kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:44<38:05, 107kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:44<27:27, 148kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:45<19:21, 209kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:45<13:30, 297kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:46<10:50, 368kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:46<08:00, 498kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:47<05:40, 698kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:48<04:50, 810kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:48<03:47, 1.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:49<02:43, 1.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:50<02:48, 1.37MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:50<02:21, 1.63MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:50<01:43, 2.22MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:52<02:04, 1.82MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:52<02:05, 1.81MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:52<01:37, 2.31MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:53<01:14, 3.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:54<01:43, 2.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:54<01:35, 2.32MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:54<01:12, 3.05MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:56<01:41, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:56<01:58, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:56<01:32, 2.36MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:56<01:08, 3.17MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:58<02:03, 1.74MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:59<02:12, 1.61MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:59<01:43, 2.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:59<01:13, 2.85MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [05:00<05:57, 589kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [05:01<04:31, 773kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [05:01<03:13, 1.08MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [05:02<03:02, 1.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [05:02<02:28, 1.39MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [05:03<01:47, 1.90MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [05:03<01:19, 2.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:04<02:45, 1.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:04<02:21, 1.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:05<01:43, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:05<01:15, 2.63MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:06<02:34, 1.29MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:06<02:28, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:07<01:53, 1.73MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:07<01:40, 1.95MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:07<01:10, 2.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:09<04:37, 696kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:09<05:29, 584kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:09<04:23, 729kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:09<03:12, 994kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:10<02:19, 1.37MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:10<01:40, 1.88MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:11<03:24, 924kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:11<02:39, 1.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:11<01:55, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:11<01:22, 2.25MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:13<11:18, 272kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:13<08:16, 371kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:13<06:03, 506kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:13<04:20, 702kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:13<03:05, 980kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:14<02:13, 1.35MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:16<08:27, 355kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:16<08:22, 358kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:16<06:29, 462kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:16<04:39, 641kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:16<03:17, 899kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:18<03:03, 959kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:18<02:27, 1.19MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:18<01:48, 1.61MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:18<01:17, 2.22MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:20<02:21, 1.22MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:20<01:57, 1.46MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:20<01:25, 1.99MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:22<01:37, 1.73MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:22<01:42, 1.64MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:22<01:19, 2.09MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:22<00:56, 2.89MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:24<04:50, 564kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:24<03:40, 742kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:24<02:36, 1.03MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:25<02:25, 1.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:26<01:58, 1.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:26<01:25, 1.84MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:28<01:40, 1.55MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:28<01:42, 1.52MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:28<01:18, 1.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:28<00:56, 2.68MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:30<01:40, 1.51MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:30<01:26, 1.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:30<01:03, 2.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:30<00:45, 3.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:31<10:13, 240kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:32<07:40, 320kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:32<05:26, 447kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:32<03:49, 631kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:33<03:21, 710kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:34<02:32, 933kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:34<01:54, 1.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:34<01:21, 1.73MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:35<01:58, 1.17MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:36<01:51, 1.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:36<01:23, 1.65MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:36<01:00, 2.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:37<01:18, 1.71MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:38<01:09, 1.94MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:38<00:51, 2.59MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:39<01:05, 1.99MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:39<00:59, 2.20MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:40<00:44, 2.90MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:41<01:00, 2.09MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:41<00:55, 2.29MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:42<00:41, 2.98MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:42<00:30, 4.05MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:43<13:49, 148kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:43<09:49, 207kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:44<06:52, 293kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:44<04:44, 417kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:45<13:57, 141kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:45<09:56, 198kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:46<06:55, 280kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:47<05:12, 366kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:47<03:49, 496kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:47<02:41, 696kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:49<02:16, 805kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:49<01:46, 1.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:49<01:16, 1.42MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:50<00:53, 1.98MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:51<13:44, 129kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:51<09:45, 181kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:51<06:53, 254kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:51<04:45, 361kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:53<03:56, 431kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:53<02:55, 579kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:53<02:04, 807kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:53<01:27, 1.13MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:55<01:52, 866kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:55<01:33, 1.04MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:55<01:11, 1.35MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:55<00:52, 1.84MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:57<00:55, 1.68MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:57<00:47, 1.96MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:57<00:38, 2.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:57<00:28, 3.21MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:57<00:22, 4.05MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:59<01:04, 1.39MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:59<01:02, 1.43MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:59<00:51, 1.74MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:59<00:38, 2.31MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [06:01<00:43, 1.95MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [06:01<00:48, 1.75MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [06:01<00:37, 2.25MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [06:01<00:27, 3.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [06:03<00:45, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [06:03<00:39, 2.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [06:03<00:29, 2.71MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:05<00:38, 2.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:05<00:42, 1.81MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:05<00:32, 2.32MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:05<00:24, 3.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:07<00:38, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:07<00:34, 2.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:07<00:25, 2.81MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:09<00:33, 2.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:09<00:37, 1.84MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:09<00:28, 2.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:09<00:20, 3.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:11<00:55, 1.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:11<00:45, 1.42MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:11<00:32, 1.94MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:13<00:36, 1.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:13<00:37, 1.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:13<00:28, 2.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:13<00:20, 2.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:15<00:45, 1.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:15<00:37, 1.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:15<00:26, 2.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:17<00:30, 1.72MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:17<00:26, 1.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:17<00:19, 2.62MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:19<00:24, 1.98MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:19<00:21, 2.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:19<00:15, 2.91MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:21<00:21, 2.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:21<00:19, 2.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:21<00:13, 3.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:23<00:18, 2.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:23<00:17, 2.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:23<00:12, 3.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:25<00:16, 2.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:25<00:15, 2.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:25<00:11, 3.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:27<00:14, 2.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:27<00:12, 2.43MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:27<00:09, 3.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:29<00:12, 2.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:29<00:11, 2.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:29<00:08, 3.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:31<00:10, 2.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:31<00:09, 2.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:31<00:07, 3.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:32<00:09, 2.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:33<00:08, 2.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:33<00:05, 3.08MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:35<00:07, 2.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:35<00:08, 1.84MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:35<00:06, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:35<00:04, 3.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:37<00:06, 1.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:37<00:06, 1.67MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:37<00:04, 2.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:37<00:02, 2.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:37<00:01, 3.99MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:40<00:04, 866kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:41<00:05, 615kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:41<00:04, 727kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:41<00:02, 982kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:41<00:00, 1.37MB/s].vector_cache/glove.6B.zip: 862MB [06:41, 2.15MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 761/400000 [00:00<00:52, 7604.71it/s]  0%|          | 1510/400000 [00:00<00:52, 7565.37it/s]  1%|          | 2251/400000 [00:00<00:52, 7517.39it/s]  1%|          | 3025/400000 [00:00<00:52, 7580.89it/s]  1%|          | 3764/400000 [00:00<00:52, 7522.02it/s]  1%|          | 4573/400000 [00:00<00:51, 7681.46it/s]  1%|▏         | 5336/400000 [00:00<00:51, 7663.53it/s]  2%|▏         | 6066/400000 [00:00<00:52, 7549.61it/s]  2%|▏         | 6779/400000 [00:00<00:53, 7399.59it/s]  2%|▏         | 7529/400000 [00:01<00:52, 7427.36it/s]  2%|▏         | 8252/400000 [00:01<00:54, 7242.59it/s]  2%|▏         | 9001/400000 [00:01<00:53, 7314.02it/s]  2%|▏         | 9747/400000 [00:01<00:53, 7355.17it/s]  3%|▎         | 10499/400000 [00:01<00:52, 7403.48it/s]  3%|▎         | 11253/400000 [00:01<00:52, 7442.91it/s]  3%|▎         | 11995/400000 [00:01<00:52, 7369.93it/s]  3%|▎         | 12751/400000 [00:01<00:52, 7424.68it/s]  3%|▎         | 13563/400000 [00:01<00:50, 7619.09it/s]  4%|▎         | 14348/400000 [00:01<00:50, 7685.47it/s]  4%|▍         | 15130/400000 [00:02<00:49, 7723.92it/s]  4%|▍         | 15929/400000 [00:02<00:49, 7801.86it/s]  4%|▍         | 16710/400000 [00:02<00:49, 7712.71it/s]  4%|▍         | 17491/400000 [00:02<00:49, 7738.02it/s]  5%|▍         | 18285/400000 [00:02<00:48, 7797.39it/s]  5%|▍         | 19066/400000 [00:02<00:49, 7667.03it/s]  5%|▍         | 19855/400000 [00:02<00:49, 7730.94it/s]  5%|▌         | 20629/400000 [00:02<00:49, 7711.43it/s]  5%|▌         | 21401/400000 [00:02<00:49, 7650.01it/s]  6%|▌         | 22167/400000 [00:02<00:49, 7609.53it/s]  6%|▌         | 22948/400000 [00:03<00:49, 7666.05it/s]  6%|▌         | 23715/400000 [00:03<00:49, 7620.54it/s]  6%|▌         | 24478/400000 [00:03<00:50, 7490.51it/s]  6%|▋         | 25228/400000 [00:03<00:50, 7490.30it/s]  6%|▋         | 25996/400000 [00:03<00:49, 7546.21it/s]  7%|▋         | 26788/400000 [00:03<00:48, 7652.20it/s]  7%|▋         | 27554/400000 [00:03<00:49, 7596.23it/s]  7%|▋         | 28315/400000 [00:03<00:49, 7580.44it/s]  7%|▋         | 29116/400000 [00:03<00:48, 7701.85it/s]  7%|▋         | 29907/400000 [00:03<00:47, 7761.41it/s]  8%|▊         | 30684/400000 [00:04<00:47, 7739.15it/s]  8%|▊         | 31475/400000 [00:04<00:47, 7788.30it/s]  8%|▊         | 32255/400000 [00:04<00:47, 7753.65it/s]  8%|▊         | 33031/400000 [00:04<00:47, 7749.19it/s]  8%|▊         | 33821/400000 [00:04<00:46, 7793.72it/s]  9%|▊         | 34620/400000 [00:04<00:46, 7851.30it/s]  9%|▉         | 35418/400000 [00:04<00:46, 7886.47it/s]  9%|▉         | 36211/400000 [00:04<00:46, 7898.93it/s]  9%|▉         | 37002/400000 [00:04<00:46, 7819.23it/s]  9%|▉         | 37785/400000 [00:04<00:46, 7788.96it/s] 10%|▉         | 38565/400000 [00:05<00:46, 7783.50it/s] 10%|▉         | 39355/400000 [00:05<00:46, 7817.43it/s] 10%|█         | 40137/400000 [00:05<00:46, 7751.69it/s] 10%|█         | 40923/400000 [00:05<00:46, 7782.15it/s] 10%|█         | 41702/400000 [00:05<00:46, 7778.38it/s] 11%|█         | 42485/400000 [00:05<00:45, 7791.90it/s] 11%|█         | 43265/400000 [00:05<00:45, 7776.15it/s] 11%|█         | 44043/400000 [00:05<00:46, 7679.57it/s] 11%|█         | 44812/400000 [00:05<00:46, 7624.54it/s] 11%|█▏        | 45606/400000 [00:05<00:45, 7715.95it/s] 12%|█▏        | 46379/400000 [00:06<00:46, 7552.70it/s] 12%|█▏        | 47172/400000 [00:06<00:46, 7659.65it/s] 12%|█▏        | 47940/400000 [00:06<00:46, 7639.58it/s] 12%|█▏        | 48718/400000 [00:06<00:45, 7679.90it/s] 12%|█▏        | 49513/400000 [00:06<00:45, 7758.53it/s] 13%|█▎        | 50290/400000 [00:06<00:45, 7677.56it/s] 13%|█▎        | 51071/400000 [00:06<00:45, 7716.38it/s] 13%|█▎        | 51844/400000 [00:06<00:45, 7657.52it/s] 13%|█▎        | 52618/400000 [00:06<00:45, 7681.00it/s] 13%|█▎        | 53387/400000 [00:06<00:45, 7663.41it/s] 14%|█▎        | 54181/400000 [00:07<00:44, 7744.19it/s] 14%|█▎        | 54967/400000 [00:07<00:44, 7776.79it/s] 14%|█▍        | 55745/400000 [00:07<00:45, 7636.63it/s] 14%|█▍        | 56510/400000 [00:07<00:45, 7621.92it/s] 14%|█▍        | 57298/400000 [00:07<00:44, 7695.16it/s] 15%|█▍        | 58069/400000 [00:07<00:45, 7590.32it/s] 15%|█▍        | 58829/400000 [00:07<00:45, 7548.50it/s] 15%|█▍        | 59585/400000 [00:07<00:45, 7481.18it/s] 15%|█▌        | 60334/400000 [00:07<00:46, 7282.92it/s] 15%|█▌        | 61076/400000 [00:08<00:46, 7323.22it/s] 15%|█▌        | 61810/400000 [00:08<00:46, 7324.27it/s] 16%|█▌        | 62544/400000 [00:08<00:46, 7245.09it/s] 16%|█▌        | 63270/400000 [00:08<00:49, 6833.58it/s] 16%|█▌        | 64012/400000 [00:08<00:48, 6998.39it/s] 16%|█▌        | 64768/400000 [00:08<00:46, 7157.52it/s] 16%|█▋        | 65508/400000 [00:08<00:46, 7227.44it/s] 17%|█▋        | 66272/400000 [00:08<00:45, 7345.27it/s] 17%|█▋        | 67041/400000 [00:08<00:44, 7444.51it/s] 17%|█▋        | 67821/400000 [00:08<00:44, 7545.59it/s] 17%|█▋        | 68595/400000 [00:09<00:43, 7602.48it/s] 17%|█▋        | 69384/400000 [00:09<00:43, 7684.90it/s] 18%|█▊        | 70164/400000 [00:09<00:42, 7717.36it/s] 18%|█▊        | 70937/400000 [00:09<00:42, 7706.23it/s] 18%|█▊        | 71754/400000 [00:09<00:41, 7838.19it/s] 18%|█▊        | 72552/400000 [00:09<00:41, 7878.19it/s] 18%|█▊        | 73378/400000 [00:09<00:40, 7988.79it/s] 19%|█▊        | 74178/400000 [00:09<00:42, 7692.47it/s] 19%|█▊        | 74962/400000 [00:09<00:42, 7735.05it/s] 19%|█▉        | 75744/400000 [00:09<00:41, 7760.34it/s] 19%|█▉        | 76522/400000 [00:10<00:41, 7751.55it/s] 19%|█▉        | 77340/400000 [00:10<00:40, 7870.70it/s] 20%|█▉        | 78132/400000 [00:10<00:40, 7884.20it/s] 20%|█▉        | 78922/400000 [00:10<00:40, 7857.62it/s] 20%|█▉        | 79724/400000 [00:10<00:40, 7904.71it/s] 20%|██        | 80515/400000 [00:10<00:40, 7900.87it/s] 20%|██        | 81319/400000 [00:10<00:40, 7939.79it/s] 21%|██        | 82114/400000 [00:10<00:40, 7906.83it/s] 21%|██        | 82905/400000 [00:10<00:42, 7527.04it/s] 21%|██        | 83665/400000 [00:10<00:41, 7546.27it/s] 21%|██        | 84423/400000 [00:11<00:42, 7402.48it/s] 21%|██▏       | 85193/400000 [00:11<00:42, 7488.89it/s] 21%|██▏       | 85949/400000 [00:11<00:41, 7507.94it/s] 22%|██▏       | 86702/400000 [00:11<00:42, 7427.43it/s] 22%|██▏       | 87446/400000 [00:11<00:42, 7360.41it/s] 22%|██▏       | 88186/400000 [00:11<00:42, 7371.47it/s] 22%|██▏       | 88924/400000 [00:11<00:42, 7366.01it/s] 22%|██▏       | 89717/400000 [00:11<00:41, 7525.02it/s] 23%|██▎       | 90471/400000 [00:11<00:41, 7524.63it/s] 23%|██▎       | 91245/400000 [00:11<00:40, 7586.71it/s] 23%|██▎       | 92005/400000 [00:12<00:40, 7580.14it/s] 23%|██▎       | 92803/400000 [00:12<00:39, 7694.31it/s] 23%|██▎       | 93576/400000 [00:12<00:39, 7700.92it/s] 24%|██▎       | 94347/400000 [00:12<00:39, 7661.68it/s] 24%|██▍       | 95120/400000 [00:12<00:39, 7682.00it/s] 24%|██▍       | 95918/400000 [00:12<00:39, 7766.97it/s] 24%|██▍       | 96699/400000 [00:12<00:38, 7778.05it/s] 24%|██▍       | 97478/400000 [00:12<00:40, 7540.43it/s] 25%|██▍       | 98234/400000 [00:12<00:40, 7475.58it/s] 25%|██▍       | 99026/400000 [00:12<00:39, 7602.68it/s] 25%|██▍       | 99798/400000 [00:13<00:39, 7635.87it/s] 25%|██▌       | 100598/400000 [00:13<00:38, 7739.67it/s] 25%|██▌       | 101374/400000 [00:13<00:38, 7727.00it/s] 26%|██▌       | 102148/400000 [00:13<00:40, 7355.26it/s] 26%|██▌       | 102888/400000 [00:13<00:41, 7211.67it/s] 26%|██▌       | 103671/400000 [00:13<00:40, 7386.52it/s] 26%|██▌       | 104469/400000 [00:13<00:39, 7553.73it/s] 26%|██▋       | 105241/400000 [00:13<00:38, 7601.02it/s] 27%|██▋       | 106004/400000 [00:13<00:38, 7578.56it/s] 27%|██▋       | 106790/400000 [00:14<00:38, 7659.26it/s] 27%|██▋       | 107560/400000 [00:14<00:38, 7670.21it/s] 27%|██▋       | 108347/400000 [00:14<00:37, 7724.94it/s] 27%|██▋       | 109121/400000 [00:14<00:37, 7697.45it/s] 27%|██▋       | 109892/400000 [00:14<00:38, 7617.34it/s] 28%|██▊       | 110660/400000 [00:14<00:37, 7635.73it/s] 28%|██▊       | 111425/400000 [00:14<00:37, 7619.76it/s] 28%|██▊       | 112203/400000 [00:14<00:37, 7666.49it/s] 28%|██▊       | 112970/400000 [00:14<00:37, 7657.67it/s] 28%|██▊       | 113736/400000 [00:14<00:37, 7557.01it/s] 29%|██▊       | 114493/400000 [00:15<00:38, 7442.97it/s] 29%|██▉       | 115278/400000 [00:15<00:37, 7558.67it/s] 29%|██▉       | 116040/400000 [00:15<00:37, 7576.78it/s] 29%|██▉       | 116804/400000 [00:15<00:37, 7593.76it/s] 29%|██▉       | 117564/400000 [00:15<00:37, 7572.44it/s] 30%|██▉       | 118322/400000 [00:15<00:37, 7531.04it/s] 30%|██▉       | 119087/400000 [00:15<00:37, 7564.78it/s] 30%|██▉       | 119856/400000 [00:15<00:36, 7599.01it/s] 30%|███       | 120617/400000 [00:15<00:37, 7533.29it/s] 30%|███       | 121371/400000 [00:15<00:37, 7508.55it/s] 31%|███       | 122123/400000 [00:16<00:36, 7511.60it/s] 31%|███       | 122896/400000 [00:16<00:36, 7575.30it/s] 31%|███       | 123656/400000 [00:16<00:36, 7580.28it/s] 31%|███       | 124421/400000 [00:16<00:36, 7600.09it/s] 31%|███▏      | 125182/400000 [00:16<00:36, 7521.84it/s] 31%|███▏      | 125949/400000 [00:16<00:36, 7562.91it/s] 32%|███▏      | 126726/400000 [00:16<00:35, 7622.29it/s] 32%|███▏      | 127491/400000 [00:16<00:35, 7629.52it/s] 32%|███▏      | 128259/400000 [00:16<00:35, 7644.49it/s] 32%|███▏      | 129024/400000 [00:16<00:35, 7599.84it/s] 32%|███▏      | 129798/400000 [00:17<00:35, 7638.78it/s] 33%|███▎      | 130580/400000 [00:17<00:35, 7691.99it/s] 33%|███▎      | 131350/400000 [00:17<00:34, 7687.64it/s] 33%|███▎      | 132123/400000 [00:17<00:34, 7699.20it/s] 33%|███▎      | 132894/400000 [00:17<00:34, 7644.87it/s] 33%|███▎      | 133659/400000 [00:17<00:35, 7435.41it/s] 34%|███▎      | 134445/400000 [00:17<00:35, 7557.06it/s] 34%|███▍      | 135230/400000 [00:17<00:34, 7640.25it/s] 34%|███▍      | 135996/400000 [00:17<00:34, 7554.74it/s] 34%|███▍      | 136753/400000 [00:17<00:35, 7336.07it/s] 34%|███▍      | 137509/400000 [00:18<00:35, 7401.10it/s] 35%|███▍      | 138309/400000 [00:18<00:34, 7567.00it/s] 35%|███▍      | 139068/400000 [00:18<00:34, 7527.99it/s] 35%|███▍      | 139833/400000 [00:18<00:34, 7562.55it/s] 35%|███▌      | 140591/400000 [00:18<00:34, 7553.63it/s] 35%|███▌      | 141381/400000 [00:18<00:33, 7653.57it/s] 36%|███▌      | 142175/400000 [00:18<00:33, 7735.58it/s] 36%|███▌      | 142959/400000 [00:18<00:33, 7764.11it/s] 36%|███▌      | 143755/400000 [00:18<00:32, 7820.65it/s] 36%|███▌      | 144538/400000 [00:18<00:33, 7527.61it/s] 36%|███▋      | 145294/400000 [00:19<00:34, 7355.10it/s] 37%|███▋      | 146101/400000 [00:19<00:33, 7555.03it/s] 37%|███▋      | 146860/400000 [00:19<00:33, 7541.29it/s] 37%|███▋      | 147632/400000 [00:19<00:33, 7592.14it/s] 37%|███▋      | 148393/400000 [00:19<00:33, 7541.99it/s] 37%|███▋      | 149162/400000 [00:19<00:33, 7585.02it/s] 37%|███▋      | 149922/400000 [00:19<00:33, 7569.56it/s] 38%|███▊      | 150711/400000 [00:19<00:32, 7660.84it/s] 38%|███▊      | 151478/400000 [00:19<00:32, 7661.90it/s] 38%|███▊      | 152245/400000 [00:20<00:32, 7625.13it/s] 38%|███▊      | 153036/400000 [00:20<00:32, 7706.30it/s] 38%|███▊      | 153814/400000 [00:20<00:31, 7726.11it/s] 39%|███▊      | 154587/400000 [00:20<00:32, 7535.67it/s] 39%|███▉      | 155369/400000 [00:20<00:32, 7615.82it/s] 39%|███▉      | 156132/400000 [00:20<00:32, 7490.66it/s] 39%|███▉      | 156891/400000 [00:20<00:32, 7517.84it/s] 39%|███▉      | 157660/400000 [00:20<00:32, 7567.25it/s] 40%|███▉      | 158418/400000 [00:20<00:33, 7296.84it/s] 40%|███▉      | 159211/400000 [00:20<00:32, 7474.07it/s] 40%|███▉      | 159983/400000 [00:21<00:31, 7543.94it/s] 40%|████      | 160783/400000 [00:21<00:31, 7674.31it/s] 40%|████      | 161556/400000 [00:21<00:31, 7688.85it/s] 41%|████      | 162327/400000 [00:21<00:31, 7517.84it/s] 41%|████      | 163103/400000 [00:21<00:31, 7585.67it/s] 41%|████      | 163869/400000 [00:21<00:31, 7607.40it/s] 41%|████      | 164665/400000 [00:21<00:30, 7707.83it/s] 41%|████▏     | 165462/400000 [00:21<00:30, 7783.64it/s] 42%|████▏     | 166247/400000 [00:21<00:29, 7802.54it/s] 42%|████▏     | 167028/400000 [00:21<00:29, 7772.70it/s] 42%|████▏     | 167806/400000 [00:22<00:30, 7637.33it/s] 42%|████▏     | 168575/400000 [00:22<00:30, 7651.38it/s] 42%|████▏     | 169348/400000 [00:22<00:30, 7670.26it/s] 43%|████▎     | 170116/400000 [00:22<00:30, 7479.94it/s] 43%|████▎     | 170905/400000 [00:22<00:30, 7596.95it/s] 43%|████▎     | 171667/400000 [00:22<00:30, 7472.15it/s] 43%|████▎     | 172441/400000 [00:22<00:30, 7549.65it/s] 43%|████▎     | 173215/400000 [00:22<00:29, 7605.63it/s] 43%|████▎     | 173983/400000 [00:22<00:29, 7627.24it/s] 44%|████▎     | 174750/400000 [00:22<00:29, 7633.57it/s] 44%|████▍     | 175514/400000 [00:23<00:29, 7604.43it/s] 44%|████▍     | 176279/400000 [00:23<00:29, 7616.60it/s] 44%|████▍     | 177041/400000 [00:23<00:29, 7529.38it/s] 44%|████▍     | 177795/400000 [00:23<00:29, 7514.98it/s] 45%|████▍     | 178587/400000 [00:23<00:29, 7630.46it/s] 45%|████▍     | 179351/400000 [00:23<00:29, 7594.62it/s] 45%|████▌     | 180139/400000 [00:23<00:28, 7675.78it/s] 45%|████▌     | 180935/400000 [00:23<00:28, 7757.68it/s] 45%|████▌     | 181726/400000 [00:23<00:27, 7800.50it/s] 46%|████▌     | 182510/400000 [00:23<00:27, 7811.70it/s] 46%|████▌     | 183292/400000 [00:24<00:28, 7590.08it/s] 46%|████▌     | 184053/400000 [00:24<00:28, 7535.98it/s] 46%|████▌     | 184808/400000 [00:24<00:28, 7498.59it/s] 46%|████▋     | 185609/400000 [00:24<00:28, 7644.98it/s] 47%|████▋     | 186405/400000 [00:24<00:27, 7734.53it/s] 47%|████▋     | 187180/400000 [00:24<00:27, 7694.97it/s] 47%|████▋     | 187960/400000 [00:24<00:27, 7723.67it/s] 47%|████▋     | 188734/400000 [00:24<00:27, 7642.01it/s] 47%|████▋     | 189499/400000 [00:24<00:27, 7634.22it/s] 48%|████▊     | 190263/400000 [00:24<00:27, 7585.34it/s] 48%|████▊     | 191041/400000 [00:25<00:27, 7640.85it/s] 48%|████▊     | 191821/400000 [00:25<00:27, 7685.88it/s] 48%|████▊     | 192606/400000 [00:25<00:26, 7733.01it/s] 48%|████▊     | 193408/400000 [00:25<00:26, 7815.29it/s] 49%|████▊     | 194198/400000 [00:25<00:26, 7838.73it/s] 49%|████▊     | 194983/400000 [00:25<00:26, 7709.51it/s] 49%|████▉     | 195755/400000 [00:25<00:26, 7712.21it/s] 49%|████▉     | 196527/400000 [00:25<00:26, 7622.06it/s] 49%|████▉     | 197307/400000 [00:25<00:26, 7672.09it/s] 50%|████▉     | 198105/400000 [00:26<00:26, 7759.09it/s] 50%|████▉     | 198882/400000 [00:26<00:26, 7707.62it/s] 50%|████▉     | 199658/400000 [00:26<00:25, 7720.87it/s] 50%|█████     | 200431/400000 [00:26<00:26, 7623.27it/s] 50%|█████     | 201194/400000 [00:26<00:26, 7522.37it/s] 50%|█████     | 201947/400000 [00:26<00:26, 7480.82it/s] 51%|█████     | 202696/400000 [00:26<00:26, 7424.03it/s] 51%|█████     | 203439/400000 [00:26<00:26, 7417.77it/s] 51%|█████     | 204208/400000 [00:26<00:26, 7494.85it/s] 51%|█████▏    | 205008/400000 [00:26<00:25, 7637.06it/s] 51%|█████▏    | 205791/400000 [00:27<00:25, 7693.83it/s] 52%|█████▏    | 206562/400000 [00:27<00:25, 7680.34it/s] 52%|█████▏    | 207337/400000 [00:27<00:25, 7699.11it/s] 52%|█████▏    | 208108/400000 [00:27<00:25, 7656.93it/s] 52%|█████▏    | 208875/400000 [00:27<00:25, 7611.50it/s] 52%|█████▏    | 209666/400000 [00:27<00:24, 7696.05it/s] 53%|█████▎    | 210437/400000 [00:27<00:24, 7597.85it/s] 53%|█████▎    | 211246/400000 [00:27<00:24, 7738.68it/s] 53%|█████▎    | 212021/400000 [00:27<00:24, 7687.47it/s] 53%|█████▎    | 212806/400000 [00:27<00:24, 7735.25it/s] 53%|█████▎    | 213585/400000 [00:28<00:24, 7750.19it/s] 54%|█████▎    | 214361/400000 [00:28<00:24, 7626.06it/s] 54%|█████▍    | 215125/400000 [00:28<00:24, 7583.03it/s] 54%|█████▍    | 215884/400000 [00:28<00:24, 7498.16it/s] 54%|█████▍    | 216635/400000 [00:28<00:24, 7491.33it/s] 54%|█████▍    | 217414/400000 [00:28<00:24, 7576.46it/s] 55%|█████▍    | 218173/400000 [00:28<00:24, 7531.39it/s] 55%|█████▍    | 218968/400000 [00:28<00:23, 7650.12it/s] 55%|█████▍    | 219734/400000 [00:28<00:23, 7579.85it/s] 55%|█████▌    | 220508/400000 [00:28<00:23, 7626.87it/s] 55%|█████▌    | 221284/400000 [00:29<00:23, 7663.57it/s] 56%|█████▌    | 222053/400000 [00:29<00:23, 7670.88it/s] 56%|█████▌    | 222845/400000 [00:29<00:22, 7741.42it/s] 56%|█████▌    | 223625/400000 [00:29<00:22, 7756.77it/s] 56%|█████▌    | 224401/400000 [00:29<00:22, 7688.08it/s] 56%|█████▋    | 225171/400000 [00:29<00:22, 7676.08it/s] 56%|█████▋    | 225939/400000 [00:29<00:22, 7637.95it/s] 57%|█████▋    | 226712/400000 [00:29<00:22, 7664.46it/s] 57%|█████▋    | 227482/400000 [00:29<00:22, 7672.25it/s] 57%|█████▋    | 228250/400000 [00:29<00:22, 7567.61it/s] 57%|█████▋    | 229027/400000 [00:30<00:22, 7626.24it/s] 57%|█████▋    | 229791/400000 [00:30<00:22, 7621.08it/s] 58%|█████▊    | 230556/400000 [00:30<00:22, 7628.25it/s] 58%|█████▊    | 231321/400000 [00:30<00:22, 7633.71it/s] 58%|█████▊    | 232108/400000 [00:30<00:21, 7702.34it/s] 58%|█████▊    | 232879/400000 [00:30<00:21, 7693.67it/s] 58%|█████▊    | 233649/400000 [00:30<00:21, 7634.34it/s] 59%|█████▊    | 234415/400000 [00:30<00:21, 7640.21it/s] 59%|█████▉    | 235180/400000 [00:30<00:21, 7587.55it/s] 59%|█████▉    | 235939/400000 [00:30<00:22, 7394.38it/s] 59%|█████▉    | 236692/400000 [00:31<00:21, 7432.33it/s] 59%|█████▉    | 237446/400000 [00:31<00:21, 7462.57it/s] 60%|█████▉    | 238193/400000 [00:31<00:21, 7461.19it/s] 60%|█████▉    | 238940/400000 [00:31<00:23, 6950.74it/s] 60%|█████▉    | 239644/400000 [00:31<00:22, 6975.20it/s] 60%|██████    | 240352/400000 [00:31<00:22, 7005.28it/s] 60%|██████    | 241057/400000 [00:31<00:23, 6910.45it/s] 60%|██████    | 241793/400000 [00:31<00:22, 7037.96it/s] 61%|██████    | 242559/400000 [00:31<00:21, 7211.56it/s] 61%|██████    | 243297/400000 [00:32<00:21, 7258.88it/s] 61%|██████    | 244075/400000 [00:32<00:21, 7405.04it/s] 61%|██████    | 244844/400000 [00:32<00:20, 7487.48it/s] 61%|██████▏   | 245625/400000 [00:32<00:20, 7579.95it/s] 62%|██████▏   | 246420/400000 [00:32<00:19, 7687.16it/s] 62%|██████▏   | 247191/400000 [00:32<00:19, 7664.06it/s] 62%|██████▏   | 247959/400000 [00:32<00:20, 7556.76it/s] 62%|██████▏   | 248735/400000 [00:32<00:19, 7614.66it/s] 62%|██████▏   | 249543/400000 [00:32<00:19, 7748.25it/s] 63%|██████▎   | 250319/400000 [00:32<00:19, 7732.99it/s] 63%|██████▎   | 251107/400000 [00:33<00:19, 7776.30it/s] 63%|██████▎   | 251904/400000 [00:33<00:18, 7832.31it/s] 63%|██████▎   | 252691/400000 [00:33<00:18, 7843.14it/s] 63%|██████▎   | 253476/400000 [00:33<00:18, 7758.79it/s] 64%|██████▎   | 254265/400000 [00:33<00:18, 7796.21it/s] 64%|██████▍   | 255045/400000 [00:33<00:19, 7464.28it/s] 64%|██████▍   | 255795/400000 [00:33<00:19, 7443.50it/s] 64%|██████▍   | 256542/400000 [00:33<00:19, 7448.72it/s] 64%|██████▍   | 257289/400000 [00:33<00:19, 7434.41it/s] 65%|██████▍   | 258053/400000 [00:33<00:18, 7494.87it/s] 65%|██████▍   | 258804/400000 [00:34<00:18, 7494.56it/s] 65%|██████▍   | 259555/400000 [00:34<00:19, 7362.67it/s] 65%|██████▌   | 260293/400000 [00:34<00:19, 7178.49it/s] 65%|██████▌   | 261055/400000 [00:34<00:19, 7303.76it/s] 65%|██████▌   | 261807/400000 [00:34<00:18, 7365.08it/s] 66%|██████▌   | 262585/400000 [00:34<00:18, 7483.42it/s] 66%|██████▌   | 263346/400000 [00:34<00:18, 7518.95it/s] 66%|██████▌   | 264136/400000 [00:34<00:17, 7629.05it/s] 66%|██████▌   | 264928/400000 [00:34<00:17, 7706.15it/s] 66%|██████▋   | 265732/400000 [00:34<00:17, 7801.81it/s] 67%|██████▋   | 266514/400000 [00:35<00:17, 7792.84it/s] 67%|██████▋   | 267328/400000 [00:35<00:16, 7892.15it/s] 67%|██████▋   | 268118/400000 [00:35<00:16, 7885.19it/s] 67%|██████▋   | 268955/400000 [00:35<00:16, 8022.24it/s] 67%|██████▋   | 269794/400000 [00:35<00:16, 8129.00it/s] 68%|██████▊   | 270608/400000 [00:35<00:16, 8079.74it/s] 68%|██████▊   | 271417/400000 [00:35<00:16, 7817.91it/s] 68%|██████▊   | 272202/400000 [00:35<00:17, 7421.63it/s] 68%|██████▊   | 272951/400000 [00:35<00:17, 7297.65it/s] 68%|██████▊   | 273695/400000 [00:35<00:17, 7336.49it/s] 69%|██████▊   | 274432/400000 [00:36<00:17, 7283.60it/s] 69%|██████▉   | 275163/400000 [00:36<00:17, 7284.00it/s] 69%|██████▉   | 275921/400000 [00:36<00:16, 7368.65it/s] 69%|██████▉   | 276729/400000 [00:36<00:16, 7567.74it/s] 69%|██████▉   | 277532/400000 [00:36<00:15, 7699.36it/s] 70%|██████▉   | 278356/400000 [00:36<00:15, 7852.40it/s] 70%|██████▉   | 279167/400000 [00:36<00:15, 7924.76it/s] 70%|██████▉   | 279962/400000 [00:36<00:15, 7916.91it/s] 70%|███████   | 280792/400000 [00:36<00:14, 8026.10it/s] 70%|███████   | 281606/400000 [00:36<00:14, 8058.40it/s] 71%|███████   | 282413/400000 [00:37<00:14, 8039.53it/s] 71%|███████   | 283238/400000 [00:37<00:14, 8101.31it/s] 71%|███████   | 284049/400000 [00:37<00:14, 7939.77it/s] 71%|███████   | 284845/400000 [00:37<00:14, 7922.37it/s] 71%|███████▏  | 285667/400000 [00:37<00:14, 8006.55it/s] 72%|███████▏  | 286482/400000 [00:37<00:14, 8046.79it/s] 72%|███████▏  | 287324/400000 [00:37<00:13, 8152.83it/s] 72%|███████▏  | 288141/400000 [00:37<00:13, 8086.87it/s] 72%|███████▏  | 288954/400000 [00:37<00:13, 8099.29it/s] 72%|███████▏  | 289765/400000 [00:38<00:14, 7855.08it/s] 73%|███████▎  | 290553/400000 [00:38<00:13, 7854.87it/s] 73%|███████▎  | 291340/400000 [00:38<00:13, 7842.10it/s] 73%|███████▎  | 292126/400000 [00:38<00:14, 7683.08it/s] 73%|███████▎  | 292934/400000 [00:38<00:13, 7796.51it/s] 73%|███████▎  | 293745/400000 [00:38<00:13, 7886.26it/s] 74%|███████▎  | 294535/400000 [00:38<00:13, 7816.56it/s] 74%|███████▍  | 295318/400000 [00:38<00:13, 7797.15it/s] 74%|███████▍  | 296099/400000 [00:38<00:13, 7705.34it/s] 74%|███████▍  | 296890/400000 [00:38<00:13, 7764.14it/s] 74%|███████▍  | 297692/400000 [00:39<00:13, 7837.68it/s] 75%|███████▍  | 298525/400000 [00:39<00:12, 7977.94it/s] 75%|███████▍  | 299339/400000 [00:39<00:12, 8023.48it/s] 75%|███████▌  | 300143/400000 [00:39<00:12, 7926.57it/s] 75%|███████▌  | 300937/400000 [00:39<00:12, 7915.47it/s] 75%|███████▌  | 301738/400000 [00:39<00:12, 7943.46it/s] 76%|███████▌  | 302533/400000 [00:39<00:12, 7933.44it/s] 76%|███████▌  | 303327/400000 [00:39<00:12, 7920.63it/s] 76%|███████▌  | 304120/400000 [00:39<00:12, 7843.79it/s] 76%|███████▌  | 304905/400000 [00:39<00:12, 7696.60it/s] 76%|███████▋  | 305676/400000 [00:40<00:12, 7622.35it/s] 77%|███████▋  | 306439/400000 [00:40<00:12, 7601.71it/s] 77%|███████▋  | 307204/400000 [00:40<00:12, 7615.62it/s] 77%|███████▋  | 307966/400000 [00:40<00:12, 7170.39it/s] 77%|███████▋  | 308734/400000 [00:40<00:12, 7314.25it/s] 77%|███████▋  | 309501/400000 [00:40<00:12, 7416.40it/s] 78%|███████▊  | 310247/400000 [00:40<00:12, 7381.59it/s] 78%|███████▊  | 310989/400000 [00:40<00:12, 7390.48it/s] 78%|███████▊  | 311730/400000 [00:40<00:12, 7132.73it/s] 78%|███████▊  | 312459/400000 [00:40<00:12, 7177.88it/s] 78%|███████▊  | 313196/400000 [00:41<00:12, 7231.50it/s] 78%|███████▊  | 313950/400000 [00:41<00:11, 7320.07it/s] 79%|███████▊  | 314697/400000 [00:41<00:11, 7362.33it/s] 79%|███████▉  | 315435/400000 [00:41<00:11, 7350.38it/s] 79%|███████▉  | 316199/400000 [00:41<00:11, 7432.37it/s] 79%|███████▉  | 316946/400000 [00:41<00:11, 7442.77it/s] 79%|███████▉  | 317714/400000 [00:41<00:10, 7511.68it/s] 80%|███████▉  | 318498/400000 [00:41<00:10, 7607.23it/s] 80%|███████▉  | 319267/400000 [00:41<00:10, 7630.30it/s] 80%|████████  | 320081/400000 [00:41<00:10, 7776.17it/s] 80%|████████  | 320860/400000 [00:42<00:10, 7775.82it/s] 80%|████████  | 321658/400000 [00:42<00:10, 7833.46it/s] 81%|████████  | 322460/400000 [00:42<00:09, 7886.62it/s] 81%|████████  | 323250/400000 [00:42<00:10, 7642.92it/s] 81%|████████  | 324017/400000 [00:42<00:10, 7581.21it/s] 81%|████████  | 324801/400000 [00:42<00:09, 7654.48it/s] 81%|████████▏ | 325580/400000 [00:42<00:09, 7693.89it/s] 82%|████████▏ | 326396/400000 [00:42<00:09, 7826.74it/s] 82%|████████▏ | 327180/400000 [00:42<00:09, 7645.63it/s] 82%|████████▏ | 327947/400000 [00:43<00:09, 7452.13it/s] 82%|████████▏ | 328705/400000 [00:43<00:09, 7487.11it/s] 82%|████████▏ | 329494/400000 [00:43<00:09, 7603.15it/s] 83%|████████▎ | 330268/400000 [00:43<00:09, 7640.97it/s] 83%|████████▎ | 331038/400000 [00:43<00:09, 7658.34it/s] 83%|████████▎ | 331827/400000 [00:43<00:08, 7723.96it/s] 83%|████████▎ | 332646/400000 [00:43<00:08, 7856.75it/s] 83%|████████▎ | 333433/400000 [00:43<00:08, 7764.31it/s] 84%|████████▎ | 334211/400000 [00:43<00:08, 7760.93it/s] 84%|████████▎ | 334988/400000 [00:43<00:08, 7655.04it/s] 84%|████████▍ | 335762/400000 [00:44<00:08, 7677.88it/s] 84%|████████▍ | 336597/400000 [00:44<00:08, 7867.86it/s] 84%|████████▍ | 337444/400000 [00:44<00:07, 8037.89it/s] 85%|████████▍ | 338250/400000 [00:44<00:07, 8032.40it/s] 85%|████████▍ | 339055/400000 [00:44<00:07, 7874.70it/s] 85%|████████▍ | 339845/400000 [00:44<00:07, 7666.97it/s] 85%|████████▌ | 340655/400000 [00:44<00:07, 7790.23it/s] 85%|████████▌ | 341459/400000 [00:44<00:07, 7863.50it/s] 86%|████████▌ | 342253/400000 [00:44<00:07, 7883.75it/s] 86%|████████▌ | 343043/400000 [00:44<00:07, 7587.87it/s] 86%|████████▌ | 343805/400000 [00:45<00:07, 7541.75it/s] 86%|████████▌ | 344562/400000 [00:45<00:07, 7517.78it/s] 86%|████████▋ | 345316/400000 [00:45<00:07, 7516.07it/s] 87%|████████▋ | 346069/400000 [00:45<00:07, 7493.69it/s] 87%|████████▋ | 346820/400000 [00:45<00:07, 7448.61it/s] 87%|████████▋ | 347581/400000 [00:45<00:06, 7495.03it/s] 87%|████████▋ | 348397/400000 [00:45<00:06, 7682.43it/s] 87%|████████▋ | 349167/400000 [00:45<00:06, 7683.85it/s] 87%|████████▋ | 349937/400000 [00:45<00:06, 7671.17it/s] 88%|████████▊ | 350709/400000 [00:45<00:06, 7684.37it/s] 88%|████████▊ | 351528/400000 [00:46<00:06, 7829.16it/s] 88%|████████▊ | 352312/400000 [00:46<00:06, 7809.40it/s] 88%|████████▊ | 353094/400000 [00:46<00:06, 7753.31it/s] 88%|████████▊ | 353892/400000 [00:46<00:05, 7818.69it/s] 89%|████████▊ | 354675/400000 [00:46<00:05, 7821.78it/s] 89%|████████▉ | 355458/400000 [00:46<00:05, 7795.19it/s] 89%|████████▉ | 356238/400000 [00:46<00:05, 7779.04it/s] 89%|████████▉ | 357042/400000 [00:46<00:05, 7851.40it/s] 89%|████████▉ | 357846/400000 [00:46<00:05, 7906.65it/s] 90%|████████▉ | 358637/400000 [00:46<00:05, 7833.49it/s] 90%|████████▉ | 359458/400000 [00:47<00:05, 7942.57it/s] 90%|█████████ | 360254/400000 [00:47<00:05, 7946.54it/s] 90%|█████████ | 361054/400000 [00:47<00:04, 7961.48it/s] 90%|█████████ | 361851/400000 [00:47<00:04, 7954.17it/s] 91%|█████████ | 362647/400000 [00:47<00:04, 7953.16it/s] 91%|█████████ | 363443/400000 [00:47<00:04, 7952.96it/s] 91%|█████████ | 364254/400000 [00:47<00:04, 7999.30it/s] 91%|█████████▏| 365055/400000 [00:47<00:04, 7968.97it/s] 91%|█████████▏| 365853/400000 [00:47<00:04, 7921.88it/s] 92%|█████████▏| 366646/400000 [00:48<00:04, 7642.22it/s] 92%|█████████▏| 367435/400000 [00:48<00:04, 7713.45it/s] 92%|█████████▏| 368209/400000 [00:48<00:04, 7530.79it/s] 92%|█████████▏| 368965/400000 [00:48<00:04, 7530.57it/s] 92%|█████████▏| 369720/400000 [00:48<00:04, 7521.36it/s] 93%|█████████▎| 370511/400000 [00:48<00:03, 7633.58it/s] 93%|█████████▎| 371276/400000 [00:48<00:03, 7554.92it/s] 93%|█████████▎| 372108/400000 [00:48<00:03, 7768.61it/s] 93%|█████████▎| 372910/400000 [00:48<00:03, 7839.91it/s] 93%|█████████▎| 373734/400000 [00:48<00:03, 7953.60it/s] 94%|█████████▎| 374531/400000 [00:49<00:03, 7818.38it/s] 94%|█████████▍| 375317/400000 [00:49<00:03, 7829.28it/s] 94%|█████████▍| 376113/400000 [00:49<00:03, 7865.95it/s] 94%|█████████▍| 376901/400000 [00:49<00:02, 7798.51it/s] 94%|█████████▍| 377709/400000 [00:49<00:02, 7880.79it/s] 95%|█████████▍| 378508/400000 [00:49<00:02, 7911.35it/s] 95%|█████████▍| 379310/400000 [00:49<00:02, 7941.92it/s] 95%|█████████▌| 380136/400000 [00:49<00:02, 8033.33it/s] 95%|█████████▌| 380946/400000 [00:49<00:02, 8050.99it/s] 95%|█████████▌| 381752/400000 [00:49<00:02, 7837.46it/s] 96%|█████████▌| 382538/400000 [00:50<00:02, 7673.00it/s] 96%|█████████▌| 383347/400000 [00:50<00:02, 7791.76it/s] 96%|█████████▌| 384171/400000 [00:50<00:01, 7918.67it/s] 96%|█████████▌| 384973/400000 [00:50<00:01, 7944.02it/s] 96%|█████████▋| 385782/400000 [00:50<00:01, 7984.41it/s] 97%|█████████▋| 386582/400000 [00:50<00:01, 7965.05it/s] 97%|█████████▋| 387391/400000 [00:50<00:01, 7999.38it/s] 97%|█████████▋| 388192/400000 [00:50<00:01, 7936.09it/s] 97%|█████████▋| 388987/400000 [00:50<00:01, 7863.09it/s] 97%|█████████▋| 389796/400000 [00:50<00:01, 7927.85it/s] 98%|█████████▊| 390591/400000 [00:51<00:01, 7932.42it/s] 98%|█████████▊| 391402/400000 [00:51<00:01, 7984.10it/s] 98%|█████████▊| 392201/400000 [00:51<00:00, 7837.41it/s] 98%|█████████▊| 393016/400000 [00:51<00:00, 7926.64it/s] 98%|█████████▊| 393833/400000 [00:51<00:00, 7997.29it/s] 99%|█████████▊| 394634/400000 [00:51<00:00, 7970.28it/s] 99%|█████████▉| 395432/400000 [00:51<00:00, 7911.38it/s] 99%|█████████▉| 396270/400000 [00:51<00:00, 8045.57it/s] 99%|█████████▉| 397082/400000 [00:51<00:00, 8066.71it/s] 99%|█████████▉| 397890/400000 [00:51<00:00, 8042.92it/s]100%|█████████▉| 398700/400000 [00:52<00:00, 8058.57it/s]100%|█████████▉| 399507/400000 [00:52<00:00, 7916.26it/s]100%|█████████▉| 399999/400000 [00:52<00:00, 7658.84it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f2ae0208c88> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011054740613792758 	 Accuracy: 57
Train Epoch: 1 	 Loss: 0.011484005180090965 	 Accuracy: 49

  model saves at 49% accuracy 

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
2020-05-14 08:25:19.741355: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 08:25:19.745716: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-14 08:25:19.745878: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558ff3c45d00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 08:25:19.745893: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f2a8cbe8b38> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6360 - accuracy: 0.5020
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6130 - accuracy: 0.5035
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6513 - accuracy: 0.5010 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8353 - accuracy: 0.4890
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7832 - accuracy: 0.4924
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7586 - accuracy: 0.4940
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6929 - accuracy: 0.4983
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6839 - accuracy: 0.4989
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6734 - accuracy: 0.4996
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6697 - accuracy: 0.4998
11000/25000 [============>.................] - ETA: 4s - loss: 7.6569 - accuracy: 0.5006
12000/25000 [=============>................] - ETA: 4s - loss: 7.6538 - accuracy: 0.5008
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6419 - accuracy: 0.5016
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6360 - accuracy: 0.5020
15000/25000 [=================>............] - ETA: 3s - loss: 7.6697 - accuracy: 0.4998
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6724 - accuracy: 0.4996
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6783 - accuracy: 0.4992
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6760 - accuracy: 0.4994
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6836 - accuracy: 0.4989
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6912 - accuracy: 0.4984
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6761 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6868 - accuracy: 0.4987
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6753 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6711 - accuracy: 0.4997
25000/25000 [==============================] - 10s 397us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f2a4107a668> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f2a41049d68> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.9341 - crf_viterbi_accuracy: 0.1867 - val_loss: 1.9430 - val_crf_viterbi_accuracy: 0.1600

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

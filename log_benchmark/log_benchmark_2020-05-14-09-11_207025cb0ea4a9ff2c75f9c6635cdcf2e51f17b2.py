
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f753d9c0f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 09:11:53.036342
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 09:11:53.040470
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 09:11:53.043942
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 09:11:53.047251
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f754978a400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352408.4062
Epoch 2/10

1/1 [==============================] - 0s 105ms/step - loss: 242471.8594
Epoch 3/10

1/1 [==============================] - 0s 99ms/step - loss: 153849.3906
Epoch 4/10

1/1 [==============================] - 0s 99ms/step - loss: 88715.3438
Epoch 5/10

1/1 [==============================] - 0s 100ms/step - loss: 50210.2578
Epoch 6/10

1/1 [==============================] - 0s 98ms/step - loss: 29827.8633
Epoch 7/10

1/1 [==============================] - 0s 98ms/step - loss: 18878.3594
Epoch 8/10

1/1 [==============================] - 0s 100ms/step - loss: 12683.8740
Epoch 9/10

1/1 [==============================] - 0s 104ms/step - loss: 9023.3760
Epoch 10/10

1/1 [==============================] - 0s 102ms/step - loss: 6775.9526

  #### Inference Need return ypred, ytrue ######################### 
[[-6.6741121e-01 -2.5518468e-01  7.7513576e-01  7.3375154e-01
   6.2975061e-01 -4.5074078e-01 -1.2669295e+00  5.1734060e-01
   1.3718954e-01  2.7957767e-01 -4.3943852e-01  1.0967569e+00
   1.1534172e+00  3.7499928e-01  1.2143240e+00 -2.8427702e-01
   1.0298209e+00  1.7242550e+00  1.2378736e+00  6.8716133e-01
  -1.2799456e+00 -2.2905976e-01 -2.3709713e-01 -2.0463184e-01
   4.5363969e-01  2.5902927e-01 -5.0170970e-01  4.7774756e-01
  -3.4799594e-01 -7.3000050e-01 -1.6794879e+00  4.1824001e-01
   1.1665633e+00 -1.3879032e+00  9.6612304e-02 -1.5833234e+00
   4.3245929e-01  4.6614912e-01 -7.2330534e-01  5.4426074e-02
   9.0325260e-01  7.0973682e-01 -4.5395976e-01  1.0214268e+00
   4.8758514e-02  1.0722500e-01  2.6560304e-01  7.1782863e-01
   1.6784775e+00 -2.9839221e-01  5.4797888e-02 -9.8550636e-01
   6.1309457e-01 -2.2532745e-01  1.7512133e+00  9.1396320e-01
   1.3012391e+00  1.5850663e-01 -3.5509282e-01  1.4752746e+00
   1.6923738e-01  9.4720049e+00  7.4067388e+00  7.3041205e+00
   7.8971686e+00  9.5708847e+00  7.4484811e+00  7.0040956e+00
   8.1763268e+00  8.1650476e+00  8.6147852e+00  8.4334707e+00
   5.6879053e+00  8.2610874e+00  6.0751143e+00  9.0227785e+00
   7.0626469e+00  7.3900747e+00  9.1840696e+00  8.2401485e+00
   7.6476755e+00  9.0221405e+00  8.4871883e+00  8.0346498e+00
   8.4115486e+00  7.2114377e+00  7.9454770e+00  8.0930109e+00
   9.2374544e+00  8.2423296e+00  7.0581441e+00  7.1884103e+00
   8.0135098e+00  6.6036029e+00  8.4003239e+00  8.9451246e+00
   8.1381273e+00  9.2600670e+00  7.5051818e+00  7.3913374e+00
   8.1277599e+00  6.1965876e+00  7.4847045e+00  6.2814984e+00
   7.6762772e+00  7.2661014e+00  7.6102042e+00  7.2820606e+00
   6.8983769e+00  9.5794811e+00  8.3361673e+00  7.5179729e+00
   7.8676333e+00  7.5712452e+00  7.7754617e+00  8.0614319e+00
   8.1096773e+00  8.1794186e+00  8.5106955e+00  6.7371206e+00
   2.0387433e-01 -4.7381401e-02 -8.9617759e-02 -1.0283238e-01
  -3.4821230e-01 -6.1530262e-02  1.2034773e+00 -6.9225752e-01
  -1.2181598e-01 -7.4097317e-01  8.7796450e-03  1.7754282e-01
   1.7184874e+00  1.2213569e+00 -7.0771003e-01 -1.4043784e-01
   1.6798660e+00  1.7140398e+00  6.6329157e-01  4.6175539e-01
  -3.4219742e-01 -1.0997519e-01 -6.7318434e-01 -4.6885005e-01
  -9.0371823e-01 -1.6416703e+00  1.9112805e-01  1.0897046e-01
   6.7362785e-03 -8.2903409e-01  5.0675809e-01  1.5863712e+00
  -4.4617805e-01 -8.7644219e-01 -6.3330555e-01 -7.1152210e-01
  -3.2445860e-01 -7.5206220e-01  1.0295012e+00  1.0532837e+00
   6.7330277e-01  1.7747043e+00  2.0516579e-01  1.7055190e+00
  -3.9449805e-01  9.2412257e-01 -9.6170187e-02  4.4833598e-01
  -1.4190083e+00  1.2311413e+00  3.5098603e-01  6.5511388e-01
   7.0399737e-01  1.3533560e+00 -2.6195273e-01  1.4414463e+00
  -1.1142656e+00  1.7789096e-02  5.1716352e-01 -3.4199011e-01
   6.8196577e-01  2.2914660e-01  4.6611184e-01  4.4895762e-01
   2.6288185e+00  1.5889466e+00  3.9054078e-01  1.4778513e+00
   1.8999025e+00  1.7382759e-01  4.1280103e-01  1.3884342e-01
   3.4428048e-01  1.4088503e+00  1.7547480e+00  6.5891773e-01
   2.4343724e+00  4.6408367e-01  2.8522682e-01  2.3661985e+00
   1.3137215e-01  3.2160997e-01  7.3363018e-01  1.8821347e-01
   9.3017834e-01  3.9448661e-01  1.8822682e+00  4.1475141e-01
   2.0532489e-01  2.6727957e-01  5.2168369e-01  2.7897620e-01
   1.6481128e+00  2.0050769e+00  2.5151980e-01  2.4377084e+00
   1.2867749e-01  1.7704782e+00  1.9130993e-01  1.9118702e-01
   4.2775482e-01  2.0047349e-01  1.3748894e+00  2.8484006e+00
   9.6946359e-02  7.8964758e-01  6.9435370e-01  2.2834096e+00
   3.8341367e-01  1.5112966e+00  1.8923550e+00  6.0422403e-01
   1.7270780e-01  1.5422295e+00  4.1017044e-01  1.9615680e+00
   1.3476360e+00  9.9955964e-01  3.7692869e-01  1.8528441e+00
   1.4575517e-01  8.5947819e+00  9.9983215e+00  7.3273158e+00
   9.3543110e+00  8.0576220e+00  9.4927673e+00  7.8325987e+00
   9.0913925e+00  1.0203980e+01  8.4012680e+00  7.9344883e+00
   7.8889995e+00  7.6482310e+00  8.6334839e+00  9.9824781e+00
   8.7721214e+00  9.6759615e+00  8.1509714e+00  9.2853184e+00
   8.7493019e+00  1.0113435e+01  7.1643705e+00  7.7884398e+00
   7.4639611e+00  7.7546768e+00  6.3580918e+00  8.6613436e+00
   8.4082918e+00  8.0296192e+00  8.4575806e+00  7.4468369e+00
   7.9857526e+00  6.9040747e+00  8.0257282e+00  7.7858095e+00
   7.7625074e+00  8.2841177e+00  7.4819503e+00  9.4498768e+00
   8.0061331e+00  7.8840995e+00  9.1464558e+00  8.4408894e+00
   9.8576202e+00  6.4914618e+00  8.5919371e+00  9.3723421e+00
   9.7345924e+00  9.2685766e+00  8.0365820e+00  9.0517511e+00
   8.0577412e+00  8.7295065e+00  8.4923344e+00  8.4452982e+00
   8.9120293e+00  7.3248763e+00  8.6315784e+00  9.2909775e+00
   9.8630714e-01  8.8603413e-01  5.8579940e-01  1.1156839e+00
   1.8538396e+00  1.8376099e+00  9.8919660e-01  7.6100111e-01
   1.9714193e+00  5.4862028e-01  1.1871789e+00  3.1850725e-01
   6.3484192e-01  2.1549921e+00  8.7252086e-01  1.0245038e+00
   1.9598507e+00  2.3145452e+00  8.6969298e-01  6.4384103e-01
   2.1929803e+00  7.7307570e-01  2.4324894e-01  1.8072178e+00
   2.0041506e+00  5.9805524e-01  8.4838736e-01  3.5690588e-01
   6.8652177e-01  1.4432836e+00  8.6351365e-01  3.0066592e-01
   1.8188457e+00  2.6498041e+00  5.4766399e-01  1.4757029e+00
   6.7758536e-01  2.0855522e+00  4.9359941e-01  7.0704091e-01
   1.4273663e+00  1.4730997e+00  8.1259549e-01  3.3828753e-01
   7.3761779e-01  1.3323832e-01  1.2118746e+00  2.2954402e+00
   1.3713619e+00  6.7398399e-01  1.5801079e+00  2.2543459e+00
   2.9392517e+00  1.0404992e+00  2.2008729e-01  1.4955862e+00
   8.2246089e-01  1.7896813e-01  1.4825583e-01  1.0338588e+00
  -5.8509784e+00  4.8436332e+00 -1.0796725e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 09:12:01.499032
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.8041
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 09:12:01.502619
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8826.27
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 09:12:01.505712
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.9197
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 09:12:01.508620
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -789.455
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140141162489392
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140140220940976
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140140220941480
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140140220941984
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140140220942488
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140140220942992

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f7545615e80> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.419358
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.389122
grad_step = 000002, loss = 0.369654
grad_step = 000003, loss = 0.349198
grad_step = 000004, loss = 0.325160
grad_step = 000005, loss = 0.299714
grad_step = 000006, loss = 0.277693
grad_step = 000007, loss = 0.266839
grad_step = 000008, loss = 0.255815
grad_step = 000009, loss = 0.242444
grad_step = 000010, loss = 0.233234
grad_step = 000011, loss = 0.221982
grad_step = 000012, loss = 0.212075
grad_step = 000013, loss = 0.204993
grad_step = 000014, loss = 0.198184
grad_step = 000015, loss = 0.190100
grad_step = 000016, loss = 0.181186
grad_step = 000017, loss = 0.171975
grad_step = 000018, loss = 0.162894
grad_step = 000019, loss = 0.154618
grad_step = 000020, loss = 0.147539
grad_step = 000021, loss = 0.141137
grad_step = 000022, loss = 0.134359
grad_step = 000023, loss = 0.126894
grad_step = 000024, loss = 0.119327
grad_step = 000025, loss = 0.112450
grad_step = 000026, loss = 0.106543
grad_step = 000027, loss = 0.101148
grad_step = 000028, loss = 0.095651
grad_step = 000029, loss = 0.089865
grad_step = 000030, loss = 0.083982
grad_step = 000031, loss = 0.078428
grad_step = 000032, loss = 0.073579
grad_step = 000033, loss = 0.069295
grad_step = 000034, loss = 0.065031
grad_step = 000035, loss = 0.060675
grad_step = 000036, loss = 0.056519
grad_step = 000037, loss = 0.052765
grad_step = 000038, loss = 0.049303
grad_step = 000039, loss = 0.045935
grad_step = 000040, loss = 0.042652
grad_step = 000041, loss = 0.039623
grad_step = 000042, loss = 0.036953
grad_step = 000043, loss = 0.034484
grad_step = 000044, loss = 0.032005
grad_step = 000045, loss = 0.029548
grad_step = 000046, loss = 0.027310
grad_step = 000047, loss = 0.025362
grad_step = 000048, loss = 0.023596
grad_step = 000049, loss = 0.021892
grad_step = 000050, loss = 0.020236
grad_step = 000051, loss = 0.018699
grad_step = 000052, loss = 0.017321
grad_step = 000053, loss = 0.016063
grad_step = 000054, loss = 0.014876
grad_step = 000055, loss = 0.013770
grad_step = 000056, loss = 0.012777
grad_step = 000057, loss = 0.011876
grad_step = 000058, loss = 0.011019
grad_step = 000059, loss = 0.010208
grad_step = 000060, loss = 0.009486
grad_step = 000061, loss = 0.008862
grad_step = 000062, loss = 0.008283
grad_step = 000063, loss = 0.007719
grad_step = 000064, loss = 0.007197
grad_step = 000065, loss = 0.006743
grad_step = 000066, loss = 0.006338
grad_step = 000067, loss = 0.005960
grad_step = 000068, loss = 0.005607
grad_step = 000069, loss = 0.005289
grad_step = 000070, loss = 0.005003
grad_step = 000071, loss = 0.004737
grad_step = 000072, loss = 0.004490
grad_step = 000073, loss = 0.004272
grad_step = 000074, loss = 0.004078
grad_step = 000075, loss = 0.003893
grad_step = 000076, loss = 0.003717
grad_step = 000077, loss = 0.003559
grad_step = 000078, loss = 0.003424
grad_step = 000079, loss = 0.003298
grad_step = 000080, loss = 0.003176
grad_step = 000081, loss = 0.003066
grad_step = 000082, loss = 0.002968
grad_step = 000083, loss = 0.002878
grad_step = 000084, loss = 0.002794
grad_step = 000085, loss = 0.002720
grad_step = 000086, loss = 0.002655
grad_step = 000087, loss = 0.002592
grad_step = 000088, loss = 0.002534
grad_step = 000089, loss = 0.002485
grad_step = 000090, loss = 0.002441
grad_step = 000091, loss = 0.002400
grad_step = 000092, loss = 0.002362
grad_step = 000093, loss = 0.002330
grad_step = 000094, loss = 0.002301
grad_step = 000095, loss = 0.002276
grad_step = 000096, loss = 0.002256
grad_step = 000097, loss = 0.002240
grad_step = 000098, loss = 0.002223
grad_step = 000099, loss = 0.002200
grad_step = 000100, loss = 0.002178
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002163
grad_step = 000102, loss = 0.002155
grad_step = 000103, loss = 0.002150
grad_step = 000104, loss = 0.002146
grad_step = 000105, loss = 0.002137
grad_step = 000106, loss = 0.002123
grad_step = 000107, loss = 0.002108
grad_step = 000108, loss = 0.002096
grad_step = 000109, loss = 0.002091
grad_step = 000110, loss = 0.002089
grad_step = 000111, loss = 0.002092
grad_step = 000112, loss = 0.002099
grad_step = 000113, loss = 0.002109
grad_step = 000114, loss = 0.002108
grad_step = 000115, loss = 0.002093
grad_step = 000116, loss = 0.002064
grad_step = 000117, loss = 0.002046
grad_step = 000118, loss = 0.002047
grad_step = 000119, loss = 0.002058
grad_step = 000120, loss = 0.002065
grad_step = 000121, loss = 0.002058
grad_step = 000122, loss = 0.002040
grad_step = 000123, loss = 0.002021
grad_step = 000124, loss = 0.002012
grad_step = 000125, loss = 0.002014
grad_step = 000126, loss = 0.002022
grad_step = 000127, loss = 0.002026
grad_step = 000128, loss = 0.002023
grad_step = 000129, loss = 0.002011
grad_step = 000130, loss = 0.001995
grad_step = 000131, loss = 0.001981
grad_step = 000132, loss = 0.001973
grad_step = 000133, loss = 0.001971
grad_step = 000134, loss = 0.001972
grad_step = 000135, loss = 0.001975
grad_step = 000136, loss = 0.001977
grad_step = 000137, loss = 0.001978
grad_step = 000138, loss = 0.001977
grad_step = 000139, loss = 0.001966
grad_step = 000140, loss = 0.001952
grad_step = 000141, loss = 0.001934
grad_step = 000142, loss = 0.001918
grad_step = 000143, loss = 0.001908
grad_step = 000144, loss = 0.001902
grad_step = 000145, loss = 0.001901
grad_step = 000146, loss = 0.001900
grad_step = 000147, loss = 0.001901
grad_step = 000148, loss = 0.001903
grad_step = 000149, loss = 0.001903
grad_step = 000150, loss = 0.001896
grad_step = 000151, loss = 0.001887
grad_step = 000152, loss = 0.001868
grad_step = 000153, loss = 0.001848
grad_step = 000154, loss = 0.001828
grad_step = 000155, loss = 0.001812
grad_step = 000156, loss = 0.001802
grad_step = 000157, loss = 0.001795
grad_step = 000158, loss = 0.001791
grad_step = 000159, loss = 0.001799
grad_step = 000160, loss = 0.001839
grad_step = 000161, loss = 0.001923
grad_step = 000162, loss = 0.001835
grad_step = 000163, loss = 0.001923
grad_step = 000164, loss = 0.002010
grad_step = 000165, loss = 0.001830
grad_step = 000166, loss = 0.001957
grad_step = 000167, loss = 0.001893
grad_step = 000168, loss = 0.001860
grad_step = 000169, loss = 0.001938
grad_step = 000170, loss = 0.001796
grad_step = 000171, loss = 0.001890
grad_step = 000172, loss = 0.001766
grad_step = 000173, loss = 0.001844
grad_step = 000174, loss = 0.001816
grad_step = 000175, loss = 0.001814
grad_step = 000176, loss = 0.001820
grad_step = 000177, loss = 0.001761
grad_step = 000178, loss = 0.001780
grad_step = 000179, loss = 0.001777
grad_step = 000180, loss = 0.001759
grad_step = 000181, loss = 0.001787
grad_step = 000182, loss = 0.001735
grad_step = 000183, loss = 0.001767
grad_step = 000184, loss = 0.001730
grad_step = 000185, loss = 0.001746
grad_step = 000186, loss = 0.001749
grad_step = 000187, loss = 0.001741
grad_step = 000188, loss = 0.001751
grad_step = 000189, loss = 0.001734
grad_step = 000190, loss = 0.001728
grad_step = 000191, loss = 0.001725
grad_step = 000192, loss = 0.001707
grad_step = 000193, loss = 0.001718
grad_step = 000194, loss = 0.001704
grad_step = 000195, loss = 0.001714
grad_step = 000196, loss = 0.001712
grad_step = 000197, loss = 0.001713
grad_step = 000198, loss = 0.001720
grad_step = 000199, loss = 0.001724
grad_step = 000200, loss = 0.001728
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001739
grad_step = 000202, loss = 0.001738
grad_step = 000203, loss = 0.001743
grad_step = 000204, loss = 0.001730
grad_step = 000205, loss = 0.001711
grad_step = 000206, loss = 0.001695
grad_step = 000207, loss = 0.001679
grad_step = 000208, loss = 0.001678
grad_step = 000209, loss = 0.001685
grad_step = 000210, loss = 0.001692
grad_step = 000211, loss = 0.001692
grad_step = 000212, loss = 0.001688
grad_step = 000213, loss = 0.001676
grad_step = 000214, loss = 0.001668
grad_step = 000215, loss = 0.001663
grad_step = 000216, loss = 0.001662
grad_step = 000217, loss = 0.001664
grad_step = 000218, loss = 0.001667
grad_step = 000219, loss = 0.001670
grad_step = 000220, loss = 0.001669
grad_step = 000221, loss = 0.001669
grad_step = 000222, loss = 0.001666
grad_step = 000223, loss = 0.001663
grad_step = 000224, loss = 0.001659
grad_step = 000225, loss = 0.001656
grad_step = 000226, loss = 0.001652
grad_step = 000227, loss = 0.001649
grad_step = 000228, loss = 0.001646
grad_step = 000229, loss = 0.001644
grad_step = 000230, loss = 0.001642
grad_step = 000231, loss = 0.001642
grad_step = 000232, loss = 0.001642
grad_step = 000233, loss = 0.001645
grad_step = 000234, loss = 0.001649
grad_step = 000235, loss = 0.001660
grad_step = 000236, loss = 0.001672
grad_step = 000237, loss = 0.001691
grad_step = 000238, loss = 0.001695
grad_step = 000239, loss = 0.001695
grad_step = 000240, loss = 0.001667
grad_step = 000241, loss = 0.001635
grad_step = 000242, loss = 0.001612
grad_step = 000243, loss = 0.001610
grad_step = 000244, loss = 0.001623
grad_step = 000245, loss = 0.001635
grad_step = 000246, loss = 0.001636
grad_step = 000247, loss = 0.001621
grad_step = 000248, loss = 0.001605
grad_step = 000249, loss = 0.001596
grad_step = 000250, loss = 0.001597
grad_step = 000251, loss = 0.001602
grad_step = 000252, loss = 0.001608
grad_step = 000253, loss = 0.001613
grad_step = 000254, loss = 0.001611
grad_step = 000255, loss = 0.001608
grad_step = 000256, loss = 0.001599
grad_step = 000257, loss = 0.001592
grad_step = 000258, loss = 0.001585
grad_step = 000259, loss = 0.001580
grad_step = 000260, loss = 0.001576
grad_step = 000261, loss = 0.001572
grad_step = 000262, loss = 0.001569
grad_step = 000263, loss = 0.001567
grad_step = 000264, loss = 0.001565
grad_step = 000265, loss = 0.001563
grad_step = 000266, loss = 0.001561
grad_step = 000267, loss = 0.001560
grad_step = 000268, loss = 0.001559
grad_step = 000269, loss = 0.001561
grad_step = 000270, loss = 0.001569
grad_step = 000271, loss = 0.001587
grad_step = 000272, loss = 0.001639
grad_step = 000273, loss = 0.001699
grad_step = 000274, loss = 0.001839
grad_step = 000275, loss = 0.001857
grad_step = 000276, loss = 0.001843
grad_step = 000277, loss = 0.001641
grad_step = 000278, loss = 0.001550
grad_step = 000279, loss = 0.001648
grad_step = 000280, loss = 0.001692
grad_step = 000281, loss = 0.001597
grad_step = 000282, loss = 0.001541
grad_step = 000283, loss = 0.001618
grad_step = 000284, loss = 0.001632
grad_step = 000285, loss = 0.001547
grad_step = 000286, loss = 0.001557
grad_step = 000287, loss = 0.001602
grad_step = 000288, loss = 0.001555
grad_step = 000289, loss = 0.001520
grad_step = 000290, loss = 0.001555
grad_step = 000291, loss = 0.001561
grad_step = 000292, loss = 0.001524
grad_step = 000293, loss = 0.001514
grad_step = 000294, loss = 0.001541
grad_step = 000295, loss = 0.001542
grad_step = 000296, loss = 0.001512
grad_step = 000297, loss = 0.001505
grad_step = 000298, loss = 0.001521
grad_step = 000299, loss = 0.001519
grad_step = 000300, loss = 0.001499
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001492
grad_step = 000302, loss = 0.001502
grad_step = 000303, loss = 0.001503
grad_step = 000304, loss = 0.001490
grad_step = 000305, loss = 0.001482
grad_step = 000306, loss = 0.001486
grad_step = 000307, loss = 0.001489
grad_step = 000308, loss = 0.001482
grad_step = 000309, loss = 0.001473
grad_step = 000310, loss = 0.001473
grad_step = 000311, loss = 0.001478
grad_step = 000312, loss = 0.001480
grad_step = 000313, loss = 0.001487
grad_step = 000314, loss = 0.001505
grad_step = 000315, loss = 0.001562
grad_step = 000316, loss = 0.001527
grad_step = 000317, loss = 0.001501
grad_step = 000318, loss = 0.001469
grad_step = 000319, loss = 0.001475
grad_step = 000320, loss = 0.001479
grad_step = 000321, loss = 0.001463
grad_step = 000322, loss = 0.001466
grad_step = 000323, loss = 0.001478
grad_step = 000324, loss = 0.001469
grad_step = 000325, loss = 0.001446
grad_step = 000326, loss = 0.001439
grad_step = 000327, loss = 0.001445
grad_step = 000328, loss = 0.001445
grad_step = 000329, loss = 0.001433
grad_step = 000330, loss = 0.001428
grad_step = 000331, loss = 0.001435
grad_step = 000332, loss = 0.001438
grad_step = 000333, loss = 0.001437
grad_step = 000334, loss = 0.001436
grad_step = 000335, loss = 0.001449
grad_step = 000336, loss = 0.001464
grad_step = 000337, loss = 0.001481
grad_step = 000338, loss = 0.001483
grad_step = 000339, loss = 0.001498
grad_step = 000340, loss = 0.001460
grad_step = 000341, loss = 0.001431
grad_step = 000342, loss = 0.001406
grad_step = 000343, loss = 0.001400
grad_step = 000344, loss = 0.001406
grad_step = 000345, loss = 0.001417
grad_step = 000346, loss = 0.001433
grad_step = 000347, loss = 0.001432
grad_step = 000348, loss = 0.001430
grad_step = 000349, loss = 0.001413
grad_step = 000350, loss = 0.001403
grad_step = 000351, loss = 0.001387
grad_step = 000352, loss = 0.001377
grad_step = 000353, loss = 0.001372
grad_step = 000354, loss = 0.001373
grad_step = 000355, loss = 0.001378
grad_step = 000356, loss = 0.001382
grad_step = 000357, loss = 0.001391
grad_step = 000358, loss = 0.001399
grad_step = 000359, loss = 0.001417
grad_step = 000360, loss = 0.001424
grad_step = 000361, loss = 0.001453
grad_step = 000362, loss = 0.001438
grad_step = 000363, loss = 0.001434
grad_step = 000364, loss = 0.001395
grad_step = 000365, loss = 0.001365
grad_step = 000366, loss = 0.001344
grad_step = 000367, loss = 0.001341
grad_step = 000368, loss = 0.001351
grad_step = 000369, loss = 0.001364
grad_step = 000370, loss = 0.001384
grad_step = 000371, loss = 0.001386
grad_step = 000372, loss = 0.001399
grad_step = 000373, loss = 0.001387
grad_step = 000374, loss = 0.001384
grad_step = 000375, loss = 0.001357
grad_step = 000376, loss = 0.001336
grad_step = 000377, loss = 0.001319
grad_step = 000378, loss = 0.001313
grad_step = 000379, loss = 0.001315
grad_step = 000380, loss = 0.001320
grad_step = 000381, loss = 0.001328
grad_step = 000382, loss = 0.001330
grad_step = 000383, loss = 0.001331
grad_step = 000384, loss = 0.001324
grad_step = 000385, loss = 0.001322
grad_step = 000386, loss = 0.001317
grad_step = 000387, loss = 0.001317
grad_step = 000388, loss = 0.001311
grad_step = 000389, loss = 0.001310
grad_step = 000390, loss = 0.001306
grad_step = 000391, loss = 0.001306
grad_step = 000392, loss = 0.001302
grad_step = 000393, loss = 0.001300
grad_step = 000394, loss = 0.001297
grad_step = 000395, loss = 0.001298
grad_step = 000396, loss = 0.001297
grad_step = 000397, loss = 0.001302
grad_step = 000398, loss = 0.001302
grad_step = 000399, loss = 0.001313
grad_step = 000400, loss = 0.001315
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001329
grad_step = 000402, loss = 0.001324
grad_step = 000403, loss = 0.001327
grad_step = 000404, loss = 0.001304
grad_step = 000405, loss = 0.001285
grad_step = 000406, loss = 0.001259
grad_step = 000407, loss = 0.001245
grad_step = 000408, loss = 0.001245
grad_step = 000409, loss = 0.001255
grad_step = 000410, loss = 0.001268
grad_step = 000411, loss = 0.001274
grad_step = 000412, loss = 0.001283
grad_step = 000413, loss = 0.001278
grad_step = 000414, loss = 0.001278
grad_step = 000415, loss = 0.001265
grad_step = 000416, loss = 0.001254
grad_step = 000417, loss = 0.001237
grad_step = 000418, loss = 0.001225
grad_step = 000419, loss = 0.001218
grad_step = 000420, loss = 0.001217
grad_step = 000421, loss = 0.001218
grad_step = 000422, loss = 0.001222
grad_step = 000423, loss = 0.001231
grad_step = 000424, loss = 0.001241
grad_step = 000425, loss = 0.001265
grad_step = 000426, loss = 0.001291
grad_step = 000427, loss = 0.001356
grad_step = 000428, loss = 0.001381
grad_step = 000429, loss = 0.001437
grad_step = 000430, loss = 0.001347
grad_step = 000431, loss = 0.001262
grad_step = 000432, loss = 0.001204
grad_step = 000433, loss = 0.001225
grad_step = 000434, loss = 0.001270
grad_step = 000435, loss = 0.001255
grad_step = 000436, loss = 0.001220
grad_step = 000437, loss = 0.001196
grad_step = 000438, loss = 0.001202
grad_step = 000439, loss = 0.001227
grad_step = 000440, loss = 0.001242
grad_step = 000441, loss = 0.001253
grad_step = 000442, loss = 0.001211
grad_step = 000443, loss = 0.001186
grad_step = 000444, loss = 0.001175
grad_step = 000445, loss = 0.001173
grad_step = 000446, loss = 0.001182
grad_step = 000447, loss = 0.001186
grad_step = 000448, loss = 0.001181
grad_step = 000449, loss = 0.001173
grad_step = 000450, loss = 0.001165
grad_step = 000451, loss = 0.001153
grad_step = 000452, loss = 0.001150
grad_step = 000453, loss = 0.001155
grad_step = 000454, loss = 0.001158
grad_step = 000455, loss = 0.001162
grad_step = 000456, loss = 0.001159
grad_step = 000457, loss = 0.001153
grad_step = 000458, loss = 0.001147
grad_step = 000459, loss = 0.001145
grad_step = 000460, loss = 0.001140
grad_step = 000461, loss = 0.001135
grad_step = 000462, loss = 0.001130
grad_step = 000463, loss = 0.001126
grad_step = 000464, loss = 0.001123
grad_step = 000465, loss = 0.001122
grad_step = 000466, loss = 0.001120
grad_step = 000467, loss = 0.001117
grad_step = 000468, loss = 0.001114
grad_step = 000469, loss = 0.001112
grad_step = 000470, loss = 0.001110
grad_step = 000471, loss = 0.001110
grad_step = 000472, loss = 0.001112
grad_step = 000473, loss = 0.001117
grad_step = 000474, loss = 0.001127
grad_step = 000475, loss = 0.001153
grad_step = 000476, loss = 0.001193
grad_step = 000477, loss = 0.001294
grad_step = 000478, loss = 0.001346
grad_step = 000479, loss = 0.001460
grad_step = 000480, loss = 0.001304
grad_step = 000481, loss = 0.001165
grad_step = 000482, loss = 0.001115
grad_step = 000483, loss = 0.001186
grad_step = 000484, loss = 0.001231
grad_step = 000485, loss = 0.001154
grad_step = 000486, loss = 0.001108
grad_step = 000487, loss = 0.001128
grad_step = 000488, loss = 0.001149
grad_step = 000489, loss = 0.001157
grad_step = 000490, loss = 0.001167
grad_step = 000491, loss = 0.001189
grad_step = 000492, loss = 0.001109
grad_step = 000493, loss = 0.001094
grad_step = 000494, loss = 0.001090
grad_step = 000495, loss = 0.001063
grad_step = 000496, loss = 0.001082
grad_step = 000497, loss = 0.001095
grad_step = 000498, loss = 0.001077
grad_step = 000499, loss = 0.001077
grad_step = 000500, loss = 0.001078
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001051
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

  date_run                              2020-05-14 09:12:19.519660
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.242905
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 09:12:19.525616
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.148208
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 09:12:19.533216
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.147085
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 09:12:19.538385
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.25207
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
0   2020-05-14 09:11:53.036342  ...    mean_absolute_error
1   2020-05-14 09:11:53.040470  ...     mean_squared_error
2   2020-05-14 09:11:53.043942  ...  median_absolute_error
3   2020-05-14 09:11:53.047251  ...               r2_score
4   2020-05-14 09:12:01.499032  ...    mean_absolute_error
5   2020-05-14 09:12:01.502619  ...     mean_squared_error
6   2020-05-14 09:12:01.505712  ...  median_absolute_error
7   2020-05-14 09:12:01.508620  ...               r2_score
8   2020-05-14 09:12:19.519660  ...    mean_absolute_error
9   2020-05-14 09:12:19.525616  ...     mean_squared_error
10  2020-05-14 09:12:19.533216  ...  median_absolute_error
11  2020-05-14 09:12:19.538385  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6f2bb9eba8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 40%|███▉      | 3948544/9912422 [00:00<00:00, 39483783.19it/s]9920512it [00:00, 36510640.18it/s]                             
0it [00:00, ?it/s]32768it [00:00, 656227.40it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1048029.82it/s]1654784it [00:00, 13084242.26it/s]                           
0it [00:00, ?it/s]8192it [00:00, 230152.78it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6ede557e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6eddb880b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6ede557e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6eddade0f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6edb3194e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6edb303c50> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6ede557e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6edda9c710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6edb3194e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6eddb88128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f2475d14208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=2e66fd342753624b5c8ac096cb6c697b4e5f47b6d1b4cf4ce7fde0f8b4c45d94
  Stored in directory: /tmp/pip-ephem-wheel-cache-u09k58aq/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f240db0f710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2965504/17464789 [====>.........................] - ETA: 0s
11517952/17464789 [==================>...........] - ETA: 0s
16506880/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 09:13:44.936431: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 09:13:44.940319: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095230000 Hz
2020-05-14 09:13:44.940473: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55635a22a130 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 09:13:44.940486: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 8.3106 - accuracy: 0.4580
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8583 - accuracy: 0.4875 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.8251 - accuracy: 0.4897
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.8123 - accuracy: 0.4905
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7586 - accuracy: 0.4940
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.8174 - accuracy: 0.4902
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7915 - accuracy: 0.4919
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7184 - accuracy: 0.4966
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6854 - accuracy: 0.4988
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6283 - accuracy: 0.5025
11000/25000 [============>.................] - ETA: 3s - loss: 7.6220 - accuracy: 0.5029
12000/25000 [=============>................] - ETA: 3s - loss: 7.6130 - accuracy: 0.5035
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6112 - accuracy: 0.5036
14000/25000 [===============>..............] - ETA: 2s - loss: 7.5801 - accuracy: 0.5056
15000/25000 [=================>............] - ETA: 2s - loss: 7.5889 - accuracy: 0.5051
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5794 - accuracy: 0.5057
17000/25000 [===================>..........] - ETA: 1s - loss: 7.5990 - accuracy: 0.5044
18000/25000 [====================>.........] - ETA: 1s - loss: 7.5985 - accuracy: 0.5044
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6093 - accuracy: 0.5037
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6030 - accuracy: 0.5041
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6192 - accuracy: 0.5031
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6346 - accuracy: 0.5021
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6433 - accuracy: 0.5015
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
25000/25000 [==============================] - 7s 272us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 09:13:58.117094
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 09:13:58.117094  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:29:12, 11.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:16:31, 15.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:44:49, 22.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:31:53, 31.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.31M/862M [00:01<5:15:40, 45.3kB/s].vector_cache/glove.6B.zip:   1%|          | 8.77M/862M [00:01<3:39:38, 64.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.0M/862M [00:01<2:33:19, 92.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.4M/862M [00:01<1:46:51, 132kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.8M/862M [00:01<1:14:31, 188kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.1M/862M [00:01<51:59, 268kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.4M/862M [00:01<36:18, 382kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.7M/862M [00:02<25:23, 544kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.9M/862M [00:02<17:46, 773kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.2M/862M [00:02<12:28, 1.10MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.4M/862M [00:02<08:47, 1.55MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.8M/862M [00:02<06:12, 2.18MB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.1M/862M [00:03<05:55, 2.28MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.2M/862M [00:03<04:23, 3.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:05<06:47, 1.98MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.4M/862M [00:05<07:37, 1.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.1M/862M [00:05<06:00, 2.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:05<04:22, 3.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:07<19:43, 677kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:07<15:10, 879kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.4M/862M [00:07<10:56, 1.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:09<10:46, 1.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:09<10:17, 1.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:09<07:46, 1.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:09<05:36, 2.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:11<10:10, 1.30MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:11<08:29, 1.56MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.6M/862M [00:11<06:16, 2.10MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<07:27, 1.76MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:13<07:34, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:13<05:59, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:13<04:21, 3.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:15<09:08, 1.43MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:15<07:43, 1.69MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.8M/862M [00:15<05:40, 2.30MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:17<07:01, 1.85MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:17<06:15, 2.07MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.9M/862M [00:17<04:42, 2.75MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:19<06:20, 2.04MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:19<05:45, 2.24MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.0M/862M [00:19<04:21, 2.96MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:21<06:05, 2.11MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:21<06:54, 1.86MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:21<05:23, 2.39MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:21<03:55, 3.27MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<10:41, 1.20MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:23<08:49, 1.45MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.2M/862M [00:23<06:27, 1.98MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:24<07:29, 1.70MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:25<06:20, 2.01MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<04:42, 2.70MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<03:27, 3.66MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<50:09, 252kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<37:42, 336kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<27:01, 468kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<20:52, 603kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<15:55, 790kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<11:27, 1.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<10:54, 1.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<08:53, 1.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<06:29, 1.93MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:29, 1.66MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:31, 1.91MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<04:49, 2.57MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:20, 1.96MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:42, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<04:17, 2.87MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:54, 2.08MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:24, 2.28MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:36<04:03, 3.03MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:44, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:04, 2.41MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<03:47, 3.22MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<02:49, 4.32MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<50:58, 239kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<36:55, 330kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<26:06, 465kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<21:04, 575kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<17:14, 702kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<12:41, 953kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<09:26, 1.28MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<8:05:23, 24.8kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<5:39:30, 35.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<3:59:05, 50.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<2:49:54, 70.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<1:59:26, 100kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<1:23:25, 143kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<1:10:43, 168kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<50:44, 235kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<35:44, 332kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<27:43, 427kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<21:52, 541kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<15:54, 743kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<11:14, 1.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<1:31:37, 128kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<1:05:07, 181kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<45:53, 256kB/s]  .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<32:08, 364kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<1:09:30, 168kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<49:51, 235kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<35:07, 332kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<27:15, 427kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<20:16, 574kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<14:27, 802kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<12:49, 902kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<10:09, 1.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<07:23, 1.56MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<07:53, 1.46MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:42, 1.71MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:58, 2.30MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:10, 1.85MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:30, 2.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<04:08, 2.75MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:34, 2.04MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:05, 2.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<03:47, 2.99MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:18, 2.13MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<04:54, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<03:40, 3.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:12, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:56, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<04:38, 2.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<03:24, 3.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<07:44, 1.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:33, 1.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<04:52, 2.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:59, 1.85MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<05:20, 2.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:00, 2.76MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:24, 2.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:55, 2.24MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<03:43, 2.96MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:10, 2.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:45, 2.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<03:35, 3.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:05, 2.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:40, 2.32MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<03:31, 3.08MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:01, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:38, 2.33MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:28, 3.10MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:59, 2.16MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:35, 2.34MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:26, 3.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:56, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:32, 2.35MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<03:24, 3.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:54, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:38, 1.88MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<04:24, 2.40MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:12, 3.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<08:13, 1.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:52, 1.53MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<05:04, 2.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:59, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:22, 1.64MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<05:00, 2.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<03:56, 2.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<7:14:32, 23.9kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<5:04:22, 34.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<3:32:39, 48.7kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<2:31:32, 68.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<1:48:10, 95.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<1:16:04, 136kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<53:12, 193kB/s]  .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<42:44, 240kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<30:58, 331kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<21:53, 467kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<17:39, 577kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<14:27, 705kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<10:32, 966kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<07:31, 1.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<09:03, 1.12MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<07:23, 1.37MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:25, 1.86MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<06:07, 1.64MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:20, 1.88MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:59, 2.51MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<05:08, 1.94MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<04:37, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:29, 2.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:47, 2.07MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:21, 2.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<03:15, 3.03MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:37, 2.13MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:15, 2.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<03:13, 3.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:33, 2.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:11, 2.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<03:09, 3.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:30, 2.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:08, 2.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<03:08, 3.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:28, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:06, 1.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:04, 2.37MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<02:56, 3.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<9:14:48, 17.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<6:29:05, 24.6kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<4:31:52, 35.1kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<3:11:49, 49.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<2:16:11, 69.8kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<1:35:36, 99.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<1:06:52, 142kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<50:12, 188kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<36:07, 261kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<25:27, 370kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<19:55, 470kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<15:53, 590kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<11:35, 808kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<09:34, 972kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<07:41, 1.21MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:36, 1.65MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<06:03, 1.53MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:02, 1.83MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<03:42, 2.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<02:43, 3.37MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<38:40, 237kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<27:59, 327kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<19:46, 462kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<15:56, 571kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<12:05, 752kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<08:38, 1.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<08:10, 1.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<06:38, 1.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<04:50, 1.86MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:30, 1.63MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:46, 1.87MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:32, 2.53MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:35, 1.94MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:07, 2.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:06, 2.86MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<02:41, 3.29MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<6:24:13, 23.0kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<4:29:08, 32.8kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<3:07:32, 46.8kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<2:16:59, 64.0kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<1:37:39, 89.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<1:08:39, 127kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<47:58, 182kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:20<37:52, 230kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<27:23, 317kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<19:20, 448kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<15:29, 557kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<12:43, 678kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<09:16, 929kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<06:35, 1.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<08:23, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<06:35, 1.30MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<04:51, 1.76MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<03:28, 2.44MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<34:25, 247kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<24:49, 342kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<17:32, 482kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<14:12, 593kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<10:47, 779kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<07:45, 1.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<07:22, 1.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<05:52, 1.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:22, 1.91MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<03:08, 2.64MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<33:26, 248kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<25:06, 330kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<17:58, 460kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<12:35, 652kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<8:01:52, 17.0kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<5:37:53, 24.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<3:55:59, 34.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<2:46:22, 49.0kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<1:57:12, 69.4kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<1:21:57, 99.0kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<59:02, 137kB/s]   .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<42:07, 192kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<29:36, 272kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<22:31, 356kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<16:34, 483kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<11:44, 679kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<10:04, 787kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<07:51, 1.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<05:41, 1.39MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<05:49, 1.35MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<04:53, 1.61MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:36, 2.17MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:21, 1.79MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<03:51, 2.02MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<02:53, 2.69MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:50, 2.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:18, 1.80MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<03:21, 2.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<02:26, 3.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:46, 1.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<04:50, 1.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:34, 2.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:16, 1.78MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:38, 2.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:48, 2.70MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<02:02, 3.70MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<31:42, 237kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<22:57, 328kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<16:10, 463kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<13:02, 572kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<09:53, 753kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<07:05, 1.05MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<06:41, 1.10MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:26, 1.36MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<03:58, 1.85MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:30, 1.62MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:53, 1.88MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<02:54, 2.51MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:44, 1.94MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:13, 2.24MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:27, 2.94MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<02:05, 3.43MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<5:14:11, 22.9kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<3:39:55, 32.6kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<2:33:18, 46.6kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<1:49:39, 64.9kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<1:18:15, 90.9kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<55:00, 129kB/s]   .vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<38:25, 184kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<29:46, 237kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<21:33, 327kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<15:11, 462kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<12:13, 571kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<09:09, 761kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<06:34, 1.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<06:12, 1.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<05:45, 1.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:23, 1.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<03:08, 2.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<50:59, 134kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<36:21, 188kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<25:32, 267kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<19:21, 350kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<14:14, 476kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<10:06, 668kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<08:37, 778kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<06:43, 997kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<04:51, 1.37MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:57, 1.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:08, 1.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<03:03, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:41, 1.78MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:56, 1.67MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:05, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<02:13, 2.93MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<6:14:57, 17.3kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<4:22:53, 24.7kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<3:03:26, 35.3kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<2:09:12, 49.8kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<1:31:43, 70.1kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<1:04:21, 99.7kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<44:53, 142kB/s]   .vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<34:16, 186kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<24:38, 258kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<17:20, 365kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<13:32, 465kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<10:43, 587kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<07:45, 809kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<05:32, 1.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<05:36, 1.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:34, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:21, 1.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:46, 1.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:54, 1.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:00, 2.04MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:33<02:10, 2.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:33, 1.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:49, 1.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:49, 2.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:22, 1.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:58, 2.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:12, 2.72MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:56, 2.02MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:40, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:00, 2.94MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:47, 2.11MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:09, 1.86MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:30, 2.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<01:48, 3.22MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<08:19, 699kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<06:25, 904kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<04:37, 1.25MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<04:34, 1.26MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:40, 1.56MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:43, 2.10MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:44<01:57, 2.90MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<22:47, 249kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<17:06, 332kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<12:11, 465kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<08:36, 655kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<06:29, 866kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<3:51:39, 24.2kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<2:42:11, 34.6kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:48<1:52:43, 49.3kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<1:22:05, 67.6kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<58:40, 94.5kB/s]  .vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<41:18, 134kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<28:44, 191kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<25:53, 212kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<18:35, 294kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<13:08, 415kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<09:11, 589kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<16:36, 326kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<12:10, 444kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<08:35, 626kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<07:14, 738kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<06:10, 864kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<04:33, 1.17MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<03:15, 1.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<04:12, 1.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:29, 1.51MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:32, 2.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:59, 1.74MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:38, 1.97MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:58, 2.62MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:35, 1.99MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:20, 2.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<01:45, 2.89MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:25, 2.09MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:13, 2.28MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:40, 3.00MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:20, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:09, 2.31MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:37, 3.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:17, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:07, 2.32MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:36, 3.05MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:15, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:00, 2.42MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:36, 3.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<01:10, 4.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<05:20, 898kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<04:13, 1.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<03:04, 1.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<03:14, 1.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:45, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:02, 2.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:31, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:14, 2.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:40, 2.75MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:15, 2.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<01:57, 2.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<01:29, 3.05MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:05, 4.13MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<17:47, 254kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<12:54, 350kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<09:04, 494kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<07:21, 604kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<05:36, 792kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<04:00, 1.10MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<03:49, 1.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<03:07, 1.40MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:17, 1.90MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:36, 1.65MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:16, 1.90MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:41, 2.53MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:10, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:57, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<01:27, 2.89MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:00, 2.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:50, 2.26MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:23, 2.98MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:55, 2.13MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:46, 2.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:20, 3.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:52, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:44, 2.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:17, 3.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:09, 3.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<2:50:55, 23.3kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<1:59:29, 33.2kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<1:23:14, 47.3kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<58:44, 66.5kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<41:51, 93.2kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<29:23, 132kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<20:55, 183kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<15:03, 254kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<10:34, 360kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<08:09, 461kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<06:29, 580kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<04:43, 793kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<03:18, 1.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<28:36, 129kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<20:22, 181kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<14:15, 257kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<10:43, 338kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<07:52, 461kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<05:33, 647kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<04:41, 759kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:39, 973kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:37, 1.35MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:38, 1.32MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:12, 1.58MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:36, 2.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:55, 1.78MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:41, 2.01MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:15, 2.70MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:40, 2.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:31, 2.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:08, 2.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:33, 2.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:25, 2.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:04, 3.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:30, 2.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:20, 2.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<00:59, 3.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<00:43, 4.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<12:21, 255kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<08:57, 351kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<06:17, 495kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<05:05, 606kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<04:11, 735kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:03, 1.00MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<02:09, 1.41MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<03:12, 939kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:33, 1.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:51, 1.61MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:58, 1.50MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:40, 1.75MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:14, 2.35MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<01:32, 1.87MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:22, 2.10MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:00, 2.81MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:21, 2.06MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:14, 2.26MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<00:55, 2.98MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:17, 2.12MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:10, 2.31MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<00:52, 3.07MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:14, 2.15MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:08, 2.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<00:51, 3.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:12, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:06, 2.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:50, 3.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:10, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:04, 2.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<00:48, 3.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:08, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:02, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<00:47, 3.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:42, 3.38MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<1:37:50, 24.5kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<1:07:56, 35.0kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<47:04, 49.5kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<33:25, 69.7kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<23:23, 99.1kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<16:10, 141kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<12:06, 187kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<08:42, 259kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<06:04, 367kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<04:41, 468kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<03:29, 626kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<02:27, 877kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:11, 968kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:42, 1.24MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:14, 1.69MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:28<00:52, 2.35MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<08:22, 245kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<06:03, 338kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<04:14, 477kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<03:22, 588kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:46, 715kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<02:02, 969kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<01:24, 1.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<08:19, 230kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<06:00, 318kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<04:11, 450kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<03:18, 559kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<02:28, 747kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:44, 1.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<01:13, 1.46MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<07:35, 235kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<05:40, 313kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<04:01, 439kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<02:47, 621kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<02:41, 637kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<02:03, 832kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:27, 1.15MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:23, 1.19MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:08, 1.44MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:49, 1.96MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:56, 1.68MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:58, 1.61MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:45, 2.06MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:44<00:31, 2.86MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<44:18, 34.0kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<31:02, 48.4kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<21:23, 69.0kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<14:56, 96.4kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<10:44, 134kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<07:31, 189kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<05:09, 269kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<04:10, 328kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:49<03:03, 447kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<02:07, 629kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:45, 742kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:21, 956kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:57, 1.32MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:56, 1.30MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<00:47, 1.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:34, 2.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:39, 1.76MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:34, 2.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:25, 2.66MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:32, 2.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:36, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:28, 2.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:19, 3.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:42, 1.46MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:35, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:25, 2.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:30, 1.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:27, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:20, 2.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:26, 2.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:29, 1.83MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:22, 2.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:17, 2.86MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<33:52, 24.5kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<23:26, 34.9kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<15:48, 49.8kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<10:55, 69.5kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<07:45, 97.3kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<05:22, 138kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:07<03:30, 197kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<08:04, 85.5kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<05:40, 121kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<03:50, 172kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<02:40, 232kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<01:55, 320kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<01:18, 452kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:59, 560kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:44, 739kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:30, 1.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:26, 1.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:21, 1.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:14, 1.84MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:15, 1.62MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:16, 1.54MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:07, 2.75MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:14, 1.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:12, 1.69MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:09, 1.85MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:05, 2.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:21<00:03, 3.83MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:53, 238kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:39, 318kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:26, 445kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:16, 628kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:12, 702kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:08, 906kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:05, 1.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:03, 1.26MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.52MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 2.07MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.74MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 857/400000 [00:00<00:46, 8569.35it/s]  0%|          | 1674/400000 [00:00<00:47, 8444.75it/s]  1%|          | 2536/400000 [00:00<00:46, 8494.80it/s]  1%|          | 3405/400000 [00:00<00:46, 8551.04it/s]  1%|          | 4270/400000 [00:00<00:46, 8580.33it/s]  1%|▏         | 5131/400000 [00:00<00:45, 8588.86it/s]  2%|▏         | 6011/400000 [00:00<00:45, 8649.31it/s]  2%|▏         | 6903/400000 [00:00<00:45, 8727.54it/s]  2%|▏         | 7771/400000 [00:00<00:45, 8711.22it/s]  2%|▏         | 8629/400000 [00:01<00:45, 8670.20it/s]  2%|▏         | 9473/400000 [00:01<00:45, 8585.02it/s]  3%|▎         | 10315/400000 [00:01<00:45, 8528.59it/s]  3%|▎         | 11189/400000 [00:01<00:45, 8588.82it/s]  3%|▎         | 12040/400000 [00:01<00:45, 8551.81it/s]  3%|▎         | 12900/400000 [00:01<00:45, 8565.18it/s]  3%|▎         | 13753/400000 [00:01<00:45, 8529.83it/s]  4%|▎         | 14604/400000 [00:01<00:45, 8522.55it/s]  4%|▍         | 15455/400000 [00:01<00:45, 8511.73it/s]  4%|▍         | 16321/400000 [00:01<00:44, 8554.53it/s]  4%|▍         | 17195/400000 [00:02<00:44, 8608.84it/s]  5%|▍         | 18056/400000 [00:02<00:44, 8588.19it/s]  5%|▍         | 18915/400000 [00:02<00:44, 8475.37it/s]  5%|▍         | 19771/400000 [00:02<00:44, 8500.50it/s]  5%|▌         | 20622/400000 [00:02<00:45, 8356.99it/s]  5%|▌         | 21491/400000 [00:02<00:44, 8453.13it/s]  6%|▌         | 22352/400000 [00:02<00:44, 8498.89it/s]  6%|▌         | 23241/400000 [00:02<00:43, 8611.55it/s]  6%|▌         | 24112/400000 [00:02<00:43, 8639.99it/s]  6%|▌         | 24977/400000 [00:02<00:43, 8607.00it/s]  6%|▋         | 25851/400000 [00:03<00:43, 8645.79it/s]  7%|▋         | 26719/400000 [00:03<00:43, 8654.23it/s]  7%|▋         | 27585/400000 [00:03<00:43, 8564.73it/s]  7%|▋         | 28442/400000 [00:03<00:43, 8534.18it/s]  7%|▋         | 29296/400000 [00:03<00:44, 8420.80it/s]  8%|▊         | 30140/400000 [00:03<00:43, 8425.12it/s]  8%|▊         | 31019/400000 [00:03<00:43, 8531.20it/s]  8%|▊         | 31882/400000 [00:03<00:43, 8559.62it/s]  8%|▊         | 32758/400000 [00:03<00:42, 8617.08it/s]  8%|▊         | 33629/400000 [00:03<00:42, 8643.97it/s]  9%|▊         | 34508/400000 [00:04<00:42, 8684.80it/s]  9%|▉         | 35377/400000 [00:04<00:42, 8646.03it/s]  9%|▉         | 36276/400000 [00:04<00:41, 8744.42it/s]  9%|▉         | 37151/400000 [00:04<00:41, 8700.85it/s] 10%|▉         | 38022/400000 [00:04<00:41, 8675.52it/s] 10%|▉         | 38919/400000 [00:04<00:41, 8759.91it/s] 10%|▉         | 39804/400000 [00:04<00:40, 8786.27it/s] 10%|█         | 40691/400000 [00:04<00:40, 8808.49it/s] 10%|█         | 41573/400000 [00:04<00:40, 8788.71it/s] 11%|█         | 42453/400000 [00:04<00:40, 8763.29it/s] 11%|█         | 43330/400000 [00:05<00:41, 8606.64it/s] 11%|█         | 44192/400000 [00:05<00:41, 8594.12it/s] 11%|█▏        | 45052/400000 [00:05<00:41, 8579.70it/s] 11%|█▏        | 45911/400000 [00:05<00:41, 8482.99it/s] 12%|█▏        | 46784/400000 [00:05<00:41, 8553.90it/s] 12%|█▏        | 47658/400000 [00:05<00:40, 8607.22it/s] 12%|█▏        | 48531/400000 [00:05<00:40, 8641.67it/s] 12%|█▏        | 49417/400000 [00:05<00:40, 8705.96it/s] 13%|█▎        | 50296/400000 [00:05<00:40, 8727.17it/s] 13%|█▎        | 51169/400000 [00:05<00:40, 8693.47it/s] 13%|█▎        | 52039/400000 [00:06<00:40, 8693.74it/s] 13%|█▎        | 52910/400000 [00:06<00:39, 8697.16it/s] 13%|█▎        | 53795/400000 [00:06<00:39, 8740.56it/s] 14%|█▎        | 54670/400000 [00:06<00:39, 8642.72it/s] 14%|█▍        | 55548/400000 [00:06<00:39, 8683.02it/s] 14%|█▍        | 56417/400000 [00:06<00:39, 8670.42it/s] 14%|█▍        | 57285/400000 [00:06<00:39, 8668.52it/s] 15%|█▍        | 58157/400000 [00:06<00:39, 8681.29it/s] 15%|█▍        | 59026/400000 [00:06<00:39, 8682.27it/s] 15%|█▍        | 59895/400000 [00:06<00:39, 8606.48it/s] 15%|█▌        | 60756/400000 [00:07<00:39, 8580.42it/s] 15%|█▌        | 61619/400000 [00:07<00:39, 8593.75it/s] 16%|█▌        | 62479/400000 [00:07<00:39, 8594.48it/s] 16%|█▌        | 63339/400000 [00:07<00:39, 8564.08it/s] 16%|█▌        | 64196/400000 [00:07<00:39, 8506.79it/s] 16%|█▋        | 65065/400000 [00:07<00:39, 8558.14it/s] 16%|█▋        | 65945/400000 [00:07<00:38, 8627.70it/s] 17%|█▋        | 66828/400000 [00:07<00:38, 8686.25it/s] 17%|█▋        | 67712/400000 [00:07<00:38, 8730.56it/s] 17%|█▋        | 68586/400000 [00:07<00:38, 8712.00it/s] 17%|█▋        | 69461/400000 [00:08<00:37, 8720.29it/s] 18%|█▊        | 70341/400000 [00:08<00:37, 8743.27it/s] 18%|█▊        | 71216/400000 [00:08<00:37, 8709.50it/s] 18%|█▊        | 72100/400000 [00:08<00:37, 8747.00it/s] 18%|█▊        | 72975/400000 [00:08<00:38, 8514.66it/s] 18%|█▊        | 73828/400000 [00:08<00:39, 8262.32it/s] 19%|█▊        | 74706/400000 [00:08<00:38, 8409.13it/s] 19%|█▉        | 75577/400000 [00:08<00:38, 8496.10it/s] 19%|█▉        | 76453/400000 [00:08<00:37, 8571.94it/s] 19%|█▉        | 77318/400000 [00:08<00:37, 8593.78it/s] 20%|█▉        | 78179/400000 [00:09<00:37, 8595.52it/s] 20%|█▉        | 79040/400000 [00:09<00:37, 8508.02it/s] 20%|█▉        | 79892/400000 [00:09<00:38, 8378.33it/s] 20%|██        | 80737/400000 [00:09<00:38, 8399.21it/s] 20%|██        | 81604/400000 [00:09<00:37, 8476.69it/s] 21%|██        | 82489/400000 [00:09<00:36, 8582.65it/s] 21%|██        | 83361/400000 [00:09<00:36, 8622.38it/s] 21%|██        | 84248/400000 [00:09<00:36, 8692.61it/s] 21%|██▏       | 85118/400000 [00:09<00:37, 8498.33it/s] 21%|██▏       | 85970/400000 [00:09<00:37, 8450.39it/s] 22%|██▏       | 86817/400000 [00:10<00:37, 8403.29it/s] 22%|██▏       | 87659/400000 [00:10<00:37, 8286.10it/s] 22%|██▏       | 88538/400000 [00:10<00:36, 8430.96it/s] 22%|██▏       | 89402/400000 [00:10<00:36, 8491.77it/s] 23%|██▎       | 90264/400000 [00:10<00:36, 8528.33it/s] 23%|██▎       | 91136/400000 [00:10<00:35, 8582.50it/s] 23%|██▎       | 92019/400000 [00:10<00:35, 8653.51it/s] 23%|██▎       | 92893/400000 [00:10<00:35, 8678.63it/s] 23%|██▎       | 93775/400000 [00:10<00:35, 8718.38it/s] 24%|██▎       | 94648/400000 [00:11<00:35, 8684.62it/s] 24%|██▍       | 95517/400000 [00:11<00:35, 8679.85it/s] 24%|██▍       | 96401/400000 [00:11<00:34, 8724.65it/s] 24%|██▍       | 97304/400000 [00:11<00:34, 8813.01it/s] 25%|██▍       | 98186/400000 [00:11<00:34, 8767.40it/s] 25%|██▍       | 99065/400000 [00:11<00:34, 8773.35it/s] 25%|██▍       | 99943/400000 [00:11<00:34, 8747.36it/s] 25%|██▌       | 100836/400000 [00:11<00:33, 8800.56it/s] 25%|██▌       | 101717/400000 [00:11<00:34, 8614.13it/s] 26%|██▌       | 102588/400000 [00:11<00:34, 8641.67it/s] 26%|██▌       | 103453/400000 [00:12<00:34, 8631.07it/s] 26%|██▌       | 104326/400000 [00:12<00:34, 8658.27it/s] 26%|██▋       | 105200/400000 [00:12<00:33, 8680.01it/s] 27%|██▋       | 106084/400000 [00:12<00:33, 8725.08it/s] 27%|██▋       | 106962/400000 [00:12<00:33, 8739.20it/s] 27%|██▋       | 107837/400000 [00:12<00:33, 8683.60it/s] 27%|██▋       | 108706/400000 [00:12<00:33, 8668.69it/s] 27%|██▋       | 109574/400000 [00:12<00:33, 8614.79it/s] 28%|██▊       | 110437/400000 [00:12<00:33, 8618.90it/s] 28%|██▊       | 111300/400000 [00:12<00:33, 8585.32it/s] 28%|██▊       | 112175/400000 [00:13<00:33, 8631.89it/s] 28%|██▊       | 113062/400000 [00:13<00:32, 8700.00it/s] 28%|██▊       | 113946/400000 [00:13<00:32, 8739.68it/s] 29%|██▊       | 114842/400000 [00:13<00:32, 8802.63it/s] 29%|██▉       | 115723/400000 [00:13<00:32, 8739.61it/s] 29%|██▉       | 116598/400000 [00:13<00:33, 8495.06it/s] 29%|██▉       | 117450/400000 [00:13<00:33, 8449.03it/s] 30%|██▉       | 118318/400000 [00:13<00:33, 8515.15it/s] 30%|██▉       | 119176/400000 [00:13<00:32, 8533.71it/s] 30%|███       | 120054/400000 [00:13<00:32, 8604.69it/s] 30%|███       | 120920/400000 [00:14<00:32, 8620.35it/s] 30%|███       | 121797/400000 [00:14<00:32, 8663.33it/s] 31%|███       | 122688/400000 [00:14<00:31, 8735.75it/s] 31%|███       | 123562/400000 [00:14<00:32, 8516.48it/s] 31%|███       | 124416/400000 [00:14<00:32, 8516.22it/s] 31%|███▏      | 125269/400000 [00:14<00:32, 8508.36it/s] 32%|███▏      | 126121/400000 [00:14<00:32, 8427.88it/s] 32%|███▏      | 126980/400000 [00:14<00:32, 8473.65it/s] 32%|███▏      | 127828/400000 [00:14<00:32, 8454.12it/s] 32%|███▏      | 128697/400000 [00:14<00:31, 8521.16it/s] 32%|███▏      | 129550/400000 [00:15<00:31, 8509.52it/s] 33%|███▎      | 130419/400000 [00:15<00:31, 8560.35it/s] 33%|███▎      | 131284/400000 [00:15<00:31, 8586.81it/s] 33%|███▎      | 132165/400000 [00:15<00:30, 8649.99it/s] 33%|███▎      | 133042/400000 [00:15<00:30, 8684.17it/s] 33%|███▎      | 133911/400000 [00:15<00:31, 8542.34it/s] 34%|███▎      | 134775/400000 [00:15<00:30, 8568.83it/s] 34%|███▍      | 135640/400000 [00:15<00:30, 8590.92it/s] 34%|███▍      | 136503/400000 [00:15<00:30, 8600.94it/s] 34%|███▍      | 137364/400000 [00:15<00:30, 8511.64it/s] 35%|███▍      | 138226/400000 [00:16<00:30, 8542.00it/s] 35%|███▍      | 139108/400000 [00:16<00:30, 8620.60it/s] 35%|███▍      | 139971/400000 [00:16<00:30, 8450.00it/s] 35%|███▌      | 140828/400000 [00:16<00:30, 8483.78it/s] 35%|███▌      | 141715/400000 [00:16<00:30, 8595.93it/s] 36%|███▌      | 142589/400000 [00:16<00:29, 8636.79it/s] 36%|███▌      | 143470/400000 [00:16<00:29, 8687.99it/s] 36%|███▌      | 144340/400000 [00:16<00:29, 8668.34it/s] 36%|███▋      | 145208/400000 [00:16<00:29, 8591.51it/s] 37%|███▋      | 146068/400000 [00:16<00:29, 8473.55it/s] 37%|███▋      | 146918/400000 [00:17<00:29, 8479.98it/s] 37%|███▋      | 147767/400000 [00:17<00:30, 8337.30it/s] 37%|███▋      | 148609/400000 [00:17<00:30, 8360.74it/s] 37%|███▋      | 149474/400000 [00:17<00:29, 8443.84it/s] 38%|███▊      | 150328/400000 [00:17<00:29, 8472.11it/s] 38%|███▊      | 151176/400000 [00:17<00:29, 8433.10it/s] 38%|███▊      | 152028/400000 [00:17<00:29, 8456.30it/s] 38%|███▊      | 152897/400000 [00:17<00:28, 8521.68it/s] 38%|███▊      | 153767/400000 [00:17<00:28, 8572.02it/s] 39%|███▊      | 154633/400000 [00:17<00:28, 8597.63it/s] 39%|███▉      | 155493/400000 [00:18<00:28, 8565.37it/s] 39%|███▉      | 156350/400000 [00:18<00:28, 8506.22it/s] 39%|███▉      | 157225/400000 [00:18<00:28, 8575.98it/s] 40%|███▉      | 158103/400000 [00:18<00:28, 8635.82it/s] 40%|███▉      | 158967/400000 [00:18<00:27, 8621.72it/s] 40%|███▉      | 159830/400000 [00:18<00:27, 8605.95it/s] 40%|████      | 160691/400000 [00:18<00:27, 8597.86it/s] 40%|████      | 161568/400000 [00:18<00:27, 8647.43it/s] 41%|████      | 162461/400000 [00:18<00:27, 8728.32it/s] 41%|████      | 163350/400000 [00:18<00:26, 8775.93it/s] 41%|████      | 164229/400000 [00:19<00:26, 8778.05it/s] 41%|████▏     | 165114/400000 [00:19<00:26, 8796.66it/s] 42%|████▏     | 166002/400000 [00:19<00:26, 8818.78it/s] 42%|████▏     | 166893/400000 [00:19<00:26, 8845.53it/s] 42%|████▏     | 167778/400000 [00:19<00:26, 8807.47it/s] 42%|████▏     | 168677/400000 [00:19<00:26, 8860.74it/s] 42%|████▏     | 169564/400000 [00:19<00:26, 8860.37it/s] 43%|████▎     | 170457/400000 [00:19<00:25, 8878.42it/s] 43%|████▎     | 171346/400000 [00:19<00:25, 8881.69it/s] 43%|████▎     | 172235/400000 [00:19<00:25, 8860.10it/s] 43%|████▎     | 173122/400000 [00:20<00:25, 8846.07it/s] 44%|████▎     | 174007/400000 [00:20<00:25, 8815.41it/s] 44%|████▎     | 174889/400000 [00:20<00:25, 8790.99it/s] 44%|████▍     | 175769/400000 [00:20<00:25, 8752.44it/s] 44%|████▍     | 176645/400000 [00:20<00:25, 8631.60it/s] 44%|████▍     | 177510/400000 [00:20<00:25, 8635.27it/s] 45%|████▍     | 178377/400000 [00:20<00:25, 8644.26it/s] 45%|████▍     | 179265/400000 [00:20<00:25, 8713.13it/s] 45%|████▌     | 180157/400000 [00:20<00:25, 8770.80it/s] 45%|████▌     | 181041/400000 [00:21<00:24, 8790.70it/s] 45%|████▌     | 181927/400000 [00:21<00:24, 8810.74it/s] 46%|████▌     | 182809/400000 [00:21<00:24, 8803.23it/s] 46%|████▌     | 183697/400000 [00:21<00:24, 8824.04it/s] 46%|████▌     | 184580/400000 [00:21<00:24, 8732.58it/s] 46%|████▋     | 185454/400000 [00:21<00:24, 8671.00it/s] 47%|████▋     | 186322/400000 [00:21<00:24, 8671.09it/s] 47%|████▋     | 187199/400000 [00:21<00:24, 8700.07it/s] 47%|████▋     | 188081/400000 [00:21<00:24, 8733.43it/s] 47%|████▋     | 188961/400000 [00:21<00:24, 8751.84it/s] 47%|████▋     | 189838/400000 [00:22<00:24, 8755.13it/s] 48%|████▊     | 190714/400000 [00:22<00:23, 8733.35it/s] 48%|████▊     | 191588/400000 [00:22<00:24, 8539.12it/s] 48%|████▊     | 192445/400000 [00:22<00:24, 8547.09it/s] 48%|████▊     | 193305/400000 [00:22<00:24, 8560.99it/s] 49%|████▊     | 194166/400000 [00:22<00:24, 8573.25it/s] 49%|████▉     | 195036/400000 [00:22<00:23, 8610.31it/s] 49%|████▉     | 195898/400000 [00:22<00:23, 8612.00it/s] 49%|████▉     | 196775/400000 [00:22<00:23, 8656.88it/s] 49%|████▉     | 197644/400000 [00:22<00:23, 8665.57it/s] 50%|████▉     | 198511/400000 [00:23<00:23, 8627.78it/s] 50%|████▉     | 199374/400000 [00:23<00:23, 8553.53it/s] 50%|█████     | 200238/400000 [00:23<00:23, 8577.81it/s] 50%|█████     | 201096/400000 [00:23<00:23, 8558.71it/s] 50%|█████     | 201953/400000 [00:23<00:23, 8547.34it/s] 51%|█████     | 202825/400000 [00:23<00:22, 8595.57it/s] 51%|█████     | 203686/400000 [00:23<00:22, 8597.74it/s] 51%|█████     | 204546/400000 [00:23<00:22, 8582.10it/s] 51%|█████▏    | 205432/400000 [00:23<00:22, 8662.47it/s] 52%|█████▏    | 206314/400000 [00:23<00:22, 8708.20it/s] 52%|█████▏    | 207199/400000 [00:24<00:22, 8749.76it/s] 52%|█████▏    | 208075/400000 [00:24<00:21, 8747.52it/s] 52%|█████▏    | 208955/400000 [00:24<00:21, 8761.37it/s] 52%|█████▏    | 209832/400000 [00:24<00:21, 8751.30it/s] 53%|█████▎    | 210708/400000 [00:24<00:21, 8671.91it/s] 53%|█████▎    | 211583/400000 [00:24<00:21, 8694.18it/s] 53%|█████▎    | 212468/400000 [00:24<00:21, 8740.31it/s] 53%|█████▎    | 213348/400000 [00:24<00:21, 8757.59it/s] 54%|█████▎    | 214236/400000 [00:24<00:21, 8792.92it/s] 54%|█████▍    | 215116/400000 [00:24<00:21, 8719.56it/s] 54%|█████▍    | 215989/400000 [00:25<00:21, 8695.26it/s] 54%|█████▍    | 216859/400000 [00:25<00:21, 8669.09it/s] 54%|█████▍    | 217742/400000 [00:25<00:20, 8714.35it/s] 55%|█████▍    | 218621/400000 [00:25<00:20, 8735.25it/s] 55%|█████▍    | 219495/400000 [00:25<00:20, 8682.40it/s] 55%|█████▌    | 220364/400000 [00:25<00:20, 8611.82it/s] 55%|█████▌    | 221236/400000 [00:25<00:20, 8643.72it/s] 56%|█████▌    | 222121/400000 [00:25<00:20, 8701.82it/s] 56%|█████▌    | 223013/400000 [00:25<00:20, 8765.04it/s] 56%|█████▌    | 223903/400000 [00:25<00:20, 8804.21it/s] 56%|█████▌    | 224787/400000 [00:26<00:19, 8814.91it/s] 56%|█████▋    | 225675/400000 [00:26<00:19, 8833.76it/s] 57%|█████▋    | 226559/400000 [00:26<00:19, 8794.44it/s] 57%|█████▋    | 227449/400000 [00:26<00:19, 8824.64it/s] 57%|█████▋    | 228338/400000 [00:26<00:19, 8842.98it/s] 57%|█████▋    | 229224/400000 [00:26<00:19, 8845.17it/s] 58%|█████▊    | 230109/400000 [00:26<00:19, 8834.17it/s] 58%|█████▊    | 230993/400000 [00:26<00:19, 8799.83it/s] 58%|█████▊    | 231874/400000 [00:26<00:19, 8786.54it/s] 58%|█████▊    | 232753/400000 [00:26<00:19, 8733.65it/s] 58%|█████▊    | 233627/400000 [00:27<00:19, 8615.91it/s] 59%|█████▊    | 234490/400000 [00:27<00:19, 8569.70it/s] 59%|█████▉    | 235348/400000 [00:27<00:19, 8538.68it/s] 59%|█████▉    | 236223/400000 [00:27<00:19, 8600.27it/s] 59%|█████▉    | 237101/400000 [00:27<00:18, 8627.91it/s] 59%|█████▉    | 237965/400000 [00:27<00:18, 8587.69it/s] 60%|█████▉    | 238843/400000 [00:27<00:18, 8642.44it/s] 60%|█████▉    | 239723/400000 [00:27<00:18, 8687.55it/s] 60%|██████    | 240607/400000 [00:27<00:18, 8731.48it/s] 60%|██████    | 241484/400000 [00:27<00:18, 8742.11it/s] 61%|██████    | 242359/400000 [00:28<00:18, 8642.44it/s] 61%|██████    | 243237/400000 [00:28<00:18, 8682.30it/s] 61%|██████    | 244120/400000 [00:28<00:17, 8724.32it/s] 61%|██████▏   | 245002/400000 [00:28<00:17, 8751.54it/s] 61%|██████▏   | 245878/400000 [00:28<00:17, 8640.06it/s] 62%|██████▏   | 246743/400000 [00:28<00:17, 8630.10it/s] 62%|██████▏   | 247617/400000 [00:28<00:17, 8661.81it/s] 62%|██████▏   | 248484/400000 [00:28<00:17, 8637.54it/s] 62%|██████▏   | 249348/400000 [00:28<00:17, 8631.64it/s] 63%|██████▎   | 250212/400000 [00:28<00:17, 8603.94it/s] 63%|██████▎   | 251073/400000 [00:29<00:17, 8575.75it/s] 63%|██████▎   | 251931/400000 [00:29<00:17, 8495.16it/s] 63%|██████▎   | 252781/400000 [00:29<00:17, 8466.26it/s] 63%|██████▎   | 253646/400000 [00:29<00:17, 8518.59it/s] 64%|██████▎   | 254499/400000 [00:29<00:17, 8229.52it/s] 64%|██████▍   | 255329/400000 [00:29<00:17, 8249.52it/s] 64%|██████▍   | 256169/400000 [00:29<00:17, 8292.32it/s] 64%|██████▍   | 257023/400000 [00:29<00:17, 8363.72it/s] 64%|██████▍   | 257910/400000 [00:29<00:16, 8506.86it/s] 65%|██████▍   | 258791/400000 [00:29<00:16, 8594.15it/s] 65%|██████▍   | 259656/400000 [00:30<00:16, 8609.66it/s] 65%|██████▌   | 260545/400000 [00:30<00:16, 8690.96it/s] 65%|██████▌   | 261430/400000 [00:30<00:15, 8736.23it/s] 66%|██████▌   | 262305/400000 [00:30<00:15, 8732.15it/s] 66%|██████▌   | 263179/400000 [00:30<00:15, 8733.60it/s] 66%|██████▌   | 264065/400000 [00:30<00:15, 8770.95it/s] 66%|██████▌   | 264943/400000 [00:30<00:15, 8702.11it/s] 66%|██████▋   | 265823/400000 [00:30<00:15, 8730.52it/s] 67%|██████▋   | 266697/400000 [00:30<00:15, 8443.01it/s] 67%|██████▋   | 267544/400000 [00:31<00:15, 8430.21it/s] 67%|██████▋   | 268389/400000 [00:31<00:15, 8419.51it/s] 67%|██████▋   | 269254/400000 [00:31<00:15, 8485.25it/s] 68%|██████▊   | 270113/400000 [00:31<00:15, 8516.34it/s] 68%|██████▊   | 270997/400000 [00:31<00:14, 8610.74it/s] 68%|██████▊   | 271904/400000 [00:31<00:14, 8742.63it/s] 68%|██████▊   | 272782/400000 [00:31<00:14, 8753.18it/s] 68%|██████▊   | 273667/400000 [00:31<00:14, 8781.98it/s] 69%|██████▊   | 274546/400000 [00:31<00:14, 8770.60it/s] 69%|██████▉   | 275427/400000 [00:31<00:14, 8781.74it/s] 69%|██████▉   | 276306/400000 [00:32<00:14, 8706.18it/s] 69%|██████▉   | 277186/400000 [00:32<00:14, 8732.98it/s] 70%|██████▉   | 278060/400000 [00:32<00:13, 8726.29it/s] 70%|██████▉   | 278933/400000 [00:32<00:13, 8701.58it/s] 70%|██████▉   | 279804/400000 [00:32<00:13, 8682.47it/s] 70%|███████   | 280673/400000 [00:32<00:13, 8678.61it/s] 70%|███████   | 281541/400000 [00:32<00:13, 8515.16it/s] 71%|███████   | 282401/400000 [00:32<00:13, 8538.19it/s] 71%|███████   | 283256/400000 [00:32<00:13, 8500.89it/s] 71%|███████   | 284114/400000 [00:32<00:13, 8521.37it/s] 71%|███████   | 284986/400000 [00:33<00:13, 8577.89it/s] 71%|███████▏  | 285856/400000 [00:33<00:13, 8614.11it/s] 72%|███████▏  | 286748/400000 [00:33<00:13, 8700.69it/s] 72%|███████▏  | 287634/400000 [00:33<00:12, 8747.21it/s] 72%|███████▏  | 288510/400000 [00:33<00:12, 8716.86it/s] 72%|███████▏  | 289408/400000 [00:33<00:12, 8794.08it/s] 73%|███████▎  | 290290/400000 [00:33<00:12, 8799.79it/s] 73%|███████▎  | 291171/400000 [00:33<00:12, 8621.41it/s] 73%|███████▎  | 292036/400000 [00:33<00:12, 8625.05it/s] 73%|███████▎  | 292900/400000 [00:33<00:12, 8552.44it/s] 73%|███████▎  | 293772/400000 [00:34<00:12, 8601.89it/s] 74%|███████▎  | 294633/400000 [00:34<00:12, 8574.85it/s] 74%|███████▍  | 295516/400000 [00:34<00:12, 8648.77it/s] 74%|███████▍  | 296391/400000 [00:34<00:11, 8678.34it/s] 74%|███████▍  | 297260/400000 [00:34<00:11, 8680.73it/s] 75%|███████▍  | 298144/400000 [00:34<00:11, 8726.25it/s] 75%|███████▍  | 299022/400000 [00:34<00:11, 8740.52it/s] 75%|███████▍  | 299899/400000 [00:34<00:11, 8747.70it/s] 75%|███████▌  | 300774/400000 [00:34<00:11, 8735.05it/s] 75%|███████▌  | 301652/400000 [00:34<00:11, 8745.91it/s] 76%|███████▌  | 302530/400000 [00:35<00:11, 8755.11it/s] 76%|███████▌  | 303406/400000 [00:35<00:11, 8754.51it/s] 76%|███████▌  | 304284/400000 [00:35<00:10, 8761.46it/s] 76%|███████▋  | 305169/400000 [00:35<00:10, 8785.81it/s] 77%|███████▋  | 306056/400000 [00:35<00:10, 8810.61it/s] 77%|███████▋  | 306941/400000 [00:35<00:10, 8821.25it/s] 77%|███████▋  | 307824/400000 [00:35<00:10, 8804.23it/s] 77%|███████▋  | 308705/400000 [00:35<00:10, 8761.10it/s] 77%|███████▋  | 309585/400000 [00:35<00:10, 8770.70it/s] 78%|███████▊  | 310463/400000 [00:35<00:10, 8610.10it/s] 78%|███████▊  | 311325/400000 [00:36<00:10, 8524.67it/s] 78%|███████▊  | 312179/400000 [00:36<00:10, 8510.11it/s] 78%|███████▊  | 313031/400000 [00:36<00:10, 8489.87it/s] 78%|███████▊  | 313886/400000 [00:36<00:10, 8506.11it/s] 79%|███████▊  | 314737/400000 [00:36<00:10, 8495.04it/s] 79%|███████▉  | 315611/400000 [00:36<00:09, 8566.32it/s] 79%|███████▉  | 316488/400000 [00:36<00:09, 8625.52it/s] 79%|███████▉  | 317361/400000 [00:36<00:09, 8656.02it/s] 80%|███████▉  | 318232/400000 [00:36<00:09, 8670.17it/s] 80%|███████▉  | 319100/400000 [00:36<00:09, 8665.34it/s] 80%|███████▉  | 319975/400000 [00:37<00:09, 8689.19it/s] 80%|████████  | 320854/400000 [00:37<00:09, 8718.91it/s] 80%|████████  | 321750/400000 [00:37<00:08, 8789.19it/s] 81%|████████  | 322630/400000 [00:37<00:08, 8778.33it/s] 81%|████████  | 323508/400000 [00:37<00:08, 8765.32it/s] 81%|████████  | 324395/400000 [00:37<00:08, 8795.91it/s] 81%|████████▏ | 325275/400000 [00:37<00:08, 8769.49it/s] 82%|████████▏ | 326153/400000 [00:37<00:08, 8752.95it/s] 82%|████████▏ | 327039/400000 [00:37<00:08, 8782.69it/s] 82%|████████▏ | 327918/400000 [00:37<00:08, 8722.42it/s] 82%|████████▏ | 328795/400000 [00:38<00:08, 8735.11it/s] 82%|████████▏ | 329669/400000 [00:38<00:08, 8712.22it/s] 83%|████████▎ | 330549/400000 [00:38<00:07, 8737.17it/s] 83%|████████▎ | 331435/400000 [00:38<00:07, 8771.20it/s] 83%|████████▎ | 332313/400000 [00:38<00:07, 8646.03it/s] 83%|████████▎ | 333181/400000 [00:38<00:07, 8655.29it/s] 84%|████████▎ | 334051/400000 [00:38<00:07, 8666.05it/s] 84%|████████▎ | 334918/400000 [00:38<00:07, 8642.48it/s] 84%|████████▍ | 335797/400000 [00:38<00:07, 8683.85it/s] 84%|████████▍ | 336666/400000 [00:38<00:07, 8602.65it/s] 84%|████████▍ | 337527/400000 [00:39<00:07, 8569.72it/s] 85%|████████▍ | 338385/400000 [00:39<00:07, 8448.08it/s] 85%|████████▍ | 339231/400000 [00:39<00:07, 8371.50it/s] 85%|████████▌ | 340069/400000 [00:39<00:07, 8259.28it/s] 85%|████████▌ | 340896/400000 [00:39<00:07, 8187.59it/s] 85%|████████▌ | 341725/400000 [00:39<00:07, 8216.47it/s] 86%|████████▌ | 342590/400000 [00:39<00:06, 8341.85it/s] 86%|████████▌ | 343460/400000 [00:39<00:06, 8444.95it/s] 86%|████████▌ | 344314/400000 [00:39<00:06, 8472.25it/s] 86%|████████▋ | 345168/400000 [00:39<00:06, 8491.57it/s] 87%|████████▋ | 346028/400000 [00:40<00:06, 8520.88it/s] 87%|████████▋ | 346893/400000 [00:40<00:06, 8558.46it/s] 87%|████████▋ | 347765/400000 [00:40<00:06, 8602.13it/s] 87%|████████▋ | 348639/400000 [00:40<00:05, 8641.24it/s] 87%|████████▋ | 349504/400000 [00:40<00:05, 8630.87it/s] 88%|████████▊ | 350371/400000 [00:40<00:05, 8641.32it/s] 88%|████████▊ | 351244/400000 [00:40<00:05, 8665.32it/s] 88%|████████▊ | 352111/400000 [00:40<00:05, 8663.05it/s] 88%|████████▊ | 352978/400000 [00:40<00:05, 8595.11it/s] 88%|████████▊ | 353838/400000 [00:40<00:05, 8533.62it/s] 89%|████████▊ | 354702/400000 [00:41<00:05, 8564.92it/s] 89%|████████▉ | 355567/400000 [00:41<00:05, 8589.13it/s] 89%|████████▉ | 356427/400000 [00:41<00:05, 8517.27it/s] 89%|████████▉ | 357279/400000 [00:41<00:05, 8353.60it/s] 90%|████████▉ | 358134/400000 [00:41<00:04, 8409.41it/s] 90%|████████▉ | 358998/400000 [00:41<00:04, 8475.90it/s] 90%|████████▉ | 359864/400000 [00:41<00:04, 8529.32it/s] 90%|█████████ | 360732/400000 [00:41<00:04, 8573.79it/s] 90%|█████████ | 361600/400000 [00:41<00:04, 8603.52it/s] 91%|█████████ | 362461/400000 [00:42<00:04, 8566.02it/s] 91%|█████████ | 363326/400000 [00:42<00:04, 8589.50it/s] 91%|█████████ | 364186/400000 [00:42<00:04, 8483.13it/s] 91%|█████████▏| 365035/400000 [00:42<00:04, 8228.41it/s] 91%|█████████▏| 365860/400000 [00:42<00:04, 8172.56it/s] 92%|█████████▏| 366679/400000 [00:42<00:04, 8061.11it/s] 92%|█████████▏| 367538/400000 [00:42<00:03, 8212.68it/s] 92%|█████████▏| 368397/400000 [00:42<00:03, 8320.92it/s] 92%|█████████▏| 369259/400000 [00:42<00:03, 8407.85it/s] 93%|█████████▎| 370122/400000 [00:42<00:03, 8472.56it/s] 93%|█████████▎| 370971/400000 [00:43<00:03, 8432.62it/s] 93%|█████████▎| 371816/400000 [00:43<00:03, 8377.22it/s] 93%|█████████▎| 372666/400000 [00:43<00:03, 8411.13it/s] 93%|█████████▎| 373549/400000 [00:43<00:03, 8530.13it/s] 94%|█████████▎| 374420/400000 [00:43<00:02, 8581.34it/s] 94%|█████████▍| 375279/400000 [00:43<00:02, 8578.56it/s] 94%|█████████▍| 376153/400000 [00:43<00:02, 8623.74it/s] 94%|█████████▍| 377037/400000 [00:43<00:02, 8687.25it/s] 94%|█████████▍| 377927/400000 [00:43<00:02, 8748.91it/s] 95%|█████████▍| 378814/400000 [00:43<00:02, 8784.32it/s] 95%|█████████▍| 379693/400000 [00:44<00:02, 8679.84it/s] 95%|█████████▌| 380574/400000 [00:44<00:02, 8717.82it/s] 95%|█████████▌| 381460/400000 [00:44<00:02, 8757.58it/s] 96%|█████████▌| 382342/400000 [00:44<00:02, 8739.95it/s] 96%|█████████▌| 383217/400000 [00:44<00:01, 8674.63it/s] 96%|█████████▌| 384085/400000 [00:44<00:01, 8657.41it/s] 96%|█████████▌| 384953/400000 [00:44<00:01, 8663.08it/s] 96%|█████████▋| 385820/400000 [00:44<00:01, 8661.82it/s] 97%|█████████▋| 386694/400000 [00:44<00:01, 8684.63it/s] 97%|█████████▋| 387563/400000 [00:44<00:01, 8683.67it/s] 97%|█████████▋| 388432/400000 [00:45<00:01, 8656.65it/s] 97%|█████████▋| 389306/400000 [00:45<00:01, 8680.53it/s] 98%|█████████▊| 390178/400000 [00:45<00:01, 8691.49it/s] 98%|█████████▊| 391055/400000 [00:45<00:01, 8714.82it/s] 98%|█████████▊| 391927/400000 [00:45<00:00, 8701.02it/s] 98%|█████████▊| 392798/400000 [00:45<00:00, 8596.42it/s] 98%|█████████▊| 393660/400000 [00:45<00:00, 8600.57it/s] 99%|█████████▊| 394521/400000 [00:45<00:00, 8592.45it/s] 99%|█████████▉| 395381/400000 [00:45<00:00, 8593.54it/s] 99%|█████████▉| 396241/400000 [00:45<00:00, 8523.10it/s] 99%|█████████▉| 397094/400000 [00:46<00:00, 8387.58it/s] 99%|█████████▉| 397936/400000 [00:46<00:00, 8397.17it/s]100%|█████████▉| 398777/400000 [00:46<00:00, 8348.90it/s]100%|█████████▉| 399613/400000 [00:46<00:00, 8229.54it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8619.52it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f3c49e41940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010838254667999748 	 Accuracy: 55
Train Epoch: 1 	 Loss: 0.010934856823056836 	 Accuracy: 68

  model saves at 68% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15867 out of table with 15852 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15867 out of table with 15852 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-14 09:23:05.487055: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 09:23:05.491729: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095230000 Hz
2020-05-14 09:23:05.491878: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fb536e1380 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 09:23:05.491892: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f3bf6820dd8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.4980 - accuracy: 0.5110
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5516 - accuracy: 0.5075 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5593 - accuracy: 0.5070
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6628 - accuracy: 0.5002
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5992 - accuracy: 0.5044
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5542 - accuracy: 0.5073
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6009 - accuracy: 0.5043
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5746 - accuracy: 0.5060
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6377 - accuracy: 0.5019
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5976 - accuracy: 0.5045
11000/25000 [============>.................] - ETA: 3s - loss: 7.6137 - accuracy: 0.5035
12000/25000 [=============>................] - ETA: 3s - loss: 7.6015 - accuracy: 0.5042
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5935 - accuracy: 0.5048
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6097 - accuracy: 0.5037
15000/25000 [=================>............] - ETA: 2s - loss: 7.6094 - accuracy: 0.5037
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6225 - accuracy: 0.5029
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6314 - accuracy: 0.5023
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6138 - accuracy: 0.5034
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6045 - accuracy: 0.5041
20000/25000 [=======================>......] - ETA: 1s - loss: 7.5930 - accuracy: 0.5048
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6206 - accuracy: 0.5030
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6374 - accuracy: 0.5019
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6526 - accuracy: 0.5009
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6513 - accuracy: 0.5010
25000/25000 [==============================] - 8s 302us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f3bb6792198> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f3bb79f1128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.7943 - crf_viterbi_accuracy: 0.3333 - val_loss: 1.8372 - val_crf_viterbi_accuracy: 0.2800

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

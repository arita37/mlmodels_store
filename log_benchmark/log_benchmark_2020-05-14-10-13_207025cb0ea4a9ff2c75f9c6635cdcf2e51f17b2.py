
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f0596219fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 10:13:42.473732
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 10:13:42.478266
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 10:13:42.482600
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 10:13:42.486509
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f05a22314a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 360239.7188
Epoch 2/10

1/1 [==============================] - 0s 102ms/step - loss: 301148.3438
Epoch 3/10

1/1 [==============================] - 0s 97ms/step - loss: 235058.0938
Epoch 4/10

1/1 [==============================] - 0s 96ms/step - loss: 169935.0156
Epoch 5/10

1/1 [==============================] - 0s 100ms/step - loss: 121120.4219
Epoch 6/10

1/1 [==============================] - 0s 92ms/step - loss: 86778.9766
Epoch 7/10

1/1 [==============================] - 0s 101ms/step - loss: 63057.2656
Epoch 8/10

1/1 [==============================] - 0s 92ms/step - loss: 46898.6328
Epoch 9/10

1/1 [==============================] - 0s 92ms/step - loss: 35974.1875
Epoch 10/10

1/1 [==============================] - 0s 95ms/step - loss: 28322.6641

  #### Inference Need return ypred, ytrue ######################### 
[[ 5.0873149e-02 -6.0636258e-01  7.6412129e-01  7.7421474e-01
   9.6719444e-01  6.2738562e-01  3.4197605e-01 -5.0473845e-01
  -2.7348618e-03 -3.6240470e-01  9.1759050e-01 -2.6278105e-01
   1.6859038e-02 -4.2658916e-01 -4.5901340e-02  8.3087385e-01
   1.0511175e-01 -8.3474535e-01  7.4663973e-01  9.3195337e-01
   3.8884988e-01  7.3417008e-01  8.4692456e-02  6.8228465e-01
  -4.1777343e-02  7.2945911e-01 -1.8079488e-01  2.7661121e-01
   6.8343049e-01  5.7064092e-01 -8.4947646e-01  4.6745223e-01
   4.0919414e-01 -8.3230174e-01 -8.4893960e-01 -8.0397391e-01
   8.0541772e-01 -2.1026471e-01 -1.8669905e-01  4.9139443e-01
  -2.6065546e-01  3.2630959e-01  8.4859949e-01 -4.9776059e-01
  -9.7835845e-01 -6.4980298e-01 -6.4940399e-01  3.8639840e-01
  -4.6136636e-01  2.1693105e-01 -9.2215538e-01  5.7718652e-01
  -2.9802972e-01 -4.8819169e-01 -7.3916622e-02 -9.8071945e-01
   9.5967203e-01  4.0285909e-01 -4.6150872e-01  7.3043156e-01
   2.0889977e-02 -8.3991349e-01 -6.2542826e-01 -8.5688286e-02
  -6.7825836e-01 -4.5180202e-01 -3.5965976e-01  7.7337575e-01
  -2.8534323e-01 -5.2254087e-01 -6.2419999e-01 -7.1007133e-01
   8.7108660e-01 -8.9046431e-01 -8.0653846e-02  2.6960003e-01
   6.7234576e-01  8.8994616e-01  4.3948439e-01  4.4582754e-01
   6.0952310e-02 -4.0650827e-01 -8.9930278e-01 -1.0777370e-01
   3.5344595e-01 -4.8682418e-01 -3.0669242e-01  4.8621213e-01
  -1.4437954e-01 -4.2855009e-01  2.6020417e-01  1.8592685e-01
   6.7694235e-01  3.9281929e-01  1.9102341e-01  7.0782590e-01
  -1.3323586e-01 -3.7900513e-01  8.4627336e-01  9.7103870e-01
  -4.2637151e-01  9.0737984e-02 -1.4333272e-01  4.1509411e-01
   9.6686655e-01 -6.2065965e-01 -6.2054026e-01 -7.9919285e-01
  -3.7257874e-01 -7.5193471e-01 -9.1791636e-01  5.9263325e-01
  -5.4342324e-01 -4.0818434e-02 -1.4144153e-02 -2.4945875e-01
  -2.0301314e-01 -8.4372991e-01 -1.6981986e-01  3.8070464e-01
  -1.0280832e-01  3.2677412e+00  2.3993609e+00  3.8946826e+00
   2.5872629e+00  3.5362735e+00  3.1945310e+00  2.8950758e+00
   2.3887262e+00  2.5499063e+00  3.5205269e+00  2.2026579e+00
   3.9055932e+00  3.8357432e+00  3.6190948e+00  2.8141754e+00
   3.3000546e+00  3.0836976e+00  3.3615911e+00  2.6961851e+00
   2.9272571e+00  2.9746287e+00  2.7903512e+00  3.6574256e+00
   2.3912833e+00  2.9071107e+00  2.8210216e+00  3.8005350e+00
   4.0542254e+00  3.1936917e+00  3.7891812e+00  3.8927724e+00
   3.1088548e+00  2.4304123e+00  2.6636879e+00  3.1663649e+00
   2.8664670e+00  2.1566269e+00  3.1510775e+00  2.3539999e+00
   3.2717748e+00  3.7561605e+00  3.2150385e+00  3.2811313e+00
   3.6527548e+00  3.9052176e+00  2.8829501e+00  3.3241758e+00
   3.6181817e+00  2.5328722e+00  3.4221809e+00  3.1558826e+00
   3.3376079e+00  3.8197837e+00  3.9652326e+00  3.7990625e+00
   3.1157582e+00  2.7683377e+00  3.0986567e+00  2.2880967e+00
   1.9159845e+00  1.6716974e+00  1.0003983e+00  1.9683645e+00
   7.8229356e-01  1.4074725e+00  1.7811809e+00  4.6839690e-01
   1.4631939e+00  5.2208054e-01  1.7762076e+00  1.7579319e+00
   1.0814086e+00  4.4648457e-01  4.5410508e-01  8.2339787e-01
   1.1667757e+00  1.0421841e+00  1.7949216e+00  1.9395936e+00
   8.7219101e-01  4.2740732e-01  8.8220924e-01  1.9793704e+00
   4.1356969e-01  4.7681141e-01  1.2213688e+00  9.9460065e-01
   7.4989492e-01  6.7524362e-01  1.6737242e+00  1.9100404e+00
   8.5999799e-01  7.2571218e-01  7.8937882e-01  6.5991604e-01
   4.2592645e-01  8.3838463e-01  1.5112839e+00  1.8918221e+00
   7.2247779e-01  1.6714859e+00  1.3895031e+00  1.3225363e+00
   1.0185342e+00  4.0109408e-01  4.7119629e-01  1.7513676e+00
   3.8951194e-01  5.6383294e-01  1.1095752e+00  1.1692120e+00
   5.0231093e-01  1.0597069e+00  1.2532406e+00  6.9405419e-01
   5.2491212e-01  1.1812111e+00  3.8887441e-01  9.4660783e-01
   5.1604712e-01  6.0294205e-01  1.4758413e+00  4.4518578e-01
   1.7594733e+00  7.0175314e-01  1.5470257e+00  4.4654489e-01
   1.2061093e+00  1.3437849e+00  1.1602315e+00  5.2596694e-01
   1.5330381e+00  5.4789078e-01  1.0511887e+00  5.5405027e-01
   1.0743968e+00  1.7411165e+00  1.5016934e+00  5.1651061e-01
   1.0299574e+00  4.2348009e-01  1.5627465e+00  9.9863493e-01
   1.9707441e+00  1.0142672e+00  7.5112474e-01  7.5424516e-01
   1.6196730e+00  1.8637247e+00  1.5805478e+00  1.8566887e+00
   1.2188618e+00  1.2418065e+00  1.9352047e+00  1.8642957e+00
   4.9494129e-01  9.0068567e-01  1.2537138e+00  1.1410958e+00
   5.4080826e-01  1.0844314e+00  1.0833049e+00  1.2337322e+00
   1.6370155e+00  1.8337655e+00  1.8507369e+00  1.4702764e+00
   6.1925685e-01  4.1721112e-01  6.5424818e-01  1.1994511e+00
   1.6217697e+00  3.9375156e-01  6.4090884e-01  3.9238751e-01
   6.6792357e-01  4.1579998e-01  1.8322816e+00  8.9651257e-01
   4.1080594e-02  3.1003132e+00  4.1659207e+00  3.6103864e+00
   4.8144150e+00  4.0914102e+00  4.2628331e+00  3.2858286e+00
   4.7249084e+00  3.4652224e+00  3.3654790e+00  4.3192091e+00
   4.6458197e+00  4.4907985e+00  4.8633590e+00  4.8826132e+00
   3.7656856e+00  2.9957829e+00  4.2667060e+00  4.4892497e+00
   3.7187009e+00  3.1579313e+00  4.5382776e+00  3.7568541e+00
   3.7595334e+00  4.2356744e+00  4.6210518e+00  3.2269835e+00
   4.2029510e+00  3.1453667e+00  4.0713015e+00  4.4550276e+00
   3.7262440e+00  3.5954022e+00  3.1397233e+00  4.4226375e+00
   3.6659203e+00  4.7482519e+00  3.8690786e+00  4.6505866e+00
   4.8043232e+00  4.0394230e+00  3.4870019e+00  3.4132872e+00
   4.6926355e+00  3.5622931e+00  4.6898971e+00  4.8071375e+00
   4.1887016e+00  3.9661503e+00  3.8546066e+00  3.8202200e+00
   3.9911590e+00  4.8675804e+00  3.6161766e+00  4.8037486e+00
   4.6181483e+00  3.4547663e+00  3.2973061e+00  3.3542938e+00
  -3.2426672e+00 -5.6821213e+00  5.8753457e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 10:13:50.725303
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.6619
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 10:13:50.729164
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9751.18
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 10:13:50.732253
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.5336
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 10:13:50.735500
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -872.287
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139661613590064
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139660672082608
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139660672083112
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139660672083616
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139660672084120
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139660672084624

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f058fb9e550> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.535224
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.502037
grad_step = 000002, loss = 0.477403
grad_step = 000003, loss = 0.452021
grad_step = 000004, loss = 0.422174
grad_step = 000005, loss = 0.389213
grad_step = 000006, loss = 0.356207
grad_step = 000007, loss = 0.326065
grad_step = 000008, loss = 0.316998
grad_step = 000009, loss = 0.317882
grad_step = 000010, loss = 0.300868
grad_step = 000011, loss = 0.279895
grad_step = 000012, loss = 0.265269
grad_step = 000013, loss = 0.257035
grad_step = 000014, loss = 0.251437
grad_step = 000015, loss = 0.244956
grad_step = 000016, loss = 0.236004
grad_step = 000017, loss = 0.225358
grad_step = 000018, loss = 0.214675
grad_step = 000019, loss = 0.204569
grad_step = 000020, loss = 0.194637
grad_step = 000021, loss = 0.185017
grad_step = 000022, loss = 0.177238
grad_step = 000023, loss = 0.171522
grad_step = 000024, loss = 0.164730
grad_step = 000025, loss = 0.155723
grad_step = 000026, loss = 0.147025
grad_step = 000027, loss = 0.140146
grad_step = 000028, loss = 0.134309
grad_step = 000029, loss = 0.128436
grad_step = 000030, loss = 0.122346
grad_step = 000031, loss = 0.116201
grad_step = 000032, loss = 0.110093
grad_step = 000033, loss = 0.104215
grad_step = 000034, loss = 0.098767
grad_step = 000035, loss = 0.093848
grad_step = 000036, loss = 0.089342
grad_step = 000037, loss = 0.084875
grad_step = 000038, loss = 0.080230
grad_step = 000039, loss = 0.075622
grad_step = 000040, loss = 0.071460
grad_step = 000041, loss = 0.067822
grad_step = 000042, loss = 0.064385
grad_step = 000043, loss = 0.060936
grad_step = 000044, loss = 0.057569
grad_step = 000045, loss = 0.054328
grad_step = 000046, loss = 0.051143
grad_step = 000047, loss = 0.048163
grad_step = 000048, loss = 0.045538
grad_step = 000049, loss = 0.043095
grad_step = 000050, loss = 0.040620
grad_step = 000051, loss = 0.038163
grad_step = 000052, loss = 0.035868
grad_step = 000053, loss = 0.033762
grad_step = 000054, loss = 0.031791
grad_step = 000055, loss = 0.029895
grad_step = 000056, loss = 0.028049
grad_step = 000057, loss = 0.026264
grad_step = 000058, loss = 0.024595
grad_step = 000059, loss = 0.023067
grad_step = 000060, loss = 0.021620
grad_step = 000061, loss = 0.020225
grad_step = 000062, loss = 0.018916
grad_step = 000063, loss = 0.017678
grad_step = 000064, loss = 0.016501
grad_step = 000065, loss = 0.015418
grad_step = 000066, loss = 0.014396
grad_step = 000067, loss = 0.013409
grad_step = 000068, loss = 0.012488
grad_step = 000069, loss = 0.011647
grad_step = 000070, loss = 0.010868
grad_step = 000071, loss = 0.010138
grad_step = 000072, loss = 0.009446
grad_step = 000073, loss = 0.008798
grad_step = 000074, loss = 0.008213
grad_step = 000075, loss = 0.007681
grad_step = 000076, loss = 0.007177
grad_step = 000077, loss = 0.006709
grad_step = 000078, loss = 0.006284
grad_step = 000079, loss = 0.005896
grad_step = 000080, loss = 0.005549
grad_step = 000081, loss = 0.005228
grad_step = 000082, loss = 0.004926
grad_step = 000083, loss = 0.004653
grad_step = 000084, loss = 0.004410
grad_step = 000085, loss = 0.004191
grad_step = 000086, loss = 0.003988
grad_step = 000087, loss = 0.003800
grad_step = 000088, loss = 0.003634
grad_step = 000089, loss = 0.003488
grad_step = 000090, loss = 0.003353
grad_step = 000091, loss = 0.003230
grad_step = 000092, loss = 0.003119
grad_step = 000093, loss = 0.003020
grad_step = 000094, loss = 0.002933
grad_step = 000095, loss = 0.002853
grad_step = 000096, loss = 0.002781
grad_step = 000097, loss = 0.002717
grad_step = 000098, loss = 0.002661
grad_step = 000099, loss = 0.002611
grad_step = 000100, loss = 0.002564
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002523
grad_step = 000102, loss = 0.002487
grad_step = 000103, loss = 0.002455
grad_step = 000104, loss = 0.002426
grad_step = 000105, loss = 0.002401
grad_step = 000106, loss = 0.002378
grad_step = 000107, loss = 0.002358
grad_step = 000108, loss = 0.002340
grad_step = 000109, loss = 0.002324
grad_step = 000110, loss = 0.002310
grad_step = 000111, loss = 0.002298
grad_step = 000112, loss = 0.002287
grad_step = 000113, loss = 0.002277
grad_step = 000114, loss = 0.002268
grad_step = 000115, loss = 0.002261
grad_step = 000116, loss = 0.002254
grad_step = 000117, loss = 0.002248
grad_step = 000118, loss = 0.002243
grad_step = 000119, loss = 0.002238
grad_step = 000120, loss = 0.002234
grad_step = 000121, loss = 0.002231
grad_step = 000122, loss = 0.002228
grad_step = 000123, loss = 0.002225
grad_step = 000124, loss = 0.002223
grad_step = 000125, loss = 0.002220
grad_step = 000126, loss = 0.002218
grad_step = 000127, loss = 0.002217
grad_step = 000128, loss = 0.002215
grad_step = 000129, loss = 0.002214
grad_step = 000130, loss = 0.002212
grad_step = 000131, loss = 0.002211
grad_step = 000132, loss = 0.002210
grad_step = 000133, loss = 0.002208
grad_step = 000134, loss = 0.002207
grad_step = 000135, loss = 0.002206
grad_step = 000136, loss = 0.002205
grad_step = 000137, loss = 0.002204
grad_step = 000138, loss = 0.002202
grad_step = 000139, loss = 0.002201
grad_step = 000140, loss = 0.002200
grad_step = 000141, loss = 0.002199
grad_step = 000142, loss = 0.002198
grad_step = 000143, loss = 0.002196
grad_step = 000144, loss = 0.002195
grad_step = 000145, loss = 0.002194
grad_step = 000146, loss = 0.002192
grad_step = 000147, loss = 0.002191
grad_step = 000148, loss = 0.002190
grad_step = 000149, loss = 0.002188
grad_step = 000150, loss = 0.002187
grad_step = 000151, loss = 0.002186
grad_step = 000152, loss = 0.002184
grad_step = 000153, loss = 0.002183
grad_step = 000154, loss = 0.002182
grad_step = 000155, loss = 0.002180
grad_step = 000156, loss = 0.002179
grad_step = 000157, loss = 0.002178
grad_step = 000158, loss = 0.002176
grad_step = 000159, loss = 0.002175
grad_step = 000160, loss = 0.002173
grad_step = 000161, loss = 0.002172
grad_step = 000162, loss = 0.002171
grad_step = 000163, loss = 0.002169
grad_step = 000164, loss = 0.002168
grad_step = 000165, loss = 0.002167
grad_step = 000166, loss = 0.002165
grad_step = 000167, loss = 0.002164
grad_step = 000168, loss = 0.002162
grad_step = 000169, loss = 0.002161
grad_step = 000170, loss = 0.002160
grad_step = 000171, loss = 0.002158
grad_step = 000172, loss = 0.002157
grad_step = 000173, loss = 0.002156
grad_step = 000174, loss = 0.002154
grad_step = 000175, loss = 0.002153
grad_step = 000176, loss = 0.002151
grad_step = 000177, loss = 0.002150
grad_step = 000178, loss = 0.002148
grad_step = 000179, loss = 0.002147
grad_step = 000180, loss = 0.002146
grad_step = 000181, loss = 0.002144
grad_step = 000182, loss = 0.002143
grad_step = 000183, loss = 0.002141
grad_step = 000184, loss = 0.002140
grad_step = 000185, loss = 0.002138
grad_step = 000186, loss = 0.002137
grad_step = 000187, loss = 0.002135
grad_step = 000188, loss = 0.002134
grad_step = 000189, loss = 0.002132
grad_step = 000190, loss = 0.002131
grad_step = 000191, loss = 0.002129
grad_step = 000192, loss = 0.002128
grad_step = 000193, loss = 0.002126
grad_step = 000194, loss = 0.002124
grad_step = 000195, loss = 0.002123
grad_step = 000196, loss = 0.002121
grad_step = 000197, loss = 0.002120
grad_step = 000198, loss = 0.002118
grad_step = 000199, loss = 0.002116
grad_step = 000200, loss = 0.002115
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002113
grad_step = 000202, loss = 0.002111
grad_step = 000203, loss = 0.002109
grad_step = 000204, loss = 0.002107
grad_step = 000205, loss = 0.002105
grad_step = 000206, loss = 0.002103
grad_step = 000207, loss = 0.002101
grad_step = 000208, loss = 0.002099
grad_step = 000209, loss = 0.002097
grad_step = 000210, loss = 0.002096
grad_step = 000211, loss = 0.002094
grad_step = 000212, loss = 0.002091
grad_step = 000213, loss = 0.002089
grad_step = 000214, loss = 0.002088
grad_step = 000215, loss = 0.002087
grad_step = 000216, loss = 0.002087
grad_step = 000217, loss = 0.002088
grad_step = 000218, loss = 0.002086
grad_step = 000219, loss = 0.002079
grad_step = 000220, loss = 0.002074
grad_step = 000221, loss = 0.002073
grad_step = 000222, loss = 0.002074
grad_step = 000223, loss = 0.002076
grad_step = 000224, loss = 0.002075
grad_step = 000225, loss = 0.002071
grad_step = 000226, loss = 0.002065
grad_step = 000227, loss = 0.002061
grad_step = 000228, loss = 0.002058
grad_step = 000229, loss = 0.002057
grad_step = 000230, loss = 0.002058
grad_step = 000231, loss = 0.002062
grad_step = 000232, loss = 0.002072
grad_step = 000233, loss = 0.002086
grad_step = 000234, loss = 0.002094
grad_step = 000235, loss = 0.002079
grad_step = 000236, loss = 0.002052
grad_step = 000237, loss = 0.002041
grad_step = 000238, loss = 0.002053
grad_step = 000239, loss = 0.002065
grad_step = 000240, loss = 0.002056
grad_step = 000241, loss = 0.002040
grad_step = 000242, loss = 0.002031
grad_step = 000243, loss = 0.002038
grad_step = 000244, loss = 0.002052
grad_step = 000245, loss = 0.002046
grad_step = 000246, loss = 0.002033
grad_step = 000247, loss = 0.002022
grad_step = 000248, loss = 0.002023
grad_step = 000249, loss = 0.002034
grad_step = 000250, loss = 0.002037
grad_step = 000251, loss = 0.002035
grad_step = 000252, loss = 0.002021
grad_step = 000253, loss = 0.002012
grad_step = 000254, loss = 0.002007
grad_step = 000255, loss = 0.002009
grad_step = 000256, loss = 0.002014
grad_step = 000257, loss = 0.002021
grad_step = 000258, loss = 0.002029
grad_step = 000259, loss = 0.002031
grad_step = 000260, loss = 0.002029
grad_step = 000261, loss = 0.002016
grad_step = 000262, loss = 0.002003
grad_step = 000263, loss = 0.001993
grad_step = 000264, loss = 0.001988
grad_step = 000265, loss = 0.001988
grad_step = 000266, loss = 0.001991
grad_step = 000267, loss = 0.001996
grad_step = 000268, loss = 0.001998
grad_step = 000269, loss = 0.001999
grad_step = 000270, loss = 0.001993
grad_step = 000271, loss = 0.001988
grad_step = 000272, loss = 0.001981
grad_step = 000273, loss = 0.001975
grad_step = 000274, loss = 0.001970
grad_step = 000275, loss = 0.001968
grad_step = 000276, loss = 0.001967
grad_step = 000277, loss = 0.001967
grad_step = 000278, loss = 0.001970
grad_step = 000279, loss = 0.001977
grad_step = 000280, loss = 0.001992
grad_step = 000281, loss = 0.002017
grad_step = 000282, loss = 0.002053
grad_step = 000283, loss = 0.002062
grad_step = 000284, loss = 0.002037
grad_step = 000285, loss = 0.001976
grad_step = 000286, loss = 0.001959
grad_step = 000287, loss = 0.001987
grad_step = 000288, loss = 0.001995
grad_step = 000289, loss = 0.001972
grad_step = 000290, loss = 0.001957
grad_step = 000291, loss = 0.001964
grad_step = 000292, loss = 0.001972
grad_step = 000293, loss = 0.001965
grad_step = 000294, loss = 0.001952
grad_step = 000295, loss = 0.001950
grad_step = 000296, loss = 0.001956
grad_step = 000297, loss = 0.001956
grad_step = 000298, loss = 0.001951
grad_step = 000299, loss = 0.001953
grad_step = 000300, loss = 0.001948
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001937
grad_step = 000302, loss = 0.001934
grad_step = 000303, loss = 0.001939
grad_step = 000304, loss = 0.001939
grad_step = 000305, loss = 0.001934
grad_step = 000306, loss = 0.001934
grad_step = 000307, loss = 0.001938
grad_step = 000308, loss = 0.001939
grad_step = 000309, loss = 0.001939
grad_step = 000310, loss = 0.001942
grad_step = 000311, loss = 0.001945
grad_step = 000312, loss = 0.001946
grad_step = 000313, loss = 0.001951
grad_step = 000314, loss = 0.001958
grad_step = 000315, loss = 0.001963
grad_step = 000316, loss = 0.001964
grad_step = 000317, loss = 0.001961
grad_step = 000318, loss = 0.001955
grad_step = 000319, loss = 0.001944
grad_step = 000320, loss = 0.001930
grad_step = 000321, loss = 0.001919
grad_step = 000322, loss = 0.001913
grad_step = 000323, loss = 0.001912
grad_step = 000324, loss = 0.001915
grad_step = 000325, loss = 0.001919
grad_step = 000326, loss = 0.001923
grad_step = 000327, loss = 0.001923
grad_step = 000328, loss = 0.001919
grad_step = 000329, loss = 0.001914
grad_step = 000330, loss = 0.001909
grad_step = 000331, loss = 0.001904
grad_step = 000332, loss = 0.001901
grad_step = 000333, loss = 0.001900
grad_step = 000334, loss = 0.001901
grad_step = 000335, loss = 0.001902
grad_step = 000336, loss = 0.001904
grad_step = 000337, loss = 0.001904
grad_step = 000338, loss = 0.001905
grad_step = 000339, loss = 0.001906
grad_step = 000340, loss = 0.001907
grad_step = 000341, loss = 0.001911
grad_step = 000342, loss = 0.001918
grad_step = 000343, loss = 0.001932
grad_step = 000344, loss = 0.001938
grad_step = 000345, loss = 0.001934
grad_step = 000346, loss = 0.001914
grad_step = 000347, loss = 0.001906
grad_step = 000348, loss = 0.001913
grad_step = 000349, loss = 0.001911
grad_step = 000350, loss = 0.001910
grad_step = 000351, loss = 0.001910
grad_step = 000352, loss = 0.001903
grad_step = 000353, loss = 0.001894
grad_step = 000354, loss = 0.001896
grad_step = 000355, loss = 0.001900
grad_step = 000356, loss = 0.001895
grad_step = 000357, loss = 0.001885
grad_step = 000358, loss = 0.001881
grad_step = 000359, loss = 0.001884
grad_step = 000360, loss = 0.001884
grad_step = 000361, loss = 0.001878
grad_step = 000362, loss = 0.001872
grad_step = 000363, loss = 0.001872
grad_step = 000364, loss = 0.001874
grad_step = 000365, loss = 0.001873
grad_step = 000366, loss = 0.001869
grad_step = 000367, loss = 0.001866
grad_step = 000368, loss = 0.001867
grad_step = 000369, loss = 0.001868
grad_step = 000370, loss = 0.001867
grad_step = 000371, loss = 0.001864
grad_step = 000372, loss = 0.001864
grad_step = 000373, loss = 0.001866
grad_step = 000374, loss = 0.001872
grad_step = 000375, loss = 0.001881
grad_step = 000376, loss = 0.001900
grad_step = 000377, loss = 0.001941
grad_step = 000378, loss = 0.002007
grad_step = 000379, loss = 0.002106
grad_step = 000380, loss = 0.002160
grad_step = 000381, loss = 0.002127
grad_step = 000382, loss = 0.001985
grad_step = 000383, loss = 0.001885
grad_step = 000384, loss = 0.001922
grad_step = 000385, loss = 0.001987
grad_step = 000386, loss = 0.001953
grad_step = 000387, loss = 0.001881
grad_step = 000388, loss = 0.001888
grad_step = 000389, loss = 0.001927
grad_step = 000390, loss = 0.001904
grad_step = 000391, loss = 0.001863
grad_step = 000392, loss = 0.001879
grad_step = 000393, loss = 0.001897
grad_step = 000394, loss = 0.001864
grad_step = 000395, loss = 0.001850
grad_step = 000396, loss = 0.001879
grad_step = 000397, loss = 0.001871
grad_step = 000398, loss = 0.001840
grad_step = 000399, loss = 0.001852
grad_step = 000400, loss = 0.001867
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001845
grad_step = 000402, loss = 0.001837
grad_step = 000403, loss = 0.001850
grad_step = 000404, loss = 0.001848
grad_step = 000405, loss = 0.001835
grad_step = 000406, loss = 0.001834
grad_step = 000407, loss = 0.001840
grad_step = 000408, loss = 0.001836
grad_step = 000409, loss = 0.001831
grad_step = 000410, loss = 0.001830
grad_step = 000411, loss = 0.001828
grad_step = 000412, loss = 0.001827
grad_step = 000413, loss = 0.001828
grad_step = 000414, loss = 0.001825
grad_step = 000415, loss = 0.001821
grad_step = 000416, loss = 0.001822
grad_step = 000417, loss = 0.001822
grad_step = 000418, loss = 0.001819
grad_step = 000419, loss = 0.001816
grad_step = 000420, loss = 0.001817
grad_step = 000421, loss = 0.001817
grad_step = 000422, loss = 0.001815
grad_step = 000423, loss = 0.001812
grad_step = 000424, loss = 0.001811
grad_step = 000425, loss = 0.001811
grad_step = 000426, loss = 0.001809
grad_step = 000427, loss = 0.001807
grad_step = 000428, loss = 0.001806
grad_step = 000429, loss = 0.001806
grad_step = 000430, loss = 0.001805
grad_step = 000431, loss = 0.001803
grad_step = 000432, loss = 0.001802
grad_step = 000433, loss = 0.001801
grad_step = 000434, loss = 0.001801
grad_step = 000435, loss = 0.001800
grad_step = 000436, loss = 0.001798
grad_step = 000437, loss = 0.001797
grad_step = 000438, loss = 0.001797
grad_step = 000439, loss = 0.001796
grad_step = 000440, loss = 0.001796
grad_step = 000441, loss = 0.001796
grad_step = 000442, loss = 0.001797
grad_step = 000443, loss = 0.001801
grad_step = 000444, loss = 0.001807
grad_step = 000445, loss = 0.001818
grad_step = 000446, loss = 0.001838
grad_step = 000447, loss = 0.001872
grad_step = 000448, loss = 0.001929
grad_step = 000449, loss = 0.002005
grad_step = 000450, loss = 0.002102
grad_step = 000451, loss = 0.002173
grad_step = 000452, loss = 0.002182
grad_step = 000453, loss = 0.002105
grad_step = 000454, loss = 0.001950
grad_step = 000455, loss = 0.001826
grad_step = 000456, loss = 0.001813
grad_step = 000457, loss = 0.001895
grad_step = 000458, loss = 0.001949
grad_step = 000459, loss = 0.001887
grad_step = 000460, loss = 0.001793
grad_step = 000461, loss = 0.001789
grad_step = 000462, loss = 0.001855
grad_step = 000463, loss = 0.001870
grad_step = 000464, loss = 0.001810
grad_step = 000465, loss = 0.001777
grad_step = 000466, loss = 0.001804
grad_step = 000467, loss = 0.001820
grad_step = 000468, loss = 0.001796
grad_step = 000469, loss = 0.001783
grad_step = 000470, loss = 0.001798
grad_step = 000471, loss = 0.001800
grad_step = 000472, loss = 0.001775
grad_step = 000473, loss = 0.001763
grad_step = 000474, loss = 0.001779
grad_step = 000475, loss = 0.001788
grad_step = 000476, loss = 0.001772
grad_step = 000477, loss = 0.001758
grad_step = 000478, loss = 0.001764
grad_step = 000479, loss = 0.001771
grad_step = 000480, loss = 0.001765
grad_step = 000481, loss = 0.001756
grad_step = 000482, loss = 0.001756
grad_step = 000483, loss = 0.001760
grad_step = 000484, loss = 0.001758
grad_step = 000485, loss = 0.001751
grad_step = 000486, loss = 0.001748
grad_step = 000487, loss = 0.001751
grad_step = 000488, loss = 0.001752
grad_step = 000489, loss = 0.001748
grad_step = 000490, loss = 0.001743
grad_step = 000491, loss = 0.001742
grad_step = 000492, loss = 0.001744
grad_step = 000493, loss = 0.001744
grad_step = 000494, loss = 0.001740
grad_step = 000495, loss = 0.001737
grad_step = 000496, loss = 0.001736
grad_step = 000497, loss = 0.001737
grad_step = 000498, loss = 0.001737
grad_step = 000499, loss = 0.001735
grad_step = 000500, loss = 0.001733
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001732
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

  date_run                              2020-05-14 10:14:12.653428
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.213225
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 10:14:12.658930
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0963168
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 10:14:12.665057
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.138928
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 10:14:12.669760
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.463568
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
0   2020-05-14 10:13:42.473732  ...    mean_absolute_error
1   2020-05-14 10:13:42.478266  ...     mean_squared_error
2   2020-05-14 10:13:42.482600  ...  median_absolute_error
3   2020-05-14 10:13:42.486509  ...               r2_score
4   2020-05-14 10:13:50.725303  ...    mean_absolute_error
5   2020-05-14 10:13:50.729164  ...     mean_squared_error
6   2020-05-14 10:13:50.732253  ...  median_absolute_error
7   2020-05-14 10:13:50.735500  ...               r2_score
8   2020-05-14 10:14:12.653428  ...    mean_absolute_error
9   2020-05-14 10:14:12.658930  ...     mean_squared_error
10  2020-05-14 10:14:12.665057  ...  median_absolute_error
11  2020-05-14 10:14:12.669760  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff45cb12cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 489751.46it/s] 93%|█████████▎| 9248768/9912422 [00:00<00:00, 698052.09it/s]9920512it [00:00, 47668882.49it/s]                           
0it [00:00, ?it/s]32768it [00:00, 710741.64it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 469709.76it/s]1654784it [00:00, 11967492.84it/s]                         
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 62843.94it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff40f4cce80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff40eafb0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff40f4cce80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff40ea520f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff40c28d4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff40c278c50> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff40f4cce80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff40ea10710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff40c28d4e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff40eafb128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f6b421c9208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=57e8b0158502c3392951f5570a4f7dbe24900555c151f59fb290845a70c24237
  Stored in directory: /tmp/pip-ephem-wheel-cache-8vwv_qke/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f6b3854f048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1851392/17464789 [==>...........................] - ETA: 0s
 8347648/17464789 [=============>................] - ETA: 0s
15327232/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 10:15:36.774969: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 10:15:36.779030: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-14 10:15:36.779170: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f213be4300 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 10:15:36.779185: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7740 - accuracy: 0.4930
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6360 - accuracy: 0.5020 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6871 - accuracy: 0.4987
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6436 - accuracy: 0.5015
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6268 - accuracy: 0.5026
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5797 - accuracy: 0.5057
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5549 - accuracy: 0.5073
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5593 - accuracy: 0.5070
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5746 - accuracy: 0.5060
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6344 - accuracy: 0.5021
11000/25000 [============>.................] - ETA: 4s - loss: 7.6374 - accuracy: 0.5019
12000/25000 [=============>................] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6501 - accuracy: 0.5011
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6250 - accuracy: 0.5027
15000/25000 [=================>............] - ETA: 3s - loss: 7.6124 - accuracy: 0.5035
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6072 - accuracy: 0.5039
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6278 - accuracy: 0.5025
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6479 - accuracy: 0.5012
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6620 - accuracy: 0.5003
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6630 - accuracy: 0.5002
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6471 - accuracy: 0.5013
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6733 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6768 - accuracy: 0.4993
25000/25000 [==============================] - 9s 364us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 10:15:52.525530
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 10:15:52.525530  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<25:24:48, 9.42kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<18:01:32, 13.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:40:20, 18.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:52:41, 26.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<6:11:54, 38.5kB/s].vector_cache/glove.6B.zip:   1%|          | 9.08M/862M [00:01<4:18:45, 54.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:01<3:00:24, 78.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.3M/862M [00:01<2:05:44, 112kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 21.9M/862M [00:01<1:27:39, 160kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.3M/862M [00:01<1:01:02, 228kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.3M/862M [00:02<42:38, 325kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.6M/862M [00:02<29:47, 462kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.7M/862M [00:02<20:51, 657kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.9M/862M [00:02<14:37, 933kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.6M/862M [00:02<10:16, 1.32MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.4M/862M [00:02<07:21, 1.83MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.5M/862M [00:04<07:03, 1.91MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<06:46, 1.98MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:04<05:09, 2.60MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<06:09, 2.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:06<05:55, 2.26MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:06<04:32, 2.94MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:08<05:56, 2.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<05:36, 2.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:08<04:12, 3.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:10<05:56, 2.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<05:31, 2.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:10<04:09, 3.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:12<06:03, 2.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<05:35, 2.36MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:12<04:11, 3.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<06:00, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<05:34, 2.35MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:14<04:10, 3.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<06:01, 2.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<05:33, 2.34MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:16<04:12, 3.08MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<06:01, 2.15MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<06:54, 1.88MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<05:29, 2.36MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:20<05:55, 2.18MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<05:31, 2.33MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:20<04:08, 3.11MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:22<05:53, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<05:26, 2.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:22<04:08, 3.09MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:24<05:54, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<05:26, 2.34MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:24<04:08, 3.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:53, 2.16MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:25, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<04:04, 3.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<05:50, 2.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<05:23, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<04:05, 3.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:49, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:22, 2.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:04, 3.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<05:47, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:20, 2.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<04:03, 3.07MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<05:45, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:18, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<04:01, 3.07MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:44, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:16, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<04:00, 3.08MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<05:41, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:14, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<03:58, 3.07MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:39, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:14, 2.33MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<03:53, 3.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<02:53, 4.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<1:32:05, 132kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<1:05:30, 185kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<45:59, 263kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<32:17, 374kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<1:08:00, 177kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<48:51, 247kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<34:26, 350kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<26:47, 448kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:46<21:07, 568kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<15:22, 779kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<12:39, 942kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<10:07, 1.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:19, 1.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<07:54, 1.50MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<06:45, 1.76MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<04:58, 2.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:16, 1.88MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:36, 2.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:11, 2.81MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:40, 2.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:09, 2.27MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<03:54, 2.99MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:28, 2.13MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:02, 2.31MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<03:49, 3.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:24, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<04:58, 2.33MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<03:46, 3.06MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:21, 2.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<04:56, 2.33MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<03:45, 3.06MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<05:19, 2.15MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<04:53, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<03:42, 3.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<05:17, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:52, 2.33MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<03:41, 3.07MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<05:15, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<04:51, 2.32MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<03:41, 3.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:13, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<04:49, 2.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<03:39, 3.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:12, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<04:45, 2.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<03:37, 3.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:09, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<04:46, 2.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:37, 3.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:07, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<04:44, 2.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<03:35, 3.06MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:05, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<04:41, 2.33MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<03:33, 3.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:03, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<04:40, 2.33MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<03:32, 3.06MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:02, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:39, 2.32MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<03:31, 3.06MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<05:00, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:37, 2.32MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<03:31, 3.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<04:58, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:35, 2.33MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<03:28, 3.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:56, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:32, 2.33MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<03:27, 3.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:54, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:31, 2.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<03:22, 3.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:51, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:29, 2.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:21, 3.12MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<04:47, 2.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<04:25, 2.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:21, 3.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:47, 2.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:25, 2.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:21, 3.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:47, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<04:23, 2.34MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:19, 3.08MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:44, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:22, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:19, 3.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:42, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:20, 2.33MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<03:15, 3.11MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:38, 2.17MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:17, 2.35MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<03:12, 3.13MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:37, 2.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:16, 2.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<03:14, 3.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:36, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:15, 2.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:11, 3.11MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:34, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:13, 2.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<03:11, 3.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:32, 2.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:12, 2.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:08, 3.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:31, 2.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:08, 2.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<03:08, 3.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:28, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:07, 2.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:05, 3.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:26, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<05:05, 1.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:03, 2.36MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<02:55, 3.26MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<1:20:39, 118kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<57:24, 166kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<40:16, 236kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<30:18, 312kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<22:10, 426kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<15:43, 600kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<13:11, 712kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<10:11, 921kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<07:21, 1.27MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<07:20, 1.27MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<06:05, 1.53MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<04:29, 2.07MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:19, 1.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:41, 1.97MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:30, 2.62MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:37, 1.99MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:11, 2.19MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:09, 2.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:21, 2.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<03:59, 2.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:01, 3.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:14, 2.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:54, 2.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<02:57, 3.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<04:11, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:51, 2.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<02:53, 3.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<04:08, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<03:49, 2.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<02:53, 3.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<04:06, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<03:47, 2.33MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<02:52, 3.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:05, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<03:45, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<02:48, 3.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:02, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<03:44, 2.33MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<02:50, 3.06MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:00, 2.15MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<03:41, 2.34MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<02:48, 3.07MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<03:58, 2.15MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<03:40, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<02:44, 3.11MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:56, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:38, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<02:46, 3.06MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:55, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<04:29, 1.87MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:30, 2.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<02:36, 3.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:19, 1.93MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:54, 2.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<02:56, 2.83MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:59, 2.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:30, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<03:30, 2.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<02:33, 3.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<06:16, 1.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<05:14, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:52, 2.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:36, 1.77MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:04, 2.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<03:01, 2.69MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:01, 2.01MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<03:39, 2.21MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<02:43, 2.95MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:48, 2.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:30, 2.29MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<02:39, 3.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:43, 2.14MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:25, 2.32MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:35, 3.05MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:40, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:23, 2.33MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:34, 3.06MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:37, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:12, 2.43MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:24, 3.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<01:47, 4.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<33:08, 234kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<24:47, 312kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<17:42, 436kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<12:24, 619kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<7:23:44, 17.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<5:11:09, 24.7kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<3:37:15, 35.2kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<2:33:08, 49.7kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<1:47:46, 70.5kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<1:15:21, 101kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:51<52:36, 143kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<1:07:09, 112kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<47:44, 158kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<33:28, 224kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<25:05, 298kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<19:06, 391kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<13:39, 546kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<09:38, 770kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<09:28, 782kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<07:24, 997kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<05:21, 1.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:27, 1.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:26, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:16, 2.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<02:22, 3.06MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<30:54, 235kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<22:22, 324kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<15:45, 459kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<12:40, 568kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<10:23, 693kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<07:34, 948kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<05:22, 1.33MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<07:02, 1.01MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<05:34, 1.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<04:04, 1.74MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<02:55, 2.41MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<19:44, 358kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<15:16, 462kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<10:59, 641kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:07<07:44, 905kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<10:14, 682kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<07:53, 885kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<05:41, 1.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<05:34, 1.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:36, 1.50MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:23, 2.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:58, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:29, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:36, 2.61MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:25, 1.98MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:05, 2.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:20, 2.89MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<03:12, 2.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<02:49, 2.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:07, 3.15MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<01:34, 4.23MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<27:59, 238kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<20:58, 317kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<15:00, 442kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<10:29, 627kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<6:24:51, 17.1kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<4:29:49, 24.4kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<3:08:17, 34.8kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<2:12:36, 49.1kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<1:33:24, 69.6kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<1:05:15, 99.3kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<46:57, 137kB/s]   .vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<33:30, 192kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<23:31, 273kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<17:52, 357kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<13:09, 484kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<09:18, 681kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<07:58, 791kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<06:13, 1.01MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:28, 1.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:35, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<03:51, 1.61MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:51, 2.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:26, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:02, 2.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:15, 2.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:01, 2.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:44, 2.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:04, 2.93MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:51, 2.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:31, 2.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<01:57, 3.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<01:25, 4.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<24:57, 239kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<18:04, 330kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<12:43, 466kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<10:15, 575kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<07:46, 757kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<05:34, 1.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<05:15, 1.11MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<04:16, 1.36MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<03:06, 1.86MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:31, 1.64MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:03, 1.88MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:16, 2.52MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:55, 1.94MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:38, 2.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<01:58, 2.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:42, 2.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:28, 2.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<01:52, 2.98MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:36, 2.13MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:23, 2.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<01:48, 3.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:33, 2.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:21, 2.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<01:45, 3.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:30, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:18, 2.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<01:44, 3.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:28, 2.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:17, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<01:43, 3.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:27, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:15, 2.33MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<01:42, 3.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:25, 2.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:13, 2.33MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:39, 3.11MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:23, 2.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:06, 2.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<01:35, 3.22MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<01:10, 4.33MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<21:12, 239kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<15:53, 319kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<11:18, 447kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:04<07:58, 631kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<07:05, 705kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<05:29, 911kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:57, 1.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:53, 1.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:13, 1.52MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:21, 2.08MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:47, 1.74MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:27, 1.98MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:48, 2.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:23, 2.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:09, 2.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:37, 2.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:15, 2.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:03, 2.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:33, 3.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:10, 2.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:00, 2.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:29, 3.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:08, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<01:57, 2.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:29, 3.07MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:06, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<01:56, 2.33MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<01:27, 3.06MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:04, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<01:54, 2.33MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:26, 3.06MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:02, 2.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<01:52, 2.33MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:25, 3.06MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:00, 2.15MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:50, 2.33MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:22, 3.11MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:58, 2.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<01:49, 2.32MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:23, 3.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:56, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:47, 2.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:20, 3.08MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:54, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:45, 2.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:19, 3.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:53, 2.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:09, 1.87MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:42, 2.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:17, 3.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:48, 2.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:41, 2.35MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:15, 3.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:47, 2.19MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:39, 2.36MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:15, 3.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:46, 2.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:38, 2.33MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:13, 3.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:44, 2.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:36, 2.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:11, 3.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:42, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:35, 2.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:11, 3.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:40, 2.17MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:28, 2.45MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:09, 3.12MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<00:50, 4.24MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<14:59, 238kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<10:49, 328kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<07:37, 463kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<06:06, 572kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<04:37, 754kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<03:18, 1.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:05, 1.11MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:30, 1.36MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:50, 1.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:03, 1.62MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:47, 1.87MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:18, 2.53MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:41, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:31, 2.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:08, 2.86MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:33, 2.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:25, 2.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:03, 2.99MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:28, 2.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:18, 2.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<00:58, 3.19MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<00:43, 4.29MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<12:07, 254kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<08:47, 350kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<06:10, 494kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<04:59, 604kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<03:47, 794kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:42, 1.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:05<02:34, 1.15MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:05, 1.40MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:31, 1.90MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<01:44, 1.65MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:30, 1.90MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:07, 2.53MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:26, 1.95MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:17, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<00:58, 2.86MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:19, 2.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:12, 2.27MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<00:53, 3.04MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:15, 2.13MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:08, 2.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<00:51, 3.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:12, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:04, 2.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<00:48, 3.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<00:35, 4.31MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<10:16, 246kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<07:42, 328kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<05:29, 458kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<04:10, 591kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<03:09, 777kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<02:15, 1.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:06, 1.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:43, 1.39MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:15, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:24, 1.65MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:13, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<00:54, 2.53MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:09, 1.95MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:02, 2.16MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:46, 2.86MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:03, 2.08MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:11, 1.84MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:55, 2.36MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:39, 3.22MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:39, 1.28MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:22, 1.54MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<01:00, 2.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:10, 1.75MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:01, 1.99MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:45, 2.65MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:59, 1.99MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:51, 2.30MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:38, 3.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<00:27, 4.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<07:33, 254kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<05:28, 349kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<03:49, 493kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<03:03, 603kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<02:19, 792kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:39, 1.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:33, 1.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:27, 1.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:05, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:46, 2.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:31, 1.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:14, 1.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:54, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:00, 1.63MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:52, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:38, 2.51MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:48, 1.94MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:43, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:32, 2.85MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:43, 2.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:39, 2.27MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:29, 3.00MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:40, 2.13MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:37, 2.31MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:27, 3.05MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:38, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:33, 2.42MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:25, 3.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:18, 4.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<05:06, 254kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<03:41, 350kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<02:34, 493kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<02:02, 604kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<01:32, 792kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:05, 1.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<01:00, 1.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:49, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:35, 1.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:39, 1.65MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:34, 1.90MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:25, 2.54MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:31, 1.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:28, 2.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:20, 2.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:27, 2.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:24, 2.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:17, 3.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:12, 4.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<03:43, 238kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<02:40, 329kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:50, 464kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:25, 574kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<01:04, 756kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:44, 1.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:40, 1.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:32, 1.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:23, 1.86MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:25, 1.63MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:21, 1.88MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:15, 2.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:18, 1.94MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:16, 2.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:12, 2.85MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:15, 2.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:14, 2.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:10, 2.99MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:13, 2.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:12, 2.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:08, 3.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:10, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:07, 3.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:08, 2.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:05, 3.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:06, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:04, 3.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:05, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.88MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:04, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:01, 3.12MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:01, 2.18MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.37MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 3.15MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 883/400000 [00:00<00:45, 8829.27it/s]  0%|          | 1695/400000 [00:00<00:46, 8603.74it/s]  1%|          | 2595/400000 [00:00<00:45, 8714.38it/s]  1%|          | 3457/400000 [00:00<00:45, 8684.81it/s]  1%|          | 4349/400000 [00:00<00:45, 8753.93it/s]  1%|▏         | 5174/400000 [00:00<00:45, 8596.32it/s]  2%|▏         | 6062/400000 [00:00<00:45, 8679.02it/s]  2%|▏         | 7032/400000 [00:00<00:43, 8961.91it/s]  2%|▏         | 7947/400000 [00:00<00:43, 9015.18it/s]  2%|▏         | 8920/400000 [00:01<00:42, 9217.75it/s]  2%|▏         | 9927/400000 [00:01<00:41, 9456.83it/s]  3%|▎         | 10904/400000 [00:01<00:40, 9547.34it/s]  3%|▎         | 11850/400000 [00:01<00:40, 9516.16it/s]  3%|▎         | 12795/400000 [00:01<00:41, 9433.80it/s]  3%|▎         | 13734/400000 [00:01<00:41, 9351.26it/s]  4%|▎         | 14724/400000 [00:01<00:40, 9508.26it/s]  4%|▍         | 15709/400000 [00:01<00:39, 9607.69it/s]  4%|▍         | 16669/400000 [00:01<00:40, 9416.84it/s]  4%|▍         | 17611/400000 [00:01<00:41, 9282.18it/s]  5%|▍         | 18596/400000 [00:02<00:40, 9443.98it/s]  5%|▍         | 19572/400000 [00:02<00:39, 9535.73it/s]  5%|▌         | 20560/400000 [00:02<00:39, 9635.17it/s]  5%|▌         | 21525/400000 [00:02<00:39, 9547.42it/s]  6%|▌         | 22481/400000 [00:02<00:39, 9460.30it/s]  6%|▌         | 23428/400000 [00:02<00:40, 9390.19it/s]  6%|▌         | 24404/400000 [00:02<00:39, 9496.74it/s]  6%|▋         | 25355/400000 [00:02<00:39, 9475.96it/s]  7%|▋         | 26337/400000 [00:02<00:39, 9574.36it/s]  7%|▋         | 27296/400000 [00:02<00:39, 9350.22it/s]  7%|▋         | 28233/400000 [00:03<00:40, 9186.69it/s]  7%|▋         | 29192/400000 [00:03<00:39, 9303.35it/s]  8%|▊         | 30162/400000 [00:03<00:39, 9417.47it/s]  8%|▊         | 31130/400000 [00:03<00:38, 9493.77it/s]  8%|▊         | 32081/400000 [00:03<00:39, 9225.91it/s]  8%|▊         | 33007/400000 [00:03<00:39, 9226.86it/s]  8%|▊         | 33947/400000 [00:03<00:39, 9276.92it/s]  9%|▊         | 34927/400000 [00:03<00:38, 9427.33it/s]  9%|▉         | 35874/400000 [00:03<00:38, 9438.26it/s]  9%|▉         | 36819/400000 [00:03<00:42, 8595.38it/s]  9%|▉         | 37719/400000 [00:04<00:41, 8711.66it/s] 10%|▉         | 38614/400000 [00:04<00:41, 8779.67it/s] 10%|▉         | 39538/400000 [00:04<00:40, 8912.38it/s] 10%|█         | 40436/400000 [00:04<00:41, 8726.91it/s] 10%|█         | 41314/400000 [00:04<00:41, 8676.22it/s] 11%|█         | 42239/400000 [00:04<00:40, 8838.56it/s] 11%|█         | 43196/400000 [00:04<00:39, 9044.59it/s] 11%|█         | 44126/400000 [00:04<00:39, 9116.93it/s] 11%|█▏        | 45051/400000 [00:04<00:38, 9155.54it/s] 11%|█▏        | 45969/400000 [00:05<00:39, 8997.27it/s] 12%|█▏        | 46871/400000 [00:05<00:39, 8948.62it/s] 12%|█▏        | 47768/400000 [00:05<00:39, 8923.23it/s] 12%|█▏        | 48739/400000 [00:05<00:38, 9142.22it/s] 12%|█▏        | 49656/400000 [00:05<00:38, 8983.25it/s] 13%|█▎        | 50557/400000 [00:05<00:38, 8977.14it/s] 13%|█▎        | 51484/400000 [00:05<00:38, 9060.04it/s] 13%|█▎        | 52459/400000 [00:05<00:37, 9254.27it/s] 13%|█▎        | 53387/400000 [00:05<00:37, 9153.17it/s] 14%|█▎        | 54304/400000 [00:05<00:38, 9072.86it/s] 14%|█▍        | 55213/400000 [00:06<00:38, 8978.43it/s] 14%|█▍        | 56112/400000 [00:06<00:38, 8927.19it/s] 14%|█▍        | 57070/400000 [00:06<00:37, 9112.88it/s] 15%|█▍        | 58032/400000 [00:06<00:36, 9255.64it/s] 15%|█▍        | 58983/400000 [00:06<00:36, 9324.72it/s] 15%|█▍        | 59917/400000 [00:06<00:36, 9191.51it/s] 15%|█▌        | 60861/400000 [00:06<00:36, 9263.58it/s] 15%|█▌        | 61789/400000 [00:06<00:37, 9042.94it/s] 16%|█▌        | 62696/400000 [00:06<00:37, 8973.32it/s] 16%|█▌        | 63608/400000 [00:06<00:37, 9014.79it/s] 16%|█▌        | 64517/400000 [00:07<00:37, 9035.68it/s] 16%|█▋        | 65514/400000 [00:07<00:35, 9296.74it/s] 17%|█▋        | 66486/400000 [00:07<00:35, 9418.00it/s] 17%|█▋        | 67475/400000 [00:07<00:34, 9554.66it/s] 17%|█▋        | 68443/400000 [00:07<00:34, 9584.38it/s] 17%|█▋        | 69403/400000 [00:07<00:35, 9308.60it/s] 18%|█▊        | 70360/400000 [00:07<00:35, 9384.83it/s] 18%|█▊        | 71327/400000 [00:07<00:34, 9468.22it/s] 18%|█▊        | 72306/400000 [00:07<00:34, 9560.51it/s] 18%|█▊        | 73264/400000 [00:07<00:34, 9525.70it/s] 19%|█▊        | 74240/400000 [00:08<00:33, 9594.57it/s] 19%|█▉        | 75201/400000 [00:08<00:34, 9541.32it/s] 19%|█▉        | 76156/400000 [00:08<00:34, 9419.56it/s] 19%|█▉        | 77099/400000 [00:08<00:35, 9195.43it/s] 20%|█▉        | 78021/400000 [00:08<00:35, 9118.37it/s] 20%|█▉        | 79006/400000 [00:08<00:34, 9325.69it/s] 20%|█▉        | 79987/400000 [00:08<00:33, 9464.38it/s] 20%|██        | 80977/400000 [00:08<00:33, 9590.38it/s] 20%|██        | 81977/400000 [00:08<00:32, 9708.39it/s] 21%|██        | 82950/400000 [00:08<00:33, 9486.12it/s] 21%|██        | 83915/400000 [00:09<00:33, 9533.10it/s] 21%|██        | 84870/400000 [00:09<00:33, 9374.07it/s] 21%|██▏       | 85830/400000 [00:09<00:33, 9439.05it/s] 22%|██▏       | 86807/400000 [00:09<00:32, 9535.23it/s] 22%|██▏       | 87762/400000 [00:09<00:32, 9501.04it/s] 22%|██▏       | 88713/400000 [00:09<00:32, 9482.90it/s] 22%|██▏       | 89690/400000 [00:09<00:32, 9565.71it/s] 23%|██▎       | 90652/400000 [00:09<00:32, 9580.15it/s] 23%|██▎       | 91611/400000 [00:09<00:32, 9561.27it/s] 23%|██▎       | 92568/400000 [00:10<00:34, 9039.86it/s] 23%|██▎       | 93479/400000 [00:10<00:33, 9049.84it/s] 24%|██▎       | 94439/400000 [00:10<00:33, 9206.41it/s] 24%|██▍       | 95401/400000 [00:10<00:32, 9325.09it/s] 24%|██▍       | 96360/400000 [00:10<00:32, 9402.12it/s] 24%|██▍       | 97303/400000 [00:10<00:32, 9360.56it/s] 25%|██▍       | 98277/400000 [00:10<00:31, 9471.03it/s] 25%|██▍       | 99266/400000 [00:10<00:31, 9590.80it/s] 25%|██▌       | 100245/400000 [00:10<00:31, 9648.09it/s] 25%|██▌       | 101211/400000 [00:10<00:32, 9318.44it/s] 26%|██▌       | 102147/400000 [00:11<00:33, 9000.56it/s] 26%|██▌       | 103119/400000 [00:11<00:32, 9203.03it/s] 26%|██▌       | 104065/400000 [00:11<00:31, 9276.71it/s] 26%|██▌       | 104996/400000 [00:11<00:32, 9104.39it/s] 26%|██▋       | 105910/400000 [00:11<00:32, 9092.10it/s] 27%|██▋       | 106822/400000 [00:11<00:32, 9099.29it/s] 27%|██▋       | 107734/400000 [00:11<00:33, 8773.10it/s] 27%|██▋       | 108652/400000 [00:11<00:32, 8888.26it/s] 27%|██▋       | 109544/400000 [00:11<00:33, 8770.44it/s] 28%|██▊       | 110424/400000 [00:11<00:33, 8761.90it/s] 28%|██▊       | 111302/400000 [00:12<00:33, 8732.45it/s] 28%|██▊       | 112258/400000 [00:12<00:32, 8963.64it/s] 28%|██▊       | 113157/400000 [00:12<00:32, 8779.11it/s] 29%|██▊       | 114075/400000 [00:12<00:32, 8892.51it/s] 29%|██▉       | 115032/400000 [00:12<00:31, 9084.71it/s] 29%|██▉       | 115943/400000 [00:12<00:32, 8827.77it/s] 29%|██▉       | 116901/400000 [00:12<00:31, 9039.83it/s] 29%|██▉       | 117817/400000 [00:12<00:31, 9074.29it/s] 30%|██▉       | 118783/400000 [00:12<00:30, 9241.82it/s] 30%|██▉       | 119742/400000 [00:12<00:30, 9341.41it/s] 30%|███       | 120679/400000 [00:13<00:30, 9212.06it/s] 30%|███       | 121603/400000 [00:13<00:30, 9183.98it/s] 31%|███       | 122523/400000 [00:13<00:30, 8955.28it/s] 31%|███       | 123421/400000 [00:13<00:31, 8889.29it/s] 31%|███       | 124399/400000 [00:13<00:30, 9137.88it/s] 31%|███▏      | 125316/400000 [00:13<00:30, 9038.95it/s] 32%|███▏      | 126280/400000 [00:13<00:29, 9209.19it/s] 32%|███▏      | 127266/400000 [00:13<00:29, 9392.01it/s] 32%|███▏      | 128257/400000 [00:13<00:28, 9540.38it/s] 32%|███▏      | 129242/400000 [00:14<00:28, 9629.68it/s] 33%|███▎      | 130207/400000 [00:14<00:28, 9515.90it/s] 33%|███▎      | 131198/400000 [00:14<00:27, 9629.22it/s] 33%|███▎      | 132175/400000 [00:14<00:27, 9669.97it/s] 33%|███▎      | 133187/400000 [00:14<00:27, 9800.34it/s] 34%|███▎      | 134169/400000 [00:14<00:27, 9694.10it/s] 34%|███▍      | 135140/400000 [00:14<00:27, 9466.16it/s] 34%|███▍      | 136122/400000 [00:14<00:27, 9568.49it/s] 34%|███▍      | 137111/400000 [00:14<00:27, 9662.49it/s] 35%|███▍      | 138096/400000 [00:14<00:26, 9715.83it/s] 35%|███▍      | 139069/400000 [00:15<00:27, 9581.23it/s] 35%|███▌      | 140029/400000 [00:15<00:27, 9413.64it/s] 35%|███▌      | 141008/400000 [00:15<00:27, 9521.99it/s] 35%|███▌      | 141981/400000 [00:15<00:26, 9581.43it/s] 36%|███▌      | 142943/400000 [00:15<00:26, 9592.75it/s] 36%|███▌      | 143923/400000 [00:15<00:26, 9653.31it/s] 36%|███▌      | 144889/400000 [00:15<00:26, 9473.35it/s] 36%|███▋      | 145882/400000 [00:15<00:26, 9605.50it/s] 37%|███▋      | 146860/400000 [00:15<00:26, 9655.95it/s] 37%|███▋      | 147841/400000 [00:15<00:25, 9701.65it/s] 37%|███▋      | 148812/400000 [00:16<00:26, 9658.78it/s] 37%|███▋      | 149779/400000 [00:16<00:26, 9301.17it/s] 38%|███▊      | 150727/400000 [00:16<00:26, 9353.56it/s] 38%|███▊      | 151709/400000 [00:16<00:26, 9487.08it/s] 38%|███▊      | 152705/400000 [00:16<00:25, 9623.42it/s] 38%|███▊      | 153670/400000 [00:16<00:25, 9480.01it/s] 39%|███▊      | 154620/400000 [00:16<00:26, 9120.81it/s] 39%|███▉      | 155584/400000 [00:16<00:26, 9269.12it/s] 39%|███▉      | 156550/400000 [00:16<00:25, 9381.53it/s] 39%|███▉      | 157491/400000 [00:16<00:25, 9359.71it/s] 40%|███▉      | 158463/400000 [00:17<00:25, 9463.89it/s] 40%|███▉      | 159412/400000 [00:17<00:25, 9366.51it/s] 40%|████      | 160393/400000 [00:17<00:25, 9494.65it/s] 40%|████      | 161368/400000 [00:17<00:24, 9568.68it/s] 41%|████      | 162342/400000 [00:17<00:24, 9619.25it/s] 41%|████      | 163318/400000 [00:17<00:24, 9660.12it/s] 41%|████      | 164285/400000 [00:17<00:25, 9392.72it/s] 41%|████▏     | 165266/400000 [00:17<00:24, 9511.71it/s] 42%|████▏     | 166219/400000 [00:17<00:24, 9499.00it/s] 42%|████▏     | 167171/400000 [00:17<00:24, 9418.00it/s] 42%|████▏     | 168132/400000 [00:18<00:24, 9473.80it/s] 42%|████▏     | 169081/400000 [00:18<00:24, 9307.12it/s] 43%|████▎     | 170013/400000 [00:18<00:25, 9112.60it/s] 43%|████▎     | 170927/400000 [00:18<00:25, 8902.55it/s] 43%|████▎     | 171820/400000 [00:18<00:25, 8819.16it/s] 43%|████▎     | 172727/400000 [00:18<00:25, 8892.74it/s] 43%|████▎     | 173619/400000 [00:18<00:25, 8900.25it/s] 44%|████▎     | 174593/400000 [00:18<00:24, 9135.59it/s] 44%|████▍     | 175589/400000 [00:18<00:23, 9368.11it/s] 44%|████▍     | 176579/400000 [00:19<00:23, 9520.38it/s] 44%|████▍     | 177584/400000 [00:19<00:22, 9672.78it/s] 45%|████▍     | 178554/400000 [00:19<00:23, 9470.26it/s] 45%|████▍     | 179511/400000 [00:19<00:23, 9498.41it/s] 45%|████▌     | 180510/400000 [00:19<00:22, 9639.41it/s] 45%|████▌     | 181484/400000 [00:19<00:22, 9669.29it/s] 46%|████▌     | 182482/400000 [00:19<00:22, 9759.19it/s] 46%|████▌     | 183460/400000 [00:19<00:22, 9604.21it/s] 46%|████▌     | 184456/400000 [00:19<00:22, 9707.78it/s] 46%|████▋     | 185428/400000 [00:19<00:22, 9641.68it/s] 47%|████▋     | 186394/400000 [00:20<00:22, 9296.63it/s] 47%|████▋     | 187398/400000 [00:20<00:22, 9507.25it/s] 47%|████▋     | 188353/400000 [00:20<00:22, 9309.33it/s] 47%|████▋     | 189288/400000 [00:20<00:22, 9261.57it/s] 48%|████▊     | 190272/400000 [00:20<00:22, 9427.81it/s] 48%|████▊     | 191248/400000 [00:20<00:21, 9523.88it/s] 48%|████▊     | 192230/400000 [00:20<00:21, 9608.72it/s] 48%|████▊     | 193193/400000 [00:20<00:21, 9449.99it/s] 49%|████▊     | 194169/400000 [00:20<00:21, 9539.81it/s] 49%|████▉     | 195149/400000 [00:20<00:21, 9613.98it/s] 49%|████▉     | 196112/400000 [00:21<00:21, 9583.26it/s] 49%|████▉     | 197091/400000 [00:21<00:21, 9643.60it/s] 50%|████▉     | 198057/400000 [00:21<00:21, 9419.16it/s] 50%|████▉     | 199009/400000 [00:21<00:21, 9446.77it/s] 50%|████▉     | 199995/400000 [00:21<00:20, 9566.07it/s] 50%|█████     | 200953/400000 [00:21<00:21, 9392.22it/s] 50%|█████     | 201894/400000 [00:21<00:22, 8850.31it/s] 51%|█████     | 202872/400000 [00:21<00:21, 9108.72it/s] 51%|█████     | 203800/400000 [00:21<00:21, 9157.00it/s] 51%|█████     | 204793/400000 [00:21<00:20, 9373.99it/s] 51%|█████▏    | 205767/400000 [00:22<00:20, 9480.37it/s] 52%|█████▏    | 206754/400000 [00:22<00:20, 9592.73it/s] 52%|█████▏    | 207721/400000 [00:22<00:20, 9613.33it/s] 52%|█████▏    | 208685/400000 [00:22<00:20, 9532.81it/s] 52%|█████▏    | 209641/400000 [00:22<00:19, 9538.85it/s] 53%|█████▎    | 210626/400000 [00:22<00:19, 9626.75it/s] 53%|█████▎    | 211590/400000 [00:22<00:19, 9491.15it/s] 53%|█████▎    | 212541/400000 [00:22<00:20, 9317.56it/s] 53%|█████▎    | 213515/400000 [00:22<00:19, 9436.56it/s] 54%|█████▎    | 214487/400000 [00:23<00:19, 9519.08it/s] 54%|█████▍    | 215441/400000 [00:23<00:19, 9258.16it/s] 54%|█████▍    | 216370/400000 [00:23<00:19, 9185.30it/s] 54%|█████▍    | 217291/400000 [00:23<00:20, 9039.96it/s] 55%|█████▍    | 218197/400000 [00:23<00:20, 8949.30it/s] 55%|█████▍    | 219159/400000 [00:23<00:19, 9138.09it/s] 55%|█████▌    | 220075/400000 [00:23<00:19, 9049.25it/s] 55%|█████▌    | 220982/400000 [00:23<00:20, 8939.20it/s] 55%|█████▌    | 221907/400000 [00:23<00:19, 9028.06it/s] 56%|█████▌    | 222819/400000 [00:23<00:19, 9053.42it/s] 56%|█████▌    | 223740/400000 [00:24<00:19, 9096.04it/s] 56%|█████▌    | 224696/400000 [00:24<00:18, 9228.92it/s] 56%|█████▋    | 225620/400000 [00:24<00:19, 9025.40it/s] 57%|█████▋    | 226548/400000 [00:24<00:19, 9098.33it/s] 57%|█████▋    | 227460/400000 [00:24<00:18, 9100.39it/s] 57%|█████▋    | 228371/400000 [00:24<00:19, 8963.69it/s] 57%|█████▋    | 229269/400000 [00:24<00:19, 8858.94it/s] 58%|█████▊    | 230156/400000 [00:24<00:19, 8795.47it/s] 58%|█████▊    | 231112/400000 [00:24<00:18, 9010.26it/s] 58%|█████▊    | 232015/400000 [00:24<00:19, 8752.54it/s] 58%|█████▊    | 232901/400000 [00:25<00:19, 8784.00it/s] 58%|█████▊    | 233850/400000 [00:25<00:18, 8983.18it/s] 59%|█████▊    | 234768/400000 [00:25<00:18, 9041.05it/s] 59%|█████▉    | 235728/400000 [00:25<00:17, 9199.36it/s] 59%|█████▉    | 236710/400000 [00:25<00:17, 9377.02it/s] 59%|█████▉    | 237650/400000 [00:25<00:17, 9271.95it/s] 60%|█████▉    | 238626/400000 [00:25<00:17, 9412.44it/s] 60%|█████▉    | 239570/400000 [00:25<00:17, 9310.49it/s] 60%|██████    | 240550/400000 [00:25<00:17, 9088.66it/s] 60%|██████    | 241504/400000 [00:25<00:17, 9218.68it/s] 61%|██████    | 242436/400000 [00:26<00:17, 9248.51it/s] 61%|██████    | 243373/400000 [00:26<00:16, 9278.44it/s] 61%|██████    | 244302/400000 [00:26<00:16, 9245.77it/s] 61%|██████▏   | 245283/400000 [00:26<00:16, 9406.73it/s] 62%|██████▏   | 246278/400000 [00:26<00:16, 9562.98it/s] 62%|██████▏   | 247236/400000 [00:26<00:16, 9516.81it/s] 62%|██████▏   | 248189/400000 [00:26<00:16, 9471.69it/s] 62%|██████▏   | 249137/400000 [00:26<00:16, 9363.13it/s] 63%|██████▎   | 250090/400000 [00:26<00:15, 9412.11it/s] 63%|██████▎   | 251091/400000 [00:26<00:15, 9581.40it/s] 63%|██████▎   | 252094/400000 [00:27<00:15, 9710.21it/s] 63%|██████▎   | 253067/400000 [00:27<00:15, 9680.43it/s] 64%|██████▎   | 254058/400000 [00:27<00:14, 9746.71it/s] 64%|██████▍   | 255034/400000 [00:27<00:14, 9747.19it/s] 64%|██████▍   | 256016/400000 [00:27<00:14, 9766.47it/s] 64%|██████▍   | 257017/400000 [00:27<00:14, 9838.16it/s] 65%|██████▍   | 258002/400000 [00:27<00:14, 9565.83it/s] 65%|██████▍   | 258968/400000 [00:27<00:14, 9592.99it/s] 65%|██████▍   | 259946/400000 [00:27<00:14, 9646.10it/s] 65%|██████▌   | 260912/400000 [00:28<00:14, 9638.57it/s] 65%|██████▌   | 261877/400000 [00:28<00:14, 9575.24it/s] 66%|██████▌   | 262836/400000 [00:28<00:14, 9436.08it/s] 66%|██████▌   | 263781/400000 [00:28<00:14, 9407.06it/s] 66%|██████▌   | 264770/400000 [00:28<00:14, 9545.95it/s] 66%|██████▋   | 265750/400000 [00:28<00:13, 9618.21it/s] 67%|██████▋   | 266720/400000 [00:28<00:13, 9641.53it/s] 67%|██████▋   | 267687/400000 [00:28<00:13, 9648.52it/s] 67%|██████▋   | 268653/400000 [00:28<00:13, 9399.52it/s] 67%|██████▋   | 269595/400000 [00:28<00:14, 9310.13it/s] 68%|██████▊   | 270588/400000 [00:29<00:13, 9487.48it/s] 68%|██████▊   | 271539/400000 [00:29<00:13, 9362.61it/s] 68%|██████▊   | 272477/400000 [00:29<00:13, 9324.30it/s] 68%|██████▊   | 273417/400000 [00:29<00:13, 9345.61it/s] 69%|██████▊   | 274387/400000 [00:29<00:13, 9447.64it/s] 69%|██████▉   | 275334/400000 [00:29<00:13, 9451.44it/s] 69%|██████▉   | 276286/400000 [00:29<00:13, 9469.47it/s] 69%|██████▉   | 277266/400000 [00:29<00:12, 9566.22it/s] 70%|██████▉   | 278224/400000 [00:29<00:13, 9348.40it/s] 70%|██████▉   | 279161/400000 [00:29<00:13, 9056.60it/s] 70%|███████   | 280070/400000 [00:30<00:13, 8969.77it/s] 70%|███████   | 280999/400000 [00:30<00:13, 9062.91it/s] 70%|███████   | 281911/400000 [00:30<00:13, 9079.14it/s] 71%|███████   | 282821/400000 [00:30<00:12, 9072.39it/s] 71%|███████   | 283735/400000 [00:30<00:12, 9092.03it/s] 71%|███████   | 284668/400000 [00:30<00:12, 9161.32it/s] 71%|███████▏  | 285585/400000 [00:30<00:12, 9106.74it/s] 72%|███████▏  | 286497/400000 [00:30<00:12, 8822.30it/s] 72%|███████▏  | 287448/400000 [00:30<00:12, 9016.61it/s] 72%|███████▏  | 288377/400000 [00:30<00:12, 9096.90it/s] 72%|███████▏  | 289289/400000 [00:31<00:12, 9092.18it/s] 73%|███████▎  | 290221/400000 [00:31<00:11, 9156.95it/s] 73%|███████▎  | 291161/400000 [00:31<00:11, 9228.11it/s] 73%|███████▎  | 292085/400000 [00:31<00:11, 9222.19it/s] 73%|███████▎  | 293050/400000 [00:31<00:11, 9345.84it/s] 73%|███████▎  | 293986/400000 [00:31<00:11, 9257.01it/s] 74%|███████▎  | 294930/400000 [00:31<00:11, 9310.41it/s] 74%|███████▍  | 295862/400000 [00:31<00:11, 9203.25it/s] 74%|███████▍  | 296784/400000 [00:31<00:11, 9008.51it/s] 74%|███████▍  | 297747/400000 [00:31<00:11, 9185.76it/s] 75%|███████▍  | 298717/400000 [00:32<00:10, 9332.98it/s] 75%|███████▍  | 299653/400000 [00:32<00:10, 9220.91it/s] 75%|███████▌  | 300586/400000 [00:32<00:10, 9252.60it/s] 75%|███████▌  | 301536/400000 [00:32<00:10, 9324.34it/s] 76%|███████▌  | 302470/400000 [00:32<00:10, 9295.01it/s] 76%|███████▌  | 303421/400000 [00:32<00:10, 9356.27it/s] 76%|███████▌  | 304381/400000 [00:32<00:10, 9426.26it/s] 76%|███████▋  | 305325/400000 [00:32<00:10, 9418.89it/s] 77%|███████▋  | 306290/400000 [00:32<00:09, 9484.76it/s] 77%|███████▋  | 307263/400000 [00:32<00:09, 9554.40it/s] 77%|███████▋  | 308229/400000 [00:33<00:09, 9583.62it/s] 77%|███████▋  | 309188/400000 [00:33<00:09, 9281.56it/s] 78%|███████▊  | 310119/400000 [00:33<00:09, 9175.11it/s] 78%|███████▊  | 311067/400000 [00:33<00:09, 9263.73it/s] 78%|███████▊  | 312063/400000 [00:33<00:09, 9461.49it/s] 78%|███████▊  | 313042/400000 [00:33<00:09, 9555.62it/s] 78%|███████▊  | 314000/400000 [00:33<00:09, 9420.68it/s] 79%|███████▊  | 314944/400000 [00:33<00:09, 9273.75it/s] 79%|███████▉  | 315874/400000 [00:33<00:09, 9068.77it/s] 79%|███████▉  | 316784/400000 [00:34<00:09, 8838.27it/s] 79%|███████▉  | 317732/400000 [00:34<00:09, 9021.18it/s] 80%|███████▉  | 318714/400000 [00:34<00:08, 9244.85it/s] 80%|███████▉  | 319642/400000 [00:34<00:08, 9251.19it/s] 80%|████████  | 320633/400000 [00:34<00:08, 9438.63it/s] 80%|████████  | 321615/400000 [00:34<00:08, 9547.63it/s] 81%|████████  | 322574/400000 [00:34<00:08, 9560.18it/s] 81%|████████  | 323539/400000 [00:34<00:07, 9586.20it/s] 81%|████████  | 324499/400000 [00:34<00:08, 9108.85it/s] 81%|████████▏ | 325420/400000 [00:34<00:08, 9138.39it/s] 82%|████████▏ | 326396/400000 [00:35<00:07, 9314.29it/s] 82%|████████▏ | 327377/400000 [00:35<00:07, 9453.04it/s] 82%|████████▏ | 328326/400000 [00:35<00:07, 9381.07it/s] 82%|████████▏ | 329276/400000 [00:35<00:07, 9415.07it/s] 83%|████████▎ | 330280/400000 [00:35<00:07, 9592.09it/s] 83%|████████▎ | 331253/400000 [00:35<00:07, 9632.52it/s] 83%|████████▎ | 332238/400000 [00:35<00:06, 9696.19it/s] 83%|████████▎ | 333209/400000 [00:35<00:06, 9691.52it/s] 84%|████████▎ | 334179/400000 [00:35<00:06, 9514.18it/s] 84%|████████▍ | 335132/400000 [00:35<00:06, 9490.26it/s] 84%|████████▍ | 336082/400000 [00:36<00:06, 9457.56it/s] 84%|████████▍ | 337029/400000 [00:36<00:06, 9391.20it/s] 84%|████████▍ | 337969/400000 [00:36<00:06, 9290.15it/s] 85%|████████▍ | 338899/400000 [00:36<00:06, 9132.80it/s] 85%|████████▍ | 339814/400000 [00:36<00:06, 9005.82it/s] 85%|████████▌ | 340727/400000 [00:36<00:06, 9040.60it/s] 85%|████████▌ | 341686/400000 [00:36<00:06, 9197.18it/s] 86%|████████▌ | 342607/400000 [00:36<00:06, 9151.25it/s] 86%|████████▌ | 343524/400000 [00:36<00:06, 9048.70it/s] 86%|████████▌ | 344430/400000 [00:37<00:06, 8832.43it/s] 86%|████████▋ | 345316/400000 [00:37<00:06, 8685.60it/s] 87%|████████▋ | 346215/400000 [00:37<00:06, 8773.71it/s] 87%|████████▋ | 347109/400000 [00:37<00:05, 8822.17it/s] 87%|████████▋ | 348012/400000 [00:37<00:05, 8881.91it/s] 87%|████████▋ | 348994/400000 [00:37<00:05, 9141.63it/s] 87%|████████▋ | 349987/400000 [00:37<00:05, 9364.61it/s] 88%|████████▊ | 350991/400000 [00:37<00:05, 9555.50it/s] 88%|████████▊ | 351958/400000 [00:37<00:05, 9589.56it/s] 88%|████████▊ | 352920/400000 [00:37<00:05, 9389.27it/s] 88%|████████▊ | 353896/400000 [00:38<00:04, 9495.09it/s] 89%|████████▊ | 354848/400000 [00:38<00:04, 9200.92it/s] 89%|████████▉ | 355791/400000 [00:38<00:04, 9265.90it/s] 89%|████████▉ | 356721/400000 [00:38<00:04, 9258.80it/s] 89%|████████▉ | 357660/400000 [00:38<00:04, 9295.21it/s] 90%|████████▉ | 358635/400000 [00:38<00:04, 9426.69it/s] 90%|████████▉ | 359593/400000 [00:38<00:04, 9471.17it/s] 90%|█████████ | 360574/400000 [00:38<00:04, 9569.02it/s] 90%|█████████ | 361561/400000 [00:38<00:03, 9656.37it/s] 91%|█████████ | 362528/400000 [00:38<00:03, 9611.66it/s] 91%|█████████ | 363532/400000 [00:39<00:03, 9734.87it/s] 91%|█████████ | 364507/400000 [00:39<00:03, 9613.30it/s] 91%|█████████▏| 365470/400000 [00:39<00:03, 9284.83it/s] 92%|█████████▏| 366402/400000 [00:39<00:03, 9288.80it/s] 92%|█████████▏| 367334/400000 [00:39<00:03, 9207.74it/s] 92%|█████████▏| 368322/400000 [00:39<00:03, 9397.19it/s] 92%|█████████▏| 369295/400000 [00:39<00:03, 9494.11it/s] 93%|█████████▎| 370247/400000 [00:39<00:03, 9416.93it/s] 93%|█████████▎| 371208/400000 [00:39<00:03, 9473.26it/s] 93%|█████████▎| 372157/400000 [00:39<00:02, 9453.86it/s] 93%|█████████▎| 373104/400000 [00:40<00:02, 9374.47it/s] 94%|█████████▎| 374043/400000 [00:40<00:02, 9323.41it/s] 94%|█████████▎| 374977/400000 [00:40<00:02, 9327.78it/s] 94%|█████████▍| 375979/400000 [00:40<00:02, 9523.11it/s] 94%|█████████▍| 376933/400000 [00:40<00:02, 9525.45it/s] 94%|█████████▍| 377932/400000 [00:40<00:02, 9659.52it/s] 95%|█████████▍| 378935/400000 [00:40<00:02, 9765.74it/s] 95%|█████████▍| 379918/400000 [00:40<00:02, 9782.62it/s] 95%|█████████▌| 380900/400000 [00:40<00:01, 9790.93it/s] 95%|█████████▌| 381880/400000 [00:40<00:01, 9418.81it/s] 96%|█████████▌| 382844/400000 [00:41<00:01, 9481.88it/s] 96%|█████████▌| 383816/400000 [00:41<00:01, 9549.84it/s] 96%|█████████▌| 384773/400000 [00:41<00:01, 9548.79it/s] 96%|█████████▋| 385730/400000 [00:41<00:01, 9239.89it/s] 97%|█████████▋| 386658/400000 [00:41<00:01, 9111.02it/s] 97%|█████████▋| 387657/400000 [00:41<00:01, 9357.01it/s] 97%|█████████▋| 388658/400000 [00:41<00:01, 9543.61it/s] 97%|█████████▋| 389652/400000 [00:41<00:01, 9656.55it/s] 98%|█████████▊| 390621/400000 [00:41<00:00, 9500.79it/s] 98%|█████████▊| 391574/400000 [00:42<00:00, 9204.52it/s] 98%|█████████▊| 392499/400000 [00:42<00:00, 9145.79it/s] 98%|█████████▊| 393417/400000 [00:42<00:00, 9117.21it/s] 99%|█████████▊| 394331/400000 [00:42<00:00, 8983.93it/s] 99%|█████████▉| 395296/400000 [00:42<00:00, 9171.40it/s] 99%|█████████▉| 396216/400000 [00:42<00:00, 9051.33it/s] 99%|█████████▉| 397146/400000 [00:42<00:00, 9123.55it/s]100%|█████████▉| 398149/400000 [00:42<00:00, 9376.16it/s]100%|█████████▉| 399139/400000 [00:42<00:00, 9524.47it/s]100%|█████████▉| 399999/400000 [00:42<00:00, 9322.67it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc2f7f34940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011196412838920844 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011183592188717131 	 Accuracy: 59

  model saves at 59% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15814 out of table with 15800 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15814 out of table with 15800 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-14 10:24:36.552123: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 10:24:36.555916: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-14 10:24:36.556051: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564807f4d710 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 10:24:36.556065: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc2ab467160> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.5746 - accuracy: 0.5060
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6973 - accuracy: 0.4980 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6973 - accuracy: 0.4980
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7471 - accuracy: 0.4947
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7341 - accuracy: 0.4956
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7075 - accuracy: 0.4973
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7236 - accuracy: 0.4963
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7030 - accuracy: 0.4976
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7228 - accuracy: 0.4963
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7203 - accuracy: 0.4965
11000/25000 [============>.................] - ETA: 4s - loss: 7.6903 - accuracy: 0.4985
12000/25000 [=============>................] - ETA: 4s - loss: 7.7062 - accuracy: 0.4974
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6654 - accuracy: 0.5001
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6699 - accuracy: 0.4998
15000/25000 [=================>............] - ETA: 3s - loss: 7.6891 - accuracy: 0.4985
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6829 - accuracy: 0.4989
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6838 - accuracy: 0.4989
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6739 - accuracy: 0.4995
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6797 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6798 - accuracy: 0.4991
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6847 - accuracy: 0.4988
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6724 - accuracy: 0.4996
25000/25000 [==============================] - 9s 371us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fc26491bb38> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fc265aff128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 914ms/step - loss: 1.4621 - crf_viterbi_accuracy: 0.0267 - val_loss: 1.4350 - val_crf_viterbi_accuracy: 0.0000e+00

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

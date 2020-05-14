
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f19dbbb1fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 11:12:35.986868
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 11:12:35.990431
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 11:12:35.993798
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 11:12:35.996820
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f19e7bc9470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354363.7812
Epoch 2/10

1/1 [==============================] - 0s 107ms/step - loss: 260299.3750
Epoch 3/10

1/1 [==============================] - 0s 116ms/step - loss: 177101.2969
Epoch 4/10

1/1 [==============================] - 0s 108ms/step - loss: 108859.9219
Epoch 5/10

1/1 [==============================] - 0s 100ms/step - loss: 66116.1562
Epoch 6/10

1/1 [==============================] - 0s 102ms/step - loss: 41661.0234
Epoch 7/10

1/1 [==============================] - 0s 98ms/step - loss: 27778.1973
Epoch 8/10

1/1 [==============================] - 0s 99ms/step - loss: 19565.2461
Epoch 9/10

1/1 [==============================] - 0s 103ms/step - loss: 14489.3486
Epoch 10/10

1/1 [==============================] - 0s 107ms/step - loss: 11192.2031

  #### Inference Need return ypred, ytrue ######################### 
[[-0.9537454   0.951337    1.3432167   0.61905277  1.3944545  -0.85175496
   0.646007    0.9219543   0.7966659   0.2982691  -1.041545    1.1579185
  -0.70019203  0.8228373  -1.8471947   0.23757106 -0.6239581  -0.44569072
  -1.0759642  -0.69307065 -0.015172    0.5554504   0.41072    -1.7025363
  -0.537452    1.2991121  -0.41066268 -0.45144004  1.0835344  -0.4476463
  -0.32855335 -0.30976388  0.45264283  0.02801765  0.48531714 -0.5651928
   0.47654897  1.2653968   0.7387666  -0.74365115 -1.1092441  -0.8104341
   0.5934396  -0.0130043  -1.1515377   0.6747274   1.4541514   0.3082651
  -0.33346754  0.16625357 -1.2403663   0.20957337  1.2254145  -1.5446013
  -1.4913762  -1.482248    0.14861906  0.6350374   2.0796938  -0.06723353
   0.6010796   1.7179728   1.1707753  -1.7721665  -1.5680475  -1.3337926
   0.41314214 -0.52846    -1.080477   -0.29225427 -0.8940464  -0.8704833
  -0.14606164  0.5642804   1.6601207  -1.0480435   1.7434272   0.36654055
  -0.0962683  -0.22415778 -0.05194837 -0.5954551  -0.888582    0.4730331
   1.0460961  -1.207459    0.46262684 -0.96082866  0.5963192  -0.01085869
  -1.5780563   0.26570526  0.29759383  0.5264572  -0.92084354  1.5210196
  -0.03786555 -0.5631665   1.5569175   0.08621994  0.8855274  -0.4834145
  -0.64639133 -0.02301171 -0.39559793  0.6337905  -0.50221777 -0.05449188
   0.0806126  -1.8544624   0.8669014   1.3474929   0.4004303  -1.3477314
  -0.28701773 -0.14126599  0.5667993  -0.3761081   0.5558971   0.9745474
   0.16402687  4.4193196   5.45304     4.898242    5.4571857   5.48268
   7.694695    6.0398183   5.448247    5.4352536   5.7960887   6.9725413
   6.3543262   6.595839    6.4481134   5.751375    5.766097    4.694015
   5.409454    4.5709233   6.5110493   6.902422    6.463096    6.481045
   5.9021773   5.3373504   7.4470024   6.124714    6.4520383   4.5518637
   4.7613664   6.400287    4.964924    5.8270073   6.7870636   4.9235597
   6.8193884   4.2218876   5.6081676   6.3903375   5.396703    4.6394825
   5.362015    4.1032305   7.1098175   5.2177496   5.398161    5.08645
   6.7988515   5.6542907   3.9115145   5.70682     5.748938    5.9650455
   4.2597575   5.5742483   5.634675    5.781832    5.6445246   6.148167
   0.8765225   0.3886925   1.8066587   1.2954329   0.6198      1.8740246
   0.19680238  0.5583092   0.42792892  2.144074    1.254259    0.5169266
   0.5673574   1.755113    0.29152226  2.142889    2.3424616   2.3471603
   1.0495541   0.530326    0.18415141  0.32572305  1.1876228   2.0179706
   1.6732435   0.84054506  0.44335377  1.3520029   1.8224995   1.8921235
   1.4755212   0.59786487  0.72802806  0.5543964   0.3813585   0.79679847
   2.5961804   0.93768495  0.2019062   0.40273774  1.240358    0.8487576
   1.2007217   0.63644546  1.1518646   1.5056977   0.7413427   0.89408755
   0.6780343   1.1236405   0.8261257   0.8351586   1.2418478   0.2904321
   0.9711046   0.9430007   2.0262375   1.2690425   0.52724963  1.036102
   2.153399    1.1954992   0.26816422  0.23979205  2.0404983   0.8551922
   2.0498734   0.34915483  1.9291345   0.62869966  0.9950116   0.3172685
   0.51210105  2.222031    0.630767    1.6226635   1.6057243   1.4315751
   2.0283966   0.55486625  2.8333378   2.727611    1.8407475   0.3533213
   0.28089052  0.2177304   0.80372477  2.0345283   1.8906305   0.13511807
   0.61726665  1.9228177   0.8245322   1.9792935   0.23159766  0.15706134
   0.7421534   1.1048843   0.8970138   1.1564004   2.7899547   0.87502855
   0.55847514  0.36281198  0.86953396  1.872757    1.2147238   1.4381549
   1.2609962   1.0727315   0.32237697  0.36323535  0.998192    0.6098435
   1.0754207   1.4425015   1.3340328   1.4917579   1.3639529   0.35311627
   0.04925805  5.63054     8.062903    7.4328656   4.7501183   7.120123
   7.293251    7.0569925   6.264742    5.507835    5.5193534   5.447053
   5.7544875   4.48299     5.5847173   6.132236    6.897039    6.870799
   5.4010816   5.561722    5.559195    6.7271266   6.5761347   5.679262
   6.9468145   5.9725175   5.900833    5.0772448   6.7540855   6.183901
   6.273005    5.7010274   7.2354026   7.7608213   7.1603155   6.2523484
   8.050468    5.650182    7.8122587   6.8105283   5.410106    7.779356
   6.4611187   5.7678847   6.4971423   8.11817     5.545142    6.9769444
   6.7660904   6.5403585   8.0631695   5.660625    6.3050394   7.397155
   6.397323    5.8441577   6.5502615   6.74154     7.1235657   7.4597945
  -4.6517906  -3.9491124   9.48033   ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 11:12:44.798882
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.7805
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 11:12:44.803158
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    9385.6
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 11:12:44.807293
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.4143
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 11:12:44.811168
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -839.547
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139748680810568
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139747470652024
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139747470652528
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139747470653032
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139747470653536
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139747470654040

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f19dbdff208> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.452018
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.425341
grad_step = 000002, loss = 0.405781
grad_step = 000003, loss = 0.383514
grad_step = 000004, loss = 0.360397
grad_step = 000005, loss = 0.338824
grad_step = 000006, loss = 0.319881
grad_step = 000007, loss = 0.303881
grad_step = 000008, loss = 0.289166
grad_step = 000009, loss = 0.276034
grad_step = 000010, loss = 0.261950
grad_step = 000011, loss = 0.249398
grad_step = 000012, loss = 0.239266
grad_step = 000013, loss = 0.229810
grad_step = 000014, loss = 0.219767
grad_step = 000015, loss = 0.209383
grad_step = 000016, loss = 0.199196
grad_step = 000017, loss = 0.190360
grad_step = 000018, loss = 0.182692
grad_step = 000019, loss = 0.174678
grad_step = 000020, loss = 0.165953
grad_step = 000021, loss = 0.157031
grad_step = 000022, loss = 0.148157
grad_step = 000023, loss = 0.139834
grad_step = 000024, loss = 0.132328
grad_step = 000025, loss = 0.125049
grad_step = 000026, loss = 0.117652
grad_step = 000027, loss = 0.110620
grad_step = 000028, loss = 0.103981
grad_step = 000029, loss = 0.097397
grad_step = 000030, loss = 0.091356
grad_step = 000031, loss = 0.085787
grad_step = 000032, loss = 0.080070
grad_step = 000033, loss = 0.074412
grad_step = 000034, loss = 0.069024
grad_step = 000035, loss = 0.064118
grad_step = 000036, loss = 0.059434
grad_step = 000037, loss = 0.054864
grad_step = 000038, loss = 0.050669
grad_step = 000039, loss = 0.046209
grad_step = 000040, loss = 0.041957
grad_step = 000041, loss = 0.038005
grad_step = 000042, loss = 0.034350
grad_step = 000043, loss = 0.031165
grad_step = 000044, loss = 0.028169
grad_step = 000045, loss = 0.025466
grad_step = 000046, loss = 0.023116
grad_step = 000047, loss = 0.021032
grad_step = 000048, loss = 0.018917
grad_step = 000049, loss = 0.016988
grad_step = 000050, loss = 0.015300
grad_step = 000051, loss = 0.013839
grad_step = 000052, loss = 0.012434
grad_step = 000053, loss = 0.011160
grad_step = 000054, loss = 0.010099
grad_step = 000055, loss = 0.009207
grad_step = 000056, loss = 0.008347
grad_step = 000057, loss = 0.007601
grad_step = 000058, loss = 0.006936
grad_step = 000059, loss = 0.006364
grad_step = 000060, loss = 0.005835
grad_step = 000061, loss = 0.005378
grad_step = 000062, loss = 0.004981
grad_step = 000063, loss = 0.004629
grad_step = 000064, loss = 0.004326
grad_step = 000065, loss = 0.004056
grad_step = 000066, loss = 0.003814
grad_step = 000067, loss = 0.003601
grad_step = 000068, loss = 0.003428
grad_step = 000069, loss = 0.003271
grad_step = 000070, loss = 0.003127
grad_step = 000071, loss = 0.003008
grad_step = 000072, loss = 0.002914
grad_step = 000073, loss = 0.002827
grad_step = 000074, loss = 0.002740
grad_step = 000075, loss = 0.002672
grad_step = 000076, loss = 0.002621
grad_step = 000077, loss = 0.002573
grad_step = 000078, loss = 0.002528
grad_step = 000079, loss = 0.002491
grad_step = 000080, loss = 0.002464
grad_step = 000081, loss = 0.002435
grad_step = 000082, loss = 0.002409
grad_step = 000083, loss = 0.002387
grad_step = 000084, loss = 0.002371
grad_step = 000085, loss = 0.002364
grad_step = 000086, loss = 0.002374
grad_step = 000087, loss = 0.002416
grad_step = 000088, loss = 0.002495
grad_step = 000089, loss = 0.002527
grad_step = 000090, loss = 0.002439
grad_step = 000091, loss = 0.002288
grad_step = 000092, loss = 0.002270
grad_step = 000093, loss = 0.002359
grad_step = 000094, loss = 0.002363
grad_step = 000095, loss = 0.002264
grad_step = 000096, loss = 0.002225
grad_step = 000097, loss = 0.002283
grad_step = 000098, loss = 0.002293
grad_step = 000099, loss = 0.002218
grad_step = 000100, loss = 0.002196
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002237
grad_step = 000102, loss = 0.002229
grad_step = 000103, loss = 0.002175
grad_step = 000104, loss = 0.002167
grad_step = 000105, loss = 0.002194
grad_step = 000106, loss = 0.002178
grad_step = 000107, loss = 0.002139
grad_step = 000108, loss = 0.002137
grad_step = 000109, loss = 0.002152
grad_step = 000110, loss = 0.002135
grad_step = 000111, loss = 0.002109
grad_step = 000112, loss = 0.002108
grad_step = 000113, loss = 0.002115
grad_step = 000114, loss = 0.002103
grad_step = 000115, loss = 0.002083
grad_step = 000116, loss = 0.002079
grad_step = 000117, loss = 0.002083
grad_step = 000118, loss = 0.002075
grad_step = 000119, loss = 0.002059
grad_step = 000120, loss = 0.002051
grad_step = 000121, loss = 0.002052
grad_step = 000122, loss = 0.002048
grad_step = 000123, loss = 0.002036
grad_step = 000124, loss = 0.002024
grad_step = 000125, loss = 0.002020
grad_step = 000126, loss = 0.002017
grad_step = 000127, loss = 0.002009
grad_step = 000128, loss = 0.001998
grad_step = 000129, loss = 0.001987
grad_step = 000130, loss = 0.001980
grad_step = 000131, loss = 0.001974
grad_step = 000132, loss = 0.001966
grad_step = 000133, loss = 0.001955
grad_step = 000134, loss = 0.001943
grad_step = 000135, loss = 0.001932
grad_step = 000136, loss = 0.001922
grad_step = 000137, loss = 0.001913
grad_step = 000138, loss = 0.001903
grad_step = 000139, loss = 0.001894
grad_step = 000140, loss = 0.001887
grad_step = 000141, loss = 0.001883
grad_step = 000142, loss = 0.001876
grad_step = 000143, loss = 0.001862
grad_step = 000144, loss = 0.001843
grad_step = 000145, loss = 0.001828
grad_step = 000146, loss = 0.001821
grad_step = 000147, loss = 0.001818
grad_step = 000148, loss = 0.001821
grad_step = 000149, loss = 0.001845
grad_step = 000150, loss = 0.001908
grad_step = 000151, loss = 0.001868
grad_step = 000152, loss = 0.001821
grad_step = 000153, loss = 0.001782
grad_step = 000154, loss = 0.001758
grad_step = 000155, loss = 0.001782
grad_step = 000156, loss = 0.001822
grad_step = 000157, loss = 0.001806
grad_step = 000158, loss = 0.001776
grad_step = 000159, loss = 0.001795
grad_step = 000160, loss = 0.001829
grad_step = 000161, loss = 0.001882
grad_step = 000162, loss = 0.001955
grad_step = 000163, loss = 0.001956
grad_step = 000164, loss = 0.001838
grad_step = 000165, loss = 0.001728
grad_step = 000166, loss = 0.001742
grad_step = 000167, loss = 0.001814
grad_step = 000168, loss = 0.001746
grad_step = 000169, loss = 0.001729
grad_step = 000170, loss = 0.001754
grad_step = 000171, loss = 0.001690
grad_step = 000172, loss = 0.001720
grad_step = 000173, loss = 0.001717
grad_step = 000174, loss = 0.001648
grad_step = 000175, loss = 0.001699
grad_step = 000176, loss = 0.001711
grad_step = 000177, loss = 0.001656
grad_step = 000178, loss = 0.001680
grad_step = 000179, loss = 0.001668
grad_step = 000180, loss = 0.001629
grad_step = 000181, loss = 0.001690
grad_step = 000182, loss = 0.001662
grad_step = 000183, loss = 0.001628
grad_step = 000184, loss = 0.001683
grad_step = 000185, loss = 0.001631
grad_step = 000186, loss = 0.001638
grad_step = 000187, loss = 0.001642
grad_step = 000188, loss = 0.001596
grad_step = 000189, loss = 0.001617
grad_step = 000190, loss = 0.001599
grad_step = 000191, loss = 0.001599
grad_step = 000192, loss = 0.001611
grad_step = 000193, loss = 0.001588
grad_step = 000194, loss = 0.001607
grad_step = 000195, loss = 0.001610
grad_step = 000196, loss = 0.001620
grad_step = 000197, loss = 0.001688
grad_step = 000198, loss = 0.001765
grad_step = 000199, loss = 0.001907
grad_step = 000200, loss = 0.002016
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001996
grad_step = 000202, loss = 0.001799
grad_step = 000203, loss = 0.001640
grad_step = 000204, loss = 0.001617
grad_step = 000205, loss = 0.001688
grad_step = 000206, loss = 0.001742
grad_step = 000207, loss = 0.001706
grad_step = 000208, loss = 0.001625
grad_step = 000209, loss = 0.001568
grad_step = 000210, loss = 0.001615
grad_step = 000211, loss = 0.001680
grad_step = 000212, loss = 0.001625
grad_step = 000213, loss = 0.001545
grad_step = 000214, loss = 0.001556
grad_step = 000215, loss = 0.001608
grad_step = 000216, loss = 0.001606
grad_step = 000217, loss = 0.001550
grad_step = 000218, loss = 0.001528
grad_step = 000219, loss = 0.001559
grad_step = 000220, loss = 0.001575
grad_step = 000221, loss = 0.001543
grad_step = 000222, loss = 0.001519
grad_step = 000223, loss = 0.001533
grad_step = 000224, loss = 0.001544
grad_step = 000225, loss = 0.001526
grad_step = 000226, loss = 0.001510
grad_step = 000227, loss = 0.001516
grad_step = 000228, loss = 0.001526
grad_step = 000229, loss = 0.001520
grad_step = 000230, loss = 0.001503
grad_step = 000231, loss = 0.001496
grad_step = 000232, loss = 0.001503
grad_step = 000233, loss = 0.001507
grad_step = 000234, loss = 0.001500
grad_step = 000235, loss = 0.001491
grad_step = 000236, loss = 0.001490
grad_step = 000237, loss = 0.001493
grad_step = 000238, loss = 0.001490
grad_step = 000239, loss = 0.001483
grad_step = 000240, loss = 0.001477
grad_step = 000241, loss = 0.001475
grad_step = 000242, loss = 0.001476
grad_step = 000243, loss = 0.001476
grad_step = 000244, loss = 0.001473
grad_step = 000245, loss = 0.001468
grad_step = 000246, loss = 0.001465
grad_step = 000247, loss = 0.001465
grad_step = 000248, loss = 0.001466
grad_step = 000249, loss = 0.001468
grad_step = 000250, loss = 0.001471
grad_step = 000251, loss = 0.001476
grad_step = 000252, loss = 0.001490
grad_step = 000253, loss = 0.001521
grad_step = 000254, loss = 0.001590
grad_step = 000255, loss = 0.001681
grad_step = 000256, loss = 0.001816
grad_step = 000257, loss = 0.001842
grad_step = 000258, loss = 0.001786
grad_step = 000259, loss = 0.001601
grad_step = 000260, loss = 0.001485
grad_step = 000261, loss = 0.001492
grad_step = 000262, loss = 0.001546
grad_step = 000263, loss = 0.001586
grad_step = 000264, loss = 0.001570
grad_step = 000265, loss = 0.001522
grad_step = 000266, loss = 0.001467
grad_step = 000267, loss = 0.001454
grad_step = 000268, loss = 0.001494
grad_step = 000269, loss = 0.001524
grad_step = 000270, loss = 0.001497
grad_step = 000271, loss = 0.001437
grad_step = 000272, loss = 0.001420
grad_step = 000273, loss = 0.001455
grad_step = 000274, loss = 0.001481
grad_step = 000275, loss = 0.001461
grad_step = 000276, loss = 0.001422
grad_step = 000277, loss = 0.001410
grad_step = 000278, loss = 0.001428
grad_step = 000279, loss = 0.001443
grad_step = 000280, loss = 0.001433
grad_step = 000281, loss = 0.001409
grad_step = 000282, loss = 0.001399
grad_step = 000283, loss = 0.001407
grad_step = 000284, loss = 0.001418
grad_step = 000285, loss = 0.001413
grad_step = 000286, loss = 0.001398
grad_step = 000287, loss = 0.001387
grad_step = 000288, loss = 0.001389
grad_step = 000289, loss = 0.001396
grad_step = 000290, loss = 0.001398
grad_step = 000291, loss = 0.001392
grad_step = 000292, loss = 0.001384
grad_step = 000293, loss = 0.001380
grad_step = 000294, loss = 0.001382
grad_step = 000295, loss = 0.001387
grad_step = 000296, loss = 0.001391
grad_step = 000297, loss = 0.001395
grad_step = 000298, loss = 0.001402
grad_step = 000299, loss = 0.001424
grad_step = 000300, loss = 0.001464
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001545
grad_step = 000302, loss = 0.001592
grad_step = 000303, loss = 0.001647
grad_step = 000304, loss = 0.001596
grad_step = 000305, loss = 0.001538
grad_step = 000306, loss = 0.001473
grad_step = 000307, loss = 0.001441
grad_step = 000308, loss = 0.001407
grad_step = 000309, loss = 0.001374
grad_step = 000310, loss = 0.001386
grad_step = 000311, loss = 0.001434
grad_step = 000312, loss = 0.001459
grad_step = 000313, loss = 0.001410
grad_step = 000314, loss = 0.001354
grad_step = 000315, loss = 0.001345
grad_step = 000316, loss = 0.001366
grad_step = 000317, loss = 0.001376
grad_step = 000318, loss = 0.001368
grad_step = 000319, loss = 0.001359
grad_step = 000320, loss = 0.001357
grad_step = 000321, loss = 0.001363
grad_step = 000322, loss = 0.001359
grad_step = 000323, loss = 0.001343
grad_step = 000324, loss = 0.001319
grad_step = 000325, loss = 0.001314
grad_step = 000326, loss = 0.001326
grad_step = 000327, loss = 0.001331
grad_step = 000328, loss = 0.001325
grad_step = 000329, loss = 0.001316
grad_step = 000330, loss = 0.001314
grad_step = 000331, loss = 0.001317
grad_step = 000332, loss = 0.001319
grad_step = 000333, loss = 0.001317
grad_step = 000334, loss = 0.001313
grad_step = 000335, loss = 0.001308
grad_step = 000336, loss = 0.001309
grad_step = 000337, loss = 0.001316
grad_step = 000338, loss = 0.001328
grad_step = 000339, loss = 0.001340
grad_step = 000340, loss = 0.001365
grad_step = 000341, loss = 0.001403
grad_step = 000342, loss = 0.001463
grad_step = 000343, loss = 0.001514
grad_step = 000344, loss = 0.001555
grad_step = 000345, loss = 0.001521
grad_step = 000346, loss = 0.001442
grad_step = 000347, loss = 0.001333
grad_step = 000348, loss = 0.001273
grad_step = 000349, loss = 0.001281
grad_step = 000350, loss = 0.001331
grad_step = 000351, loss = 0.001375
grad_step = 000352, loss = 0.001368
grad_step = 000353, loss = 0.001328
grad_step = 000354, loss = 0.001274
grad_step = 000355, loss = 0.001250
grad_step = 000356, loss = 0.001261
grad_step = 000357, loss = 0.001290
grad_step = 000358, loss = 0.001318
grad_step = 000359, loss = 0.001317
grad_step = 000360, loss = 0.001300
grad_step = 000361, loss = 0.001266
grad_step = 000362, loss = 0.001240
grad_step = 000363, loss = 0.001232
grad_step = 000364, loss = 0.001240
grad_step = 000365, loss = 0.001254
grad_step = 000366, loss = 0.001264
grad_step = 000367, loss = 0.001269
grad_step = 000368, loss = 0.001264
grad_step = 000369, loss = 0.001254
grad_step = 000370, loss = 0.001239
grad_step = 000371, loss = 0.001225
grad_step = 000372, loss = 0.001215
grad_step = 000373, loss = 0.001209
grad_step = 000374, loss = 0.001206
grad_step = 000375, loss = 0.001207
grad_step = 000376, loss = 0.001211
grad_step = 000377, loss = 0.001218
grad_step = 000378, loss = 0.001235
grad_step = 000379, loss = 0.001264
grad_step = 000380, loss = 0.001325
grad_step = 000381, loss = 0.001392
grad_step = 000382, loss = 0.001491
grad_step = 000383, loss = 0.001506
grad_step = 000384, loss = 0.001491
grad_step = 000385, loss = 0.001364
grad_step = 000386, loss = 0.001247
grad_step = 000387, loss = 0.001192
grad_step = 000388, loss = 0.001218
grad_step = 000389, loss = 0.001282
grad_step = 000390, loss = 0.001311
grad_step = 000391, loss = 0.001289
grad_step = 000392, loss = 0.001221
grad_step = 000393, loss = 0.001172
grad_step = 000394, loss = 0.001176
grad_step = 000395, loss = 0.001216
grad_step = 000396, loss = 0.001243
grad_step = 000397, loss = 0.001229
grad_step = 000398, loss = 0.001206
grad_step = 000399, loss = 0.001191
grad_step = 000400, loss = 0.001190
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001181
grad_step = 000402, loss = 0.001167
grad_step = 000403, loss = 0.001152
grad_step = 000404, loss = 0.001149
grad_step = 000405, loss = 0.001160
grad_step = 000406, loss = 0.001166
grad_step = 000407, loss = 0.001159
grad_step = 000408, loss = 0.001144
grad_step = 000409, loss = 0.001137
grad_step = 000410, loss = 0.001139
grad_step = 000411, loss = 0.001141
grad_step = 000412, loss = 0.001139
grad_step = 000413, loss = 0.001137
grad_step = 000414, loss = 0.001134
grad_step = 000415, loss = 0.001136
grad_step = 000416, loss = 0.001147
grad_step = 000417, loss = 0.001175
grad_step = 000418, loss = 0.001218
grad_step = 000419, loss = 0.001296
grad_step = 000420, loss = 0.001388
grad_step = 000421, loss = 0.001487
grad_step = 000422, loss = 0.001451
grad_step = 000423, loss = 0.001346
grad_step = 000424, loss = 0.001191
grad_step = 000425, loss = 0.001115
grad_step = 000426, loss = 0.001150
grad_step = 000427, loss = 0.001233
grad_step = 000428, loss = 0.001271
grad_step = 000429, loss = 0.001205
grad_step = 000430, loss = 0.001118
grad_step = 000431, loss = 0.001096
grad_step = 000432, loss = 0.001139
grad_step = 000433, loss = 0.001184
grad_step = 000434, loss = 0.001183
grad_step = 000435, loss = 0.001146
grad_step = 000436, loss = 0.001104
grad_step = 000437, loss = 0.001088
grad_step = 000438, loss = 0.001098
grad_step = 000439, loss = 0.001112
grad_step = 000440, loss = 0.001113
grad_step = 000441, loss = 0.001109
grad_step = 000442, loss = 0.001108
grad_step = 000443, loss = 0.001101
grad_step = 000444, loss = 0.001089
grad_step = 000445, loss = 0.001071
grad_step = 000446, loss = 0.001061
grad_step = 000447, loss = 0.001065
grad_step = 000448, loss = 0.001074
grad_step = 000449, loss = 0.001081
grad_step = 000450, loss = 0.001075
grad_step = 000451, loss = 0.001067
grad_step = 000452, loss = 0.001065
grad_step = 000453, loss = 0.001075
grad_step = 000454, loss = 0.001090
grad_step = 000455, loss = 0.001110
grad_step = 000456, loss = 0.001126
grad_step = 000457, loss = 0.001148
grad_step = 000458, loss = 0.001167
grad_step = 000459, loss = 0.001198
grad_step = 000460, loss = 0.001199
grad_step = 000461, loss = 0.001195
grad_step = 000462, loss = 0.001127
grad_step = 000463, loss = 0.001061
grad_step = 000464, loss = 0.001027
grad_step = 000465, loss = 0.001041
grad_step = 000466, loss = 0.001075
grad_step = 000467, loss = 0.001085
grad_step = 000468, loss = 0.001074
grad_step = 000469, loss = 0.001051
grad_step = 000470, loss = 0.001034
grad_step = 000471, loss = 0.001025
grad_step = 000472, loss = 0.001020
grad_step = 000473, loss = 0.001020
grad_step = 000474, loss = 0.001029
grad_step = 000475, loss = 0.001045
grad_step = 000476, loss = 0.001059
grad_step = 000477, loss = 0.001066
grad_step = 000478, loss = 0.001062
grad_step = 000479, loss = 0.001059
grad_step = 000480, loss = 0.001053
grad_step = 000481, loss = 0.001051
grad_step = 000482, loss = 0.001042
grad_step = 000483, loss = 0.001033
grad_step = 000484, loss = 0.001015
grad_step = 000485, loss = 0.000998
grad_step = 000486, loss = 0.000986
grad_step = 000487, loss = 0.000983
grad_step = 000488, loss = 0.000984
grad_step = 000489, loss = 0.000982
grad_step = 000490, loss = 0.000981
grad_step = 000491, loss = 0.000980
grad_step = 000492, loss = 0.000983
grad_step = 000493, loss = 0.000989
grad_step = 000494, loss = 0.001002
grad_step = 000495, loss = 0.001020
grad_step = 000496, loss = 0.001052
grad_step = 000497, loss = 0.001100
grad_step = 000498, loss = 0.001176
grad_step = 000499, loss = 0.001243
grad_step = 000500, loss = 0.001303
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001242
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

  date_run                              2020-05-14 11:13:09.096445
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.233879
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 11:13:09.103510
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.156985
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 11:13:09.111416
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.127274
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 11:13:09.117916
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.38545
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
0   2020-05-14 11:12:35.986868  ...    mean_absolute_error
1   2020-05-14 11:12:35.990431  ...     mean_squared_error
2   2020-05-14 11:12:35.993798  ...  median_absolute_error
3   2020-05-14 11:12:35.996820  ...               r2_score
4   2020-05-14 11:12:44.798882  ...    mean_absolute_error
5   2020-05-14 11:12:44.803158  ...     mean_squared_error
6   2020-05-14 11:12:44.807293  ...  median_absolute_error
7   2020-05-14 11:12:44.811168  ...               r2_score
8   2020-05-14 11:13:09.096445  ...    mean_absolute_error
9   2020-05-14 11:13:09.103510  ...     mean_squared_error
10  2020-05-14 11:13:09.111416  ...  median_absolute_error
11  2020-05-14 11:13:09.117916  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53d634bba8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▌      | 3579904/9912422 [00:00<00:00, 35643728.23it/s]9920512it [00:00, 36747214.94it/s]                             
0it [00:00, ?it/s]32768it [00:00, 651045.47it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 483394.17it/s]1654784it [00:00, 12214774.85it/s]                         
0it [00:00, ?it/s]8192it [00:00, 222404.64it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5388d04e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f53883360b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5388d04e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f538828b0f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5385ac64e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5385ab1c50> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5388d04e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5388249710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5385ac64e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5388336128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f656b2f71d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=4d563df4dca979546ff8d08070bdd1f1e728fb843c7292077f3e1880bed7423a
  Stored in directory: /tmp/pip-ephem-wheel-cache-5yrl8jvj/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f6561462048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1843200/17464789 [==>...........................] - ETA: 0s
 8962048/17464789 [==============>...............] - ETA: 0s
 9781248/17464789 [===============>..............] - ETA: 0s
12132352/17464789 [===================>..........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 11:14:36.053011: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 11:14:36.058287: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-14 11:14:36.058446: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55803766a5e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 11:14:36.058461: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 8.0346 - accuracy: 0.4760
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6436 - accuracy: 0.5015 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6922 - accuracy: 0.4983
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6820 - accuracy: 0.4990
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6513 - accuracy: 0.5010
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6845 - accuracy: 0.4988
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7088 - accuracy: 0.4972
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6803 - accuracy: 0.4991
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
11000/25000 [============>.................] - ETA: 4s - loss: 7.5927 - accuracy: 0.5048
12000/25000 [=============>................] - ETA: 4s - loss: 7.5951 - accuracy: 0.5047
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6301 - accuracy: 0.5024
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6316 - accuracy: 0.5023
15000/25000 [=================>............] - ETA: 3s - loss: 7.6503 - accuracy: 0.5011
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6580 - accuracy: 0.5006
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6576 - accuracy: 0.5006
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6470 - accuracy: 0.5013
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6392 - accuracy: 0.5018
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6505 - accuracy: 0.5010
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6498 - accuracy: 0.5011
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6548 - accuracy: 0.5008
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6540 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 10s 390us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 11:14:53.106909
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 11:14:53.106909  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:02<72:45:03, 3.29kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:02<51:09:33, 4.68kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:02<35:51:38, 6.68kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<25:05:48, 9.53kB/s].vector_cache/glove.6B.zip:   0%|          | 3.60M/862M [00:03<17:30:56, 13.6kB/s].vector_cache/glove.6B.zip:   1%|          | 7.62M/862M [00:03<12:12:19, 19.4kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:03<8:29:52, 27.8kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.1M/862M [00:03<5:55:02, 39.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.4M/862M [00:03<4:07:21, 56.6kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:03<2:52:12, 80.9kB/s].vector_cache/glove.6B.zip:   3%|▎         | 30.0M/862M [00:03<2:00:09, 115kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:03<1:23:40, 165kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.5M/862M [00:03<58:27, 235kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 43.0M/862M [00:03<40:47, 335kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.9M/862M [00:04<28:31, 476kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:04<19:57, 677kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:04<15:59, 844kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:06<13:03, 1.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:06<11:20, 1.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:07<08:29, 1.58MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:08<08:15, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:08<07:23, 1.80MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.6M/862M [00:09<05:33, 2.40MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:10<06:40, 1.99MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:10<07:25, 1.79MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.1M/862M [00:11<05:53, 2.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<04:14, 3.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:12<12:44:43, 17.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:12<8:56:26, 24.6kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:13<6:15:06, 35.1kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:14<4:24:56, 49.6kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:14<3:06:44, 70.4kB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.3M/862M [00:14<2:10:47, 100kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:16<1:34:24, 139kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:16<1:07:08, 195kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:16<47:15, 276kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<33:08, 393kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:18<1:12:19, 180kB/s].vector_cache/glove.6B.zip:  10%|▉         | 81.9M/862M [00:18<51:56, 250kB/s]  .vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:18<36:36, 354kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:20<28:37, 452kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:20<22:42, 570kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:20<16:26, 786kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:20<11:41, 1.10MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:22<13:04, 984kB/s] .vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:22<10:28, 1.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:22<07:36, 1.69MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:24<08:18, 1.54MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:24<07:08, 1.79MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.9M/862M [00:24<05:19, 2.40MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:26<06:43, 1.89MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:26<06:00, 2.12MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:26<04:31, 2.81MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<06:09, 2.05MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:28<05:37, 2.25MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:28<04:11, 3.01MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<05:55, 2.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:30<05:26, 2.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:30<04:04, 3.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<05:50, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:32<05:23, 2.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:32<04:04, 3.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<03:34, 3.48MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<8:58:27, 23.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:34<6:17:20, 33.0kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<4:23:22, 47.1kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<3:12:03, 64.6kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:36<2:15:29, 91.5kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:36<1:38:42, 126kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:36<1:09:17, 178kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<51:02, 242kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<36:47, 335kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:38<26:04, 472kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<18:20, 669kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<1:02:08, 197kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<44:44, 274kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:40<31:33, 387kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<24:54, 489kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<18:28, 660kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:42<13:13, 920kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:22, 1.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<52:18, 232kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<37:50, 320kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:44<26:44, 452kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<21:31, 560kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<16:18, 739kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:46<11:41, 1.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<10:59, 1.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<08:55, 1.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:48<06:32, 1.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<07:22, 1.61MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<06:23, 1.87MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:50<04:42, 2.52MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<06:04, 1.95MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<05:29, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:52<04:05, 2.89MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<05:39, 2.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<05:10, 2.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:54<03:52, 3.03MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:55<05:29, 2.13MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<05:04, 2.31MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:56<03:50, 3.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<05:25, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<04:59, 2.33MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:58<03:47, 3.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<05:22, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<04:57, 2.34MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<03:45, 3.07MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<05:21, 2.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<04:42, 2.44MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<03:33, 3.22MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<02:38, 4.33MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<44:56, 254kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<33:50, 338kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<24:10, 473kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<17:01, 669kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<17:28, 650kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<13:24, 848kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<09:39, 1.17MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<09:23, 1.20MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<08:54, 1.27MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:07<06:49, 1.65MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<06:34, 1.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<05:46, 1.95MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:09<04:16, 2.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:11<05:36, 1.99MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<05:05, 2.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:11<03:50, 2.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<05:18, 2.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<04:39, 2.38MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:13<03:39, 3.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<02:42, 4.07MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<09:18, 1.18MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<07:39, 1.44MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<05:37, 1.95MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:17<06:31, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:17<05:41, 1.92MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:17<04:15, 2.56MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<05:32, 1.96MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<04:59, 2.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:19<03:43, 2.91MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<03:18, 3.28MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<7:44:17, 23.3kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<5:25:18, 33.2kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<3:46:53, 47.5kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<2:45:55, 64.8kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<1:58:25, 90.8kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:23<1:23:22, 129kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<58:15, 184kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<48:36, 220kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<35:10, 304kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:25<24:48, 429kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<19:43, 538kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<14:57, 709kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<10:41, 991kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<07:35, 1.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:29<5:16:37, 33.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<3:42:36, 47.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:29<2:35:41, 67.6kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<1:51:03, 94.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<1:18:44, 133kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:31<55:15, 189kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:32<41:00, 254kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<29:45, 350kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:33<21:02, 493kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<17:07, 604kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<13:03, 792kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<09:22, 1.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:36<08:58, 1.15MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<07:19, 1.40MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:37<05:22, 1.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<06:10, 1.65MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<05:21, 1.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:39<03:58, 2.56MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<05:12, 1.95MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<04:41, 2.16MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:41<03:29, 2.90MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<04:50, 2.08MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<04:14, 2.37MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<03:12, 3.14MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<02:22, 4.22MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<39:22, 254kB/s] .vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<28:34, 350kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<20:12, 493kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:46<16:27, 604kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<12:30, 793kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<08:59, 1.10MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:48<08:35, 1.15MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<07:01, 1.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<05:09, 1.91MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:50<05:54, 1.66MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<05:08, 1.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<03:50, 2.54MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<04:58, 1.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<04:29, 2.17MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<03:22, 2.87MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<04:38, 2.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<04:14, 2.27MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<03:12, 2.99MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<04:30, 2.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<04:07, 2.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:56<03:07, 3.05MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:58<04:25, 2.15MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:58<03:54, 2.43MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<02:58, 3.20MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<02:11, 4.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:00<37:12, 254kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:00<26:59, 350kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<19:03, 494kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:02<15:29, 606kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:02<12:46, 734kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<09:24, 995kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<06:39, 1.40MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:04<9:02:41, 17.2kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<6:20:36, 24.5kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<4:25:53, 34.9kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<3:06:04, 49.7kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:06<8:48:14, 17.5kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<6:09:50, 25.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<4:17:43, 35.7kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:08<3:06:56, 49.1kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:08<2:12:42, 69.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:08<1:33:13, 98.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:10<1:06:22, 137kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:10<47:22, 192kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<33:18, 273kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:12<25:30, 355kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:12<19:42, 459kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:12<14:11, 636kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<10:00, 899kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<13:36, 660kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<10:26, 859kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<07:29, 1.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:16<07:19, 1.22MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:16<06:01, 1.48MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<04:24, 2.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:18<05:10, 1.71MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:18<04:22, 2.02MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:18<03:23, 2.60MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<02:27, 3.57MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:20<29:38, 296kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:20<21:38, 405kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:20<15:19, 570kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:22<12:45, 683kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:22<09:47, 888kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<07:03, 1.23MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<06:57, 1.24MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:24<05:45, 1.50MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<04:14, 2.03MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:25<04:58, 1.72MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:26<04:21, 1.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:26<03:15, 2.61MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:27<04:17, 1.98MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<03:52, 2.19MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:28<02:55, 2.90MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:29<04:01, 2.09MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<03:40, 2.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:30<02:47, 3.02MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<03:55, 2.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<03:36, 2.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:32<02:41, 3.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<03:50, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<04:24, 1.88MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<03:30, 2.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:35<03:46, 2.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:35<03:30, 2.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:36<02:37, 3.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:37<03:43, 2.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:37<04:17, 1.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<03:25, 2.38MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<02:28, 3.28MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:39<7:47:59, 17.3kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:39<5:28:09, 24.6kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<3:49:10, 35.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:41<2:41:35, 49.6kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:41<1:54:42, 69.9kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<1:20:31, 99.4kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<56:14, 142kB/s]   .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:43<42:40, 186kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<30:41, 259kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<21:37, 366kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:45<16:54, 466kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<12:38, 623kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<09:01, 869kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<08:08, 960kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<06:29, 1.20MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<04:41, 1.66MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:49<05:05, 1.52MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:49<04:21, 1.77MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<03:12, 2.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<04:04, 1.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<04:25, 1.73MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<03:29, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<02:30, 3.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:53<7:21:22, 17.2kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:53<5:09:28, 24.6kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<3:36:04, 35.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:55<2:32:18, 49.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:55<1:48:08, 69.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<1:15:56, 99.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<52:53, 141kB/s]   .vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:57<7:44:37, 16.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:57<5:25:44, 22.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<3:47:24, 32.7kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<2:38:50, 46.6kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:59<7:40:37, 16.1kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:59<5:22:14, 22.9kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<3:44:03, 32.7kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:01<3:00:41, 40.6kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:01<2:07:57, 57.3kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:01<1:29:46, 81.5kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:03<1:03:34, 114kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:03<45:13, 160kB/s]  .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<31:42, 228kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:05<23:45, 303kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:05<17:21, 414kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<12:17, 583kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<10:14, 695kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<07:53, 902kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<05:41, 1.25MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<05:37, 1.25MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<04:39, 1.51MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<03:26, 2.04MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:11<04:02, 1.73MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:11<03:32, 1.97MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<02:39, 2.62MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:13<03:29, 1.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:13<03:48, 1.81MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:13<02:58, 2.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<02:08, 3.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<08:00, 855kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:15<06:19, 1.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:15<04:33, 1.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<04:46, 1.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:17<04:43, 1.43MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:17<03:36, 1.88MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<02:35, 2.59MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:18<05:44, 1.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<04:42, 1.42MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:19<03:27, 1.93MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<02:40, 2.48MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:21<6:53:04, 16.1kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:21<4:50:15, 22.8kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:21<3:23:02, 32.6kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<2:22:08, 46.2kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:23<1:40:09, 65.5kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:23<1:10:02, 93.3kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:25<50:08, 130kB/s]   .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:25<36:24, 178kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:25<25:44, 252kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<17:57, 358kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:27<21:46, 295kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:27<15:53, 404kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<11:14, 568kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<09:19, 682kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<07:03, 900kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<05:03, 1.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:36, 1.74MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:31<26:15, 239kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:31<19:00, 330kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:31<13:24, 466kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:33<10:48, 576kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:33<08:11, 758kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<05:52, 1.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:35<05:32, 1.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:35<04:30, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<03:16, 1.87MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:37<03:43, 1.63MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<03:51, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<03:00, 2.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:09, 2.79MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<5:48:25, 17.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<4:04:15, 24.6kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:39<2:50:23, 35.1kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<1:59:57, 49.6kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<1:24:29, 70.3kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<59:01, 100kB/s]   .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<42:27, 138kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:43<30:17, 194kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<21:15, 275kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<16:09, 359kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<11:53, 487kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<08:26, 684kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<07:13, 794kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<05:38, 1.02MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:47<04:04, 1.40MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<04:10, 1.36MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<03:30, 1.61MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:49<02:35, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:50<03:07, 1.79MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<02:45, 2.03MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:51<02:03, 2.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:52<02:44, 2.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<02:29, 2.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<01:52, 2.93MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:54<02:35, 2.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<02:23, 2.29MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<01:48, 3.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<02:31, 2.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:56<02:18, 2.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<01:45, 3.06MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<02:23, 2.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<02:41, 1.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<02:06, 2.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<01:31, 3.44MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:00<04:10, 1.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:00<03:30, 1.50MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<02:34, 2.03MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:02<02:56, 1.76MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<02:35, 1.99MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<01:55, 2.67MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<02:33, 2.00MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:04<02:18, 2.21MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<01:44, 2.92MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<02:24, 2.10MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<02:11, 2.29MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<01:39, 3.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:08<02:19, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:08<02:08, 2.32MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<01:35, 3.09MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:10<02:16, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:10<02:05, 2.34MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<01:35, 3.07MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<02:14, 2.16MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<02:03, 2.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<01:33, 3.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:14<02:12, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:14<02:02, 2.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<01:32, 3.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:16<02:10, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:16<02:21, 2.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<01:54, 2.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<01:26, 3.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:18<02:04, 2.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<01:56, 2.38MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<01:28, 3.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<02:05, 2.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:20<01:55, 2.36MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<01:26, 3.13MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<02:04, 2.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:22<01:54, 2.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<01:26, 3.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<01:14, 3.59MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<3:30:41, 21.0kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<2:27:17, 30.0kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<1:41:57, 42.8kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<1:19:47, 54.6kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<56:43, 76.8kB/s]  .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<39:48, 109kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<28:13, 152kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<20:11, 212kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<14:08, 301kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:30<10:47, 392kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:30<07:58, 529kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<05:39, 741kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:32<04:54, 847kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:32<03:51, 1.08MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<02:47, 1.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<02:54, 1.41MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<02:26, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<01:48, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:36<02:12, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:36<02:21, 1.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:36<01:51, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<01:19, 2.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:38<05:43, 690kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:38<04:20, 907kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:38<03:11, 1.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:15, 1.72MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:39<05:45, 673kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:40<04:25, 875kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:40<03:10, 1.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:41<03:06, 1.23MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:42<02:30, 1.52MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:42<01:49, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<01:18, 2.85MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<15:01, 249kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:44<10:53, 343kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<07:39, 484kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:45<06:10, 595kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:45<04:41, 781kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:46<03:21, 1.08MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<03:10, 1.13MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<02:57, 1.22MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:48<02:13, 1.61MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<01:35, 2.24MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:49<03:30, 1.01MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:49<02:49, 1.25MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:50<02:02, 1.72MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:51<02:13, 1.56MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:51<01:54, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:24, 2.45MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:53<01:46, 1.92MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:53<01:56, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<01:30, 2.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<01:05, 3.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:55<02:10, 1.53MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<01:51, 1.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<01:22, 2.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:57<01:43, 1.90MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:57<01:32, 2.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:57<01:08, 2.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:59<01:33, 2.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:59<01:24, 2.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<01:03, 2.98MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:01<01:28, 2.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:01<01:20, 2.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:01<01:00, 3.05MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:03<01:25, 2.14MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:03<01:18, 2.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<00:59, 3.07MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:05<01:23, 2.15MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:05<01:16, 2.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<00:56, 3.12MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<01:20, 2.17MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<01:32, 1.90MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<01:12, 2.41MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<00:51, 3.31MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<02:32, 1.12MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<02:04, 1.37MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<01:30, 1.87MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<01:41, 1.64MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:11<01:28, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<01:05, 2.52MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:13<01:23, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:13<01:15, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<00:56, 2.85MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<00:48, 3.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:15<2:03:53, 21.4kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:15<1:26:25, 30.5kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<59:19, 43.5kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:17<48:11, 53.5kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:17<34:13, 75.2kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<23:57, 107kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<16:50, 149kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<12:01, 208kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<08:23, 295kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:21<06:21, 384kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:21<04:41, 519kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<03:18, 727kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:23<02:50, 835kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:23<02:13, 1.06MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:23<01:36, 1.46MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:25<01:38, 1.40MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:25<01:23, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<01:00, 2.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:27<01:13, 1.82MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:27<01:05, 2.06MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<00:47, 2.76MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:29<01:03, 2.05MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:29<00:57, 2.25MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<00:42, 3.01MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:30<00:59, 2.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:31<01:07, 1.86MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:31<00:53, 2.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:37, 3.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:32<1:57:16, 17.3kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:33<1:22:00, 24.7kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:33<56:48, 35.2kB/s]  .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<39:03, 50.3kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<35:11, 55.7kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:35<24:45, 78.9kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<17:08, 112kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:36<12:13, 155kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:36<08:43, 216kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<06:03, 306kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:38<04:35, 397kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:38<03:23, 535kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<02:22, 752kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:40<02:02, 856kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:40<01:47, 975kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:41<01:19, 1.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:55, 1.83MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:42<01:35, 1.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:42<01:17, 1.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<00:55, 1.79MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:44<01:00, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:44<01:02, 1.55MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<00:48, 1.99MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:33, 2.75MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:46<1:29:53, 17.2kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<1:02:49, 24.6kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<43:16, 35.0kB/s]  .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<29:55, 49.5kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<21:12, 69.7kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<14:46, 99.1kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<10:06, 141kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<07:40, 184kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:50<05:29, 256kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:50<03:48, 362kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<02:54, 462kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<02:10, 617kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:31, 862kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:54<01:20, 954kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:54<01:03, 1.19MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:45, 1.64MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:56<00:47, 1.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:56<00:40, 1.76MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:29, 2.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:58<00:36, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:58<00:31, 2.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:22, 2.93MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:16, 3.94MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<04:29, 238kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<03:13, 329kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<02:13, 465kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:02<01:44, 574kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:02<01:20, 742kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<00:58, 1.01MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:40, 1.40MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:03<00:32, 1.72MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:04<42:11, 22.1kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:04<29:10, 31.5kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<19:17, 45.0kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<15:07, 57.2kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<10:43, 80.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:06<07:25, 114kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<05:00, 159kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:08<03:34, 221kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<02:26, 314kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:10<01:47, 406kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:10<01:19, 547kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:54, 768kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<00:45, 872kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<00:35, 1.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:24, 1.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:14<00:24, 1.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:14<00:20, 1.69MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:14, 2.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:16, 1.85MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:14, 2.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:10, 2.76MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:17<00:13, 2.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:11, 2.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:18<00:08, 2.99MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<00:10, 2.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:20<00:09, 2.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:20<00:06, 3.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:21<00:08, 2.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:22<00:07, 2.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:22<00:05, 3.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:23<00:06, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:24<00:06, 2.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:24<00:04, 3.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:25<00:04, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:04, 2.35MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:02, 3.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:03, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:02, 2.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:28<00:01, 3.08MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:01, 2.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:00, 2.35MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:30<00:00, 3.09MB/s].vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 736/400000 [00:00<00:54, 7357.59it/s]  0%|          | 1470/400000 [00:00<00:54, 7352.27it/s]  1%|          | 2262/400000 [00:00<00:52, 7512.82it/s]  1%|          | 3007/400000 [00:00<00:52, 7491.76it/s]  1%|          | 3792/400000 [00:00<00:52, 7594.23it/s]  1%|          | 4566/400000 [00:00<00:51, 7636.21it/s]  1%|▏         | 5276/400000 [00:00<00:52, 7466.60it/s]  2%|▏         | 6050/400000 [00:00<00:52, 7546.30it/s]  2%|▏         | 6809/400000 [00:00<00:52, 7557.42it/s]  2%|▏         | 7558/400000 [00:01<00:52, 7536.78it/s]  2%|▏         | 8316/400000 [00:01<00:51, 7547.33it/s]  2%|▏         | 9057/400000 [00:01<00:52, 7438.56it/s]  2%|▏         | 9792/400000 [00:01<00:52, 7371.28it/s]  3%|▎         | 10523/400000 [00:01<00:54, 7187.18it/s]  3%|▎         | 11258/400000 [00:01<00:53, 7234.82it/s]  3%|▎         | 12002/400000 [00:01<00:53, 7292.86it/s]  3%|▎         | 12748/400000 [00:01<00:52, 7339.30it/s]  3%|▎         | 13493/400000 [00:01<00:52, 7371.30it/s]  4%|▎         | 14230/400000 [00:01<00:52, 7365.75it/s]  4%|▎         | 14981/400000 [00:02<00:51, 7408.31it/s]  4%|▍         | 15759/400000 [00:02<00:51, 7514.19it/s]  4%|▍         | 16511/400000 [00:02<00:51, 7510.69it/s]  4%|▍         | 17272/400000 [00:02<00:50, 7539.20it/s]  5%|▍         | 18041/400000 [00:02<00:50, 7581.82it/s]  5%|▍         | 18803/400000 [00:02<00:50, 7590.50it/s]  5%|▍         | 19582/400000 [00:02<00:49, 7649.20it/s]  5%|▌         | 20348/400000 [00:02<00:50, 7522.34it/s]  5%|▌         | 21101/400000 [00:02<00:50, 7517.03it/s]  5%|▌         | 21871/400000 [00:02<00:49, 7566.27it/s]  6%|▌         | 22628/400000 [00:03<00:51, 7331.05it/s]  6%|▌         | 23363/400000 [00:03<00:52, 7218.40it/s]  6%|▌         | 24125/400000 [00:03<00:51, 7331.76it/s]  6%|▌         | 24860/400000 [00:03<00:51, 7284.38it/s]  6%|▋         | 25604/400000 [00:03<00:51, 7328.04it/s]  7%|▋         | 26382/400000 [00:03<00:50, 7457.55it/s]  7%|▋         | 27130/400000 [00:03<00:49, 7462.97it/s]  7%|▋         | 27902/400000 [00:03<00:49, 7535.18it/s]  7%|▋         | 28657/400000 [00:03<00:50, 7407.58it/s]  7%|▋         | 29399/400000 [00:03<00:50, 7399.69it/s]  8%|▊         | 30196/400000 [00:04<00:48, 7560.65it/s]  8%|▊         | 31015/400000 [00:04<00:47, 7735.33it/s]  8%|▊         | 31871/400000 [00:04<00:46, 7965.47it/s]  8%|▊         | 32671/400000 [00:04<00:46, 7946.14it/s]  8%|▊         | 33468/400000 [00:04<00:47, 7728.51it/s]  9%|▊         | 34244/400000 [00:04<00:48, 7513.94it/s]  9%|▉         | 35014/400000 [00:04<00:48, 7568.10it/s]  9%|▉         | 35831/400000 [00:04<00:47, 7737.50it/s]  9%|▉         | 36613/400000 [00:04<00:46, 7762.03it/s]  9%|▉         | 37392/400000 [00:04<00:47, 7616.10it/s] 10%|▉         | 38156/400000 [00:05<00:49, 7288.89it/s] 10%|▉         | 38890/400000 [00:05<00:49, 7303.90it/s] 10%|▉         | 39658/400000 [00:05<00:48, 7410.68it/s] 10%|█         | 40411/400000 [00:05<00:48, 7444.54it/s] 10%|█         | 41158/400000 [00:05<00:48, 7386.44it/s] 10%|█         | 41898/400000 [00:05<00:49, 7183.35it/s] 11%|█         | 42619/400000 [00:05<00:50, 7133.33it/s] 11%|█         | 43336/400000 [00:05<00:49, 7142.23it/s] 11%|█         | 44107/400000 [00:05<00:48, 7302.18it/s] 11%|█         | 44901/400000 [00:06<00:47, 7480.66it/s] 11%|█▏        | 45716/400000 [00:06<00:46, 7667.55it/s] 12%|█▏        | 46522/400000 [00:06<00:45, 7779.89it/s] 12%|█▏        | 47303/400000 [00:06<00:46, 7523.03it/s] 12%|█▏        | 48059/400000 [00:06<00:47, 7480.13it/s] 12%|█▏        | 48829/400000 [00:06<00:46, 7543.80it/s] 12%|█▏        | 49603/400000 [00:06<00:46, 7600.87it/s] 13%|█▎        | 50414/400000 [00:06<00:45, 7744.28it/s] 13%|█▎        | 51191/400000 [00:06<00:45, 7635.47it/s] 13%|█▎        | 51957/400000 [00:06<00:45, 7573.49it/s] 13%|█▎        | 52719/400000 [00:07<00:45, 7587.04it/s] 13%|█▎        | 53542/400000 [00:07<00:44, 7767.41it/s] 14%|█▎        | 54321/400000 [00:07<00:44, 7772.38it/s] 14%|█▍        | 55203/400000 [00:07<00:42, 8057.78it/s] 14%|█▍        | 56013/400000 [00:07<00:44, 7803.48it/s] 14%|█▍        | 56798/400000 [00:07<00:44, 7680.23it/s] 14%|█▍        | 57607/400000 [00:07<00:43, 7798.64it/s] 15%|█▍        | 58448/400000 [00:07<00:42, 7971.07it/s] 15%|█▍        | 59249/400000 [00:07<00:42, 7963.50it/s] 15%|█▌        | 60048/400000 [00:07<00:44, 7708.47it/s] 15%|█▌        | 60823/400000 [00:08<00:46, 7322.86it/s] 15%|█▌        | 61570/400000 [00:08<00:45, 7366.21it/s] 16%|█▌        | 62312/400000 [00:08<00:47, 7169.69it/s] 16%|█▌        | 63101/400000 [00:08<00:45, 7369.13it/s] 16%|█▌        | 63922/400000 [00:08<00:44, 7601.38it/s] 16%|█▌        | 64691/400000 [00:08<00:43, 7625.78it/s] 16%|█▋        | 65458/400000 [00:08<00:44, 7601.25it/s] 17%|█▋        | 66262/400000 [00:08<00:43, 7725.57it/s] 17%|█▋        | 67077/400000 [00:08<00:42, 7848.10it/s] 17%|█▋        | 67864/400000 [00:08<00:42, 7845.51it/s] 17%|█▋        | 68652/400000 [00:09<00:42, 7854.62it/s] 17%|█▋        | 69487/400000 [00:09<00:41, 7995.13it/s] 18%|█▊        | 70288/400000 [00:09<00:41, 7990.04it/s] 18%|█▊        | 71096/400000 [00:09<00:41, 8013.57it/s] 18%|█▊        | 71899/400000 [00:09<00:41, 7966.54it/s] 18%|█▊        | 72697/400000 [00:09<00:42, 7707.77it/s] 18%|█▊        | 73470/400000 [00:09<00:42, 7648.04it/s] 19%|█▊        | 74320/400000 [00:09<00:41, 7884.33it/s] 19%|█▉        | 75175/400000 [00:09<00:40, 8070.51it/s] 19%|█▉        | 75986/400000 [00:10<00:40, 8032.91it/s] 19%|█▉        | 76792/400000 [00:10<00:41, 7852.14it/s] 19%|█▉        | 77580/400000 [00:10<00:41, 7804.88it/s] 20%|█▉        | 78363/400000 [00:10<00:41, 7808.60it/s] 20%|█▉        | 79146/400000 [00:10<00:41, 7762.59it/s] 20%|█▉        | 79930/400000 [00:10<00:41, 7784.75it/s] 20%|██        | 80710/400000 [00:10<00:41, 7692.73it/s] 20%|██        | 81492/400000 [00:10<00:41, 7730.00it/s] 21%|██        | 82270/400000 [00:10<00:41, 7744.18it/s] 21%|██        | 83089/400000 [00:10<00:40, 7871.65it/s] 21%|██        | 83889/400000 [00:11<00:39, 7908.07it/s] 21%|██        | 84681/400000 [00:11<00:40, 7881.37it/s] 21%|██▏       | 85504/400000 [00:11<00:39, 7981.01it/s] 22%|██▏       | 86303/400000 [00:11<00:39, 7912.58it/s] 22%|██▏       | 87095/400000 [00:11<00:39, 7847.33it/s] 22%|██▏       | 87912/400000 [00:11<00:39, 7940.02it/s] 22%|██▏       | 88707/400000 [00:11<00:40, 7749.95it/s] 22%|██▏       | 89514/400000 [00:11<00:39, 7842.79it/s] 23%|██▎       | 90300/400000 [00:11<00:40, 7639.51it/s] 23%|██▎       | 91080/400000 [00:11<00:40, 7685.04it/s] 23%|██▎       | 91851/400000 [00:12<00:40, 7657.51it/s] 23%|██▎       | 92618/400000 [00:12<00:40, 7522.73it/s] 23%|██▎       | 93372/400000 [00:12<00:41, 7318.56it/s] 24%|██▎       | 94156/400000 [00:12<00:40, 7466.27it/s] 24%|██▎       | 94979/400000 [00:12<00:39, 7678.49it/s] 24%|██▍       | 95838/400000 [00:12<00:38, 7929.26it/s] 24%|██▍       | 96636/400000 [00:12<00:38, 7912.39it/s] 24%|██▍       | 97479/400000 [00:12<00:37, 8058.62it/s] 25%|██▍       | 98288/400000 [00:12<00:37, 8051.79it/s] 25%|██▍       | 99098/400000 [00:12<00:37, 8064.98it/s] 25%|██▍       | 99906/400000 [00:13<00:37, 7936.08it/s] 25%|██▌       | 100702/400000 [00:13<00:38, 7745.31it/s] 25%|██▌       | 101499/400000 [00:13<00:38, 7808.86it/s] 26%|██▌       | 102282/400000 [00:13<00:38, 7779.21it/s] 26%|██▌       | 103062/400000 [00:13<00:39, 7556.45it/s] 26%|██▌       | 103832/400000 [00:13<00:38, 7597.56it/s] 26%|██▌       | 104594/400000 [00:13<00:39, 7517.53it/s] 26%|██▋       | 105349/400000 [00:13<00:39, 7523.52it/s] 27%|██▋       | 106178/400000 [00:13<00:37, 7736.35it/s] 27%|██▋       | 107037/400000 [00:13<00:36, 7972.35it/s] 27%|██▋       | 107857/400000 [00:14<00:36, 8038.75it/s] 27%|██▋       | 108664/400000 [00:14<00:36, 7922.27it/s] 27%|██▋       | 109496/400000 [00:14<00:36, 8033.93it/s] 28%|██▊       | 110376/400000 [00:14<00:35, 8248.29it/s] 28%|██▊       | 111204/400000 [00:14<00:35, 8171.68it/s] 28%|██▊       | 112024/400000 [00:14<00:35, 8013.34it/s] 28%|██▊       | 112828/400000 [00:14<00:36, 7777.29it/s] 28%|██▊       | 113609/400000 [00:14<00:37, 7604.01it/s] 29%|██▊       | 114409/400000 [00:14<00:37, 7718.29it/s] 29%|██▉       | 115260/400000 [00:15<00:35, 7939.80it/s] 29%|██▉       | 116081/400000 [00:15<00:35, 8016.05it/s] 29%|██▉       | 116886/400000 [00:15<00:35, 7895.22it/s] 29%|██▉       | 117713/400000 [00:15<00:35, 8002.06it/s] 30%|██▉       | 118518/400000 [00:15<00:35, 8013.71it/s] 30%|██▉       | 119321/400000 [00:15<00:35, 7946.89it/s] 30%|███       | 120133/400000 [00:15<00:35, 7995.94it/s] 30%|███       | 120941/400000 [00:15<00:34, 8020.81it/s] 30%|███       | 121744/400000 [00:15<00:34, 7995.90it/s] 31%|███       | 122586/400000 [00:15<00:34, 8117.55it/s] 31%|███       | 123437/400000 [00:16<00:33, 8229.86it/s] 31%|███       | 124261/400000 [00:16<00:33, 8196.27it/s] 31%|███▏      | 125110/400000 [00:16<00:33, 8282.07it/s] 31%|███▏      | 125939/400000 [00:16<00:33, 8158.65it/s] 32%|███▏      | 126756/400000 [00:16<00:34, 8013.99it/s] 32%|███▏      | 127564/400000 [00:16<00:33, 8033.12it/s] 32%|███▏      | 128390/400000 [00:16<00:33, 8097.21it/s] 32%|███▏      | 129201/400000 [00:16<00:33, 8029.23it/s] 33%|███▎      | 130005/400000 [00:16<00:34, 7852.55it/s] 33%|███▎      | 130792/400000 [00:16<00:34, 7769.95it/s] 33%|███▎      | 131571/400000 [00:17<00:35, 7642.29it/s] 33%|███▎      | 132337/400000 [00:17<00:35, 7594.51it/s] 33%|███▎      | 133098/400000 [00:17<00:35, 7594.91it/s] 33%|███▎      | 133868/400000 [00:17<00:34, 7624.42it/s] 34%|███▎      | 134703/400000 [00:17<00:33, 7825.85it/s] 34%|███▍      | 135502/400000 [00:17<00:33, 7873.45it/s] 34%|███▍      | 136315/400000 [00:17<00:33, 7947.88it/s] 34%|███▍      | 137160/400000 [00:17<00:32, 8090.90it/s] 34%|███▍      | 137974/400000 [00:17<00:32, 8103.03it/s] 35%|███▍      | 138801/400000 [00:17<00:32, 8150.32it/s] 35%|███▍      | 139617/400000 [00:18<00:32, 7920.05it/s] 35%|███▌      | 140411/400000 [00:18<00:33, 7811.11it/s] 35%|███▌      | 141215/400000 [00:18<00:32, 7876.19it/s] 36%|███▌      | 142039/400000 [00:18<00:32, 7980.36it/s] 36%|███▌      | 142840/400000 [00:18<00:32, 7986.74it/s] 36%|███▌      | 143640/400000 [00:18<00:32, 7884.52it/s] 36%|███▌      | 144430/400000 [00:18<00:32, 7859.45it/s] 36%|███▋      | 145236/400000 [00:18<00:32, 7918.21it/s] 37%|███▋      | 146029/400000 [00:18<00:32, 7859.92it/s] 37%|███▋      | 146834/400000 [00:19<00:31, 7914.68it/s] 37%|███▋      | 147686/400000 [00:19<00:31, 8085.43it/s] 37%|███▋      | 148496/400000 [00:19<00:31, 7964.56it/s] 37%|███▋      | 149298/400000 [00:19<00:31, 7980.08it/s] 38%|███▊      | 150154/400000 [00:19<00:30, 8144.56it/s] 38%|███▊      | 150970/400000 [00:19<00:30, 8142.31it/s] 38%|███▊      | 151786/400000 [00:19<00:30, 8072.97it/s] 38%|███▊      | 152595/400000 [00:19<00:31, 7892.13it/s] 38%|███▊      | 153394/400000 [00:19<00:31, 7919.99it/s] 39%|███▊      | 154188/400000 [00:19<00:31, 7881.27it/s] 39%|███▊      | 154977/400000 [00:20<00:31, 7827.35it/s] 39%|███▉      | 155762/400000 [00:20<00:31, 7833.68it/s] 39%|███▉      | 156559/400000 [00:20<00:30, 7872.36it/s] 39%|███▉      | 157361/400000 [00:20<00:30, 7915.88it/s] 40%|███▉      | 158179/400000 [00:20<00:30, 7991.60it/s] 40%|███▉      | 159009/400000 [00:20<00:29, 8080.12it/s] 40%|███▉      | 159818/400000 [00:20<00:30, 7922.06it/s] 40%|████      | 160614/400000 [00:20<00:30, 7932.42it/s] 40%|████      | 161408/400000 [00:20<00:30, 7832.76it/s] 41%|████      | 162277/400000 [00:20<00:29, 8070.54it/s] 41%|████      | 163087/400000 [00:21<00:29, 8022.71it/s] 41%|████      | 163901/400000 [00:21<00:29, 8056.65it/s] 41%|████      | 164708/400000 [00:21<00:29, 7902.78it/s] 41%|████▏     | 165500/400000 [00:21<00:30, 7643.98it/s] 42%|████▏     | 166268/400000 [00:21<00:30, 7637.66it/s] 42%|████▏     | 167061/400000 [00:21<00:30, 7720.42it/s] 42%|████▏     | 167848/400000 [00:21<00:29, 7761.47it/s] 42%|████▏     | 168626/400000 [00:21<00:29, 7722.29it/s] 42%|████▏     | 169448/400000 [00:21<00:29, 7864.27it/s] 43%|████▎     | 170281/400000 [00:21<00:28, 7996.89it/s] 43%|████▎     | 171128/400000 [00:22<00:28, 8131.20it/s] 43%|████▎     | 171943/400000 [00:22<00:28, 8102.11it/s] 43%|████▎     | 172755/400000 [00:22<00:28, 8092.36it/s] 43%|████▎     | 173568/400000 [00:22<00:27, 8102.40it/s] 44%|████▎     | 174379/400000 [00:22<00:27, 8059.80it/s] 44%|████▍     | 175186/400000 [00:22<00:28, 8000.40it/s] 44%|████▍     | 175987/400000 [00:22<00:28, 7917.07it/s] 44%|████▍     | 176780/400000 [00:22<00:28, 7859.26it/s] 44%|████▍     | 177567/400000 [00:22<00:28, 7752.27it/s] 45%|████▍     | 178354/400000 [00:22<00:28, 7786.19it/s] 45%|████▍     | 179134/400000 [00:23<00:28, 7625.77it/s] 45%|████▍     | 179898/400000 [00:23<00:29, 7548.16it/s] 45%|████▌     | 180676/400000 [00:23<00:28, 7615.18it/s] 45%|████▌     | 181513/400000 [00:23<00:27, 7824.38it/s] 46%|████▌     | 182334/400000 [00:23<00:27, 7934.43it/s] 46%|████▌     | 183130/400000 [00:23<00:27, 7932.61it/s] 46%|████▌     | 183925/400000 [00:23<00:27, 7746.69it/s] 46%|████▌     | 184702/400000 [00:23<00:27, 7707.36it/s] 46%|████▋     | 185476/400000 [00:23<00:27, 7716.07it/s] 47%|████▋     | 186295/400000 [00:23<00:27, 7852.31it/s] 47%|████▋     | 187086/400000 [00:24<00:27, 7869.18it/s] 47%|████▋     | 187891/400000 [00:24<00:26, 7922.03it/s] 47%|████▋     | 188684/400000 [00:24<00:27, 7787.77it/s] 47%|████▋     | 189468/400000 [00:24<00:26, 7800.90it/s] 48%|████▊     | 190249/400000 [00:24<00:27, 7739.67it/s] 48%|████▊     | 191079/400000 [00:24<00:26, 7897.45it/s] 48%|████▊     | 191870/400000 [00:24<00:26, 7884.26it/s] 48%|████▊     | 192660/400000 [00:24<00:26, 7712.35it/s] 48%|████▊     | 193447/400000 [00:24<00:26, 7756.75it/s] 49%|████▊     | 194233/400000 [00:25<00:26, 7786.74it/s] 49%|████▉     | 195031/400000 [00:25<00:26, 7843.39it/s] 49%|████▉     | 195860/400000 [00:25<00:25, 7970.06it/s] 49%|████▉     | 196658/400000 [00:25<00:25, 7960.58it/s] 49%|████▉     | 197458/400000 [00:25<00:25, 7970.88it/s] 50%|████▉     | 198274/400000 [00:25<00:25, 8024.68it/s] 50%|████▉     | 199083/400000 [00:25<00:24, 8042.53it/s] 50%|████▉     | 199891/400000 [00:25<00:24, 8053.40it/s] 50%|█████     | 200697/400000 [00:25<00:24, 7987.02it/s] 50%|█████     | 201503/400000 [00:25<00:24, 8006.84it/s] 51%|█████     | 202313/400000 [00:26<00:24, 8034.00it/s] 51%|█████     | 203125/400000 [00:26<00:24, 8055.36it/s] 51%|█████     | 203944/400000 [00:26<00:24, 8094.15it/s] 51%|█████     | 204754/400000 [00:26<00:24, 8048.63it/s] 51%|█████▏    | 205560/400000 [00:26<00:24, 7887.68it/s] 52%|█████▏    | 206350/400000 [00:26<00:24, 7758.23it/s] 52%|█████▏    | 207145/400000 [00:26<00:24, 7811.28it/s] 52%|█████▏    | 207932/400000 [00:26<00:24, 7828.49it/s] 52%|█████▏    | 208735/400000 [00:26<00:24, 7886.53it/s] 52%|█████▏    | 209525/400000 [00:26<00:24, 7877.76it/s] 53%|█████▎    | 210338/400000 [00:27<00:23, 7949.81it/s] 53%|█████▎    | 211136/400000 [00:27<00:23, 7956.73it/s] 53%|█████▎    | 211978/400000 [00:27<00:23, 8089.11it/s] 53%|█████▎    | 212788/400000 [00:27<00:23, 8037.78it/s] 53%|█████▎    | 213610/400000 [00:27<00:23, 8088.94it/s] 54%|█████▎    | 214420/400000 [00:27<00:23, 8045.94it/s] 54%|█████▍    | 215225/400000 [00:27<00:22, 8035.00it/s] 54%|█████▍    | 216030/400000 [00:27<00:22, 8036.35it/s] 54%|█████▍    | 216834/400000 [00:27<00:22, 7967.53it/s] 54%|█████▍    | 217673/400000 [00:27<00:22, 8087.91it/s] 55%|█████▍    | 218483/400000 [00:28<00:22, 8002.13it/s] 55%|█████▍    | 219284/400000 [00:28<00:22, 7950.47it/s] 55%|█████▌    | 220080/400000 [00:28<00:22, 7914.15it/s] 55%|█████▌    | 220876/400000 [00:28<00:22, 7926.70it/s] 55%|█████▌    | 221698/400000 [00:28<00:22, 8011.33it/s] 56%|█████▌    | 222505/400000 [00:28<00:22, 8028.28it/s] 56%|█████▌    | 223309/400000 [00:28<00:22, 8026.52it/s] 56%|█████▌    | 224126/400000 [00:28<00:21, 8067.30it/s] 56%|█████▌    | 224951/400000 [00:28<00:21, 8118.69it/s] 56%|█████▋    | 225764/400000 [00:28<00:21, 7949.99it/s] 57%|█████▋    | 226560/400000 [00:29<00:21, 7920.43it/s] 57%|█████▋    | 227394/400000 [00:29<00:21, 8039.25it/s] 57%|█████▋    | 228199/400000 [00:29<00:21, 8025.92it/s] 57%|█████▋    | 229003/400000 [00:29<00:21, 7946.67it/s] 57%|█████▋    | 229799/400000 [00:29<00:21, 7935.74it/s] 58%|█████▊    | 230594/400000 [00:29<00:21, 7932.76it/s] 58%|█████▊    | 231388/400000 [00:29<00:21, 7868.93it/s] 58%|█████▊    | 232176/400000 [00:29<00:22, 7457.21it/s] 58%|█████▊    | 232927/400000 [00:29<00:22, 7440.67it/s] 58%|█████▊    | 233714/400000 [00:29<00:21, 7562.55it/s] 59%|█████▊    | 234474/400000 [00:30<00:21, 7528.82it/s] 59%|█████▉    | 235277/400000 [00:30<00:21, 7672.15it/s] 59%|█████▉    | 236064/400000 [00:30<00:21, 7728.17it/s] 59%|█████▉    | 236839/400000 [00:30<00:21, 7687.84it/s] 59%|█████▉    | 237654/400000 [00:30<00:20, 7819.28it/s] 60%|█████▉    | 238495/400000 [00:30<00:20, 7986.23it/s] 60%|█████▉    | 239296/400000 [00:30<00:20, 7965.21it/s] 60%|██████    | 240108/400000 [00:30<00:19, 8009.20it/s] 60%|██████    | 240910/400000 [00:30<00:20, 7865.66it/s] 60%|██████    | 241698/400000 [00:31<00:20, 7815.18it/s] 61%|██████    | 242495/400000 [00:31<00:20, 7859.88it/s] 61%|██████    | 243282/400000 [00:31<00:20, 7776.59it/s] 61%|██████    | 244061/400000 [00:31<00:20, 7726.04it/s] 61%|██████    | 244835/400000 [00:31<00:20, 7657.03it/s] 61%|██████▏   | 245626/400000 [00:31<00:19, 7730.44it/s] 62%|██████▏   | 246437/400000 [00:31<00:19, 7839.16it/s] 62%|██████▏   | 247291/400000 [00:31<00:19, 8034.82it/s] 62%|██████▏   | 248137/400000 [00:31<00:18, 8155.46it/s] 62%|██████▏   | 248955/400000 [00:31<00:18, 8085.65it/s] 62%|██████▏   | 249795/400000 [00:32<00:18, 8174.65it/s] 63%|██████▎   | 250614/400000 [00:32<00:18, 8020.22it/s] 63%|██████▎   | 251418/400000 [00:32<00:18, 7925.88it/s] 63%|██████▎   | 252212/400000 [00:32<00:18, 7897.90it/s] 63%|██████▎   | 253006/400000 [00:32<00:18, 7907.49it/s] 63%|██████▎   | 253798/400000 [00:32<00:18, 7788.48it/s] 64%|██████▎   | 254603/400000 [00:32<00:18, 7863.41it/s] 64%|██████▍   | 255391/400000 [00:32<00:18, 7779.40it/s] 64%|██████▍   | 256170/400000 [00:32<00:18, 7642.54it/s] 64%|██████▍   | 256936/400000 [00:32<00:18, 7552.50it/s] 64%|██████▍   | 257693/400000 [00:33<00:18, 7526.13it/s] 65%|██████▍   | 258506/400000 [00:33<00:18, 7695.26it/s] 65%|██████▍   | 259348/400000 [00:33<00:17, 7898.21it/s] 65%|██████▌   | 260200/400000 [00:33<00:17, 8073.22it/s] 65%|██████▌   | 261010/400000 [00:33<00:17, 7989.67it/s] 65%|██████▌   | 261845/400000 [00:33<00:17, 8092.14it/s] 66%|██████▌   | 262660/400000 [00:33<00:16, 8107.73it/s] 66%|██████▌   | 263472/400000 [00:33<00:16, 8084.62it/s] 66%|██████▌   | 264286/400000 [00:33<00:16, 8100.46it/s] 66%|██████▋   | 265097/400000 [00:33<00:17, 7914.46it/s] 66%|██████▋   | 265894/400000 [00:34<00:16, 7928.44it/s] 67%|██████▋   | 266688/400000 [00:34<00:17, 7810.95it/s] 67%|██████▋   | 267471/400000 [00:34<00:16, 7800.16it/s] 67%|██████▋   | 268258/400000 [00:34<00:16, 7820.44it/s] 67%|██████▋   | 269088/400000 [00:34<00:16, 7956.87it/s] 67%|██████▋   | 269889/400000 [00:34<00:16, 7972.23it/s] 68%|██████▊   | 270700/400000 [00:34<00:16, 8012.15it/s] 68%|██████▊   | 271502/400000 [00:34<00:16, 7903.37it/s] 68%|██████▊   | 272294/400000 [00:34<00:16, 7564.46it/s] 68%|██████▊   | 273055/400000 [00:34<00:16, 7510.25it/s] 68%|██████▊   | 273845/400000 [00:35<00:16, 7620.70it/s] 69%|██████▊   | 274652/400000 [00:35<00:16, 7749.31it/s] 69%|██████▉   | 275444/400000 [00:35<00:15, 7797.29it/s] 69%|██████▉   | 276240/400000 [00:35<00:15, 7843.23it/s] 69%|██████▉   | 277026/400000 [00:35<00:15, 7739.50it/s] 69%|██████▉   | 277856/400000 [00:35<00:15, 7897.40it/s] 70%|██████▉   | 278658/400000 [00:35<00:15, 7933.60it/s] 70%|██████▉   | 279453/400000 [00:35<00:15, 7933.50it/s] 70%|███████   | 280248/400000 [00:35<00:15, 7867.60it/s] 70%|███████   | 281036/400000 [00:36<00:15, 7795.54it/s] 70%|███████   | 281874/400000 [00:36<00:14, 7960.84it/s] 71%|███████   | 282681/400000 [00:36<00:14, 7991.92it/s] 71%|███████   | 283540/400000 [00:36<00:14, 8161.71it/s] 71%|███████   | 284402/400000 [00:36<00:13, 8293.52it/s] 71%|███████▏  | 285233/400000 [00:36<00:14, 7963.21it/s] 72%|███████▏  | 286034/400000 [00:36<00:14, 7892.10it/s] 72%|███████▏  | 286911/400000 [00:36<00:13, 8134.95it/s] 72%|███████▏  | 287729/400000 [00:36<00:13, 8025.65it/s] 72%|███████▏  | 288584/400000 [00:36<00:13, 8174.45it/s] 72%|███████▏  | 289405/400000 [00:37<00:13, 8127.89it/s] 73%|███████▎  | 290220/400000 [00:37<00:13, 8057.73it/s] 73%|███████▎  | 291028/400000 [00:37<00:13, 8027.65it/s] 73%|███████▎  | 291835/400000 [00:37<00:13, 8036.43it/s] 73%|███████▎  | 292640/400000 [00:37<00:13, 7962.75it/s] 73%|███████▎  | 293437/400000 [00:37<00:13, 7890.02it/s] 74%|███████▎  | 294240/400000 [00:37<00:13, 7931.47it/s] 74%|███████▍  | 295052/400000 [00:37<00:13, 7985.36it/s] 74%|███████▍  | 295851/400000 [00:37<00:13, 7940.34it/s] 74%|███████▍  | 296654/400000 [00:37<00:12, 7965.11it/s] 74%|███████▍  | 297451/400000 [00:38<00:13, 7737.77it/s] 75%|███████▍  | 298227/400000 [00:38<00:13, 7737.60it/s] 75%|███████▍  | 299037/400000 [00:38<00:12, 7842.66it/s] 75%|███████▍  | 299851/400000 [00:38<00:12, 7927.70it/s] 75%|███████▌  | 300669/400000 [00:38<00:12, 8000.83it/s] 75%|███████▌  | 301470/400000 [00:38<00:12, 7954.20it/s] 76%|███████▌  | 302267/400000 [00:38<00:12, 7916.00it/s] 76%|███████▌  | 303060/400000 [00:38<00:12, 7886.55it/s] 76%|███████▌  | 303850/400000 [00:38<00:12, 7879.83it/s] 76%|███████▌  | 304748/400000 [00:38<00:11, 8178.65it/s] 76%|███████▋  | 305569/400000 [00:39<00:11, 8089.58it/s] 77%|███████▋  | 306381/400000 [00:39<00:11, 8089.03it/s] 77%|███████▋  | 307192/400000 [00:39<00:11, 7989.04it/s] 77%|███████▋  | 307993/400000 [00:39<00:11, 7962.49it/s] 77%|███████▋  | 308833/400000 [00:39<00:11, 8086.18it/s] 77%|███████▋  | 309643/400000 [00:39<00:11, 7984.92it/s] 78%|███████▊  | 310443/400000 [00:39<00:11, 7870.09it/s] 78%|███████▊  | 311253/400000 [00:39<00:11, 7936.97it/s] 78%|███████▊  | 312048/400000 [00:39<00:11, 7703.50it/s] 78%|███████▊  | 312868/400000 [00:39<00:11, 7844.67it/s] 78%|███████▊  | 313675/400000 [00:40<00:10, 7907.16it/s] 79%|███████▊  | 314468/400000 [00:40<00:10, 7858.59it/s] 79%|███████▉  | 315259/400000 [00:40<00:10, 7872.42it/s] 79%|███████▉  | 316048/400000 [00:40<00:10, 7830.33it/s] 79%|███████▉  | 316849/400000 [00:40<00:10, 7882.80it/s] 79%|███████▉  | 317638/400000 [00:40<00:10, 7836.85it/s] 80%|███████▉  | 318456/400000 [00:40<00:10, 7934.76it/s] 80%|███████▉  | 319300/400000 [00:40<00:09, 8078.95it/s] 80%|████████  | 320109/400000 [00:40<00:09, 8029.77it/s] 80%|████████  | 320994/400000 [00:40<00:09, 8259.20it/s] 80%|████████  | 321823/400000 [00:41<00:09, 8019.16it/s] 81%|████████  | 322628/400000 [00:41<00:09, 8022.95it/s] 81%|████████  | 323437/400000 [00:41<00:09, 8041.93it/s] 81%|████████  | 324243/400000 [00:41<00:09, 8008.62it/s] 81%|████████▏ | 325045/400000 [00:41<00:09, 7951.18it/s] 81%|████████▏ | 325841/400000 [00:41<00:09, 7870.67it/s] 82%|████████▏ | 326666/400000 [00:41<00:09, 7977.55it/s] 82%|████████▏ | 327465/400000 [00:41<00:09, 7924.33it/s] 82%|████████▏ | 328259/400000 [00:41<00:09, 7856.41it/s] 82%|████████▏ | 329053/400000 [00:42<00:09, 7878.28it/s] 82%|████████▏ | 329845/400000 [00:42<00:08, 7888.60it/s] 83%|████████▎ | 330635/400000 [00:42<00:08, 7879.83it/s] 83%|████████▎ | 331487/400000 [00:42<00:08, 8060.49it/s] 83%|████████▎ | 332297/400000 [00:42<00:08, 8071.33it/s] 83%|████████▎ | 333105/400000 [00:42<00:08, 7998.71it/s] 83%|████████▎ | 333906/400000 [00:42<00:08, 7913.58it/s] 84%|████████▎ | 334714/400000 [00:42<00:08, 7962.36it/s] 84%|████████▍ | 335544/400000 [00:42<00:07, 8059.88it/s] 84%|████████▍ | 336362/400000 [00:42<00:07, 8093.78it/s] 84%|████████▍ | 337189/400000 [00:43<00:07, 8142.64it/s] 85%|████████▍ | 338004/400000 [00:43<00:07, 8126.42it/s] 85%|████████▍ | 338817/400000 [00:43<00:07, 8005.90it/s] 85%|████████▍ | 339651/400000 [00:43<00:07, 8101.39it/s] 85%|████████▌ | 340462/400000 [00:43<00:07, 8103.96it/s] 85%|████████▌ | 341273/400000 [00:43<00:07, 8072.28it/s] 86%|████████▌ | 342081/400000 [00:43<00:07, 7973.04it/s] 86%|████████▌ | 342879/400000 [00:43<00:07, 7933.86it/s] 86%|████████▌ | 343716/400000 [00:43<00:06, 8059.48it/s] 86%|████████▌ | 344536/400000 [00:43<00:06, 8099.84it/s] 86%|████████▋ | 345347/400000 [00:44<00:06, 7991.15it/s] 87%|████████▋ | 346147/400000 [00:44<00:06, 7912.85it/s] 87%|████████▋ | 346989/400000 [00:44<00:06, 8058.11it/s] 87%|████████▋ | 347796/400000 [00:44<00:06, 8059.59it/s] 87%|████████▋ | 348603/400000 [00:44<00:06, 7752.04it/s] 87%|████████▋ | 349382/400000 [00:44<00:06, 7689.63it/s] 88%|████████▊ | 350175/400000 [00:44<00:06, 7757.56it/s] 88%|████████▊ | 350995/400000 [00:44<00:06, 7882.20it/s] 88%|████████▊ | 351785/400000 [00:44<00:06, 7735.29it/s] 88%|████████▊ | 352561/400000 [00:44<00:06, 7732.52it/s] 88%|████████▊ | 353336/400000 [00:45<00:06, 7718.91it/s] 89%|████████▊ | 354118/400000 [00:45<00:05, 7748.20it/s] 89%|████████▊ | 354932/400000 [00:45<00:05, 7860.55it/s] 89%|████████▉ | 355740/400000 [00:45<00:05, 7923.29it/s] 89%|████████▉ | 356571/400000 [00:45<00:05, 8033.48it/s] 89%|████████▉ | 357376/400000 [00:45<00:05, 8003.91it/s] 90%|████████▉ | 358178/400000 [00:45<00:05, 8004.49it/s] 90%|████████▉ | 359067/400000 [00:45<00:04, 8250.18it/s] 90%|████████▉ | 359895/400000 [00:45<00:04, 8158.54it/s] 90%|█████████ | 360713/400000 [00:45<00:04, 8092.04it/s] 90%|█████████ | 361569/400000 [00:46<00:04, 8225.31it/s] 91%|█████████ | 362394/400000 [00:46<00:04, 8184.02it/s] 91%|█████████ | 363222/400000 [00:46<00:04, 8211.84it/s] 91%|█████████ | 364044/400000 [00:46<00:04, 8198.16it/s] 91%|█████████ | 364865/400000 [00:46<00:04, 8125.66it/s] 91%|█████████▏| 365679/400000 [00:46<00:04, 7974.03it/s] 92%|█████████▏| 366478/400000 [00:46<00:04, 7867.15it/s] 92%|█████████▏| 367303/400000 [00:46<00:04, 7977.26it/s] 92%|█████████▏| 368148/400000 [00:46<00:03, 8111.55it/s] 92%|█████████▏| 368988/400000 [00:47<00:03, 8193.94it/s] 92%|█████████▏| 369809/400000 [00:47<00:03, 8163.36it/s] 93%|█████████▎| 370627/400000 [00:47<00:03, 7996.03it/s] 93%|█████████▎| 371439/400000 [00:47<00:03, 8030.49it/s] 93%|█████████▎| 372282/400000 [00:47<00:03, 8145.95it/s] 93%|█████████▎| 373098/400000 [00:47<00:03, 8046.78it/s] 93%|█████████▎| 373907/400000 [00:47<00:03, 8058.86it/s] 94%|█████████▎| 374746/400000 [00:47<00:03, 8154.28it/s] 94%|█████████▍| 375563/400000 [00:47<00:03, 7888.09it/s] 94%|█████████▍| 376405/400000 [00:47<00:02, 8038.70it/s] 94%|█████████▍| 377241/400000 [00:48<00:02, 8128.83it/s] 95%|█████████▍| 378056/400000 [00:48<00:02, 8112.63it/s] 95%|█████████▍| 378869/400000 [00:48<00:02, 7839.71it/s] 95%|█████████▍| 379722/400000 [00:48<00:02, 8034.70it/s] 95%|█████████▌| 380585/400000 [00:48<00:02, 8203.68it/s] 95%|█████████▌| 381435/400000 [00:48<00:02, 8289.94it/s] 96%|█████████▌| 382267/400000 [00:48<00:02, 8261.58it/s] 96%|█████████▌| 383095/400000 [00:48<00:02, 8149.26it/s] 96%|█████████▌| 383938/400000 [00:48<00:01, 8230.09it/s] 96%|█████████▌| 384779/400000 [00:48<00:01, 8282.38it/s] 96%|█████████▋| 385609/400000 [00:49<00:01, 8280.68it/s] 97%|█████████▋| 386438/400000 [00:49<00:01, 8098.03it/s] 97%|█████████▋| 387250/400000 [00:49<00:01, 7907.04it/s] 97%|█████████▋| 388061/400000 [00:49<00:01, 7964.84it/s] 97%|█████████▋| 388909/400000 [00:49<00:01, 8112.12it/s] 97%|█████████▋| 389722/400000 [00:49<00:01, 8048.34it/s] 98%|█████████▊| 390529/400000 [00:49<00:01, 7839.06it/s] 98%|█████████▊| 391316/400000 [00:49<00:01, 7829.64it/s] 98%|█████████▊| 392123/400000 [00:49<00:00, 7899.96it/s] 98%|█████████▊| 392961/400000 [00:49<00:00, 8036.56it/s] 98%|█████████▊| 393795/400000 [00:50<00:00, 8122.78it/s] 99%|█████████▊| 394648/400000 [00:50<00:00, 8238.60it/s] 99%|█████████▉| 395474/400000 [00:50<00:00, 8046.15it/s] 99%|█████████▉| 396281/400000 [00:50<00:00, 8011.82it/s] 99%|█████████▉| 397084/400000 [00:50<00:00, 7953.11it/s] 99%|█████████▉| 397881/400000 [00:50<00:00, 7787.84it/s]100%|█████████▉| 398662/400000 [00:50<00:00, 7605.05it/s]100%|█████████▉| 399425/400000 [00:50<00:00, 7517.91it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 7860.61it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc0e31cbc88> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011081281138080384 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.010973746162593166 	 Accuracy: 67

  model saves at 67% accuracy 

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
2020-05-14 11:24:02.204218: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 11:24:02.208138: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-14 11:24:02.208305: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5587e8c39680 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 11:24:02.208321: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc0967090f0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5593 - accuracy: 0.5070
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6130 - accuracy: 0.5035
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6564 - accuracy: 0.5007 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6053 - accuracy: 0.5040
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6574 - accuracy: 0.5006
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6998 - accuracy: 0.4978
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7652 - accuracy: 0.4936
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.7011 - accuracy: 0.4978
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7382 - accuracy: 0.4953
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7586 - accuracy: 0.4940
11000/25000 [============>.................] - ETA: 4s - loss: 7.7252 - accuracy: 0.4962
12000/25000 [=============>................] - ETA: 4s - loss: 7.6883 - accuracy: 0.4986
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6595 - accuracy: 0.5005
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6732 - accuracy: 0.4996
15000/25000 [=================>............] - ETA: 3s - loss: 7.6523 - accuracy: 0.5009
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6369 - accuracy: 0.5019
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6477 - accuracy: 0.5012
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6538 - accuracy: 0.5008
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6553 - accuracy: 0.5007
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6467 - accuracy: 0.5013
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6440 - accuracy: 0.5015
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6576 - accuracy: 0.5006
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6546 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 10s 403us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fc0e31cbc88> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fc04fc26c88> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.6706 - crf_viterbi_accuracy: 0.1467 - val_loss: 1.5848 - val_crf_viterbi_accuracy: 0.1333

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

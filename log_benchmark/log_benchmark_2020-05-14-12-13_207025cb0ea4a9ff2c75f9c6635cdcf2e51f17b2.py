
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fee5956af28> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 12:13:37.321762
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 12:13:37.325844
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 12:13:37.328970
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 12:13:37.332497
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fee65335390> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356273.3750
Epoch 2/10

1/1 [==============================] - 0s 100ms/step - loss: 240169.0625
Epoch 3/10

1/1 [==============================] - 0s 92ms/step - loss: 131563.9531
Epoch 4/10

1/1 [==============================] - 0s 92ms/step - loss: 62163.7617
Epoch 5/10

1/1 [==============================] - 0s 94ms/step - loss: 30296.9727
Epoch 6/10

1/1 [==============================] - 0s 89ms/step - loss: 16296.2559
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 9939.9258
Epoch 8/10

1/1 [==============================] - 0s 102ms/step - loss: 6700.1855
Epoch 9/10

1/1 [==============================] - 0s 89ms/step - loss: 4891.1387
Epoch 10/10

1/1 [==============================] - 0s 90ms/step - loss: 3807.7781

  #### Inference Need return ypred, ytrue ######################### 
[[-5.63799977e-01  1.14126377e+01  1.20294123e+01  1.10256748e+01
   1.12960367e+01  1.03806095e+01  1.01928244e+01  1.11236095e+01
   1.20614824e+01  1.10579662e+01  1.16753263e+01  1.43011646e+01
   1.28523645e+01  1.09133873e+01  1.17718019e+01  1.30912924e+01
   1.16258669e+01  1.05572624e+01  1.12246180e+01  1.32366867e+01
   1.28805389e+01  1.19801826e+01  1.15807848e+01  1.28702593e+01
   1.03981771e+01  1.25749521e+01  1.16573668e+01  1.10082064e+01
   1.10126696e+01  1.30081711e+01  1.15277596e+01  1.30025949e+01
   1.02994137e+01  8.75386333e+00  1.15143270e+01  1.09551010e+01
   1.05543089e+01  1.13954382e+01  1.28582392e+01  1.22142715e+01
   9.72999859e+00  1.00026455e+01  1.02719193e+01  1.23995609e+01
   1.32988024e+01  1.27920094e+01  1.09285450e+01  1.08536921e+01
   9.31351089e+00  1.20427160e+01  1.17109985e+01  1.29976540e+01
   1.26067066e+01  1.19579153e+01  1.14111805e+01  1.09972382e+01
   9.50477982e+00  1.23619986e+01  9.71091938e+00  1.41493492e+01
  -6.14170194e-01  1.38322985e+00 -6.34678543e-01 -5.45345664e-01
   2.63984275e+00 -1.80641413e-01 -4.70418632e-01  8.51916313e-01
   1.09361124e+00  1.54064977e+00 -6.61988258e-01 -3.56612206e-02
   2.39433718e+00 -2.40270674e-01 -1.28549504e+00  5.56206703e-01
   4.27961379e-01  4.24388289e-01 -7.88675010e-01 -1.88794523e-01
  -1.18146375e-01  2.11687297e-01  2.99478471e-02  9.24262285e-01
  -8.64707649e-01  7.97763407e-01 -9.03322399e-01 -4.34092551e-01
  -4.20053363e-01  9.72130179e-01 -1.09659731e+00 -6.70832396e-01
   9.06102419e-01 -7.20486641e-01  4.24313620e-02 -6.83949590e-02
   2.45105624e+00  1.58160210e-01  3.49203259e-01 -1.79550886e+00
  -2.61229110e+00 -1.29955614e+00 -2.31978226e+00 -2.15570480e-01
   7.57269681e-01 -1.30017412e+00 -4.57368433e-01  6.46255136e-01
  -1.12206793e+00 -3.39690924e-01 -5.19518077e-01 -9.23042893e-02
  -2.57809669e-01 -8.02326441e-01  4.41309929e-01 -1.26763046e+00
  -1.06187239e-01 -1.01239395e+00  1.27295703e-01  5.72457671e-01
  -7.28687167e-01 -2.64213490e+00  2.68959701e-01 -1.54337978e+00
   1.17899573e+00  1.22238362e+00  6.85995579e-01  2.19178629e+00
  -1.35654795e+00 -1.11726773e+00  1.80734301e+00  1.02896702e+00
  -4.07997966e-01 -2.21909285e+00  1.34308994e-01 -1.04770392e-01
  -1.61094159e-01 -3.88425589e-03  1.96632195e+00  9.80732799e-01
   2.31516838e+00  2.23391938e+00  2.37797320e-01 -6.12807333e-01
  -3.51789594e-01  1.31334114e+00 -7.54353583e-01  1.75953615e+00
   2.51049340e-01  4.34708595e-02 -2.61322170e-01  5.10451853e-01
  -6.46756947e-01  1.30906671e-01 -1.85486913e+00  5.63275337e-01
  -1.39622569e+00 -1.39609098e+00  1.12397099e+00 -3.88341784e-01
   9.43412781e-01  1.54379499e+00  7.89142728e-01  1.17774403e+00
  -2.20248371e-01 -1.65368140e-01  6.98284149e-01  3.40921402e+00
  -1.24301577e+00  2.50157446e-01 -1.50427938e+00  1.21691716e+00
  -1.71594870e+00  1.47318006e+00 -5.17509937e-01 -7.40496397e-01
   2.20932341e+00 -6.64665878e-01  2.30194658e-01  7.00967729e-01
   6.80593789e-01  1.19200964e+01  1.06801090e+01  1.06501608e+01
   1.22089758e+01  1.12688532e+01  1.13960924e+01  1.15982857e+01
   9.90451241e+00  1.09404345e+01  1.08843212e+01  1.18474693e+01
   1.25525141e+01  1.27639217e+01  1.20749903e+01  1.12183638e+01
   1.11305017e+01  1.24031191e+01  1.14288311e+01  1.16671963e+01
   1.30551071e+01  1.14845095e+01  1.01203384e+01  1.04182549e+01
   1.13652315e+01  1.10419941e+01  1.07018681e+01  1.08822241e+01
   1.21333504e+01  1.18736792e+01  1.10117455e+01  1.24977551e+01
   1.07704229e+01  1.06730165e+01  1.18302231e+01  1.01157255e+01
   1.00437422e+01  1.18778477e+01  1.12472315e+01  1.25078373e+01
   1.38488970e+01  1.27816153e+01  1.29730282e+01  1.19383898e+01
   1.18098469e+01  1.08989153e+01  9.58083534e+00  1.25149279e+01
   8.73427582e+00  1.05063372e+01  1.15669451e+01  1.11156464e+01
   1.19298124e+01  1.13319340e+01  1.19254351e+01  1.06639900e+01
   1.11108780e+01  1.00962334e+01  1.09134731e+01  1.22118063e+01
   1.05111504e+00  1.05235600e+00  2.54571438e-01  2.72204876e-01
   1.38932288e-01  6.14716828e-01  1.25490308e-01  2.62093163e+00
   3.27908993e+00  1.70711219e-01  1.81259477e+00  3.89461088e+00
   2.76403189e+00  1.24282694e+00  1.91378903e+00  1.62196815e+00
   2.05634236e-01  1.75185537e+00  1.82962418e+00  1.01920843e-01
   1.91437650e+00  3.87269855e-01  1.58994889e+00  1.79935396e+00
   2.75115907e-01  2.47991920e+00  1.48332429e+00  7.35869288e-01
   1.37665701e+00  1.77126968e+00  6.89944267e-01  2.30842888e-01
   2.05313921e-01  3.50824237e-01  9.05200839e-02  5.04599810e-01
   1.78879964e+00  7.19089270e-01  5.91759920e-01  2.58246469e+00
   1.73459291e-01  1.50505781e+00  1.83941388e+00  3.06769013e-01
   7.19317198e-01  5.04062057e-01  2.66844654e+00  9.16631937e-01
   7.40502059e-01  7.68078387e-01  1.23142409e+00  1.18786979e+00
   2.28156209e-01  8.74911904e-01  3.66015375e-01  1.38524997e+00
   1.09845400e-01  2.28956342e-01  2.47438121e+00  6.34525418e-01
   2.99045801e-01  1.02309012e+00  2.60040140e+00  8.15212846e-01
   7.80413270e-01  2.64519739e+00  5.08086622e-01  9.08937097e-01
   5.31007111e-01  1.76540685e+00  2.30190802e+00  2.94106054e+00
   4.28877294e-01  8.74727011e-01  1.91085351e+00  1.87886477e-01
   5.28837740e-01  5.18527567e-01  1.04153955e+00  1.56211638e+00
   1.95005310e+00  2.14754581e+00  1.14945304e+00  2.33227897e+00
   6.79907680e-01  1.66518927e-01  8.85104656e-01  6.67807937e-01
   8.57013524e-01  1.73718572e-01  1.67917573e+00  2.70401764e+00
   1.02437615e-01  2.72874784e+00  9.55767870e-01  2.33441091e+00
   2.67255783e+00  5.74803829e-01  2.29087996e+00  4.84281480e-01
   1.59153342e-01  1.89571214e+00  2.53184199e-01  4.66287374e-01
   9.83082354e-01  2.28688359e-01  3.42949200e+00  1.63206506e+00
   2.65490961e+00  3.04253435e+00  7.29322076e-01  3.46897840e-01
   4.80944991e-01  2.19997406e-01  5.32086551e-01  4.96115685e-02
   2.81019330e+00  4.34160531e-01  1.17854989e+00  6.56762362e-01
   1.18305235e+01 -8.88113785e+00 -1.11645803e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 12:13:45.877686
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   90.0788
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 12:13:45.881412
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8139.58
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 12:13:45.884574
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                     89.77
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 12:13:45.887653
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -727.957
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140661318616456
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140658806284752
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140658806285256
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140658806285760
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140658806286264
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140658806286768

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fee44f44dd8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.613611
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.575541
grad_step = 000002, loss = 0.547351
grad_step = 000003, loss = 0.519516
grad_step = 000004, loss = 0.490852
grad_step = 000005, loss = 0.465363
grad_step = 000006, loss = 0.450472
grad_step = 000007, loss = 0.438291
grad_step = 000008, loss = 0.424538
grad_step = 000009, loss = 0.410142
grad_step = 000010, loss = 0.397019
grad_step = 000011, loss = 0.386550
grad_step = 000012, loss = 0.377015
grad_step = 000013, loss = 0.366457
grad_step = 000014, loss = 0.354248
grad_step = 000015, loss = 0.341174
grad_step = 000016, loss = 0.328650
grad_step = 000017, loss = 0.317828
grad_step = 000018, loss = 0.308592
grad_step = 000019, loss = 0.298900
grad_step = 000020, loss = 0.287951
grad_step = 000021, loss = 0.277065
grad_step = 000022, loss = 0.267179
grad_step = 000023, loss = 0.256024
grad_step = 000024, loss = 0.244173
grad_step = 000025, loss = 0.231913
grad_step = 000026, loss = 0.220727
grad_step = 000027, loss = 0.212012
grad_step = 000028, loss = 0.203007
grad_step = 000029, loss = 0.193791
grad_step = 000030, loss = 0.184744
grad_step = 000031, loss = 0.176010
grad_step = 000032, loss = 0.167716
grad_step = 000033, loss = 0.159820
grad_step = 000034, loss = 0.151975
grad_step = 000035, loss = 0.143923
grad_step = 000036, loss = 0.136174
grad_step = 000037, loss = 0.129259
grad_step = 000038, loss = 0.122836
grad_step = 000039, loss = 0.116144
grad_step = 000040, loss = 0.109471
grad_step = 000041, loss = 0.103540
grad_step = 000042, loss = 0.098081
grad_step = 000043, loss = 0.092527
grad_step = 000044, loss = 0.087093
grad_step = 000045, loss = 0.082075
grad_step = 000046, loss = 0.077404
grad_step = 000047, loss = 0.072920
grad_step = 000048, loss = 0.068583
grad_step = 000049, loss = 0.064429
grad_step = 000050, loss = 0.060466
grad_step = 000051, loss = 0.056777
grad_step = 000052, loss = 0.053270
grad_step = 000053, loss = 0.049867
grad_step = 000054, loss = 0.046717
grad_step = 000055, loss = 0.043773
grad_step = 000056, loss = 0.040913
grad_step = 000057, loss = 0.038209
grad_step = 000058, loss = 0.035672
grad_step = 000059, loss = 0.033272
grad_step = 000060, loss = 0.031037
grad_step = 000061, loss = 0.028909
grad_step = 000062, loss = 0.026895
grad_step = 000063, loss = 0.025071
grad_step = 000064, loss = 0.023306
grad_step = 000065, loss = 0.021622
grad_step = 000066, loss = 0.020119
grad_step = 000067, loss = 0.018697
grad_step = 000068, loss = 0.017338
grad_step = 000069, loss = 0.016085
grad_step = 000070, loss = 0.014937
grad_step = 000071, loss = 0.013865
grad_step = 000072, loss = 0.012837
grad_step = 000073, loss = 0.011902
grad_step = 000074, loss = 0.011056
grad_step = 000075, loss = 0.010241
grad_step = 000076, loss = 0.009496
grad_step = 000077, loss = 0.008815
grad_step = 000078, loss = 0.008180
grad_step = 000079, loss = 0.007588
grad_step = 000080, loss = 0.007042
grad_step = 000081, loss = 0.006553
grad_step = 000082, loss = 0.006096
grad_step = 000083, loss = 0.005678
grad_step = 000084, loss = 0.005297
grad_step = 000085, loss = 0.004949
grad_step = 000086, loss = 0.004633
grad_step = 000087, loss = 0.004343
grad_step = 000088, loss = 0.004089
grad_step = 000089, loss = 0.003854
grad_step = 000090, loss = 0.003643
grad_step = 000091, loss = 0.003454
grad_step = 000092, loss = 0.003288
grad_step = 000093, loss = 0.003136
grad_step = 000094, loss = 0.003003
grad_step = 000095, loss = 0.002887
grad_step = 000096, loss = 0.002782
grad_step = 000097, loss = 0.002690
grad_step = 000098, loss = 0.002609
grad_step = 000099, loss = 0.002539
grad_step = 000100, loss = 0.002476
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002423
grad_step = 000102, loss = 0.002376
grad_step = 000103, loss = 0.002335
grad_step = 000104, loss = 0.002299
grad_step = 000105, loss = 0.002268
grad_step = 000106, loss = 0.002242
grad_step = 000107, loss = 0.002219
grad_step = 000108, loss = 0.002199
grad_step = 000109, loss = 0.002182
grad_step = 000110, loss = 0.002167
grad_step = 000111, loss = 0.002155
grad_step = 000112, loss = 0.002145
grad_step = 000113, loss = 0.002135
grad_step = 000114, loss = 0.002127
grad_step = 000115, loss = 0.002121
grad_step = 000116, loss = 0.002115
grad_step = 000117, loss = 0.002110
grad_step = 000118, loss = 0.002106
grad_step = 000119, loss = 0.002102
grad_step = 000120, loss = 0.002098
grad_step = 000121, loss = 0.002095
grad_step = 000122, loss = 0.002092
grad_step = 000123, loss = 0.002090
grad_step = 000124, loss = 0.002087
grad_step = 000125, loss = 0.002085
grad_step = 000126, loss = 0.002082
grad_step = 000127, loss = 0.002080
grad_step = 000128, loss = 0.002077
grad_step = 000129, loss = 0.002075
grad_step = 000130, loss = 0.002072
grad_step = 000131, loss = 0.002070
grad_step = 000132, loss = 0.002067
grad_step = 000133, loss = 0.002064
grad_step = 000134, loss = 0.002061
grad_step = 000135, loss = 0.002058
grad_step = 000136, loss = 0.002055
grad_step = 000137, loss = 0.002051
grad_step = 000138, loss = 0.002048
grad_step = 000139, loss = 0.002045
grad_step = 000140, loss = 0.002041
grad_step = 000141, loss = 0.002037
grad_step = 000142, loss = 0.002033
grad_step = 000143, loss = 0.002030
grad_step = 000144, loss = 0.002026
grad_step = 000145, loss = 0.002022
grad_step = 000146, loss = 0.002017
grad_step = 000147, loss = 0.002013
grad_step = 000148, loss = 0.002009
grad_step = 000149, loss = 0.002005
grad_step = 000150, loss = 0.002001
grad_step = 000151, loss = 0.001997
grad_step = 000152, loss = 0.001992
grad_step = 000153, loss = 0.001988
grad_step = 000154, loss = 0.001984
grad_step = 000155, loss = 0.001981
grad_step = 000156, loss = 0.001977
grad_step = 000157, loss = 0.001972
grad_step = 000158, loss = 0.001968
grad_step = 000159, loss = 0.001964
grad_step = 000160, loss = 0.001960
grad_step = 000161, loss = 0.001957
grad_step = 000162, loss = 0.001953
grad_step = 000163, loss = 0.001949
grad_step = 000164, loss = 0.001945
grad_step = 000165, loss = 0.001941
grad_step = 000166, loss = 0.001936
grad_step = 000167, loss = 0.001933
grad_step = 000168, loss = 0.001929
grad_step = 000169, loss = 0.001926
grad_step = 000170, loss = 0.001985
grad_step = 000171, loss = 0.001926
grad_step = 000172, loss = 0.001934
grad_step = 000173, loss = 0.001933
grad_step = 000174, loss = 0.001921
grad_step = 000175, loss = 0.001912
grad_step = 000176, loss = 0.001912
grad_step = 000177, loss = 0.001908
grad_step = 000178, loss = 0.001904
grad_step = 000179, loss = 0.001903
grad_step = 000180, loss = 0.001893
grad_step = 000181, loss = 0.001887
grad_step = 000182, loss = 0.001890
grad_step = 000183, loss = 0.001886
grad_step = 000184, loss = 0.001879
grad_step = 000185, loss = 0.001874
grad_step = 000186, loss = 0.001873
grad_step = 000187, loss = 0.001871
grad_step = 000188, loss = 0.001865
grad_step = 000189, loss = 0.001863
grad_step = 000190, loss = 0.001861
grad_step = 000191, loss = 0.001856
grad_step = 000192, loss = 0.001851
grad_step = 000193, loss = 0.001848
grad_step = 000194, loss = 0.001847
grad_step = 000195, loss = 0.001844
grad_step = 000196, loss = 0.001840
grad_step = 000197, loss = 0.001837
grad_step = 000198, loss = 0.001834
grad_step = 000199, loss = 0.001832
grad_step = 000200, loss = 0.001828
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001824
grad_step = 000202, loss = 0.001820
grad_step = 000203, loss = 0.001818
grad_step = 000204, loss = 0.001816
grad_step = 000205, loss = 0.001813
grad_step = 000206, loss = 0.001810
grad_step = 000207, loss = 0.001807
grad_step = 000208, loss = 0.001804
grad_step = 000209, loss = 0.001803
grad_step = 000210, loss = 0.001805
grad_step = 000211, loss = 0.001813
grad_step = 000212, loss = 0.001837
grad_step = 000213, loss = 0.001892
grad_step = 000214, loss = 0.001989
grad_step = 000215, loss = 0.002054
grad_step = 000216, loss = 0.002025
grad_step = 000217, loss = 0.001860
grad_step = 000218, loss = 0.001783
grad_step = 000219, loss = 0.001857
grad_step = 000220, loss = 0.001923
grad_step = 000221, loss = 0.001875
grad_step = 000222, loss = 0.001785
grad_step = 000223, loss = 0.001788
grad_step = 000224, loss = 0.001849
grad_step = 000225, loss = 0.001851
grad_step = 000226, loss = 0.001788
grad_step = 000227, loss = 0.001764
grad_step = 000228, loss = 0.001799
grad_step = 000229, loss = 0.001815
grad_step = 000230, loss = 0.001781
grad_step = 000231, loss = 0.001752
grad_step = 000232, loss = 0.001767
grad_step = 000233, loss = 0.001786
grad_step = 000234, loss = 0.001773
grad_step = 000235, loss = 0.001743
grad_step = 000236, loss = 0.001742
grad_step = 000237, loss = 0.001760
grad_step = 000238, loss = 0.001760
grad_step = 000239, loss = 0.001738
grad_step = 000240, loss = 0.001727
grad_step = 000241, loss = 0.001734
grad_step = 000242, loss = 0.001742
grad_step = 000243, loss = 0.001734
grad_step = 000244, loss = 0.001720
grad_step = 000245, loss = 0.001715
grad_step = 000246, loss = 0.001719
grad_step = 000247, loss = 0.001722
grad_step = 000248, loss = 0.001717
grad_step = 000249, loss = 0.001707
grad_step = 000250, loss = 0.001701
grad_step = 000251, loss = 0.001701
grad_step = 000252, loss = 0.001703
grad_step = 000253, loss = 0.001702
grad_step = 000254, loss = 0.001697
grad_step = 000255, loss = 0.001691
grad_step = 000256, loss = 0.001686
grad_step = 000257, loss = 0.001682
grad_step = 000258, loss = 0.001681
grad_step = 000259, loss = 0.001681
grad_step = 000260, loss = 0.001681
grad_step = 000261, loss = 0.001682
grad_step = 000262, loss = 0.001683
grad_step = 000263, loss = 0.001685
grad_step = 000264, loss = 0.001687
grad_step = 000265, loss = 0.001689
grad_step = 000266, loss = 0.001690
grad_step = 000267, loss = 0.001692
grad_step = 000268, loss = 0.001695
grad_step = 000269, loss = 0.001702
grad_step = 000270, loss = 0.001708
grad_step = 000271, loss = 0.001714
grad_step = 000272, loss = 0.001712
grad_step = 000273, loss = 0.001709
grad_step = 000274, loss = 0.001698
grad_step = 000275, loss = 0.001684
grad_step = 000276, loss = 0.001665
grad_step = 000277, loss = 0.001646
grad_step = 000278, loss = 0.001633
grad_step = 000279, loss = 0.001626
grad_step = 000280, loss = 0.001625
grad_step = 000281, loss = 0.001627
grad_step = 000282, loss = 0.001630
grad_step = 000283, loss = 0.001636
grad_step = 000284, loss = 0.001650
grad_step = 000285, loss = 0.001672
grad_step = 000286, loss = 0.001710
grad_step = 000287, loss = 0.001741
grad_step = 000288, loss = 0.001772
grad_step = 000289, loss = 0.001750
grad_step = 000290, loss = 0.001716
grad_step = 000291, loss = 0.001675
grad_step = 000292, loss = 0.001640
grad_step = 000293, loss = 0.001612
grad_step = 000294, loss = 0.001611
grad_step = 000295, loss = 0.001632
grad_step = 000296, loss = 0.001649
grad_step = 000297, loss = 0.001645
grad_step = 000298, loss = 0.001622
grad_step = 000299, loss = 0.001603
grad_step = 000300, loss = 0.001594
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001593
grad_step = 000302, loss = 0.001584
grad_step = 000303, loss = 0.001575
grad_step = 000304, loss = 0.001575
grad_step = 000305, loss = 0.001582
grad_step = 000306, loss = 0.001591
grad_step = 000307, loss = 0.001594
grad_step = 000308, loss = 0.001599
grad_step = 000309, loss = 0.001602
grad_step = 000310, loss = 0.001613
grad_step = 000311, loss = 0.001613
grad_step = 000312, loss = 0.001604
grad_step = 000313, loss = 0.001576
grad_step = 000314, loss = 0.001551
grad_step = 000315, loss = 0.001536
grad_step = 000316, loss = 0.001534
grad_step = 000317, loss = 0.001542
grad_step = 000318, loss = 0.001549
grad_step = 000319, loss = 0.001548
grad_step = 000320, loss = 0.001544
grad_step = 000321, loss = 0.001544
grad_step = 000322, loss = 0.001544
grad_step = 000323, loss = 0.001550
grad_step = 000324, loss = 0.001552
grad_step = 000325, loss = 0.001554
grad_step = 000326, loss = 0.001546
grad_step = 000327, loss = 0.001543
grad_step = 000328, loss = 0.001541
grad_step = 000329, loss = 0.001542
grad_step = 000330, loss = 0.001536
grad_step = 000331, loss = 0.001527
grad_step = 000332, loss = 0.001513
grad_step = 000333, loss = 0.001500
grad_step = 000334, loss = 0.001490
grad_step = 000335, loss = 0.001486
grad_step = 000336, loss = 0.001490
grad_step = 000337, loss = 0.001511
grad_step = 000338, loss = 0.001578
grad_step = 000339, loss = 0.001632
grad_step = 000340, loss = 0.001704
grad_step = 000341, loss = 0.001527
grad_step = 000342, loss = 0.001459
grad_step = 000343, loss = 0.001548
grad_step = 000344, loss = 0.001546
grad_step = 000345, loss = 0.001472
grad_step = 000346, loss = 0.001457
grad_step = 000347, loss = 0.001518
grad_step = 000348, loss = 0.001536
grad_step = 000349, loss = 0.001465
grad_step = 000350, loss = 0.001447
grad_step = 000351, loss = 0.001482
grad_step = 000352, loss = 0.001511
grad_step = 000353, loss = 0.001523
grad_step = 000354, loss = 0.001502
grad_step = 000355, loss = 0.001554
grad_step = 000356, loss = 0.001659
grad_step = 000357, loss = 0.001675
grad_step = 000358, loss = 0.001676
grad_step = 000359, loss = 0.001626
grad_step = 000360, loss = 0.001533
grad_step = 000361, loss = 0.001437
grad_step = 000362, loss = 0.001422
grad_step = 000363, loss = 0.001482
grad_step = 000364, loss = 0.001522
grad_step = 000365, loss = 0.001502
grad_step = 000366, loss = 0.001437
grad_step = 000367, loss = 0.001415
grad_step = 000368, loss = 0.001439
grad_step = 000369, loss = 0.001438
grad_step = 000370, loss = 0.001426
grad_step = 000371, loss = 0.001423
grad_step = 000372, loss = 0.001411
grad_step = 000373, loss = 0.001395
grad_step = 000374, loss = 0.001404
grad_step = 000375, loss = 0.001441
grad_step = 000376, loss = 0.001469
grad_step = 000377, loss = 0.001497
grad_step = 000378, loss = 0.001471
grad_step = 000379, loss = 0.001445
grad_step = 000380, loss = 0.001417
grad_step = 000381, loss = 0.001394
grad_step = 000382, loss = 0.001380
grad_step = 000383, loss = 0.001390
grad_step = 000384, loss = 0.001405
grad_step = 000385, loss = 0.001393
grad_step = 000386, loss = 0.001373
grad_step = 000387, loss = 0.001365
grad_step = 000388, loss = 0.001372
grad_step = 000389, loss = 0.001377
grad_step = 000390, loss = 0.001369
grad_step = 000391, loss = 0.001362
grad_step = 000392, loss = 0.001355
grad_step = 000393, loss = 0.001349
grad_step = 000394, loss = 0.001346
grad_step = 000395, loss = 0.001351
grad_step = 000396, loss = 0.001363
grad_step = 000397, loss = 0.001369
grad_step = 000398, loss = 0.001384
grad_step = 000399, loss = 0.001390
grad_step = 000400, loss = 0.001412
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001397
grad_step = 000402, loss = 0.001391
grad_step = 000403, loss = 0.001359
grad_step = 000404, loss = 0.001341
grad_step = 000405, loss = 0.001337
grad_step = 000406, loss = 0.001343
grad_step = 000407, loss = 0.001355
grad_step = 000408, loss = 0.001351
grad_step = 000409, loss = 0.001349
grad_step = 000410, loss = 0.001333
grad_step = 000411, loss = 0.001323
grad_step = 000412, loss = 0.001315
grad_step = 000413, loss = 0.001311
grad_step = 000414, loss = 0.001311
grad_step = 000415, loss = 0.001315
grad_step = 000416, loss = 0.001324
grad_step = 000417, loss = 0.001340
grad_step = 000418, loss = 0.001381
grad_step = 000419, loss = 0.001417
grad_step = 000420, loss = 0.001500
grad_step = 000421, loss = 0.001467
grad_step = 000422, loss = 0.001458
grad_step = 000423, loss = 0.001438
grad_step = 000424, loss = 0.001503
grad_step = 000425, loss = 0.001581
grad_step = 000426, loss = 0.001493
grad_step = 000427, loss = 0.001409
grad_step = 000428, loss = 0.001377
grad_step = 000429, loss = 0.001335
grad_step = 000430, loss = 0.001329
grad_step = 000431, loss = 0.001351
grad_step = 000432, loss = 0.001366
grad_step = 000433, loss = 0.001386
grad_step = 000434, loss = 0.001373
grad_step = 000435, loss = 0.001325
grad_step = 000436, loss = 0.001288
grad_step = 000437, loss = 0.001292
grad_step = 000438, loss = 0.001308
grad_step = 000439, loss = 0.001315
grad_step = 000440, loss = 0.001317
grad_step = 000441, loss = 0.001306
grad_step = 000442, loss = 0.001296
grad_step = 000443, loss = 0.001292
grad_step = 000444, loss = 0.001287
grad_step = 000445, loss = 0.001280
grad_step = 000446, loss = 0.001279
grad_step = 000447, loss = 0.001288
grad_step = 000448, loss = 0.001289
grad_step = 000449, loss = 0.001284
grad_step = 000450, loss = 0.001281
grad_step = 000451, loss = 0.001283
grad_step = 000452, loss = 0.001280
grad_step = 000453, loss = 0.001277
grad_step = 000454, loss = 0.001280
grad_step = 000455, loss = 0.001292
grad_step = 000456, loss = 0.001300
grad_step = 000457, loss = 0.001303
grad_step = 000458, loss = 0.001296
grad_step = 000459, loss = 0.001274
grad_step = 000460, loss = 0.001257
grad_step = 000461, loss = 0.001247
grad_step = 000462, loss = 0.001243
grad_step = 000463, loss = 0.001245
grad_step = 000464, loss = 0.001251
grad_step = 000465, loss = 0.001253
grad_step = 000466, loss = 0.001248
grad_step = 000467, loss = 0.001239
grad_step = 000468, loss = 0.001234
grad_step = 000469, loss = 0.001233
grad_step = 000470, loss = 0.001231
grad_step = 000471, loss = 0.001229
grad_step = 000472, loss = 0.001230
grad_step = 000473, loss = 0.001232
grad_step = 000474, loss = 0.001237
grad_step = 000475, loss = 0.001247
grad_step = 000476, loss = 0.001270
grad_step = 000477, loss = 0.001313
grad_step = 000478, loss = 0.001373
grad_step = 000479, loss = 0.001389
grad_step = 000480, loss = 0.001360
grad_step = 000481, loss = 0.001265
grad_step = 000482, loss = 0.001238
grad_step = 000483, loss = 0.001270
grad_step = 000484, loss = 0.001282
grad_step = 000485, loss = 0.001273
grad_step = 000486, loss = 0.001294
grad_step = 000487, loss = 0.001333
grad_step = 000488, loss = 0.001374
grad_step = 000489, loss = 0.001411
grad_step = 000490, loss = 0.001475
grad_step = 000491, loss = 0.001446
grad_step = 000492, loss = 0.001397
grad_step = 000493, loss = 0.001313
grad_step = 000494, loss = 0.001231
grad_step = 000495, loss = 0.001222
grad_step = 000496, loss = 0.001260
grad_step = 000497, loss = 0.001291
grad_step = 000498, loss = 0.001297
grad_step = 000499, loss = 0.001283
grad_step = 000500, loss = 0.001229
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001195
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

  date_run                              2020-05-14 12:14:03.016897
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.273514
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 12:14:03.022661
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.189257
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 12:14:03.029058
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.161273
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 12:14:03.033533
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.87583
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
0   2020-05-14 12:13:37.321762  ...    mean_absolute_error
1   2020-05-14 12:13:37.325844  ...     mean_squared_error
2   2020-05-14 12:13:37.328970  ...  median_absolute_error
3   2020-05-14 12:13:37.332497  ...               r2_score
4   2020-05-14 12:13:45.877686  ...    mean_absolute_error
5   2020-05-14 12:13:45.881412  ...     mean_squared_error
6   2020-05-14 12:13:45.884574  ...  median_absolute_error
7   2020-05-14 12:13:45.887653  ...               r2_score
8   2020-05-14 12:14:03.016897  ...    mean_absolute_error
9   2020-05-14 12:14:03.022661  ...     mean_squared_error
10  2020-05-14 12:14:03.029058  ...  median_absolute_error
11  2020-05-14 12:14:03.033533  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb10ec62ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:32, 303371.35it/s]  2%|▏         | 212992/9912422 [00:00<00:24, 394210.84it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 546362.49it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 772340.64it/s] 77%|███████▋  | 7675904/9912422 [00:00<00:02, 1092539.95it/s]9920512it [00:00, 10703893.99it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 149490.59it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 306614.10it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 395889.03it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 548988.37it/s]1654784it [00:00, 2730811.95it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 52512.93it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb0c161de48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb0be4680b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb0c161de48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb0c0ba20b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb0be3dd4a8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb0be3c8710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb0c161de48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb0c0b616d8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb0be3dd4a8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb10ec25eb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f8b53842208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=261da82a26e711a0ff74d42ce34b97004d36bc03e3cd7ecc06d6d3b1b365f5ff
  Stored in directory: /tmp/pip-ephem-wheel-cache-46ocpmuj/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f8aeb63d710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 44s
   57344/17464789 [..............................] - ETA: 37s
   73728/17464789 [..............................] - ETA: 44s
  122880/17464789 [..............................] - ETA: 35s
  180224/17464789 [..............................] - ETA: 30s
  229376/17464789 [..............................] - ETA: 28s
  278528/17464789 [..............................] - ETA: 27s
  335872/17464789 [..............................] - ETA: 25s
  352256/17464789 [..............................] - ETA: 27s
  417792/17464789 [..............................] - ETA: 25s
  475136/17464789 [..............................] - ETA: 24s
  524288/17464789 [..............................] - ETA: 24s
  581632/17464789 [..............................] - ETA: 23s
  630784/17464789 [>.............................] - ETA: 23s
  679936/17464789 [>.............................] - ETA: 23s
  737280/17464789 [>.............................] - ETA: 22s
  786432/17464789 [>.............................] - ETA: 22s
  851968/17464789 [>.............................] - ETA: 21s
  909312/17464789 [>.............................] - ETA: 21s
  925696/17464789 [>.............................] - ETA: 22s
  974848/17464789 [>.............................] - ETA: 22s
 1032192/17464789 [>.............................] - ETA: 21s
 1081344/17464789 [>.............................] - ETA: 21s
 1138688/17464789 [>.............................] - ETA: 21s
 1187840/17464789 [=>............................] - ETA: 21s
 1236992/17464789 [=>............................] - ETA: 21s
 1294336/17464789 [=>............................] - ETA: 21s
 1343488/17464789 [=>............................] - ETA: 21s
 1417216/17464789 [=>............................] - ETA: 20s
 1466368/17464789 [=>............................] - ETA: 20s
 1515520/17464789 [=>............................] - ETA: 20s
 1572864/17464789 [=>............................] - ETA: 20s
 1622016/17464789 [=>............................] - ETA: 20s
 1671168/17464789 [=>............................] - ETA: 20s
 1728512/17464789 [=>............................] - ETA: 19s
 1777664/17464789 [==>...........................] - ETA: 19s
 1851392/17464789 [==>...........................] - ETA: 19s
 1900544/17464789 [==>...........................] - ETA: 19s
 1949696/17464789 [==>...........................] - ETA: 19s
 2007040/17464789 [==>...........................] - ETA: 19s
 2056192/17464789 [==>...........................] - ETA: 19s
 2113536/17464789 [==>...........................] - ETA: 19s
 2162688/17464789 [==>...........................] - ETA: 19s
 2211840/17464789 [==>...........................] - ETA: 18s
 2269184/17464789 [==>...........................] - ETA: 18s
 2318336/17464789 [==>...........................] - ETA: 18s
 2392064/17464789 [===>..........................] - ETA: 18s
 2441216/17464789 [===>..........................] - ETA: 18s
 2490368/17464789 [===>..........................] - ETA: 18s
 2547712/17464789 [===>..........................] - ETA: 18s
 2596864/17464789 [===>..........................] - ETA: 18s
 2646016/17464789 [===>..........................] - ETA: 18s
 2703360/17464789 [===>..........................] - ETA: 18s
 2768896/17464789 [===>..........................] - ETA: 17s
 2826240/17464789 [===>..........................] - ETA: 17s
 2875392/17464789 [===>..........................] - ETA: 17s
 2949120/17464789 [====>.........................] - ETA: 17s
 2998272/17464789 [====>.........................] - ETA: 17s
 3047424/17464789 [====>.........................] - ETA: 17s
 3121152/17464789 [====>.........................] - ETA: 17s
 3170304/17464789 [====>.........................] - ETA: 17s
 3244032/17464789 [====>.........................] - ETA: 17s
 3293184/17464789 [====>.........................] - ETA: 16s
 3366912/17464789 [====>.........................] - ETA: 16s
 3432448/17464789 [====>.........................] - ETA: 16s
 3481600/17464789 [====>.........................] - ETA: 16s
 3555328/17464789 [=====>........................] - ETA: 16s
 3620864/17464789 [=====>........................] - ETA: 16s
 3678208/17464789 [=====>........................] - ETA: 16s
 3743744/17464789 [=====>........................] - ETA: 16s
 3817472/17464789 [=====>........................] - ETA: 15s
 3883008/17464789 [=====>........................] - ETA: 15s
 3940352/17464789 [=====>........................] - ETA: 15s
 4005888/17464789 [=====>........................] - ETA: 15s
 4079616/17464789 [======>.......................] - ETA: 15s
 4145152/17464789 [======>.......................] - ETA: 15s
 4218880/17464789 [======>.......................] - ETA: 15s
 4284416/17464789 [======>.......................] - ETA: 15s
 4358144/17464789 [======>.......................] - ETA: 14s
 4423680/17464789 [======>.......................] - ETA: 14s
 4513792/17464789 [======>.......................] - ETA: 14s
 4579328/17464789 [======>.......................] - ETA: 14s
 4653056/17464789 [======>.......................] - ETA: 14s
 4718592/17464789 [=======>......................] - ETA: 14s
 4792320/17464789 [=======>......................] - ETA: 14s
 4874240/17464789 [=======>......................] - ETA: 13s
 4947968/17464789 [=======>......................] - ETA: 13s
 5013504/17464789 [=======>......................] - ETA: 13s
 5103616/17464789 [=======>......................] - ETA: 13s
 5177344/17464789 [=======>......................] - ETA: 13s
 5242880/17464789 [========>.....................] - ETA: 13s
 5332992/17464789 [========>.....................] - ETA: 13s
 5414912/17464789 [========>.....................] - ETA: 12s
 5488640/17464789 [========>.....................] - ETA: 12s
 5570560/17464789 [========>.....................] - ETA: 12s
 5660672/17464789 [========>.....................] - ETA: 12s
 5734400/17464789 [========>.....................] - ETA: 12s
 5816320/17464789 [========>.....................] - ETA: 12s
 5906432/17464789 [=========>....................] - ETA: 12s
 5971968/17464789 [=========>....................] - ETA: 12s
 6062080/17464789 [=========>....................] - ETA: 11s
 6152192/17464789 [=========>....................] - ETA: 11s
 6234112/17464789 [=========>....................] - ETA: 11s
 6307840/17464789 [=========>....................] - ETA: 11s
 6389760/17464789 [=========>....................] - ETA: 11s
 6479872/17464789 [==========>...................] - ETA: 11s
 6569984/17464789 [==========>...................] - ETA: 11s
 6651904/17464789 [==========>...................] - ETA: 10s
 6742016/17464789 [==========>...................] - ETA: 10s
 6823936/17464789 [==========>...................] - ETA: 10s
 6930432/17464789 [==========>...................] - ETA: 10s
 7020544/17464789 [===========>..................] - ETA: 10s
 7102464/17464789 [===========>..................] - ETA: 10s
 7192576/17464789 [===========>..................] - ETA: 10s
 7299072/17464789 [===========>..................] - ETA: 10s
 7380992/17464789 [===========>..................] - ETA: 9s 
 7471104/17464789 [===========>..................] - ETA: 9s
 7577600/17464789 [============>.................] - ETA: 9s
 7659520/17464789 [============>.................] - ETA: 9s
 7766016/17464789 [============>.................] - ETA: 9s
 7856128/17464789 [============>.................] - ETA: 9s
 7962624/17464789 [============>.................] - ETA: 9s
 8044544/17464789 [============>.................] - ETA: 9s
 8151040/17464789 [=============>................] - ETA: 8s
 8257536/17464789 [=============>................] - ETA: 8s
 8355840/17464789 [=============>................] - ETA: 8s
 8445952/17464789 [=============>................] - ETA: 8s
 8552448/17464789 [=============>................] - ETA: 8s
 8658944/17464789 [=============>................] - ETA: 8s
 8757248/17464789 [==============>...............] - ETA: 8s
 8863744/17464789 [==============>...............] - ETA: 7s
 8970240/17464789 [==============>...............] - ETA: 7s
 9076736/17464789 [==============>...............] - ETA: 7s
 9175040/17464789 [==============>...............] - ETA: 7s
 9281536/17464789 [==============>...............] - ETA: 7s
 9388032/17464789 [===============>..............] - ETA: 7s
 9494528/17464789 [===============>..............] - ETA: 7s
 9592832/17464789 [===============>..............] - ETA: 7s
 9715712/17464789 [===============>..............] - ETA: 6s
 9822208/17464789 [===============>..............] - ETA: 6s
 9928704/17464789 [================>.............] - ETA: 6s
10043392/17464789 [================>.............] - ETA: 6s
10149888/17464789 [================>.............] - ETA: 6s
10256384/17464789 [================>.............] - ETA: 6s
10379264/17464789 [================>.............] - ETA: 6s
10502144/17464789 [=================>............] - ETA: 6s
10625024/17464789 [=================>............] - ETA: 5s
10739712/17464789 [=================>............] - ETA: 5s
10862592/17464789 [=================>............] - ETA: 5s
10985472/17464789 [=================>............] - ETA: 5s
11108352/17464789 [==================>...........] - ETA: 5s
11247616/17464789 [==================>...........] - ETA: 5s
11386880/17464789 [==================>...........] - ETA: 5s
11526144/17464789 [==================>...........] - ETA: 4s
11665408/17464789 [===================>..........] - ETA: 4s
11804672/17464789 [===================>..........] - ETA: 4s
11943936/17464789 [===================>..........] - ETA: 4s
12083200/17464789 [===================>..........] - ETA: 4s
12238848/17464789 [====================>.........] - ETA: 4s
12394496/17464789 [====================>.........] - ETA: 4s
12558336/17464789 [====================>.........] - ETA: 3s
12713984/17464789 [====================>.........] - ETA: 3s
12886016/17464789 [=====================>........] - ETA: 3s
13058048/17464789 [=====================>........] - ETA: 3s
13230080/17464789 [=====================>........] - ETA: 3s
13410304/17464789 [======================>.......] - ETA: 3s
13598720/17464789 [======================>.......] - ETA: 2s
13787136/17464789 [======================>.......] - ETA: 2s
13983744/17464789 [=======================>......] - ETA: 2s
14172160/17464789 [=======================>......] - ETA: 2s
14368768/17464789 [=======================>......] - ETA: 2s
14573568/17464789 [========================>.....] - ETA: 2s
14786560/17464789 [========================>.....] - ETA: 1s
14991360/17464789 [========================>.....] - ETA: 1s
15220736/17464789 [=========================>....] - ETA: 1s
15441920/17464789 [=========================>....] - ETA: 1s
15671296/17464789 [=========================>....] - ETA: 1s
15900672/17464789 [==========================>...] - ETA: 1s
16138240/17464789 [==========================>...] - ETA: 0s
16384000/17464789 [===========================>..] - ETA: 0s
16629760/17464789 [===========================>..] - ETA: 0s
16875520/17464789 [===========================>..] - ETA: 0s
17129472/17464789 [============================>.] - ETA: 0s
17391616/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 12s 1us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 12:15:43.377413: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 12:15:43.381822: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-14 12:15:43.381940: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56165c82c310 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 12:15:43.381951: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.5286 - accuracy: 0.5090
 2000/25000 [=>............................] - ETA: 7s - loss: 7.6206 - accuracy: 0.5030 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.4980 - accuracy: 0.5110
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.5401 - accuracy: 0.5082
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5348 - accuracy: 0.5086
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5644 - accuracy: 0.5067
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5615 - accuracy: 0.5069
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6110 - accuracy: 0.5036
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.5985 - accuracy: 0.5044
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6068 - accuracy: 0.5039
11000/25000 [============>.................] - ETA: 3s - loss: 7.6318 - accuracy: 0.5023
12000/25000 [=============>................] - ETA: 3s - loss: 7.6334 - accuracy: 0.5022
13000/25000 [==============>...............] - ETA: 2s - loss: 7.5994 - accuracy: 0.5044
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6108 - accuracy: 0.5036
15000/25000 [=================>............] - ETA: 2s - loss: 7.6206 - accuracy: 0.5030
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6407 - accuracy: 0.5017
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6342 - accuracy: 0.5021
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6692 - accuracy: 0.4998
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6569 - accuracy: 0.5006
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6720 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6812 - accuracy: 0.4990
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6652 - accuracy: 0.5001
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6533 - accuracy: 0.5009
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
25000/25000 [==============================] - 7s 264us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 12:15:56.162821
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 12:15:56.162821  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<15:39:07, 15.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:59:15, 21.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 1.79M/862M [00:00<7:40:55, 31.1kB/s].vector_cache/glove.6B.zip:   0%|          | 3.92M/862M [00:00<5:22:02, 44.4kB/s].vector_cache/glove.6B.zip:   1%|          | 7.81M/862M [00:00<3:44:31, 63.4kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.1M/862M [00:01<2:36:28, 90.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.8M/862M [00:01<1:49:20, 129kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 16.7M/862M [00:01<1:16:36, 184kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.1M/862M [00:01<53:50, 261kB/s]  .vector_cache/glove.6B.zip:   2%|▏         | 19.6M/862M [00:01<37:53, 371kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.9M/862M [00:01<26:39, 526kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.4M/862M [00:01<18:46, 744kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.7M/862M [00:01<13:07, 1.06MB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.7M/862M [00:01<09:10, 1.50MB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.2M/862M [00:01<06:31, 2.10MB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.3M/862M [00:02<04:45, 2.87MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.0M/862M [00:02<03:33, 3.84MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.0M/862M [00:02<02:36, 5.19MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.3M/862M [00:02<02:00, 6.76MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:02<01:33, 8.63MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:02<05:37, 2.40MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:02<04:01, 3.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:05<39:32, 340kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:05<29:38, 453kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.4M/862M [00:05<20:47, 643kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:07<23:35, 566kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:07<24:58, 535kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:07<22:18, 599kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:07<21:58, 608kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:07<23:43, 563kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:07<19:42, 678kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:07<14:10, 941kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.2M/862M [00:07<10:09, 1.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:09<11:20, 1.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<08:40, 1.53MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<08:07, 1.63MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:11<10:30, 1.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:11<07:41, 1.72MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.4M/862M [00:11<05:36, 2.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<09:37, 1.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:13<11:08, 1.18MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:13<08:08, 1.61MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<08:04, 1.62MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:15<08:07, 1.61MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:15<05:51, 2.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<07:46, 1.67MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:17<07:34, 1.72MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.7M/862M [00:17<05:25, 2.39MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<09:07, 1.42MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:18<08:42, 1.49MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:19<06:23, 2.02MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.9M/862M [00:19<04:44, 2.72MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<08:31, 1.51MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<07:16, 1.77MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:21<05:10, 2.48MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:23<42:58, 298kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:23<31:54, 401kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.4M/862M [00:23<22:23, 570kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:25<22:22, 570kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:25<19:56, 639kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:25<14:21, 886kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:25<10:17, 1.24MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<07:33, 1.68MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<17:57, 706kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<14:40, 863kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<10:28, 1.21MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<07:35, 1.66MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<12:38, 997kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<11:01, 1.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<08:04, 1.56MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:29<06:07, 2.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<04:54, 2.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<04:03, 3.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<03:20, 3.74MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<1:28:30, 142kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<1:03:43, 197kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<44:32, 280kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<39:12, 318kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<29:20, 425kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<20:34, 603kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<21:14, 584kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<16:43, 741kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:36<11:55, 1.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<08:39, 1.43MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:29, 1.90MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<31:23, 393kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<25:35, 482kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<18:40, 660kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<13:08, 934kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:40<32:00, 383kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<23:27, 522kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<16:28, 741kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:42<19:44, 618kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<15:32, 785kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:42<10:58, 1.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<13:01, 931kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<10:55, 1.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:44<07:45, 1.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<10:43, 1.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<09:16, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:46<06:35, 1.82MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<11:14, 1.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<10:23, 1.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:48<07:36, 1.57MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:48<05:44, 2.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:49<04:36, 2.59MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<03:33, 3.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<20:29, 582kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<15:45, 756kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:50<11:05, 1.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:52<17:28, 678kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:52<13:15, 893kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:52<09:24, 1.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<10:37, 1.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<09:07, 1.29MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:54<06:28, 1.81MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:57<14:37, 801kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:57<11:56, 980kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:57<08:36, 1.36MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:59<08:26, 1.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<07:33, 1.54MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:59<05:22, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<11:11, 1.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<08:56, 1.29MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:01<06:20, 1.82MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<13:18, 865kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<10:59, 1.05MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:03<07:49, 1.47MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<09:39, 1.18MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<08:22, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:05<05:57, 1.91MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<11:20, 1.00MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<09:35, 1.18MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:07<06:56, 1.63MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:07<05:10, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:07<04:07, 2.74MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<10:02, 1.13MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<08:22, 1.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<05:56, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<17:22, 647kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<13:47, 814kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:11<09:45, 1.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<10:40, 1.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<09:22, 1.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:13<06:48, 1.64MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:13<05:02, 2.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:13<03:48, 2.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:15<18:12, 609kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<15:39, 708kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<11:51, 935kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<08:32, 1.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:15<06:31, 1.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:15<05:01, 2.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<03:51, 2.85MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<58:44, 188kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<41:42, 264kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<30:46, 356kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<23:12, 472kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:19<16:18, 669kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<15:15, 714kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<12:17, 885kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:21<08:44, 1.24MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<08:59, 1.20MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<07:51, 1.38MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<05:37, 1.91MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<07:23, 1.45MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<06:41, 1.61MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:25<04:48, 2.23MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<06:45, 1.58MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<06:16, 1.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:27<04:32, 2.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:29<07:15, 1.46MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<05:55, 1.79MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:29<04:13, 2.50MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<09:26, 1.12MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<07:38, 1.38MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:31<05:35, 1.88MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:31<04:09, 2.52MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<06:54, 1.52MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<05:22, 1.95MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<05:23, 1.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<07:03, 1.47MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<05:33, 1.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:35<04:01, 2.58MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<06:36, 1.57MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<05:54, 1.75MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<04:25, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<05:00, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<09:09, 1.12MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<08:14, 1.25MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<06:04, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:39<04:25, 2.31MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<06:55, 1.47MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<07:20, 1.39MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<05:31, 1.85MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:41<04:03, 2.50MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:43<05:44, 1.77MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:43<06:20, 1.60MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<04:59, 2.02MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:43<03:44, 2.70MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:45<04:53, 2.06MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<04:42, 2.14MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:45<03:32, 2.83MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:45<02:42, 3.70MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:47<05:31, 1.81MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:47<05:07, 1.95MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:47<03:53, 2.56MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:47<02:56, 3.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<05:13, 1.90MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<04:29, 2.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:49<03:18, 2.99MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<02:29, 3.97MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<16:25, 600kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<13:05, 753kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:51<09:22, 1.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:51<06:42, 1.46MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:52<10:23, 941kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<09:26, 1.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<07:07, 1.37MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:53<05:09, 1.89MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<03:45, 2.58MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<6:42:18, 24.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<4:42:12, 34.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:57<3:17:44, 48.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<2:19:41, 69.0kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:57<1:37:19, 98.5kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:59<1:16:00, 126kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<54:08, 177kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:59<37:44, 252kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:02<1:05:01, 146kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:02<46:53, 203kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<32:42, 289kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<31:39, 298kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<23:36, 400kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:04<16:37, 566kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<14:04, 666kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<11:15, 832kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:06<07:55, 1.18MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:07<12:05, 770kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:07<10:01, 928kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:07<07:06, 1.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:09<07:53, 1.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:09<06:36, 1.40MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:09<04:40, 1.96MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:11<20:12, 454kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:11<15:09, 605kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:11<10:36, 858kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:13<50:25, 180kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<36:22, 250kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:15<26:34, 340kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:15<19:42, 458kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:15<13:46, 651kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:17<27:44, 323kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:17<20:28, 437kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:19<15:30, 573kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:19<11:47, 753kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:19<08:23, 1.06MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:21<08:07, 1.09MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:21<06:45, 1.31MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:21<04:47, 1.83MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:23<08:29, 1.03MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:23<07:01, 1.25MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:23<05:05, 1.71MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:23<03:39, 2.38MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:25<1:38:02, 88.6kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:26<1:09:56, 124kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:26<48:58, 177kB/s]  .vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:28<36:07, 239kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:28<26:17, 328kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:28<18:22, 466kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:30<19:21, 441kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:30<14:46, 578kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:30<10:23, 818kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:32<10:41, 793kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:32<08:49, 960kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:32<06:15, 1.35MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:34<07:08, 1.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:34<06:01, 1.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:34<04:20, 1.93MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:34<03:13, 2.58MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:36<07:35, 1.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:36<06:02, 1.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:38<05:25, 1.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:38<04:29, 1.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:40<04:20, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:40<03:56, 2.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:42<03:55, 2.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:42<03:31, 2.31MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:42<02:31, 3.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:44<13:38, 591kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:44<10:24, 774kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:44<07:17, 1.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:46<19:29, 410kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:46<14:31, 550kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:46<10:09, 781kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:48<20:36, 385kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:48<15:16, 518kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:50<11:45, 669kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:50<08:54, 882kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:50<06:16, 1.24MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:52<09:31, 817kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:52<07:17, 1.07MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:52<05:07, 1.51MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:54<25:19, 305kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:54<18:33, 416kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:56<13:59, 547kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:56<10:16, 744kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:58<08:17, 914kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:58<06:30, 1.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:00<05:37, 1.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:00<04:26, 1.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:01<04:13, 1.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:02<03:28, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:03<03:31, 2.09MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:04<03:13, 2.29MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:05<03:18, 2.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:06<03:18, 2.21MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:06<02:21, 3.08MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:07<17:49, 406kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:07<13:02, 554kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:09<10:09, 705kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:10<08:07, 882kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:10<05:44, 1.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:11<06:52, 1.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:11<05:35, 1.27MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:12<03:57, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:13<06:56, 1.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:13<05:38, 1.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:15<04:55, 1.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:15<04:15, 1.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:16<03:03, 2.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:17<04:42, 1.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:17<04:05, 1.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:17<02:55, 2.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:19<04:42, 1.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:19<04:04, 1.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:19<02:54, 2.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:21<05:18, 1.27MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:21<04:30, 1.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:21<03:20, 2.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:22<02:27, 2.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:23<04:32, 1.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:23<03:53, 1.72MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:23<02:47, 2.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:25<04:31, 1.46MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:25<03:53, 1.70MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:25<02:48, 2.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:27<03:44, 1.75MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:27<03:01, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:27<02:11, 2.97MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:29<04:21, 1.49MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:29<03:37, 1.79MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:29<02:36, 2.46MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:31<04:17, 1.50MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:31<04:01, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:32<02:54, 2.20MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:33<03:40, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:33<03:21, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:33<02:23, 2.63MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:35<05:06, 1.23MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:35<04:37, 1.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:35<03:17, 1.89MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:37<04:26, 1.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:37<03:59, 1.55MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:37<02:49, 2.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:39<08:41, 706kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:39<06:52, 892kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:39<04:52, 1.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:42<05:56, 1.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:42<06:14, 972kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:42<04:36, 1.31MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:44<04:09, 1.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:44<04:55, 1.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:44<03:35, 1.66MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:44<02:36, 2.28MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:44<01:58, 2.99MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:46<5:39:48, 17.5kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:46<3:58:26, 24.9kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:46<2:45:34, 35.5kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:48<1:59:02, 49.2kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:48<1:23:58, 69.8kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:48<58:14, 99.6kB/s]  .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:50<50:21, 115kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:50<35:55, 161kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:50<24:57, 230kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:52<22:40, 252kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:52<16:35, 345kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:52<11:39, 488kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:52<08:14, 687kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:54<09:39, 585kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:54<07:20, 770kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:54<05:12, 1.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:56<05:06, 1.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:56<04:48, 1.16MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:56<03:25, 1.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:58<03:49, 1.44MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:58<03:18, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:58<02:20, 2.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:59<06:07, 890kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:00<04:52, 1.12MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [04:00<03:25, 1.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [04:02<07:55, 678kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:02<06:19, 849kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:02<04:27, 1.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:04<05:41, 934kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:04<04:24, 1.20MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:04<03:05, 1.70MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:06<23:22, 224kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:06<16:56, 309kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:06<11:46, 440kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:08<14:12, 364kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:08<10:31, 491kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:10<08:00, 637kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:10<06:09, 828kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:12<04:59, 1.01MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:12<04:03, 1.24MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:14<03:31, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:14<03:01, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:16<02:47, 1.75MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:16<02:29, 1.97MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:18<02:25, 2.00MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:18<02:12, 2.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:18<01:34, 3.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:20<04:22, 1.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:20<03:34, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:22<03:09, 1.49MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:22<02:39, 1.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:22<01:53, 2.46MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:24<05:12, 888kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:24<04:03, 1.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:24<02:51, 1.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:26<04:11, 1.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:26<03:25, 1.33MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:26<02:25, 1.85MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:28<03:32, 1.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:28<02:45, 1.63MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:28<01:57, 2.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:30<03:31, 1.25MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:30<02:50, 1.55MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:30<02:03, 2.13MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:32<02:37, 1.66MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:32<02:34, 1.69MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:32<01:52, 2.31MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:32<01:24, 3.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:32<01:04, 3.98MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:34<03:24, 1.25MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:34<02:47, 1.52MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:35<01:58, 2.13MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:36<05:02, 831kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:36<04:06, 1.02MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:37<02:52, 1.43MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:38<04:47, 860kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:38<03:57, 1.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:39<02:46, 1.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:40<04:58, 813kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:40<04:03, 994kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:40<02:50, 1.40MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:42<05:22, 741kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:42<04:06, 968kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:44<03:24, 1.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:44<02:46, 1.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:44<01:56, 1.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:46<07:43, 497kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:46<05:57, 643kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:46<04:08, 911kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:49<04:12, 891kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:49<03:26, 1.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:49<02:28, 1.50MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:49<01:48, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:51<02:39, 1.38MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:51<02:11, 1.68MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:51<01:34, 2.31MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:53<02:05, 1.73MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:53<02:14, 1.61MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:53<01:45, 2.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:53<01:18, 2.73MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:53<00:58, 3.63MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:55<03:44, 945kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:55<03:00, 1.18MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:55<02:08, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:55<01:35, 2.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:57<03:00, 1.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:57<02:36, 1.33MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:57<01:52, 1.83MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:57<01:24, 2.44MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:59<02:16, 1.50MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:59<02:05, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:59<01:35, 2.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:59<01:11, 2.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:01<01:48, 1.84MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:01<01:46, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:01<01:15, 2.62MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:03<03:14, 1.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:03<02:44, 1.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:03<01:55, 1.67MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:05<03:15, 980kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:05<02:42, 1.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:05<01:53, 1.65MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:07<03:27, 905kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:07<02:38, 1.18MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:07<01:51, 1.66MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:10<02:04, 1.45MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:10<01:55, 1.57MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:10<01:20, 2.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:12<02:40, 1.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:12<02:17, 1.29MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:12<01:36, 1.81MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:14<03:11, 901kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:14<02:38, 1.09MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:14<01:51, 1.53MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:16<02:27, 1.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:16<02:07, 1.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:16<01:29, 1.85MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:18<02:55, 940kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:18<02:42, 1.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:18<01:56, 1.40MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:20<01:55, 1.39MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:20<01:43, 1.55MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:20<01:12, 2.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:22<02:29, 1.04MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:22<02:07, 1.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:22<01:29, 1.71MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:24<02:00, 1.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:24<01:50, 1.38MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:24<01:18, 1.91MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:26<01:33, 1.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:26<02:03, 1.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:26<02:16, 1.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:26<01:39, 1.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:28<01:31, 1.57MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:28<02:06, 1.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:28<02:29, 961kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:28<02:00, 1.19MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:28<01:26, 1.64MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:28<01:03, 2.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:30<01:48, 1.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:30<02:20, 996kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:30<02:04, 1.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:30<01:28, 1.56MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:32<01:33, 1.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:32<01:47, 1.26MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:32<01:18, 1.72MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:34<01:17, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:34<01:08, 1.91MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:34<00:48, 2.63MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:36<01:21, 1.56MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:36<01:20, 1.58MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:36<00:57, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:38<01:09, 1.77MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:38<01:03, 1.93MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:38<00:44, 2.69MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:40<01:54, 1.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:40<01:33, 1.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:40<01:04, 1.80MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:42<02:36, 737kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:42<02:06, 911kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:42<01:26, 1.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:44<02:08, 866kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:44<01:44, 1.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:44<01:11, 1.50MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:46<02:47, 639kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:46<02:06, 843kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:46<01:26, 1.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:48<02:39, 643kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:48<02:04, 822kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:48<01:25, 1.16MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:50<02:57, 555kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:50<02:45, 594kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:50<02:05, 781kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:50<01:26, 1.10MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:52<01:50, 851kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:52<01:32, 1.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:52<01:06, 1.39MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:54<01:02, 1.45MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:54<00:57, 1.57MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:54<00:41, 2.14MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:54<00:30, 2.83MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:56<00:52, 1.63MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:56<01:03, 1.35MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:56<00:52, 1.64MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:56<00:38, 2.19MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:58<00:41, 1.98MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:58<00:59, 1.37MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:58<00:56, 1.45MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:58<00:40, 2.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:00<00:44, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:00<01:01, 1.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:00<01:05, 1.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:00<00:49, 1.56MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [06:00<00:34, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:02<00:50, 1.45MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [06:02<00:41, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:02<00:28, 2.45MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:05<04:54, 237kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:05<03:35, 323kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:05<02:29, 456kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:05<01:43, 643kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:07<01:43, 635kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:07<01:18, 832kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:07<00:53, 1.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:10<01:40, 613kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:10<01:17, 791kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:10<00:53, 1.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:11<00:47, 1.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:12<00:35, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:12<00:31, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:12<00:29, 1.88MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:12<00:24, 2.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:12<00:18, 2.93MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:12<00:14, 3.62MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:12<00:12, 4.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:14<01:08, 776kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:14<00:55, 955kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:14<00:39, 1.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:14<00:29, 1.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:14<00:22, 2.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:14<00:17, 2.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:14<00:14, 3.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:15<05:56, 137kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:16<04:15, 190kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:16<02:57, 269kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:16<02:03, 379kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:16<01:26, 530kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:16<01:03, 718kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:16<00:45, 987kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:17<01:54, 391kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:18<01:38, 453kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:18<01:15, 589kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:18<01:03, 693kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:18<00:56, 786kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:18<00:48, 899kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:18<00:43, 1.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:18<00:39, 1.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:18<00:35, 1.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:18<00:28, 1.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:18<00:21, 1.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:19<00:16, 2.59MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:19<00:21, 1.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:20<00:26, 1.54MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:20<00:26, 1.51MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:20<00:26, 1.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:20<00:31, 1.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:20<00:29, 1.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:20<00:26, 1.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:20<00:20, 1.92MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:20<00:15, 2.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:20<00:11, 3.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:22<00:32, 1.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:22<00:25, 1.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:22<00:18, 1.89MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:22<00:13, 2.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:22<00:10, 3.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:24<00:23, 1.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:24<00:21, 1.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:24<00:15, 2.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:24<00:11, 2.65MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:24<00:08, 3.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:25<00:09, 3.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:25<00:08, 3.28MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:26<00:28, 999kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:26<00:23, 1.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:26<00:16, 1.60MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:26<00:12, 2.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:26<00:08, 2.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:27<00:07, 3.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:28<01:03, 380kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:28<00:48, 493kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:28<00:33, 688kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:28<00:23, 952kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:28<00:15, 1.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:28<00:11, 1.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:29<00:09, 2.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:30<05:36, 59.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:30<03:56, 83.2kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:30<02:37, 118kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:30<01:44, 168kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:30<01:09, 239kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:30<00:46, 337kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:32<02:33, 103kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:32<01:47, 144kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:32<01:11, 205kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:32<00:46, 290kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:32<00:30, 410kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:32<00:20, 573kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:34<03:41, 52.4kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:34<02:33, 74.2kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:34<01:37, 106kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:34<00:59, 150kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:34<00:36, 214kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:36<00:41, 182kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:36<00:29, 251kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:36<00:17, 355kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:36<00:09, 501kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:36<00:05, 704kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:38<00:10, 330kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:38<00:06, 450kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:38<00:03, 630kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:38<00:00, 881kB/s].vector_cache/glove.6B.zip: 862MB [06:38, 2.16MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 959/400000 [00:00<00:41, 9580.80it/s]  0%|          | 1936/400000 [00:00<00:41, 9633.40it/s]  1%|          | 2923/400000 [00:00<00:40, 9703.01it/s]  1%|          | 3886/400000 [00:00<00:40, 9680.56it/s]  1%|          | 4872/400000 [00:00<00:40, 9733.01it/s]  1%|▏         | 5832/400000 [00:00<00:40, 9690.55it/s]  2%|▏         | 6822/400000 [00:00<00:40, 9750.68it/s]  2%|▏         | 7784/400000 [00:00<00:40, 9710.70it/s]  2%|▏         | 8756/400000 [00:00<00:40, 9712.73it/s]  2%|▏         | 9707/400000 [00:01<00:40, 9650.90it/s]  3%|▎         | 10646/400000 [00:01<00:40, 9526.76it/s]  3%|▎         | 11602/400000 [00:01<00:40, 9534.14it/s]  3%|▎         | 12573/400000 [00:01<00:40, 9585.43it/s]  3%|▎         | 13528/400000 [00:01<00:40, 9571.86it/s]  4%|▎         | 14479/400000 [00:01<00:40, 9485.07it/s]  4%|▍         | 15424/400000 [00:01<00:41, 9320.40it/s]  4%|▍         | 16401/400000 [00:01<00:40, 9449.07it/s]  4%|▍         | 17374/400000 [00:01<00:40, 9529.71it/s]  5%|▍         | 18347/400000 [00:01<00:39, 9586.68it/s]  5%|▍         | 19306/400000 [00:02<00:39, 9541.15it/s]  5%|▌         | 20268/400000 [00:02<00:39, 9562.97it/s]  5%|▌         | 21225/400000 [00:02<00:39, 9555.86it/s]  6%|▌         | 22195/400000 [00:02<00:39, 9596.15it/s]  6%|▌         | 23190/400000 [00:02<00:38, 9698.11it/s]  6%|▌         | 24186/400000 [00:02<00:38, 9773.44it/s]  6%|▋         | 25175/400000 [00:02<00:38, 9807.36it/s]  7%|▋         | 26156/400000 [00:02<00:38, 9772.96it/s]  7%|▋         | 27134/400000 [00:02<00:38, 9769.27it/s]  7%|▋         | 28112/400000 [00:02<00:39, 9461.28it/s]  7%|▋         | 29061/400000 [00:03<00:40, 9235.65it/s]  7%|▋         | 29988/400000 [00:03<00:40, 9116.37it/s]  8%|▊         | 30928/400000 [00:03<00:40, 9197.87it/s]  8%|▊         | 31869/400000 [00:03<00:39, 9258.69it/s]  8%|▊         | 32797/400000 [00:03<00:39, 9245.06it/s]  8%|▊         | 33759/400000 [00:03<00:39, 9351.52it/s]  9%|▊         | 34696/400000 [00:03<00:40, 9123.82it/s]  9%|▉         | 35611/400000 [00:03<00:40, 9065.05it/s]  9%|▉         | 36519/400000 [00:03<00:41, 8658.55it/s]  9%|▉         | 37390/400000 [00:03<00:42, 8586.38it/s] 10%|▉         | 38280/400000 [00:04<00:41, 8677.48it/s] 10%|▉         | 39151/400000 [00:04<00:41, 8678.75it/s] 10%|█         | 40021/400000 [00:04<00:41, 8641.93it/s] 10%|█         | 40887/400000 [00:04<00:41, 8602.37it/s] 10%|█         | 41817/400000 [00:04<00:40, 8800.20it/s] 11%|█         | 42782/400000 [00:04<00:39, 9038.79it/s] 11%|█         | 43744/400000 [00:04<00:38, 9204.90it/s] 11%|█         | 44721/400000 [00:04<00:37, 9364.63it/s] 11%|█▏        | 45705/400000 [00:04<00:37, 9501.87it/s] 12%|█▏        | 46658/400000 [00:04<00:37, 9343.84it/s] 12%|█▏        | 47595/400000 [00:05<00:38, 9194.49it/s] 12%|█▏        | 48517/400000 [00:05<00:38, 9103.91it/s] 12%|█▏        | 49484/400000 [00:05<00:37, 9265.63it/s] 13%|█▎        | 50461/400000 [00:05<00:37, 9409.79it/s] 13%|█▎        | 51438/400000 [00:05<00:36, 9513.74it/s] 13%|█▎        | 52438/400000 [00:05<00:36, 9653.35it/s] 13%|█▎        | 53405/400000 [00:05<00:37, 9333.67it/s] 14%|█▎        | 54342/400000 [00:05<00:37, 9131.28it/s] 14%|█▍        | 55290/400000 [00:05<00:37, 9231.91it/s] 14%|█▍        | 56308/400000 [00:06<00:36, 9496.49it/s] 14%|█▍        | 57262/400000 [00:06<00:36, 9292.05it/s] 15%|█▍        | 58195/400000 [00:06<00:37, 9091.53it/s] 15%|█▍        | 59108/400000 [00:06<00:37, 9087.62it/s] 15%|█▌        | 60020/400000 [00:06<00:37, 9000.56it/s] 15%|█▌        | 60922/400000 [00:06<00:38, 8709.51it/s] 15%|█▌        | 61797/400000 [00:06<00:40, 8440.51it/s] 16%|█▌        | 62646/400000 [00:06<00:40, 8352.64it/s] 16%|█▌        | 63530/400000 [00:06<00:39, 8491.58it/s] 16%|█▌        | 64413/400000 [00:06<00:39, 8590.18it/s] 16%|█▋        | 65341/400000 [00:07<00:38, 8785.99it/s] 17%|█▋        | 66268/400000 [00:07<00:37, 8922.20it/s] 17%|█▋        | 67196/400000 [00:07<00:36, 9025.05it/s] 17%|█▋        | 68167/400000 [00:07<00:36, 9202.77it/s] 17%|█▋        | 69115/400000 [00:07<00:35, 9282.53it/s] 18%|█▊        | 70045/400000 [00:07<00:35, 9234.65it/s] 18%|█▊        | 70981/400000 [00:07<00:35, 9271.64it/s] 18%|█▊        | 71910/400000 [00:07<00:35, 9180.64it/s] 18%|█▊        | 72829/400000 [00:07<00:35, 9114.64it/s] 18%|█▊        | 73757/400000 [00:07<00:35, 9163.30it/s] 19%|█▊        | 74693/400000 [00:08<00:35, 9221.25it/s] 19%|█▉        | 75636/400000 [00:08<00:34, 9282.37it/s] 19%|█▉        | 76565/400000 [00:08<00:35, 9219.55it/s] 19%|█▉        | 77497/400000 [00:08<00:34, 9248.42it/s] 20%|█▉        | 78436/400000 [00:08<00:34, 9288.84it/s] 20%|█▉        | 79399/400000 [00:08<00:34, 9386.88it/s] 20%|██        | 80348/400000 [00:08<00:33, 9416.56it/s] 20%|██        | 81290/400000 [00:08<00:34, 9178.19it/s] 21%|██        | 82210/400000 [00:08<00:35, 8913.16it/s] 21%|██        | 83120/400000 [00:08<00:35, 8966.90it/s] 21%|██        | 84035/400000 [00:09<00:35, 9018.19it/s] 21%|██        | 84939/400000 [00:09<00:35, 8968.79it/s] 21%|██▏       | 85837/400000 [00:09<00:35, 8869.30it/s] 22%|██▏       | 86725/400000 [00:09<00:36, 8649.98it/s] 22%|██▏       | 87615/400000 [00:09<00:35, 8720.90it/s] 22%|██▏       | 88519/400000 [00:09<00:35, 8813.71it/s] 22%|██▏       | 89404/400000 [00:09<00:35, 8822.78it/s] 23%|██▎       | 90288/400000 [00:09<00:35, 8818.02it/s] 23%|██▎       | 91188/400000 [00:09<00:34, 8869.91it/s] 23%|██▎       | 92076/400000 [00:10<00:35, 8790.68it/s] 23%|██▎       | 92956/400000 [00:10<00:34, 8788.17it/s] 23%|██▎       | 93863/400000 [00:10<00:34, 8870.51it/s] 24%|██▎       | 94778/400000 [00:10<00:34, 8950.71it/s] 24%|██▍       | 95679/400000 [00:10<00:33, 8965.84it/s] 24%|██▍       | 96615/400000 [00:10<00:33, 9077.83it/s] 24%|██▍       | 97546/400000 [00:10<00:33, 9146.22it/s] 25%|██▍       | 98462/400000 [00:10<00:32, 9138.66it/s] 25%|██▍       | 99377/400000 [00:10<00:32, 9140.91it/s] 25%|██▌       | 100323/400000 [00:10<00:32, 9233.89it/s] 25%|██▌       | 101247/400000 [00:11<00:32, 9204.54it/s] 26%|██▌       | 102200/400000 [00:11<00:32, 9297.76it/s] 26%|██▌       | 103131/400000 [00:11<00:32, 9187.83it/s] 26%|██▌       | 104051/400000 [00:11<00:32, 9036.28it/s] 26%|██▋       | 105029/400000 [00:11<00:31, 9245.30it/s] 26%|██▋       | 105956/400000 [00:11<00:31, 9248.59it/s] 27%|██▋       | 106883/400000 [00:11<00:31, 9184.96it/s] 27%|██▋       | 107803/400000 [00:11<00:32, 8964.46it/s] 27%|██▋       | 108702/400000 [00:11<00:32, 8865.83it/s] 27%|██▋       | 109602/400000 [00:11<00:32, 8904.68it/s] 28%|██▊       | 110539/400000 [00:12<00:32, 9036.79it/s] 28%|██▊       | 111459/400000 [00:12<00:31, 9083.84it/s] 28%|██▊       | 112387/400000 [00:12<00:31, 9138.76it/s] 28%|██▊       | 113302/400000 [00:12<00:31, 9139.51it/s] 29%|██▊       | 114224/400000 [00:12<00:31, 9162.48it/s] 29%|██▉       | 115141/400000 [00:12<00:31, 9018.49it/s] 29%|██▉       | 116044/400000 [00:12<00:31, 8997.07it/s] 29%|██▉       | 117005/400000 [00:12<00:30, 9170.68it/s] 29%|██▉       | 117935/400000 [00:12<00:30, 9208.22it/s] 30%|██▉       | 118857/400000 [00:12<00:30, 9159.03it/s] 30%|██▉       | 119774/400000 [00:13<00:30, 9041.66it/s] 30%|███       | 120679/400000 [00:13<00:31, 8994.58it/s] 30%|███       | 121580/400000 [00:13<00:31, 8953.83it/s] 31%|███       | 122476/400000 [00:13<00:31, 8870.52it/s] 31%|███       | 123364/400000 [00:13<00:31, 8748.39it/s] 31%|███       | 124301/400000 [00:13<00:30, 8925.25it/s] 31%|███▏      | 125260/400000 [00:13<00:30, 9113.47it/s] 32%|███▏      | 126174/400000 [00:13<00:30, 9069.38it/s] 32%|███▏      | 127083/400000 [00:13<00:30, 9005.84it/s] 32%|███▏      | 127985/400000 [00:13<00:30, 8977.96it/s] 32%|███▏      | 128886/400000 [00:14<00:30, 8984.75it/s] 32%|███▏      | 129834/400000 [00:14<00:29, 9125.77it/s] 33%|███▎      | 130757/400000 [00:14<00:29, 9154.63it/s] 33%|███▎      | 131674/400000 [00:14<00:29, 8991.41it/s] 33%|███▎      | 132575/400000 [00:14<00:30, 8824.33it/s] 33%|███▎      | 133516/400000 [00:14<00:29, 8989.49it/s] 34%|███▎      | 134468/400000 [00:14<00:29, 9140.40it/s] 34%|███▍      | 135397/400000 [00:14<00:28, 9182.43it/s] 34%|███▍      | 136320/400000 [00:14<00:28, 9195.29it/s] 34%|███▍      | 137252/400000 [00:14<00:28, 9230.86it/s] 35%|███▍      | 138176/400000 [00:15<00:28, 9161.74it/s] 35%|███▍      | 139093/400000 [00:15<00:28, 9039.01it/s] 35%|███▌      | 140017/400000 [00:15<00:28, 9096.61it/s] 35%|███▌      | 140943/400000 [00:15<00:28, 9142.60it/s] 35%|███▌      | 141874/400000 [00:15<00:28, 9190.19it/s] 36%|███▌      | 142800/400000 [00:15<00:27, 9209.02it/s] 36%|███▌      | 143786/400000 [00:15<00:27, 9394.39it/s] 36%|███▌      | 144727/400000 [00:15<00:27, 9377.91it/s] 36%|███▋      | 145666/400000 [00:15<00:27, 9373.37it/s] 37%|███▋      | 146611/400000 [00:15<00:26, 9394.74it/s] 37%|███▋      | 147574/400000 [00:16<00:26, 9461.42it/s] 37%|███▋      | 148521/400000 [00:16<00:26, 9387.92it/s] 37%|███▋      | 149461/400000 [00:16<00:26, 9343.21it/s] 38%|███▊      | 150402/400000 [00:16<00:26, 9361.15it/s] 38%|███▊      | 151339/400000 [00:16<00:26, 9212.80it/s] 38%|███▊      | 152305/400000 [00:16<00:26, 9341.78it/s] 38%|███▊      | 153311/400000 [00:16<00:25, 9545.30it/s] 39%|███▊      | 154268/400000 [00:16<00:25, 9490.77it/s] 39%|███▉      | 155219/400000 [00:16<00:25, 9433.20it/s] 39%|███▉      | 156164/400000 [00:17<00:26, 9352.46it/s] 39%|███▉      | 157101/400000 [00:17<00:26, 9167.04it/s] 40%|███▉      | 158028/400000 [00:17<00:26, 9194.82it/s] 40%|███▉      | 159005/400000 [00:17<00:25, 9357.33it/s] 40%|███▉      | 159943/400000 [00:17<00:25, 9347.30it/s] 40%|████      | 160913/400000 [00:17<00:25, 9449.66it/s] 40%|████      | 161880/400000 [00:17<00:25, 9512.69it/s] 41%|████      | 162902/400000 [00:17<00:24, 9712.37it/s] 41%|████      | 163931/400000 [00:17<00:23, 9877.86it/s] 41%|████      | 164921/400000 [00:17<00:24, 9776.33it/s] 41%|████▏     | 165901/400000 [00:18<00:24, 9735.32it/s] 42%|████▏     | 166876/400000 [00:18<00:24, 9640.55it/s] 42%|████▏     | 167860/400000 [00:18<00:23, 9696.72it/s] 42%|████▏     | 168890/400000 [00:18<00:23, 9868.38it/s] 42%|████▏     | 169879/400000 [00:18<00:23, 9724.60it/s] 43%|████▎     | 170853/400000 [00:18<00:24, 9471.84it/s] 43%|████▎     | 171803/400000 [00:18<00:24, 9465.83it/s] 43%|████▎     | 172801/400000 [00:18<00:23, 9612.61it/s] 43%|████▎     | 173764/400000 [00:18<00:23, 9582.13it/s] 44%|████▎     | 174724/400000 [00:18<00:24, 9351.42it/s] 44%|████▍     | 175662/400000 [00:19<00:24, 9298.25it/s] 44%|████▍     | 176655/400000 [00:19<00:23, 9477.92it/s] 44%|████▍     | 177671/400000 [00:19<00:22, 9671.88it/s] 45%|████▍     | 178642/400000 [00:19<00:22, 9681.04it/s] 45%|████▍     | 179612/400000 [00:19<00:22, 9642.88it/s] 45%|████▌     | 180594/400000 [00:19<00:22, 9693.73it/s] 45%|████▌     | 181565/400000 [00:19<00:22, 9688.49it/s] 46%|████▌     | 182555/400000 [00:19<00:22, 9749.86it/s] 46%|████▌     | 183531/400000 [00:19<00:22, 9665.42it/s] 46%|████▌     | 184499/400000 [00:19<00:22, 9651.11it/s] 46%|████▋     | 185496/400000 [00:20<00:22, 9741.42it/s] 47%|████▋     | 186471/400000 [00:20<00:21, 9729.90it/s] 47%|████▋     | 187445/400000 [00:20<00:22, 9575.23it/s] 47%|████▋     | 188421/400000 [00:20<00:21, 9629.73it/s] 47%|████▋     | 189385/400000 [00:20<00:22, 9499.68it/s] 48%|████▊     | 190369/400000 [00:20<00:21, 9598.88it/s] 48%|████▊     | 191367/400000 [00:20<00:21, 9709.93it/s] 48%|████▊     | 192366/400000 [00:20<00:21, 9790.04it/s] 48%|████▊     | 193360/400000 [00:20<00:21, 9833.18it/s] 49%|████▊     | 194344/400000 [00:20<00:20, 9802.64it/s] 49%|████▉     | 195325/400000 [00:21<00:21, 9420.67it/s] 49%|████▉     | 196324/400000 [00:21<00:21, 9583.74it/s] 49%|████▉     | 197317/400000 [00:21<00:20, 9684.39it/s] 50%|████▉     | 198338/400000 [00:21<00:20, 9833.02it/s] 50%|████▉     | 199324/400000 [00:21<00:20, 9609.28it/s] 50%|█████     | 200288/400000 [00:21<00:21, 9493.35it/s] 50%|█████     | 201291/400000 [00:21<00:20, 9648.05it/s] 51%|█████     | 202258/400000 [00:21<00:20, 9593.22it/s] 51%|█████     | 203219/400000 [00:21<00:20, 9376.09it/s] 51%|█████     | 204159/400000 [00:22<00:21, 9054.74it/s] 51%|█████▏    | 205069/400000 [00:22<00:21, 8861.15it/s] 51%|█████▏    | 205970/400000 [00:22<00:21, 8902.80it/s] 52%|█████▏    | 206863/400000 [00:22<00:21, 8850.95it/s] 52%|█████▏    | 207766/400000 [00:22<00:21, 8901.39it/s] 52%|█████▏    | 208658/400000 [00:22<00:21, 8867.44it/s] 52%|█████▏    | 209591/400000 [00:22<00:21, 9000.17it/s] 53%|█████▎    | 210620/400000 [00:22<00:20, 9350.54it/s] 53%|█████▎    | 211592/400000 [00:22<00:19, 9457.06it/s] 53%|█████▎    | 212609/400000 [00:22<00:19, 9658.00it/s] 53%|█████▎    | 213579/400000 [00:23<00:19, 9484.62it/s] 54%|█████▎    | 214543/400000 [00:23<00:19, 9529.44it/s] 54%|█████▍    | 215530/400000 [00:23<00:19, 9628.34it/s] 54%|█████▍    | 216495/400000 [00:23<00:19, 9592.45it/s] 54%|█████▍    | 217456/400000 [00:23<00:19, 9457.88it/s] 55%|█████▍    | 218404/400000 [00:23<00:19, 9292.13it/s] 55%|█████▍    | 219372/400000 [00:23<00:19, 9404.32it/s] 55%|█████▌    | 220354/400000 [00:23<00:18, 9524.54it/s] 55%|█████▌    | 221344/400000 [00:23<00:18, 9631.84it/s] 56%|█████▌    | 222309/400000 [00:23<00:18, 9612.03it/s] 56%|█████▌    | 223298/400000 [00:24<00:18, 9692.61it/s] 56%|█████▌    | 224299/400000 [00:24<00:17, 9785.56it/s] 56%|█████▋    | 225343/400000 [00:24<00:17, 9972.87it/s] 57%|█████▋    | 226367/400000 [00:24<00:17, 10050.96it/s] 57%|█████▋    | 227374/400000 [00:24<00:18, 9576.22it/s]  57%|█████▋    | 228387/400000 [00:24<00:17, 9735.69it/s] 57%|█████▋    | 229410/400000 [00:24<00:17, 9878.58it/s] 58%|█████▊    | 230466/400000 [00:24<00:16, 10072.92it/s] 58%|█████▊    | 231477/400000 [00:24<00:16, 10057.31it/s] 58%|█████▊    | 232494/400000 [00:24<00:16, 10089.61it/s] 58%|█████▊    | 233505/400000 [00:25<00:16, 9842.36it/s]  59%|█████▊    | 234591/400000 [00:25<00:16, 10124.36it/s] 59%|█████▉    | 235608/400000 [00:25<00:16, 9867.00it/s]  59%|█████▉    | 236599/400000 [00:25<00:16, 9816.41it/s] 59%|█████▉    | 237584/400000 [00:25<00:16, 9696.51it/s] 60%|█████▉    | 238604/400000 [00:25<00:16, 9840.94it/s] 60%|█████▉    | 239597/400000 [00:25<00:16, 9866.62it/s] 60%|██████    | 240616/400000 [00:25<00:16, 9960.57it/s] 60%|██████    | 241646/400000 [00:25<00:15, 10057.36it/s] 61%|██████    | 242653/400000 [00:25<00:15, 10041.20it/s] 61%|██████    | 243658/400000 [00:26<00:15, 9930.96it/s]  61%|██████    | 244652/400000 [00:26<00:16, 9687.90it/s] 61%|██████▏   | 245623/400000 [00:26<00:16, 9637.86it/s] 62%|██████▏   | 246663/400000 [00:26<00:15, 9854.20it/s] 62%|██████▏   | 247651/400000 [00:26<00:15, 9706.45it/s] 62%|██████▏   | 248624/400000 [00:26<00:16, 9351.34it/s] 62%|██████▏   | 249640/400000 [00:26<00:15, 9577.98it/s] 63%|██████▎   | 250635/400000 [00:26<00:15, 9685.02it/s] 63%|██████▎   | 251674/400000 [00:26<00:15, 9882.93it/s] 63%|██████▎   | 252666/400000 [00:27<00:15, 9668.86it/s] 63%|██████▎   | 253637/400000 [00:27<00:15, 9638.28it/s] 64%|██████▎   | 254646/400000 [00:27<00:14, 9768.59it/s] 64%|██████▍   | 255672/400000 [00:27<00:14, 9907.61it/s] 64%|██████▍   | 256665/400000 [00:27<00:14, 9811.99it/s] 64%|██████▍   | 257648/400000 [00:27<00:14, 9577.09it/s] 65%|██████▍   | 258609/400000 [00:27<00:15, 9322.21it/s] 65%|██████▍   | 259545/400000 [00:27<00:15, 9237.55it/s] 65%|██████▌   | 260517/400000 [00:27<00:14, 9375.15it/s] 65%|██████▌   | 261576/400000 [00:27<00:14, 9708.33it/s] 66%|██████▌   | 262552/400000 [00:28<00:14, 9550.00it/s] 66%|██████▌   | 263542/400000 [00:28<00:14, 9651.81it/s] 66%|██████▌   | 264511/400000 [00:28<00:14, 9595.34it/s] 66%|██████▋   | 265535/400000 [00:28<00:13, 9778.57it/s] 67%|██████▋   | 266558/400000 [00:28<00:13, 9908.59it/s] 67%|██████▋   | 267551/400000 [00:28<00:13, 9860.28it/s] 67%|██████▋   | 268569/400000 [00:28<00:13, 9953.91it/s] 67%|██████▋   | 269566/400000 [00:28<00:13, 9878.46it/s] 68%|██████▊   | 270607/400000 [00:28<00:12, 10030.33it/s] 68%|██████▊   | 271612/400000 [00:28<00:12, 10001.73it/s] 68%|██████▊   | 272614/400000 [00:29<00:12, 9844.47it/s]  68%|██████▊   | 273626/400000 [00:29<00:12, 9923.78it/s] 69%|██████▊   | 274657/400000 [00:29<00:12, 10036.14it/s] 69%|██████▉   | 275666/400000 [00:29<00:12, 10048.90it/s] 69%|██████▉   | 276672/400000 [00:29<00:12, 9669.15it/s]  69%|██████▉   | 277658/400000 [00:29<00:12, 9722.81it/s] 70%|██████▉   | 278633/400000 [00:29<00:12, 9713.15it/s] 70%|██████▉   | 279658/400000 [00:29<00:12, 9865.39it/s] 70%|███████   | 280679/400000 [00:29<00:11, 9965.42it/s] 70%|███████   | 281724/400000 [00:29<00:11, 10105.74it/s] 71%|███████   | 282737/400000 [00:30<00:11, 10001.64it/s] 71%|███████   | 283739/400000 [00:30<00:11, 10000.60it/s] 71%|███████   | 284744/400000 [00:30<00:11, 10012.96it/s] 71%|███████▏  | 285746/400000 [00:30<00:11, 9605.79it/s]  72%|███████▏  | 286732/400000 [00:30<00:11, 9678.82it/s] 72%|███████▏  | 287744/400000 [00:30<00:11, 9807.00it/s] 72%|███████▏  | 288728/400000 [00:30<00:11, 9688.27it/s] 72%|███████▏  | 289724/400000 [00:30<00:11, 9767.26it/s] 73%|███████▎  | 290722/400000 [00:30<00:11, 9828.07it/s] 73%|███████▎  | 291735/400000 [00:31<00:10, 9914.54it/s] 73%|███████▎  | 292728/400000 [00:31<00:11, 9734.96it/s] 73%|███████▎  | 293703/400000 [00:31<00:10, 9699.42it/s] 74%|███████▎  | 294677/400000 [00:31<00:10, 9710.71it/s] 74%|███████▍  | 295730/400000 [00:31<00:10, 9940.72it/s] 74%|███████▍  | 296726/400000 [00:31<00:10, 9908.16it/s] 74%|███████▍  | 297719/400000 [00:31<00:10, 9865.66it/s] 75%|███████▍  | 298707/400000 [00:31<00:10, 9789.93it/s] 75%|███████▍  | 299687/400000 [00:31<00:10, 9635.57it/s] 75%|███████▌  | 300652/400000 [00:31<00:10, 9599.57it/s] 75%|███████▌  | 301632/400000 [00:32<00:10, 9656.64it/s] 76%|███████▌  | 302599/400000 [00:32<00:10, 9531.99it/s] 76%|███████▌  | 303554/400000 [00:32<00:10, 9386.56it/s] 76%|███████▌  | 304494/400000 [00:32<00:10, 9174.16it/s] 76%|███████▋  | 305500/400000 [00:32<00:10, 9422.31it/s] 77%|███████▋  | 306501/400000 [00:32<00:09, 9589.50it/s] 77%|███████▋  | 307463/400000 [00:32<00:10, 9246.61it/s] 77%|███████▋  | 308393/400000 [00:32<00:10, 9102.84it/s] 77%|███████▋  | 309423/400000 [00:32<00:09, 9430.88it/s] 78%|███████▊  | 310477/400000 [00:32<00:09, 9736.22it/s] 78%|███████▊  | 311458/400000 [00:33<00:09, 9686.54it/s] 78%|███████▊  | 312432/400000 [00:33<00:09, 9525.53it/s] 78%|███████▊  | 313421/400000 [00:33<00:09, 9614.37it/s] 79%|███████▊  | 314458/400000 [00:33<00:08, 9828.29it/s] 79%|███████▉  | 315444/400000 [00:33<00:08, 9655.94it/s] 79%|███████▉  | 316447/400000 [00:33<00:08, 9763.44it/s] 79%|███████▉  | 317426/400000 [00:33<00:08, 9652.71it/s] 80%|███████▉  | 318394/400000 [00:33<00:08, 9364.94it/s] 80%|███████▉  | 319334/400000 [00:33<00:08, 9209.71it/s] 80%|████████  | 320258/400000 [00:34<00:08, 9121.00it/s] 80%|████████  | 321264/400000 [00:34<00:08, 9382.08it/s] 81%|████████  | 322206/400000 [00:34<00:08, 9348.52it/s] 81%|████████  | 323173/400000 [00:34<00:08, 9440.64it/s] 81%|████████  | 324198/400000 [00:34<00:07, 9669.52it/s] 81%|████████▏ | 325168/400000 [00:34<00:07, 9363.77it/s] 82%|████████▏ | 326109/400000 [00:34<00:07, 9355.77it/s] 82%|████████▏ | 327048/400000 [00:34<00:07, 9311.23it/s] 82%|████████▏ | 327989/400000 [00:34<00:07, 9339.57it/s] 82%|████████▏ | 328925/400000 [00:34<00:07, 9340.46it/s] 82%|████████▏ | 329861/400000 [00:35<00:07, 9314.35it/s] 83%|████████▎ | 330795/400000 [00:35<00:07, 9319.64it/s] 83%|████████▎ | 331728/400000 [00:35<00:07, 9268.79it/s] 83%|████████▎ | 332656/400000 [00:35<00:07, 9236.18it/s] 83%|████████▎ | 333598/400000 [00:35<00:07, 9289.69it/s] 84%|████████▎ | 334530/400000 [00:35<00:07, 9297.54it/s] 84%|████████▍ | 335525/400000 [00:35<00:06, 9483.19it/s] 84%|████████▍ | 336475/400000 [00:35<00:06, 9474.24it/s] 84%|████████▍ | 337424/400000 [00:35<00:06, 9380.13it/s] 85%|████████▍ | 338363/400000 [00:35<00:06, 9248.48it/s] 85%|████████▍ | 339327/400000 [00:36<00:06, 9362.44it/s] 85%|████████▌ | 340265/400000 [00:36<00:06, 9165.54it/s] 85%|████████▌ | 341184/400000 [00:36<00:06, 9023.88it/s] 86%|████████▌ | 342127/400000 [00:36<00:06, 9140.91it/s] 86%|████████▌ | 343113/400000 [00:36<00:06, 9342.84it/s] 86%|████████▌ | 344075/400000 [00:36<00:05, 9423.96it/s] 86%|████████▋ | 345034/400000 [00:36<00:05, 9471.87it/s] 86%|████████▋ | 345983/400000 [00:36<00:05, 9373.22it/s] 87%|████████▋ | 346922/400000 [00:36<00:05, 9376.92it/s] 87%|████████▋ | 347890/400000 [00:36<00:05, 9464.11it/s] 87%|████████▋ | 348838/400000 [00:37<00:05, 9410.63it/s] 87%|████████▋ | 349780/400000 [00:37<00:05, 9154.25it/s] 88%|████████▊ | 350720/400000 [00:37<00:05, 9223.80it/s] 88%|████████▊ | 351696/400000 [00:37<00:05, 9376.68it/s] 88%|████████▊ | 352649/400000 [00:37<00:05, 9420.63it/s] 88%|████████▊ | 353593/400000 [00:37<00:04, 9405.20it/s] 89%|████████▊ | 354544/400000 [00:37<00:04, 9433.60it/s] 89%|████████▉ | 355488/400000 [00:37<00:04, 9358.21it/s] 89%|████████▉ | 356425/400000 [00:37<00:04, 9233.12it/s] 89%|████████▉ | 357350/400000 [00:37<00:04, 9150.94it/s] 90%|████████▉ | 358276/400000 [00:38<00:04, 9182.72it/s] 90%|████████▉ | 359252/400000 [00:38<00:04, 9346.25it/s] 90%|█████████ | 360188/400000 [00:38<00:04, 9325.86it/s] 90%|█████████ | 361122/400000 [00:38<00:04, 9178.71it/s] 91%|█████████ | 362041/400000 [00:38<00:04, 9141.65it/s] 91%|█████████ | 362995/400000 [00:38<00:03, 9256.72it/s] 91%|█████████ | 363931/400000 [00:38<00:03, 9287.39it/s] 91%|█████████ | 364861/400000 [00:38<00:03, 8984.53it/s] 91%|█████████▏| 365763/400000 [00:38<00:03, 8965.98it/s] 92%|█████████▏| 366662/400000 [00:38<00:03, 8940.22it/s] 92%|█████████▏| 367610/400000 [00:39<00:03, 9095.17it/s] 92%|█████████▏| 368622/400000 [00:39<00:03, 9380.05it/s] 92%|█████████▏| 369564/400000 [00:39<00:03, 9233.33it/s] 93%|█████████▎| 370560/400000 [00:39<00:03, 9436.22it/s] 93%|█████████▎| 371511/400000 [00:39<00:03, 9455.50it/s] 93%|█████████▎| 372486/400000 [00:39<00:02, 9538.78it/s] 93%|█████████▎| 373442/400000 [00:39<00:02, 9532.45it/s] 94%|█████████▎| 374397/400000 [00:39<00:02, 9409.44it/s] 94%|█████████▍| 375348/400000 [00:39<00:02, 9438.12it/s] 94%|█████████▍| 376312/400000 [00:40<00:02, 9495.74it/s] 94%|█████████▍| 377263/400000 [00:40<00:02, 9404.49it/s] 95%|█████████▍| 378205/400000 [00:40<00:02, 9319.99it/s] 95%|█████████▍| 379138/400000 [00:40<00:02, 9268.39it/s] 95%|█████████▌| 380066/400000 [00:40<00:02, 9180.12it/s] 95%|█████████▌| 380985/400000 [00:40<00:02, 9156.40it/s] 95%|█████████▌| 381967/400000 [00:40<00:01, 9344.39it/s] 96%|█████████▌| 382903/400000 [00:40<00:01, 9169.08it/s] 96%|█████████▌| 383887/400000 [00:40<00:01, 9359.90it/s] 96%|█████████▌| 384894/400000 [00:40<00:01, 9560.01it/s] 96%|█████████▋| 385853/400000 [00:41<00:01, 9368.65it/s] 97%|█████████▋| 386850/400000 [00:41<00:01, 9539.59it/s] 97%|█████████▋| 387807/400000 [00:41<00:01, 9512.54it/s] 97%|█████████▋| 388761/400000 [00:41<00:01, 9488.90it/s] 97%|█████████▋| 389753/400000 [00:41<00:01, 9612.83it/s] 98%|█████████▊| 390786/400000 [00:41<00:00, 9817.05it/s] 98%|█████████▊| 391800/400000 [00:41<00:00, 9911.58it/s] 98%|█████████▊| 392793/400000 [00:41<00:00, 9893.14it/s] 98%|█████████▊| 393784/400000 [00:41<00:00, 9597.29it/s] 99%|█████████▊| 394749/400000 [00:41<00:00, 9610.40it/s] 99%|█████████▉| 395729/400000 [00:42<00:00, 9666.27it/s] 99%|█████████▉| 396698/400000 [00:42<00:00, 9512.31it/s] 99%|█████████▉| 397651/400000 [00:42<00:00, 9407.48it/s]100%|█████████▉| 398605/400000 [00:42<00:00, 9445.34it/s]100%|█████████▉| 399568/400000 [00:42<00:00, 9498.56it/s]100%|█████████▉| 399999/400000 [00:42<00:00, 9410.40it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f23fb2c6940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011388329903625793 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011378215308173444 	 Accuracy: 53

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
2020-05-14 12:25:11.029536: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 12:25:11.033447: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-14 12:25:11.033584: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561f472761d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 12:25:11.033597: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f23ae7f9128> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8506 - accuracy: 0.4880
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8890 - accuracy: 0.4855 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.8506 - accuracy: 0.4880
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8161 - accuracy: 0.4902
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7525 - accuracy: 0.4944
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7177 - accuracy: 0.4967
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7258 - accuracy: 0.4961
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7318 - accuracy: 0.4958
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6956 - accuracy: 0.4981
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6682 - accuracy: 0.4999
11000/25000 [============>.................] - ETA: 3s - loss: 7.6680 - accuracy: 0.4999
12000/25000 [=============>................] - ETA: 3s - loss: 7.6768 - accuracy: 0.4993
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6761 - accuracy: 0.4994
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6765 - accuracy: 0.4994
15000/25000 [=================>............] - ETA: 2s - loss: 7.6820 - accuracy: 0.4990
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7145 - accuracy: 0.4969
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6964 - accuracy: 0.4981
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6879 - accuracy: 0.4986
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6900 - accuracy: 0.4985
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6927 - accuracy: 0.4983
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6761 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6834 - accuracy: 0.4989
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6793 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6820 - accuracy: 0.4990
25000/25000 [==============================] - 7s 264us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f235f16b6a0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f23a0d152b0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 2.3422 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 2.2745 - val_crf_viterbi_accuracy: 0.0267

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


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f228e3b4f28> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 13:14:50.817342
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 13:14:50.822466
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 13:14:50.827103
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 13:14:50.831520
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f229a17f390> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 358852.3750
Epoch 2/10

1/1 [==============================] - 0s 119ms/step - loss: 319741.4062
Epoch 3/10

1/1 [==============================] - 0s 135ms/step - loss: 229466.7500
Epoch 4/10

1/1 [==============================] - 0s 125ms/step - loss: 133800.4375
Epoch 5/10

1/1 [==============================] - 0s 117ms/step - loss: 68270.6484
Epoch 6/10

1/1 [==============================] - 0s 138ms/step - loss: 35509.2148
Epoch 7/10

1/1 [==============================] - 0s 125ms/step - loss: 20097.8359
Epoch 8/10

1/1 [==============================] - 0s 130ms/step - loss: 12579.5928
Epoch 9/10

1/1 [==============================] - 0s 114ms/step - loss: 8550.9229
Epoch 10/10

1/1 [==============================] - 0s 137ms/step - loss: 6265.2314

  #### Inference Need return ypred, ytrue ######################### 
[[-0.01795267 -1.6196356  -0.84558856 -0.06077769  0.1547194  -0.42236963
  -0.30189127 -0.26442188 -1.1260598  -1.867121   -0.26136512  1.7035122
   0.03100315 -0.1175375   1.6167655  -1.3413961   0.04204132 -0.10791245
   1.2653576  -1.8422327  -0.4299841   0.746021   -0.4459067   0.92642397
   0.9310646  -0.06964457 -0.13958123  0.25017762  1.1861134  -1.2200669
   1.0200713   0.80737877  0.04708428 -1.2048101  -0.13737366 -0.85706276
  -0.5205083   1.4055872  -0.32727656 -0.11623548 -0.51582503 -0.5083667
  -0.6182989  -0.9036714  -1.4036645  -0.56241524 -0.09033255 -1.283907
  -1.0866442  -1.6306369   0.14196838 -1.3705319   0.6080022  -0.14425233
   1.1165755   0.59750175  0.5387828  -2.0983362   0.23217538 -0.7963657
  -0.51828897 -0.7639815  -0.22728232 -0.01363552  1.5944364  -0.10193041
  -0.607634   -0.7856497  -0.39022443  0.39444014 -0.28697294 -1.1157899
  -0.41529015 -1.2722902  -0.29012138  0.94375235 -1.9265436  -0.61754304
  -1.3287681   0.356603    0.4478635   0.23023243 -0.13822697  0.97317994
  -0.16874349  0.14915386 -0.76841056  1.5817139   0.04024515  0.16765973
  -0.03638494  0.55485445 -0.4051473   0.42848963 -0.92685586  0.46033084
  -0.49439996 -0.9656558  -0.01907885  0.8975533  -0.57109207 -0.7128756
  -0.84847367  0.03199813 -0.7593636  -0.14204876 -0.32871625  1.0231977
  -0.7238272   0.09013325 -0.2736081   0.37983853  0.7703392  -0.30307552
  -0.29336143  0.09668262 -0.9023361  -1.1117765  -0.0284126  -2.0965014
   0.43909103  9.09597     7.8918424   8.662748    9.349091   10.095687
   7.7963066   9.125607    8.310938    9.346902    9.776264    9.370284
   9.249746    7.2486377   8.824764    8.694184    8.910182   10.882226
  11.3669615   7.912839   11.035294    9.476991   10.335366    8.664355
   7.414055    8.280579   10.5987015  10.474753    9.156225    8.971584
   9.252587    7.9880576   8.166661    8.938322    7.7962017   7.924223
   9.267136    7.847081    8.33658     9.720699   10.254504    9.494961
   7.608375    7.734467    8.925184    8.252288    9.355045    8.9602165
   9.211224    8.317344    8.015005    8.963975    8.549334    9.284761
   7.3897185   9.460845    8.415116    9.981606    8.840143    7.4339294
   1.8526537   0.97624916  2.258295    2.364037    1.4080443   0.51536095
   0.62429965  2.01959     0.6035877   1.6220224   0.26701772  2.914225
   0.3992223   1.6579053   0.97509557  0.43803096  0.51170045  0.39281738
   0.57041323  0.76475894  2.134537    0.5143993   1.5154301   2.1202765
   1.3040663   0.57737833  2.5846663   0.49395537  1.153903    1.766351
   1.1254853   0.4349844   3.8762822   0.5065356   2.200863    0.76311153
   2.0958028   0.5009158   1.564135    0.46044815  0.6017101   0.5024073
   0.31877774  0.45175004  0.23710275  0.26556015  1.8599048   0.7354806
   0.39759803  0.78128743  0.90632594  0.7422417   0.75593984  0.67278355
   1.7989236   0.17061698  0.53491896  1.830132    0.9475796   1.9537897
   2.3493438   0.2318877   0.32298076  0.46522337  0.9518527   1.4919231
   0.35373676  1.0369356   2.8448682   1.9575225   1.1537645   1.7810636
   0.6600004   1.9946394   1.3781978   1.0417515   0.92709315  0.75457567
   2.5911202   0.46662807  0.620893    0.17717862  0.5305078   1.7871774
   0.6491466   0.70342106  0.38336575  0.6063102   0.6629783   1.6502192
   2.2080054   0.44406784  1.6998639   1.6247255   0.36603814  0.50484407
   1.7662305   2.4390066   0.91038543  0.44246268  0.92487854  3.0483356
   0.5366505   1.0015615   1.2385823   0.59664327  1.8890588   1.5113786
   0.2790624   1.5073584   1.6825187   2.0146618   0.09945071  0.7519109
   0.8842411   1.1456409   1.5028415   0.7785423   0.39904153  0.28930438
   0.14183515  8.605106    9.363953    7.9618587   8.414795    7.4675317
   8.799736    7.374784    9.549359    9.792651    8.878232    8.305487
   8.373839    9.104346    8.343905    8.914617    8.419186    7.504505
   7.7510457   8.911696    7.0110006   8.468297    7.6202006   9.076249
   9.804967    8.692822    8.3085      9.865893    8.417411    9.245841
   9.265876    8.536336    7.9267993   9.82112     8.926154    8.088796
   8.874872    8.578003    8.530648    9.118124    8.739369    7.6412487
   9.474339    8.484506    9.514893   10.115614    9.75082     9.102464
   9.056436    8.435266    9.965829    9.810461    8.247222    8.325154
   8.76349     7.789854    8.349519    8.598916    9.012404   10.051338
  -0.5675472   0.3236321  10.60234   ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 13:15:01.951401
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.2445
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 13:15:01.956172
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8721.07
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 13:15:01.960884
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.6032
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 13:15:01.966749
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -780.033
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139786052737792
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139785091166728
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139785091167232
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139785091167736
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139785091168240
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139785091168744

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f2295fffe80> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.645898
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.610276
grad_step = 000002, loss = 0.581213
grad_step = 000003, loss = 0.550877
grad_step = 000004, loss = 0.521385
grad_step = 000005, loss = 0.494982
grad_step = 000006, loss = 0.474447
grad_step = 000007, loss = 0.465064
grad_step = 000008, loss = 0.447466
grad_step = 000009, loss = 0.425799
grad_step = 000010, loss = 0.409764
grad_step = 000011, loss = 0.396988
grad_step = 000012, loss = 0.382656
grad_step = 000013, loss = 0.366782
grad_step = 000014, loss = 0.351701
grad_step = 000015, loss = 0.339789
grad_step = 000016, loss = 0.330997
grad_step = 000017, loss = 0.321082
grad_step = 000018, loss = 0.307824
grad_step = 000019, loss = 0.294338
grad_step = 000020, loss = 0.283178
grad_step = 000021, loss = 0.273503
grad_step = 000022, loss = 0.263279
grad_step = 000023, loss = 0.251871
grad_step = 000024, loss = 0.240447
grad_step = 000025, loss = 0.230447
grad_step = 000026, loss = 0.221662
grad_step = 000027, loss = 0.212450
grad_step = 000028, loss = 0.202525
grad_step = 000029, loss = 0.193044
grad_step = 000030, loss = 0.184549
grad_step = 000031, loss = 0.176445
grad_step = 000032, loss = 0.168093
grad_step = 000033, loss = 0.159714
grad_step = 000034, loss = 0.151930
grad_step = 000035, loss = 0.144761
grad_step = 000036, loss = 0.137680
grad_step = 000037, loss = 0.130687
grad_step = 000038, loss = 0.124149
grad_step = 000039, loss = 0.117972
grad_step = 000040, loss = 0.111852
grad_step = 000041, loss = 0.105857
grad_step = 000042, loss = 0.100258
grad_step = 000043, loss = 0.094966
grad_step = 000044, loss = 0.088912
grad_step = 000045, loss = 0.082799
grad_step = 000046, loss = 0.077099
grad_step = 000047, loss = 0.071870
grad_step = 000048, loss = 0.067097
grad_step = 000049, loss = 0.062820
grad_step = 000050, loss = 0.058985
grad_step = 000051, loss = 0.055346
grad_step = 000052, loss = 0.051764
grad_step = 000053, loss = 0.048231
grad_step = 000054, loss = 0.044753
grad_step = 000055, loss = 0.041430
grad_step = 000056, loss = 0.038288
grad_step = 000057, loss = 0.035326
grad_step = 000058, loss = 0.032592
grad_step = 000059, loss = 0.030052
grad_step = 000060, loss = 0.027706
grad_step = 000061, loss = 0.025559
grad_step = 000062, loss = 0.023531
grad_step = 000063, loss = 0.021625
grad_step = 000064, loss = 0.019871
grad_step = 000065, loss = 0.018216
grad_step = 000066, loss = 0.016682
grad_step = 000067, loss = 0.015251
grad_step = 000068, loss = 0.013941
grad_step = 000069, loss = 0.012747
grad_step = 000070, loss = 0.011667
grad_step = 000071, loss = 0.010699
grad_step = 000072, loss = 0.009808
grad_step = 000073, loss = 0.009008
grad_step = 000074, loss = 0.008268
grad_step = 000075, loss = 0.007587
grad_step = 000076, loss = 0.006959
grad_step = 000077, loss = 0.006393
grad_step = 000078, loss = 0.005889
grad_step = 000079, loss = 0.005442
grad_step = 000080, loss = 0.005051
grad_step = 000081, loss = 0.004699
grad_step = 000082, loss = 0.004388
grad_step = 000083, loss = 0.004105
grad_step = 000084, loss = 0.003850
grad_step = 000085, loss = 0.003621
grad_step = 000086, loss = 0.003417
grad_step = 000087, loss = 0.003237
grad_step = 000088, loss = 0.003080
grad_step = 000089, loss = 0.002941
grad_step = 000090, loss = 0.002819
grad_step = 000091, loss = 0.002710
grad_step = 000092, loss = 0.002611
grad_step = 000093, loss = 0.002519
grad_step = 000094, loss = 0.002436
grad_step = 000095, loss = 0.002362
grad_step = 000096, loss = 0.002296
grad_step = 000097, loss = 0.002237
grad_step = 000098, loss = 0.002185
grad_step = 000099, loss = 0.002137
grad_step = 000100, loss = 0.002092
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002050
grad_step = 000102, loss = 0.002011
grad_step = 000103, loss = 0.001976
grad_step = 000104, loss = 0.001945
grad_step = 000105, loss = 0.001917
grad_step = 000106, loss = 0.001890
grad_step = 000107, loss = 0.001866
grad_step = 000108, loss = 0.001845
grad_step = 000109, loss = 0.001828
grad_step = 000110, loss = 0.001817
grad_step = 000111, loss = 0.001807
grad_step = 000112, loss = 0.001792
grad_step = 000113, loss = 0.001766
grad_step = 000114, loss = 0.001740
grad_step = 000115, loss = 0.001724
grad_step = 000116, loss = 0.001717
grad_step = 000117, loss = 0.001714
grad_step = 000118, loss = 0.001708
grad_step = 000119, loss = 0.001698
grad_step = 000120, loss = 0.001681
grad_step = 000121, loss = 0.001662
grad_step = 000122, loss = 0.001645
grad_step = 000123, loss = 0.001632
grad_step = 000124, loss = 0.001624
grad_step = 000125, loss = 0.001618
grad_step = 000126, loss = 0.001620
grad_step = 000127, loss = 0.001627
grad_step = 000128, loss = 0.001647
grad_step = 000129, loss = 0.001669
grad_step = 000130, loss = 0.001688
grad_step = 000131, loss = 0.001663
grad_step = 000132, loss = 0.001607
grad_step = 000133, loss = 0.001547
grad_step = 000134, loss = 0.001522
grad_step = 000135, loss = 0.001534
grad_step = 000136, loss = 0.001560
grad_step = 000137, loss = 0.001578
grad_step = 000138, loss = 0.001563
grad_step = 000139, loss = 0.001528
grad_step = 000140, loss = 0.001484
grad_step = 000141, loss = 0.001456
grad_step = 000142, loss = 0.001449
grad_step = 000143, loss = 0.001457
grad_step = 000144, loss = 0.001472
grad_step = 000145, loss = 0.001487
grad_step = 000146, loss = 0.001500
grad_step = 000147, loss = 0.001497
grad_step = 000148, loss = 0.001483
grad_step = 000149, loss = 0.001447
grad_step = 000150, loss = 0.001409
grad_step = 000151, loss = 0.001377
grad_step = 000152, loss = 0.001358
grad_step = 000153, loss = 0.001352
grad_step = 000154, loss = 0.001355
grad_step = 000155, loss = 0.001366
grad_step = 000156, loss = 0.001387
grad_step = 000157, loss = 0.001428
grad_step = 000158, loss = 0.001479
grad_step = 000159, loss = 0.001543
grad_step = 000160, loss = 0.001538
grad_step = 000161, loss = 0.001474
grad_step = 000162, loss = 0.001350
grad_step = 000163, loss = 0.001280
grad_step = 000164, loss = 0.001304
grad_step = 000165, loss = 0.001369
grad_step = 000166, loss = 0.001405
grad_step = 000167, loss = 0.001359
grad_step = 000168, loss = 0.001293
grad_step = 000169, loss = 0.001254
grad_step = 000170, loss = 0.001259
grad_step = 000171, loss = 0.001284
grad_step = 000172, loss = 0.001303
grad_step = 000173, loss = 0.001312
grad_step = 000174, loss = 0.001300
grad_step = 000175, loss = 0.001263
grad_step = 000176, loss = 0.001223
grad_step = 000177, loss = 0.001210
grad_step = 000178, loss = 0.001227
grad_step = 000179, loss = 0.001250
grad_step = 000180, loss = 0.001256
grad_step = 000181, loss = 0.001246
grad_step = 000182, loss = 0.001234
grad_step = 000183, loss = 0.001221
grad_step = 000184, loss = 0.001207
grad_step = 000185, loss = 0.001191
grad_step = 000186, loss = 0.001180
grad_step = 000187, loss = 0.001180
grad_step = 000188, loss = 0.001189
grad_step = 000189, loss = 0.001205
grad_step = 000190, loss = 0.001222
grad_step = 000191, loss = 0.001248
grad_step = 000192, loss = 0.001285
grad_step = 000193, loss = 0.001355
grad_step = 000194, loss = 0.001431
grad_step = 000195, loss = 0.001432
grad_step = 000196, loss = 0.001308
grad_step = 000197, loss = 0.001197
grad_step = 000198, loss = 0.001212
grad_step = 000199, loss = 0.001247
grad_step = 000200, loss = 0.001220
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001238
grad_step = 000202, loss = 0.001268
grad_step = 000203, loss = 0.001214
grad_step = 000204, loss = 0.001149
grad_step = 000205, loss = 0.001159
grad_step = 000206, loss = 0.001202
grad_step = 000207, loss = 0.001205
grad_step = 000208, loss = 0.001171
grad_step = 000209, loss = 0.001165
grad_step = 000210, loss = 0.001184
grad_step = 000211, loss = 0.001181
grad_step = 000212, loss = 0.001148
grad_step = 000213, loss = 0.001130
grad_step = 000214, loss = 0.001144
grad_step = 000215, loss = 0.001160
grad_step = 000216, loss = 0.001148
grad_step = 000217, loss = 0.001136
grad_step = 000218, loss = 0.001142
grad_step = 000219, loss = 0.001156
grad_step = 000220, loss = 0.001156
grad_step = 000221, loss = 0.001144
grad_step = 000222, loss = 0.001142
grad_step = 000223, loss = 0.001151
grad_step = 000224, loss = 0.001159
grad_step = 000225, loss = 0.001158
grad_step = 000226, loss = 0.001158
grad_step = 000227, loss = 0.001167
grad_step = 000228, loss = 0.001184
grad_step = 000229, loss = 0.001200
grad_step = 000230, loss = 0.001211
grad_step = 000231, loss = 0.001219
grad_step = 000232, loss = 0.001231
grad_step = 000233, loss = 0.001236
grad_step = 000234, loss = 0.001229
grad_step = 000235, loss = 0.001197
grad_step = 000236, loss = 0.001160
grad_step = 000237, loss = 0.001130
grad_step = 000238, loss = 0.001113
grad_step = 000239, loss = 0.001106
grad_step = 000240, loss = 0.001106
grad_step = 000241, loss = 0.001115
grad_step = 000242, loss = 0.001130
grad_step = 000243, loss = 0.001144
grad_step = 000244, loss = 0.001153
grad_step = 000245, loss = 0.001152
grad_step = 000246, loss = 0.001147
grad_step = 000247, loss = 0.001140
grad_step = 000248, loss = 0.001131
grad_step = 000249, loss = 0.001122
grad_step = 000250, loss = 0.001110
grad_step = 000251, loss = 0.001099
grad_step = 000252, loss = 0.001091
grad_step = 000253, loss = 0.001088
grad_step = 000254, loss = 0.001087
grad_step = 000255, loss = 0.001087
grad_step = 000256, loss = 0.001086
grad_step = 000257, loss = 0.001086
grad_step = 000258, loss = 0.001087
grad_step = 000259, loss = 0.001091
grad_step = 000260, loss = 0.001098
grad_step = 000261, loss = 0.001110
grad_step = 000262, loss = 0.001131
grad_step = 000263, loss = 0.001169
grad_step = 000264, loss = 0.001235
grad_step = 000265, loss = 0.001337
grad_step = 000266, loss = 0.001489
grad_step = 000267, loss = 0.001620
grad_step = 000268, loss = 0.001642
grad_step = 000269, loss = 0.001445
grad_step = 000270, loss = 0.001171
grad_step = 000271, loss = 0.001077
grad_step = 000272, loss = 0.001203
grad_step = 000273, loss = 0.001334
grad_step = 000274, loss = 0.001285
grad_step = 000275, loss = 0.001133
grad_step = 000276, loss = 0.001090
grad_step = 000277, loss = 0.001172
grad_step = 000278, loss = 0.001219
grad_step = 000279, loss = 0.001174
grad_step = 000280, loss = 0.001109
grad_step = 000281, loss = 0.001095
grad_step = 000282, loss = 0.001129
grad_step = 000283, loss = 0.001153
grad_step = 000284, loss = 0.001140
grad_step = 000285, loss = 0.001097
grad_step = 000286, loss = 0.001072
grad_step = 000287, loss = 0.001097
grad_step = 000288, loss = 0.001126
grad_step = 000289, loss = 0.001117
grad_step = 000290, loss = 0.001078
grad_step = 000291, loss = 0.001059
grad_step = 000292, loss = 0.001081
grad_step = 000293, loss = 0.001102
grad_step = 000294, loss = 0.001096
grad_step = 000295, loss = 0.001072
grad_step = 000296, loss = 0.001058
grad_step = 000297, loss = 0.001065
grad_step = 000298, loss = 0.001075
grad_step = 000299, loss = 0.001076
grad_step = 000300, loss = 0.001068
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001059
grad_step = 000302, loss = 0.001058
grad_step = 000303, loss = 0.001058
grad_step = 000304, loss = 0.001058
grad_step = 000305, loss = 0.001059
grad_step = 000306, loss = 0.001058
grad_step = 000307, loss = 0.001057
grad_step = 000308, loss = 0.001053
grad_step = 000309, loss = 0.001048
grad_step = 000310, loss = 0.001046
grad_step = 000311, loss = 0.001048
grad_step = 000312, loss = 0.001051
grad_step = 000313, loss = 0.001051
grad_step = 000314, loss = 0.001050
grad_step = 000315, loss = 0.001046
grad_step = 000316, loss = 0.001042
grad_step = 000317, loss = 0.001040
grad_step = 000318, loss = 0.001040
grad_step = 000319, loss = 0.001041
grad_step = 000320, loss = 0.001041
grad_step = 000321, loss = 0.001041
grad_step = 000322, loss = 0.001041
grad_step = 000323, loss = 0.001040
grad_step = 000324, loss = 0.001040
grad_step = 000325, loss = 0.001040
grad_step = 000326, loss = 0.001040
grad_step = 000327, loss = 0.001040
grad_step = 000328, loss = 0.001039
grad_step = 000329, loss = 0.001039
grad_step = 000330, loss = 0.001039
grad_step = 000331, loss = 0.001039
grad_step = 000332, loss = 0.001041
grad_step = 000333, loss = 0.001044
grad_step = 000334, loss = 0.001050
grad_step = 000335, loss = 0.001059
grad_step = 000336, loss = 0.001074
grad_step = 000337, loss = 0.001098
grad_step = 000338, loss = 0.001137
grad_step = 000339, loss = 0.001195
grad_step = 000340, loss = 0.001273
grad_step = 000341, loss = 0.001370
grad_step = 000342, loss = 0.001443
grad_step = 000343, loss = 0.001462
grad_step = 000344, loss = 0.001374
grad_step = 000345, loss = 0.001212
grad_step = 000346, loss = 0.001075
grad_step = 000347, loss = 0.001043
grad_step = 000348, loss = 0.001110
grad_step = 000349, loss = 0.001189
grad_step = 000350, loss = 0.001196
grad_step = 000351, loss = 0.001125
grad_step = 000352, loss = 0.001047
grad_step = 000353, loss = 0.001032
grad_step = 000354, loss = 0.001074
grad_step = 000355, loss = 0.001118
grad_step = 000356, loss = 0.001116
grad_step = 000357, loss = 0.001072
grad_step = 000358, loss = 0.001029
grad_step = 000359, loss = 0.001022
grad_step = 000360, loss = 0.001045
grad_step = 000361, loss = 0.001070
grad_step = 000362, loss = 0.001072
grad_step = 000363, loss = 0.001049
grad_step = 000364, loss = 0.001024
grad_step = 000365, loss = 0.001014
grad_step = 000366, loss = 0.001023
grad_step = 000367, loss = 0.001039
grad_step = 000368, loss = 0.001044
grad_step = 000369, loss = 0.001036
grad_step = 000370, loss = 0.001021
grad_step = 000371, loss = 0.001010
grad_step = 000372, loss = 0.001009
grad_step = 000373, loss = 0.001016
grad_step = 000374, loss = 0.001024
grad_step = 000375, loss = 0.001026
grad_step = 000376, loss = 0.001022
grad_step = 000377, loss = 0.001014
grad_step = 000378, loss = 0.001007
grad_step = 000379, loss = 0.001004
grad_step = 000380, loss = 0.001005
grad_step = 000381, loss = 0.001008
grad_step = 000382, loss = 0.001010
grad_step = 000383, loss = 0.001010
grad_step = 000384, loss = 0.001008
grad_step = 000385, loss = 0.001006
grad_step = 000386, loss = 0.001003
grad_step = 000387, loss = 0.001002
grad_step = 000388, loss = 0.001001
grad_step = 000389, loss = 0.001002
grad_step = 000390, loss = 0.001003
grad_step = 000391, loss = 0.001005
grad_step = 000392, loss = 0.001006
grad_step = 000393, loss = 0.001007
grad_step = 000394, loss = 0.001009
grad_step = 000395, loss = 0.001012
grad_step = 000396, loss = 0.001017
grad_step = 000397, loss = 0.001025
grad_step = 000398, loss = 0.001038
grad_step = 000399, loss = 0.001055
grad_step = 000400, loss = 0.001081
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001106
grad_step = 000402, loss = 0.001133
grad_step = 000403, loss = 0.001145
grad_step = 000404, loss = 0.001146
grad_step = 000405, loss = 0.001143
grad_step = 000406, loss = 0.001147
grad_step = 000407, loss = 0.001161
grad_step = 000408, loss = 0.001175
grad_step = 000409, loss = 0.001150
grad_step = 000410, loss = 0.001098
grad_step = 000411, loss = 0.001031
grad_step = 000412, loss = 0.000991
grad_step = 000413, loss = 0.000994
grad_step = 000414, loss = 0.001023
grad_step = 000415, loss = 0.001050
grad_step = 000416, loss = 0.001055
grad_step = 000417, loss = 0.001042
grad_step = 000418, loss = 0.001025
grad_step = 000419, loss = 0.001014
grad_step = 000420, loss = 0.001009
grad_step = 000421, loss = 0.001006
grad_step = 000422, loss = 0.000998
grad_step = 000423, loss = 0.000990
grad_step = 000424, loss = 0.000987
grad_step = 000425, loss = 0.000991
grad_step = 000426, loss = 0.001001
grad_step = 000427, loss = 0.001011
grad_step = 000428, loss = 0.001017
grad_step = 000429, loss = 0.001016
grad_step = 000430, loss = 0.001011
grad_step = 000431, loss = 0.001005
grad_step = 000432, loss = 0.001000
grad_step = 000433, loss = 0.000998
grad_step = 000434, loss = 0.000996
grad_step = 000435, loss = 0.000993
grad_step = 000436, loss = 0.000987
grad_step = 000437, loss = 0.000981
grad_step = 000438, loss = 0.000976
grad_step = 000439, loss = 0.000973
grad_step = 000440, loss = 0.000972
grad_step = 000441, loss = 0.000972
grad_step = 000442, loss = 0.000973
grad_step = 000443, loss = 0.000973
grad_step = 000444, loss = 0.000972
grad_step = 000445, loss = 0.000971
grad_step = 000446, loss = 0.000969
grad_step = 000447, loss = 0.000967
grad_step = 000448, loss = 0.000966
grad_step = 000449, loss = 0.000965
grad_step = 000450, loss = 0.000964
grad_step = 000451, loss = 0.000964
grad_step = 000452, loss = 0.000964
grad_step = 000453, loss = 0.000964
grad_step = 000454, loss = 0.000965
grad_step = 000455, loss = 0.000966
grad_step = 000456, loss = 0.000968
grad_step = 000457, loss = 0.000974
grad_step = 000458, loss = 0.000985
grad_step = 000459, loss = 0.001009
grad_step = 000460, loss = 0.001057
grad_step = 000461, loss = 0.001154
grad_step = 000462, loss = 0.001338
grad_step = 000463, loss = 0.001626
grad_step = 000464, loss = 0.002026
grad_step = 000465, loss = 0.002252
grad_step = 000466, loss = 0.002095
grad_step = 000467, loss = 0.001495
grad_step = 000468, loss = 0.001060
grad_step = 000469, loss = 0.001260
grad_step = 000470, loss = 0.001581
grad_step = 000471, loss = 0.001421
grad_step = 000472, loss = 0.001052
grad_step = 000473, loss = 0.001117
grad_step = 000474, loss = 0.001376
grad_step = 000475, loss = 0.001262
grad_step = 000476, loss = 0.001000
grad_step = 000477, loss = 0.001091
grad_step = 000478, loss = 0.001244
grad_step = 000479, loss = 0.001125
grad_step = 000480, loss = 0.000966
grad_step = 000481, loss = 0.001064
grad_step = 000482, loss = 0.001153
grad_step = 000483, loss = 0.001043
grad_step = 000484, loss = 0.000957
grad_step = 000485, loss = 0.001051
grad_step = 000486, loss = 0.001085
grad_step = 000487, loss = 0.000990
grad_step = 000488, loss = 0.000960
grad_step = 000489, loss = 0.001036
grad_step = 000490, loss = 0.001030
grad_step = 000491, loss = 0.000962
grad_step = 000492, loss = 0.000967
grad_step = 000493, loss = 0.001013
grad_step = 000494, loss = 0.000989
grad_step = 000495, loss = 0.000952
grad_step = 000496, loss = 0.000969
grad_step = 000497, loss = 0.000988
grad_step = 000498, loss = 0.000962
grad_step = 000499, loss = 0.000950
grad_step = 000500, loss = 0.000967
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000966
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

  date_run                              2020-05-14 13:15:35.382976
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.253102
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 13:15:35.390249
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.172335
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 13:15:35.404047
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.13987
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 13:15:35.415514
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.61869
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
0   2020-05-14 13:14:50.817342  ...    mean_absolute_error
1   2020-05-14 13:14:50.822466  ...     mean_squared_error
2   2020-05-14 13:14:50.827103  ...  median_absolute_error
3   2020-05-14 13:14:50.831520  ...               r2_score
4   2020-05-14 13:15:01.951401  ...    mean_absolute_error
5   2020-05-14 13:15:01.956172  ...     mean_squared_error
6   2020-05-14 13:15:01.960884  ...  median_absolute_error
7   2020-05-14 13:15:01.966749  ...               r2_score
8   2020-05-14 13:15:35.382976  ...    mean_absolute_error
9   2020-05-14 13:15:35.390249  ...     mean_squared_error
10  2020-05-14 13:15:35.404047  ...  median_absolute_error
11  2020-05-14 13:15:35.415514  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff3ada0dfd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 29%|██▊       | 2826240/9912422 [00:00<00:00, 28037819.57it/s]9920512it [00:00, 32185552.47it/s]                             
0it [00:00, ?it/s]32768it [00:00, 630595.65it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 462109.12it/s]1654784it [00:00, 11653333.47it/s]                         
0it [00:00, ?it/s]8192it [00:00, 156694.87it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff36040ee80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff3ada17ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff36040ee80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff3ada17ef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff35d1d14e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff3ada17ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff36040ee80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff3ada17ef0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff35d1d14e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff3ada17ef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f61c1b761d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=78640f5d0087e780406de9f8de8929b7302db8da1e465080b5ede6210c84d4b9
  Stored in directory: /tmp/pip-ephem-wheel-cache-50alaq3f/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f6159971710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2572288/17464789 [===>..........................] - ETA: 0s
 8511488/17464789 [=============>................] - ETA: 0s
15032320/17464789 [========================>.....] - ETA: 0s
17244160/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 13:17:12.153279: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 13:17:12.157958: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-14 13:17:12.158121: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5600950d11b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 13:17:12.158138: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 16s - loss: 7.7433 - accuracy: 0.4950
 2000/25000 [=>............................] - ETA: 12s - loss: 7.5516 - accuracy: 0.5075
 3000/25000 [==>...........................] - ETA: 10s - loss: 7.6257 - accuracy: 0.5027
 4000/25000 [===>..........................] - ETA: 9s - loss: 7.6590 - accuracy: 0.5005 
 5000/25000 [=====>........................] - ETA: 8s - loss: 7.6298 - accuracy: 0.5024
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.5848 - accuracy: 0.5053
 7000/25000 [=======>......................] - ETA: 7s - loss: 7.6732 - accuracy: 0.4996
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.7050 - accuracy: 0.4975
 9000/25000 [=========>....................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6988 - accuracy: 0.4979
11000/25000 [============>.................] - ETA: 5s - loss: 7.7252 - accuracy: 0.4962
12000/25000 [=============>................] - ETA: 5s - loss: 7.6845 - accuracy: 0.4988
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6879 - accuracy: 0.4986
14000/25000 [===============>..............] - ETA: 4s - loss: 7.6568 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 3s - loss: 7.6339 - accuracy: 0.5021
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6254 - accuracy: 0.5027
17000/25000 [===================>..........] - ETA: 3s - loss: 7.6116 - accuracy: 0.5036
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6070 - accuracy: 0.5039
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6239 - accuracy: 0.5028
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6413 - accuracy: 0.5016
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6440 - accuracy: 0.5015
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6485 - accuracy: 0.5012
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6526 - accuracy: 0.5009
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
25000/25000 [==============================] - 12s 465us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 13:17:32.987737
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 13:17:32.987737  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<16:37:25, 14.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<11:52:13, 20.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<8:21:47, 28.6kB/s]  .vector_cache/glove.6B.zip:   0%|          | 885k/862M [00:00<5:51:38, 40.8kB/s].vector_cache/glove.6B.zip:   0%|          | 1.81M/862M [00:01<4:06:22, 58.2kB/s].vector_cache/glove.6B.zip:   1%|          | 4.67M/862M [00:01<2:52:02, 83.1kB/s].vector_cache/glove.6B.zip:   1%|          | 8.14M/862M [00:01<2:00:03, 119kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 11.9M/862M [00:01<1:23:47, 169kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.4M/862M [00:01<58:31, 241kB/s]  .vector_cache/glove.6B.zip:   2%|▏         | 18.5M/862M [00:01<40:57, 343kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.1M/862M [00:01<28:36, 489kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.3M/862M [00:01<20:01, 695kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.1M/862M [00:01<14:03, 985kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.9M/862M [00:01<09:52, 1.39MB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.2M/862M [00:02<07:00, 1.96MB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.5M/862M [00:02<04:58, 2.74MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.6M/862M [00:02<03:34, 3.81MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.8M/862M [00:02<02:36, 5.17MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.9M/862M [00:03<03:18, 4.07MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.9M/862M [00:03<02:31, 5.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<05:14, 2.56MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:05<06:19, 2.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.1M/862M [00:05<05:05, 2.63MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:05<03:42, 3.60MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<13:15, 1.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:07<12:30, 1.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:07<10:19, 1.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:07<07:52, 1.69MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.4M/862M [00:07<05:50, 2.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:09<07:17, 1.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:09<08:30, 1.56MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:09<07:19, 1.81MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:09<05:39, 2.34MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:09<04:08, 3.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<10:05, 1.31MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:11<09:47, 1.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:11<08:22, 1.58MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:11<06:16, 2.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:11<04:33, 2.88MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<13:55, 944kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:13<12:43, 1.03MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.5M/862M [00:13<09:29, 1.38MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.5M/862M [00:13<07:02, 1.86MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:15<07:41, 1.70MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:15<08:17, 1.57MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:15<06:22, 2.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.7M/862M [00:15<04:51, 2.68MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:17<06:21, 2.05MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:17<10:26, 1.24MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:17<09:23, 1.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:17<07:16, 1.79MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:17<05:20, 2.43MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:19<07:02, 1.84MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:19<07:56, 1.63MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:19<06:36, 1.95MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:19<05:10, 2.49MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:19<03:51, 3.35MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:21<07:36, 1.69MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:21<07:56, 1.62MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:21<07:07, 1.80MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:21<05:25, 2.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:21<04:03, 3.16MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:21<03:08, 4.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:23<42:49, 299kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:23<32:59, 388kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.4M/862M [00:23<23:44, 538kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.5M/862M [00:23<16:55, 754kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:25<14:39, 868kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:25<13:17, 957kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:25<09:57, 1.28MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<07:19, 1.73MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<05:20, 2.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<15:10, 834kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<13:36, 930kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<10:09, 1.24MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<07:29, 1.69MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<05:27, 2.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<15:28, 813kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<23:15, 541kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<19:14, 654kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:30<14:04, 893kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<10:11, 1.23MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<07:17, 1.72MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<56:39, 221kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<40:21, 310kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<28:25, 439kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<23:28, 530kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<19:25, 641kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<14:14, 874kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<10:06, 1.23MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<12:33, 985kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<11:26, 1.08MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<08:35, 1.44MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<06:15, 1.97MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<07:47, 1.58MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<08:07, 1.51MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<06:14, 1.97MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<04:40, 2.62MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<06:13, 1.97MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<06:59, 1.75MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<05:27, 2.24MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<04:01, 3.03MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:47, 1.79MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<07:20, 1.66MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<06:21, 1.91MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<04:58, 2.44MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<03:46, 3.22MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:59, 2.02MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<06:49, 1.77MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<06:31, 1.85MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<05:07, 2.36MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<03:48, 3.17MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<06:30, 1.85MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:45<07:22, 1.63MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:45<06:39, 1.81MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<05:09, 2.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<03:46, 3.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<08:05, 1.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<08:25, 1.42MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<06:33, 1.82MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<04:43, 2.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:53, 1.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<09:13, 1.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<07:01, 1.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<05:00, 2.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<39:48, 297kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<30:58, 382kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<22:41, 521kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<16:14, 727kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<13:28, 872kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<12:35, 934kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<10:03, 1.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<07:23, 1.59MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<05:17, 2.21MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<23:00, 508kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<19:18, 605kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<14:38, 798kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<10:38, 1.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<07:34, 1.54MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<24:04, 483kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<18:55, 614kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<14:16, 814kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<10:13, 1.13MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<09:42, 1.19MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<09:24, 1.23MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<07:05, 1.63MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:16, 2.19MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:21, 1.80MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<07:01, 1.64MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:35, 2.05MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<04:13, 2.71MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<03:17, 3.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<02:36, 4.38MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<27:05, 421kB/s] .vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<21:38, 527kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<16:29, 692kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<11:58, 951kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<08:36, 1.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<09:28, 1.20MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<09:19, 1.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<07:05, 1.60MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<05:17, 2.14MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<03:56, 2.87MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<08:52, 1.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<08:52, 1.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:47, 1.66MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<05:04, 2.22MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<03:48, 2.94MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<07:58, 1.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<08:04, 1.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<06:12, 1.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:08<04:30, 2.47MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<06:55, 1.61MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<06:33, 1.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<05:14, 2.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<04:00, 2.77MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<03:06, 3.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<02:27, 4.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<38:00, 291kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<29:10, 379kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<21:03, 525kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<14:50, 742kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<19:38, 560kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<15:48, 695kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<11:35, 947kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<09:52, 1.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<09:03, 1.21MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<06:48, 1.60MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<04:54, 2.22MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<08:39, 1.26MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<08:12, 1.32MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<06:16, 1.73MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:18<04:30, 2.40MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<14:47, 730kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<12:33, 859kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<09:14, 1.17MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<06:41, 1.61MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<07:38, 1.40MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<07:26, 1.44MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<05:43, 1.87MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<04:06, 2.59MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<13:15, 804kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<11:28, 929kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<08:28, 1.26MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<06:07, 1.73MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<07:26, 1.42MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<07:16, 1.46MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<05:36, 1.88MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<04:02, 2.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<10:24, 1.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<09:21, 1.12MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<07:04, 1.49MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<06:37, 1.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<06:47, 1.54MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<05:17, 1.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<03:47, 2.73MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<21:39, 479kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<19:04, 544kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<14:22, 722kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<10:19, 1.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<09:10, 1.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<07:50, 1.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:46, 1.78MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<04:11, 2.44MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<09:25, 1.09MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<08:24, 1.22MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<06:21, 1.61MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<04:34, 2.23MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<09:26, 1.08MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<08:28, 1.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<06:18, 1.61MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:37, 2.19MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<06:17, 1.60MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<06:14, 1.62MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:48, 2.09MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<03:28, 2.90MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<19:12, 523kB/s] .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<15:16, 657kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<11:08, 900kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<07:55, 1.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<09:58, 1.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<08:45, 1.14MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<06:34, 1.51MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<04:41, 2.11MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<20:19, 487kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<16:02, 617kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<11:35, 853kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<08:15, 1.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<09:17, 1.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<07:41, 1.28MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<06:03, 1.62MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:24, 2.22MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<05:43, 1.70MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<07:28, 1.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<06:04, 1.60MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:27, 2.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:46, 1.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:41, 1.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:23, 2.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<03:09, 3.04MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<43:05, 223kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<31:51, 302kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<22:36, 425kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<15:58, 600kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<14:15, 670kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<11:39, 820kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<08:33, 1.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<06:04, 1.56MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<25:42, 369kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<19:36, 484kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<14:07, 670kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<09:58, 945kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<13:06, 718kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<12:29, 754kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<09:33, 984kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:50, 1.37MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<07:13, 1.29MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:02<06:40, 1.40MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<05:00, 1.86MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:38, 2.55MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<07:11, 1.29MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<09:23, 989kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<07:21, 1.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:19, 1.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<05:51, 1.57MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<05:45, 1.60MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<04:25, 2.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:17, 2.79MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<04:48, 1.90MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<04:58, 1.84MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<03:52, 2.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<02:48, 3.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<14:35, 622kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<11:47, 769kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<08:39, 1.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<06:06, 1.47MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<2:15:44, 66.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<1:36:33, 93.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<1:07:52, 132kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<48:47, 183kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<35:38, 251kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<25:17, 352kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<19:07, 463kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<16:28, 538kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<12:18, 720kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<08:46, 1.01MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:18<08:31, 1.03MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<07:29, 1.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<05:34, 1.58MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:01, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<07:00, 1.24MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<06:01, 1.45MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:29, 1.94MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:50, 1.79MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<04:54, 1.77MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<03:48, 2.27MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:07, 2.09MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<04:22, 1.96MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<03:25, 2.50MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<02:31, 3.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:08, 1.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<05:43, 1.49MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<04:19, 1.97MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:12, 2.65MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:49, 1.75MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:51, 1.74MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<03:42, 2.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<02:44, 3.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:06, 1.64MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:05, 1.65MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<03:54, 2.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<02:55, 2.86MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:17, 1.94MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:29, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<03:29, 2.37MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<03:50, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:09, 1.98MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<03:15, 2.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<02:24, 3.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<05:22, 1.52MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<05:09, 1.58MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:57, 2.06MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:09, 1.95MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:06, 1.97MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<03:39, 2.21MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<06:04, 1.33MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:27, 1.81MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<03:11, 2.51MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<13:10, 609kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<09:42, 826kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<06:51, 1.16MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<09:43, 818kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<09:14, 860kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<07:02, 1.13MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<05:06, 1.55MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<03:41, 2.14MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<10:21, 762kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<08:54, 886kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<07:06, 1.11MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<05:15, 1.50MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<03:45, 2.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<09:43, 804kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<08:05, 964kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<05:55, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:18, 1.80MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<05:18, 1.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<05:00, 1.55MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<03:54, 1.98MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<02:52, 2.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:50<04:07, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<04:07, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:08, 2.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<02:18, 3.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<05:24, 1.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<04:49, 1.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<04:05, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:03, 2.48MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:39, 2.06MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<03:49, 1.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:00, 2.50MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<02:14, 3.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:53, 1.92MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<03:56, 1.89MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:04, 2.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<02:17, 3.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<03:52, 1.91MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:58<03:58, 1.86MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<03:13, 2.29MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<02:29, 2.96MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<01:52, 3.91MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:01<08:52, 827kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:01<13:23, 547kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:01<11:05, 661kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:01<08:07, 902kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<05:49, 1.25MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:03<05:49, 1.25MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:03<05:03, 1.44MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:03<04:12, 1.73MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<03:03, 2.36MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<02:16, 3.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:05<13:40, 527kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:05<10:44, 670kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<07:47, 921kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<06:40, 1.07MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<05:48, 1.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:07<04:21, 1.63MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<04:15, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<04:26, 1.59MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<03:38, 1.94MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:09<02:48, 2.50MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<02:11, 3.21MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<01:45, 3.97MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:11<04:36, 1.52MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:11<05:07, 1.36MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:11<04:00, 1.74MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:11<03:04, 2.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<02:15, 3.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<05:35, 1.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:13<04:57, 1.39MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:13<03:45, 1.84MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<02:48, 2.45MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<02:07, 3.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<06:09, 1.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:15<05:40, 1.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:15<04:18, 1.59MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:04, 2.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<06:32, 1.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:17<05:34, 1.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:17<04:28, 1.51MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:17<03:18, 2.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<02:26, 2.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:18<05:06, 1.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<04:45, 1.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:19<03:35, 1.87MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<02:34, 2.58MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<08:01, 827kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<06:40, 994kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:21<05:16, 1.26MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<03:45, 1.75MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:46, 2.37MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<3:19:45, 32.9kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<2:21:59, 46.3kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:23<1:39:42, 65.8kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:23<1:10:13, 93.3kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:23<49:11, 133kB/s]   .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<34:27, 189kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<28:15, 230kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:25<21:06, 308kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:25<15:01, 432kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:25<10:44, 603kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<07:34, 849kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<12:24, 518kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:27<09:42, 662kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:27<07:02, 910kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<05:03, 1.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:40, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:28<12:40, 502kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<10:07, 628kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:29<07:23, 859kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<05:15, 1.20MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:30<05:46, 1.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:31<05:02, 1.25MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:31<03:46, 1.66MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:31<02:44, 2.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<04:12, 1.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:33<03:58, 1.56MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:33<03:02, 2.04MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<02:17, 2.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<01:47, 3.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<04:47, 1.28MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<04:07, 1.49MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<03:04, 1.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<03:21, 1.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<03:08, 1.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<02:26, 2.48MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<01:48, 3.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<03:33, 1.69MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<03:44, 1.61MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<03:21, 1.79MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:39<02:31, 2.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<01:52, 3.19MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:40<03:50, 1.54MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<03:41, 1.61MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<02:47, 2.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<02:04, 2.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:42<03:13, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<03:14, 1.81MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<02:32, 2.30MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<01:53, 3.09MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<03:04, 1.89MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<02:57, 1.96MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<02:32, 2.28MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<01:53, 3.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:47<03:11, 1.80MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:47<07:54, 726kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:47<06:52, 834kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<05:08, 1.11MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:47<03:41, 1.55MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:49<04:08, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:49<03:48, 1.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:49<02:58, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:49<02:13, 2.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<01:38, 3.41MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:51<08:25, 665kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:51<06:48, 822kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<04:58, 1.12MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<03:32, 1.57MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:53<05:58, 927kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:53<05:05, 1.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:53<03:43, 1.48MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:53<02:43, 2.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:55<03:36, 1.51MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:55<03:15, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:55<02:28, 2.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:55<01:50, 2.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:57<02:55, 1.85MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:57<02:56, 1.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:57<02:16, 2.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:59<02:30, 2.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:59<02:38, 2.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<02:00, 2.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:59<01:31, 3.47MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:01<02:39, 1.98MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:01<02:42, 1.94MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:01<02:05, 2.50MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<01:30, 3.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:03<11:55, 435kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:03<09:10, 565kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:03<06:35, 784kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<04:39, 1.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:05<06:01, 851kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:05<05:03, 1.01MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:05<03:43, 1.37MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:07<03:28, 1.45MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:07<04:05, 1.23MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:07<03:11, 1.58MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:07<02:21, 2.12MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<01:42, 2.92MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:09<12:51, 387kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:09<09:47, 509kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:09<07:01, 706kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:11<05:44, 856kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:11<04:48, 1.02MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:11<03:33, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<03:19, 1.46MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:13<03:06, 1.56MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:13<02:20, 2.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:14<02:28, 1.93MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:15<02:20, 2.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:15<01:45, 2.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:16<02:11, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:17<02:41, 1.75MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:17<02:10, 2.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:17<01:34, 2.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:18<03:51, 1.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:19<03:26, 1.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:19<02:35, 1.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:19<01:51, 2.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<04:35, 993kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:21<03:57, 1.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:21<02:56, 1.54MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<02:06, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<06:05, 738kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:23<05:00, 898kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:23<03:40, 1.22MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<02:36, 1.70MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<05:31, 801kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:25<04:35, 965kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:25<03:20, 1.32MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:25<02:25, 1.81MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<01:47, 2.43MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<10:34, 412kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<08:05, 539kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:27<05:47, 749kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<04:04, 1.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<07:04, 607kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<05:37, 761kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:29<04:09, 1.03MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:29<02:58, 1.43MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:30<03:22, 1.25MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:30<03:03, 1.38MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:31<02:17, 1.83MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:32<02:19, 1.79MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:32<02:26, 1.70MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:32<02:09, 1.92MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:33<01:43, 2.41MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:33<01:16, 3.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:35<03:08, 1.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:35<05:32, 736kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:36<04:53, 835kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<03:37, 1.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:36<02:36, 1.55MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:37<02:45, 1.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:37<02:26, 1.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:37<01:50, 2.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:38<01:21, 2.93MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:38<01:01, 3.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:39<1:52:26, 35.1kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<1:19:15, 49.7kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<55:27, 70.9kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:40<38:39, 101kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:41<28:02, 138kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:41<20:15, 191kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:41<14:14, 271kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:42<10:00, 383kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:43<08:08, 468kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:43<06:20, 601kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:43<04:37, 822kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:43<03:17, 1.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<03:14, 1.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<03:04, 1.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<02:41, 1.39MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<02:09, 1.73MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:46<01:34, 2.34MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:47<01:56, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:47<01:50, 1.99MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:47<01:37, 2.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<01:15, 2.88MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:48<00:56, 3.82MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:49<02:10, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:49<02:07, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<01:37, 2.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<01:44, 2.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<02:27, 1.44MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<02:30, 1.41MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:51<02:12, 1.60MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:51<01:43, 2.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:52<01:16, 2.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:53<01:58, 1.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<01:55, 1.79MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:53<01:28, 2.32MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:54<01:04, 3.17MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:55<02:53, 1.17MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<02:35, 1.31MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<01:56, 1.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:55<01:24, 2.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:56<01:03, 3.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<19:22, 172kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<14:01, 237kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<10:05, 329kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<07:07, 463kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:57<05:00, 653kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:59<05:02, 647kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:59<04:07, 790kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<03:02, 1.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:59<02:10, 1.48MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:01<02:29, 1.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:01<02:43, 1.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:02<02:09, 1.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:02<01:33, 2.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:03<02:01, 1.54MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:03<01:50, 1.69MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:04<01:25, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:04<01:02, 2.93MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:05<01:40, 1.83MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:05<02:10, 1.40MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:06<01:45, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:06<01:16, 2.35MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:07<01:47, 1.66MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<01:44, 1.71MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:08<01:20, 2.22MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:09<01:26, 2.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<01:29, 1.96MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:10<01:09, 2.51MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:10<00:49, 3.45MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<17:11, 165kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<12:28, 228kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:12<08:47, 321kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:12<06:07, 456kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<05:48, 477kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<04:31, 614kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:14<03:15, 846kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<02:42, 996kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<02:14, 1.21MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<01:38, 1.63MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<01:40, 1.57MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<02:03, 1.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:18<01:39, 1.59MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:18<01:11, 2.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<01:35, 1.62MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<01:30, 1.69MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<01:09, 2.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:22<01:26, 1.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:22<03:32, 704kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:22<03:06, 805kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:22<02:27, 1.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:22<01:46, 1.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:23<01:17, 1.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:23<00:57, 2.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:24<06:16, 388kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:24<04:37, 525kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:24<03:27, 701kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:24<02:26, 979kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:26<02:12, 1.06MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:26<02:19, 1.01MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:26<01:48, 1.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:27<01:17, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<00:56, 2.42MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:28<03:43, 613kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:28<02:56, 777kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:28<02:15, 1.01MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:28<01:38, 1.38MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:29<01:10, 1.90MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:30<02:03, 1.08MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:30<02:04, 1.07MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:31<01:51, 1.19MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:31<01:25, 1.55MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:31<01:03, 2.08MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:31<00:46, 2.81MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:33<01:55, 1.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:33<02:05, 1.02MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:33<01:37, 1.32MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:33<01:12, 1.76MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:33<00:51, 2.43MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:35<03:14, 640kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:35<02:35, 799kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:35<01:53, 1.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:35<01:19, 1.52MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:37<01:57, 1.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:37<01:42, 1.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:37<01:14, 1.60MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:37<00:54, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:38<01:14, 1.57MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:39<01:10, 1.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:39<00:53, 2.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<00:38, 2.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:40<03:07, 599kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:41<02:28, 752kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:41<01:47, 1.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:42<01:32, 1.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:43<01:22, 1.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:43<01:01, 1.74MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:44<01:00, 1.71MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:45<01:17, 1.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:45<01:01, 1.68MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:45<00:47, 2.18MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:45<00:34, 2.96MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:47<01:21, 1.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:47<02:07, 785kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:47<01:45, 948kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:48<01:16, 1.29MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:48<00:53, 1.80MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:49<01:40, 955kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:49<01:17, 1.24MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:49<00:57, 1.64MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:49<00:40, 2.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:51<01:32, 989kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:52<02:34, 591kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:52<02:09, 705kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:52<01:34, 958kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:52<01:06, 1.33MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:54<01:24, 1.04MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:54<02:24, 607kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:55<01:58, 738kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:55<01:32, 943kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:55<01:05, 1.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:55<00:47, 1.79MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:57<01:14, 1.12MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:57<01:19, 1.05MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:57<01:01, 1.35MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:57<00:46, 1.78MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:57<00:33, 2.39MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:57<00:25, 3.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:59<01:16, 1.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:59<01:01, 1.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:59<00:44, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:59<00:31, 2.41MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [06:01<02:15, 554kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [06:01<01:45, 711kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:01<01:17, 955kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:01<00:55, 1.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:01<00:38, 1.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:03<02:38, 447kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:03<02:01, 580kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:03<01:26, 801kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:05<01:10, 951kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:05<00:59, 1.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:05<00:43, 1.51MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:05<00:30, 2.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:07<00:45, 1.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:07<00:42, 1.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:07<00:31, 1.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:07<00:21, 2.69MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:09<01:45, 555kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:09<01:22, 704kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:09<00:59, 966kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:09<00:39, 1.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:11<27:01, 33.5kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:11<18:57, 47.6kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:11<13:15, 67.6kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:11<09:00, 96.4kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:13<06:18, 133kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:13<04:31, 184kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:13<03:08, 260kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:13<02:04, 370kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:14<05:57, 129kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:15<04:16, 179kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:15<02:57, 253kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:15<01:59, 360kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:17<01:52, 372kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:17<01:48, 386kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:17<01:21, 510kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:17<00:59, 694kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:17<00:41, 962kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:18<00:28, 1.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:19<00:44, 859kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:20<00:45, 829kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:20<00:34, 1.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:20<00:25, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:20<00:17, 2.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:21<00:25, 1.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:21<00:20, 1.63MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:21<00:15, 2.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:21<00:11, 2.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:21<00:09, 3.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:22<00:08, 3.81MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:22<00:06, 4.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:24<00:52, 567kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:24<00:55, 535kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:24<00:43, 678kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:24<00:32, 900kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:24<00:21, 1.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:24<00:15, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:26<00:28, 906kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:26<00:23, 1.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:26<00:19, 1.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:26<00:15, 1.57MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:26<00:12, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:26<00:10, 2.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:26<00:08, 2.68MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:26<00:06, 3.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:28<00:16, 1.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:28<00:15, 1.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:28<00:11, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:28<00:09, 2.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:28<00:06, 2.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:28<00:04, 3.69MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:30<00:14, 1.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:30<00:12, 1.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:30<00:10, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:30<00:08, 2.00MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:30<00:06, 2.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:30<00:04, 3.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:30<00:03, 3.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:32<00:07, 1.69MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:32<00:06, 1.82MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:32<00:04, 2.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:32<00:03, 3.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:34<00:06, 1.32MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:34<00:07, 1.26MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:34<00:06, 1.42MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:34<00:04, 1.70MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:34<00:03, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:34<00:02, 2.84MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:34<00:01, 3.64MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:36<00:03, 1.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:36<00:02, 1.63MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:36<00:01, 2.05MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:36<00:01, 2.71MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:36<00:00, 3.66MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:38<00:01, 441kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:38<00:00, 538kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:38<00:00, 717kB/s].vector_cache/glove.6B.zip: 862MB [06:38, 2.16MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 616/400000 [00:00<01:04, 6153.23it/s]  0%|          | 1104/400000 [00:00<01:09, 5705.85it/s]  0%|          | 1587/400000 [00:00<01:13, 5410.22it/s]  1%|          | 2140/400000 [00:00<01:13, 5444.86it/s]  1%|          | 2619/400000 [00:00<01:15, 5228.71it/s]  1%|          | 3053/400000 [00:00<01:20, 4924.91it/s]  1%|          | 3658/400000 [00:00<01:15, 5215.27it/s]  1%|          | 4289/400000 [00:00<01:11, 5500.97it/s]  1%|          | 4861/400000 [00:00<01:11, 5564.02it/s]  1%|▏         | 5572/400000 [00:01<01:06, 5951.75it/s]  2%|▏         | 6284/400000 [00:01<01:02, 6257.19it/s]  2%|▏         | 6911/400000 [00:01<01:06, 5899.41it/s]  2%|▏         | 7548/400000 [00:01<01:05, 6031.73it/s]  2%|▏         | 8156/400000 [00:01<01:05, 5956.61it/s]  2%|▏         | 8775/400000 [00:01<01:04, 6024.27it/s]  2%|▏         | 9380/400000 [00:01<01:08, 5674.94it/s]  2%|▏         | 9954/400000 [00:01<01:14, 5206.04it/s]  3%|▎         | 10487/400000 [00:01<01:18, 4980.41it/s]  3%|▎         | 10997/400000 [00:01<01:17, 5013.70it/s]  3%|▎         | 11506/400000 [00:02<01:18, 4921.60it/s]  3%|▎         | 12004/400000 [00:02<01:19, 4886.99it/s]  3%|▎         | 12605/400000 [00:02<01:14, 5176.45it/s]  3%|▎         | 13318/400000 [00:02<01:08, 5639.06it/s]  4%|▎         | 14031/400000 [00:02<01:04, 6015.06it/s]  4%|▎         | 14716/400000 [00:02<01:01, 6242.71it/s]  4%|▍         | 15361/400000 [00:02<01:01, 6303.31it/s]  4%|▍         | 16003/400000 [00:02<01:04, 5909.44it/s]  4%|▍         | 16607/400000 [00:02<01:05, 5841.87it/s]  4%|▍         | 17201/400000 [00:03<01:08, 5590.55it/s]  4%|▍         | 17769/400000 [00:03<01:11, 5345.33it/s]  5%|▍         | 18312/400000 [00:03<01:14, 5137.65it/s]  5%|▍         | 18834/400000 [00:03<01:15, 5059.67it/s]  5%|▍         | 19422/400000 [00:03<01:12, 5279.83it/s]  5%|▍         | 19957/400000 [00:03<01:13, 5139.66it/s]  5%|▌         | 20478/400000 [00:03<01:13, 5158.16it/s]  5%|▌         | 20998/400000 [00:03<01:15, 4988.39it/s]  5%|▌         | 21546/400000 [00:03<01:13, 5125.50it/s]  6%|▌         | 22136/400000 [00:04<01:10, 5335.47it/s]  6%|▌         | 22715/400000 [00:04<01:09, 5462.66it/s]  6%|▌         | 23266/400000 [00:04<01:11, 5294.00it/s]  6%|▌         | 23800/400000 [00:04<01:15, 4976.71it/s]  6%|▌         | 24510/400000 [00:04<01:08, 5466.31it/s]  6%|▋         | 25076/400000 [00:04<01:08, 5478.99it/s]  6%|▋         | 25638/400000 [00:04<01:08, 5433.36it/s]  7%|▋         | 26258/400000 [00:04<01:06, 5641.05it/s]  7%|▋         | 26831/400000 [00:04<01:08, 5465.01it/s]  7%|▋         | 27550/400000 [00:04<01:03, 5888.25it/s]  7%|▋         | 28269/400000 [00:05<00:59, 6224.65it/s]  7%|▋         | 28907/400000 [00:05<00:59, 6268.67it/s]  7%|▋         | 29545/400000 [00:05<00:59, 6194.37it/s]  8%|▊         | 30180/400000 [00:05<00:59, 6239.67it/s]  8%|▊         | 30892/400000 [00:05<00:56, 6479.83it/s]  8%|▊         | 31599/400000 [00:05<00:55, 6644.33it/s]  8%|▊         | 32269/400000 [00:05<00:56, 6481.71it/s]  8%|▊         | 32923/400000 [00:05<00:56, 6493.85it/s]  8%|▊         | 33576/400000 [00:05<00:56, 6490.48it/s]  9%|▊         | 34300/400000 [00:05<00:54, 6696.32it/s]  9%|▉         | 35024/400000 [00:06<00:53, 6849.11it/s]  9%|▉         | 35724/400000 [00:06<00:53, 6872.28it/s]  9%|▉         | 36414/400000 [00:06<00:55, 6576.84it/s]  9%|▉         | 37076/400000 [00:06<00:55, 6587.28it/s]  9%|▉         | 37770/400000 [00:06<00:54, 6687.44it/s] 10%|▉         | 38442/400000 [00:06<00:58, 6144.55it/s] 10%|▉         | 39070/400000 [00:06<00:58, 6184.05it/s] 10%|▉         | 39716/400000 [00:06<00:57, 6263.63it/s] 10%|█         | 40348/400000 [00:06<00:58, 6161.74it/s] 10%|█         | 41068/400000 [00:07<00:55, 6439.00it/s] 10%|█         | 41784/400000 [00:07<00:53, 6638.64it/s] 11%|█         | 42454/400000 [00:07<00:53, 6624.28it/s] 11%|█         | 43121/400000 [00:07<00:56, 6365.12it/s] 11%|█         | 43763/400000 [00:07<01:02, 5706.59it/s] 11%|█         | 44350/400000 [00:07<01:05, 5471.14it/s] 11%|█         | 44972/400000 [00:07<01:02, 5675.31it/s] 11%|█▏        | 45592/400000 [00:07<01:01, 5806.47it/s] 12%|█▏        | 46182/400000 [00:07<01:04, 5521.01it/s] 12%|█▏        | 46901/400000 [00:08<00:59, 5932.75it/s] 12%|█▏        | 47636/400000 [00:08<00:55, 6295.61it/s] 12%|█▏        | 48309/400000 [00:08<00:54, 6419.85it/s] 12%|█▏        | 49031/400000 [00:08<00:52, 6639.62it/s] 12%|█▏        | 49716/400000 [00:08<00:52, 6699.38it/s] 13%|█▎        | 50451/400000 [00:08<00:50, 6881.71it/s] 13%|█▎        | 51146/400000 [00:08<00:52, 6650.80it/s] 13%|█▎        | 51818/400000 [00:08<00:53, 6482.89it/s] 13%|█▎        | 52472/400000 [00:08<00:54, 6391.71it/s] 13%|█▎        | 53116/400000 [00:08<00:55, 6295.96it/s] 13%|█▎        | 53850/400000 [00:09<00:52, 6575.17it/s] 14%|█▎        | 54513/400000 [00:09<00:57, 6033.61it/s] 14%|█▍        | 55129/400000 [00:09<00:57, 6021.69it/s] 14%|█▍        | 55740/400000 [00:09<00:57, 5983.94it/s] 14%|█▍        | 56364/400000 [00:09<00:56, 6056.90it/s] 14%|█▍        | 56975/400000 [00:09<00:56, 6025.92it/s] 14%|█▍        | 57581/400000 [00:09<00:56, 6015.26it/s] 15%|█▍        | 58280/400000 [00:09<00:54, 6276.27it/s] 15%|█▍        | 58928/400000 [00:09<00:53, 6333.87it/s] 15%|█▍        | 59565/400000 [00:10<00:54, 6233.66it/s] 15%|█▌        | 60242/400000 [00:10<00:53, 6384.99it/s] 15%|█▌        | 60884/400000 [00:10<00:53, 6392.34it/s] 15%|█▌        | 61618/400000 [00:10<00:50, 6649.62it/s] 16%|█▌        | 62342/400000 [00:10<00:49, 6815.61it/s] 16%|█▌        | 63037/400000 [00:10<00:49, 6854.98it/s] 16%|█▌        | 63726/400000 [00:10<00:49, 6803.99it/s] 16%|█▌        | 64409/400000 [00:10<00:51, 6535.39it/s] 16%|█▋        | 65067/400000 [00:10<00:51, 6484.04it/s] 16%|█▋        | 65719/400000 [00:10<00:52, 6334.34it/s] 17%|█▋        | 66387/400000 [00:11<00:51, 6433.14it/s] 17%|█▋        | 67033/400000 [00:11<00:52, 6357.86it/s] 17%|█▋        | 67671/400000 [00:11<00:53, 6246.99it/s] 17%|█▋        | 68303/400000 [00:11<00:52, 6266.60it/s] 17%|█▋        | 68931/400000 [00:11<00:53, 6242.21it/s] 17%|█▋        | 69641/400000 [00:11<00:51, 6474.88it/s] 18%|█▊        | 70364/400000 [00:11<00:49, 6682.07it/s] 18%|█▊        | 71036/400000 [00:11<00:50, 6462.44it/s] 18%|█▊        | 71721/400000 [00:11<00:49, 6572.07it/s] 18%|█▊        | 72382/400000 [00:11<00:49, 6581.30it/s] 18%|█▊        | 73043/400000 [00:12<00:50, 6443.20it/s] 18%|█▊        | 73774/400000 [00:12<00:48, 6680.83it/s] 19%|█▊        | 74446/400000 [00:12<00:49, 6600.08it/s] 19%|█▉        | 75109/400000 [00:12<00:50, 6370.99it/s] 19%|█▉        | 75750/400000 [00:12<00:52, 6164.87it/s] 19%|█▉        | 76457/400000 [00:12<00:50, 6410.52it/s] 19%|█▉        | 77104/400000 [00:12<00:56, 5679.13it/s] 19%|█▉        | 77692/400000 [00:12<00:58, 5523.67it/s] 20%|█▉        | 78349/400000 [00:12<00:55, 5749.82it/s] 20%|█▉        | 78937/400000 [00:13<00:57, 5546.74it/s] 20%|█▉        | 79659/400000 [00:13<00:53, 5959.71it/s] 20%|██        | 80298/400000 [00:13<00:52, 6081.98it/s] 20%|██        | 80918/400000 [00:13<00:52, 6109.44it/s] 20%|██        | 81617/400000 [00:13<00:50, 6348.35it/s] 21%|██        | 82332/400000 [00:13<00:48, 6567.99it/s] 21%|██        | 83056/400000 [00:13<00:46, 6755.39it/s] 21%|██        | 83780/400000 [00:13<00:45, 6891.77it/s] 21%|██        | 84475/400000 [00:13<00:48, 6537.32it/s] 21%|██▏       | 85173/400000 [00:14<00:47, 6663.33it/s] 21%|██▏       | 85896/400000 [00:14<00:46, 6823.44it/s] 22%|██▏       | 86591/400000 [00:14<00:45, 6859.38it/s] 22%|██▏       | 87281/400000 [00:14<00:45, 6867.57it/s] 22%|██▏       | 87971/400000 [00:14<00:47, 6554.99it/s] 22%|██▏       | 88659/400000 [00:14<00:46, 6648.37it/s] 22%|██▏       | 89370/400000 [00:14<00:45, 6779.87it/s] 23%|██▎       | 90092/400000 [00:14<00:44, 6904.44it/s] 23%|██▎       | 90809/400000 [00:14<00:44, 6981.33it/s] 23%|██▎       | 91510/400000 [00:14<00:48, 6361.41it/s] 23%|██▎       | 92159/400000 [00:15<00:50, 6125.81it/s] 23%|██▎       | 92858/400000 [00:15<00:48, 6361.72it/s] 23%|██▎       | 93589/400000 [00:15<00:46, 6617.22it/s] 24%|██▎       | 94261/400000 [00:15<00:49, 6192.98it/s] 24%|██▎       | 94893/400000 [00:15<00:49, 6174.43it/s] 24%|██▍       | 95584/400000 [00:15<00:47, 6376.09it/s] 24%|██▍       | 96320/400000 [00:15<00:45, 6641.22it/s] 24%|██▍       | 96993/400000 [00:15<00:46, 6526.03it/s] 24%|██▍       | 97714/400000 [00:15<00:45, 6716.14it/s] 25%|██▍       | 98392/400000 [00:16<00:45, 6638.67it/s] 25%|██▍       | 99061/400000 [00:16<00:45, 6589.04it/s] 25%|██▍       | 99792/400000 [00:16<00:44, 6788.00it/s] 25%|██▌       | 100475/400000 [00:16<00:46, 6475.89it/s] 25%|██▌       | 101129/400000 [00:16<00:49, 5979.73it/s] 25%|██▌       | 101802/400000 [00:16<00:48, 6184.59it/s] 26%|██▌       | 102530/400000 [00:16<00:45, 6476.07it/s] 26%|██▌       | 103255/400000 [00:16<00:44, 6688.90it/s] 26%|██▌       | 103939/400000 [00:16<00:43, 6729.76it/s] 26%|██▌       | 104651/400000 [00:16<00:43, 6835.77it/s] 26%|██▋       | 105340/400000 [00:17<00:44, 6605.93it/s] 27%|██▋       | 106068/400000 [00:17<00:43, 6794.23it/s] 27%|██▋       | 106802/400000 [00:17<00:42, 6949.01it/s] 27%|██▋       | 107525/400000 [00:17<00:41, 7029.37it/s] 27%|██▋       | 108232/400000 [00:17<00:45, 6466.25it/s] 27%|██▋       | 108890/400000 [00:17<00:48, 5983.87it/s] 27%|██▋       | 109614/400000 [00:17<00:46, 6310.86it/s] 28%|██▊       | 110332/400000 [00:17<00:44, 6546.73it/s] 28%|██▊       | 111061/400000 [00:17<00:42, 6751.03it/s] 28%|██▊       | 111747/400000 [00:18<00:48, 5917.35it/s] 28%|██▊       | 112468/400000 [00:18<00:45, 6252.07it/s] 28%|██▊       | 113117/400000 [00:18<00:47, 6059.59it/s] 28%|██▊       | 113741/400000 [00:18<00:49, 5841.27it/s] 29%|██▊       | 114340/400000 [00:18<00:49, 5782.67it/s] 29%|██▊       | 114929/400000 [00:18<00:50, 5695.20it/s] 29%|██▉       | 115662/400000 [00:18<00:46, 6102.68it/s] 29%|██▉       | 116286/400000 [00:18<00:47, 6021.73it/s] 29%|██▉       | 117025/400000 [00:18<00:44, 6374.91it/s] 29%|██▉       | 117675/400000 [00:19<00:45, 6269.13it/s] 30%|██▉       | 118378/400000 [00:19<00:43, 6478.92it/s] 30%|██▉       | 119034/400000 [00:19<00:43, 6421.55it/s] 30%|██▉       | 119682/400000 [00:19<00:44, 6355.80it/s] 30%|███       | 120322/400000 [00:19<00:45, 6090.84it/s] 30%|███       | 120937/400000 [00:19<00:48, 5702.70it/s] 30%|███       | 121614/400000 [00:19<00:46, 5984.38it/s] 31%|███       | 122243/400000 [00:19<00:45, 6067.39it/s] 31%|███       | 122876/400000 [00:19<00:45, 6142.70it/s] 31%|███       | 123512/400000 [00:19<00:44, 6204.74it/s] 31%|███       | 124225/400000 [00:20<00:42, 6455.93it/s] 31%|███       | 124917/400000 [00:20<00:41, 6587.02it/s] 31%|███▏      | 125586/400000 [00:20<00:41, 6615.47it/s] 32%|███▏      | 126251/400000 [00:20<00:44, 6216.82it/s] 32%|███▏      | 126880/400000 [00:20<00:45, 5995.40it/s] 32%|███▏      | 127581/400000 [00:20<00:43, 6265.05it/s] 32%|███▏      | 128288/400000 [00:20<00:41, 6482.50it/s] 32%|███▏      | 128944/400000 [00:20<00:42, 6427.55it/s] 32%|███▏      | 129677/400000 [00:20<00:40, 6672.37it/s] 33%|███▎      | 130414/400000 [00:21<00:39, 6866.59it/s] 33%|███▎      | 131120/400000 [00:21<00:38, 6922.91it/s] 33%|███▎      | 131817/400000 [00:21<00:38, 6928.17it/s] 33%|███▎      | 132513/400000 [00:21<00:38, 6874.08it/s] 33%|███▎      | 133203/400000 [00:21<00:38, 6846.84it/s] 33%|███▎      | 133890/400000 [00:21<00:39, 6710.28it/s] 34%|███▎      | 134587/400000 [00:21<00:39, 6783.98it/s] 34%|███▍      | 135267/400000 [00:21<00:39, 6754.09it/s] 34%|███▍      | 135944/400000 [00:21<00:43, 6083.58it/s] 34%|███▍      | 136599/400000 [00:21<00:42, 6215.97it/s] 34%|███▍      | 137323/400000 [00:22<00:40, 6491.38it/s] 35%|███▍      | 138021/400000 [00:22<00:39, 6630.12it/s] 35%|███▍      | 138716/400000 [00:22<00:38, 6721.49it/s] 35%|███▍      | 139394/400000 [00:22<00:39, 6542.98it/s] 35%|███▌      | 140115/400000 [00:22<00:38, 6728.47it/s] 35%|███▌      | 140854/400000 [00:22<00:37, 6913.76it/s] 35%|███▌      | 141561/400000 [00:22<00:37, 6957.69it/s] 36%|███▌      | 142300/400000 [00:22<00:36, 7081.51it/s] 36%|███▌      | 143012/400000 [00:22<00:37, 6832.80it/s] 36%|███▌      | 143700/400000 [00:23<00:39, 6533.24it/s] 36%|███▌      | 144359/400000 [00:23<00:40, 6354.39it/s] 36%|███▋      | 145047/400000 [00:23<00:39, 6501.87it/s] 36%|███▋      | 145722/400000 [00:23<00:38, 6571.74it/s] 37%|███▋      | 146383/400000 [00:23<00:43, 5837.05it/s] 37%|███▋      | 146986/400000 [00:23<00:42, 5890.49it/s] 37%|███▋      | 147588/400000 [00:23<00:42, 5895.24it/s] 37%|███▋      | 148324/400000 [00:23<00:40, 6268.76it/s] 37%|███▋      | 148964/400000 [00:23<00:39, 6278.17it/s] 37%|███▋      | 149676/400000 [00:23<00:38, 6507.38it/s] 38%|███▊      | 150410/400000 [00:24<00:37, 6714.19it/s] 38%|███▊      | 151115/400000 [00:24<00:36, 6810.15it/s] 38%|███▊      | 151802/400000 [00:24<00:37, 6661.71it/s] 38%|███▊      | 152539/400000 [00:24<00:36, 6857.03it/s] 38%|███▊      | 153230/400000 [00:24<00:36, 6767.60it/s] 38%|███▊      | 153962/400000 [00:24<00:35, 6922.62it/s] 39%|███▊      | 154684/400000 [00:24<00:35, 7007.39it/s] 39%|███▉      | 155421/400000 [00:24<00:34, 7111.87it/s] 39%|███▉      | 156155/400000 [00:24<00:33, 7178.04it/s] 39%|███▉      | 156875/400000 [00:25<00:35, 6867.11it/s] 39%|███▉      | 157616/400000 [00:25<00:34, 7019.83it/s] 40%|███▉      | 158322/400000 [00:25<00:35, 6725.23it/s] 40%|███▉      | 159000/400000 [00:25<00:35, 6725.99it/s] 40%|███▉      | 159677/400000 [00:25<00:36, 6503.64it/s] 40%|████      | 160332/400000 [00:25<00:38, 6231.10it/s] 40%|████      | 161068/400000 [00:25<00:36, 6531.45it/s] 40%|████      | 161776/400000 [00:25<00:35, 6684.76it/s] 41%|████      | 162518/400000 [00:25<00:34, 6887.43it/s] 41%|████      | 163247/400000 [00:25<00:33, 7001.45it/s] 41%|████      | 163952/400000 [00:26<00:37, 6297.74it/s] 41%|████      | 164668/400000 [00:26<00:36, 6529.86it/s] 41%|████▏     | 165335/400000 [00:26<00:35, 6564.22it/s] 42%|████▏     | 166022/400000 [00:26<00:35, 6650.77it/s] 42%|████▏     | 166729/400000 [00:26<00:34, 6769.65it/s] 42%|████▏     | 167413/400000 [00:26<00:34, 6789.55it/s] 42%|████▏     | 168117/400000 [00:26<00:33, 6860.47it/s] 42%|████▏     | 168835/400000 [00:26<00:33, 6952.30it/s] 42%|████▏     | 169582/400000 [00:26<00:32, 7097.67it/s] 43%|████▎     | 170295/400000 [00:26<00:32, 7087.26it/s] 43%|████▎     | 171006/400000 [00:27<00:32, 6997.94it/s] 43%|████▎     | 171708/400000 [00:27<00:33, 6878.73it/s] 43%|████▎     | 172398/400000 [00:27<00:34, 6519.77it/s] 43%|████▎     | 173118/400000 [00:27<00:33, 6709.23it/s] 43%|████▎     | 173794/400000 [00:27<00:34, 6492.88it/s] 44%|████▎     | 174456/400000 [00:27<00:34, 6529.14it/s] 44%|████▍     | 175167/400000 [00:27<00:33, 6691.33it/s] 44%|████▍     | 175912/400000 [00:27<00:32, 6900.41it/s] 44%|████▍     | 176658/400000 [00:27<00:31, 7056.90it/s] 44%|████▍     | 177368/400000 [00:28<00:33, 6583.02it/s] 45%|████▍     | 178054/400000 [00:28<00:33, 6661.64it/s] 45%|████▍     | 178750/400000 [00:28<00:32, 6746.68it/s] 45%|████▍     | 179442/400000 [00:28<00:32, 6795.54it/s] 45%|████▌     | 180126/400000 [00:28<00:33, 6584.75it/s] 45%|████▌     | 180789/400000 [00:28<00:33, 6502.89it/s] 45%|████▌     | 181443/400000 [00:28<00:34, 6426.98it/s] 46%|████▌     | 182088/400000 [00:28<00:34, 6375.27it/s] 46%|████▌     | 182728/400000 [00:28<00:35, 6052.06it/s] 46%|████▌     | 183441/400000 [00:29<00:34, 6338.40it/s] 46%|████▌     | 184082/400000 [00:29<00:36, 5899.85it/s] 46%|████▌     | 184699/400000 [00:29<00:36, 5978.31it/s] 46%|████▋     | 185321/400000 [00:29<00:35, 6046.00it/s] 46%|████▋     | 185932/400000 [00:29<00:37, 5748.79it/s] 47%|████▋     | 186514/400000 [00:29<00:38, 5603.53it/s] 47%|████▋     | 187081/400000 [00:29<00:40, 5271.45it/s] 47%|████▋     | 187617/400000 [00:29<00:41, 5121.19it/s] 47%|████▋     | 188137/400000 [00:29<00:41, 5144.46it/s] 47%|████▋     | 188782/400000 [00:29<00:38, 5475.46it/s] 47%|████▋     | 189339/400000 [00:30<00:38, 5428.06it/s] 47%|████▋     | 189889/400000 [00:30<00:39, 5295.03it/s] 48%|████▊     | 190424/400000 [00:30<00:41, 5089.40it/s] 48%|████▊     | 190939/400000 [00:30<00:41, 5072.91it/s] 48%|████▊     | 191545/400000 [00:30<00:39, 5332.19it/s] 48%|████▊     | 192094/400000 [00:30<00:38, 5376.48it/s] 48%|████▊     | 192637/400000 [00:30<00:40, 5166.88it/s] 48%|████▊     | 193159/400000 [00:30<00:40, 5052.74it/s] 48%|████▊     | 193669/400000 [00:30<00:43, 4783.98it/s] 49%|████▊     | 194311/400000 [00:31<00:39, 5178.38it/s] 49%|████▊     | 194850/400000 [00:31<00:39, 5238.82it/s] 49%|████▉     | 195467/400000 [00:31<00:37, 5486.31it/s] 49%|████▉     | 196026/400000 [00:31<00:40, 5043.64it/s] 49%|████▉     | 196679/400000 [00:31<00:37, 5412.42it/s] 49%|████▉     | 197409/400000 [00:31<00:34, 5866.27it/s] 50%|████▉     | 198077/400000 [00:31<00:33, 6087.17it/s] 50%|████▉     | 198801/400000 [00:31<00:31, 6390.97it/s] 50%|████▉     | 199538/400000 [00:31<00:30, 6655.62it/s] 50%|█████     | 200219/400000 [00:32<00:34, 5788.61it/s] 50%|█████     | 200829/400000 [00:32<00:38, 5211.54it/s] 50%|█████     | 201383/400000 [00:32<00:38, 5107.90it/s] 50%|█████     | 201917/400000 [00:32<00:39, 5021.68it/s] 51%|█████     | 202458/400000 [00:32<00:38, 5129.14it/s] 51%|█████     | 202983/400000 [00:32<00:39, 4948.48it/s] 51%|█████     | 203488/400000 [00:32<00:41, 4770.79it/s] 51%|█████     | 203974/400000 [00:32<00:42, 4631.38it/s] 51%|█████     | 204472/400000 [00:32<00:41, 4728.59it/s] 51%|█████     | 204979/400000 [00:33<00:40, 4825.63it/s] 51%|█████▏    | 205535/400000 [00:33<00:38, 5022.88it/s] 52%|█████▏    | 206208/400000 [00:33<00:35, 5435.96it/s] 52%|█████▏    | 206776/400000 [00:33<00:35, 5503.88it/s] 52%|█████▏    | 207336/400000 [00:33<00:36, 5234.18it/s] 52%|█████▏    | 207869/400000 [00:33<00:40, 4750.16it/s] 52%|█████▏    | 208424/400000 [00:33<00:38, 4964.78it/s] 52%|█████▏    | 208997/400000 [00:33<00:36, 5168.30it/s] 52%|█████▏    | 209592/400000 [00:33<00:35, 5380.03it/s] 53%|█████▎    | 210189/400000 [00:34<00:34, 5542.82it/s] 53%|█████▎    | 210825/400000 [00:34<00:32, 5763.41it/s] 53%|█████▎    | 211410/400000 [00:34<00:33, 5609.64it/s] 53%|█████▎    | 211978/400000 [00:34<00:34, 5426.05it/s] 53%|█████▎    | 212653/400000 [00:34<00:32, 5765.23it/s] 53%|█████▎    | 213240/400000 [00:34<00:33, 5613.57it/s] 53%|█████▎    | 213810/400000 [00:34<00:33, 5519.18it/s] 54%|█████▎    | 214368/400000 [00:34<00:33, 5514.11it/s] 54%|█████▎    | 214940/400000 [00:34<00:33, 5573.76it/s] 54%|█████▍    | 215501/400000 [00:34<00:33, 5523.28it/s] 54%|█████▍    | 216056/400000 [00:35<00:35, 5231.22it/s] 54%|█████▍    | 216636/400000 [00:35<00:34, 5387.15it/s] 54%|█████▍    | 217354/400000 [00:35<00:31, 5821.68it/s] 55%|█████▍    | 218072/400000 [00:35<00:29, 6171.66it/s] 55%|█████▍    | 218726/400000 [00:35<00:28, 6267.21it/s] 55%|█████▍    | 219364/400000 [00:35<00:31, 5657.06it/s] 55%|█████▌    | 220014/400000 [00:35<00:30, 5884.81it/s] 55%|█████▌    | 220696/400000 [00:35<00:29, 6135.00it/s] 55%|█████▌    | 221324/400000 [00:35<00:29, 5971.91it/s] 55%|█████▌    | 221995/400000 [00:36<00:28, 6175.46it/s] 56%|█████▌    | 222622/400000 [00:36<00:32, 5427.32it/s] 56%|█████▌    | 223188/400000 [00:36<00:32, 5424.48it/s] 56%|█████▌    | 223770/400000 [00:36<00:31, 5535.32it/s] 56%|█████▌    | 224336/400000 [00:36<00:34, 5084.14it/s] 56%|█████▌    | 224860/400000 [00:36<00:36, 4773.75it/s] 56%|█████▋    | 225353/400000 [00:36<00:36, 4760.12it/s] 57%|█████▋    | 226057/400000 [00:36<00:32, 5271.78it/s] 57%|█████▋    | 226713/400000 [00:36<00:30, 5600.90it/s] 57%|█████▋    | 227378/400000 [00:37<00:29, 5873.12it/s] 57%|█████▋    | 227985/400000 [00:37<00:30, 5550.90it/s] 57%|█████▋    | 228558/400000 [00:37<00:31, 5423.09it/s] 57%|█████▋    | 229132/400000 [00:37<00:31, 5501.34it/s] 57%|█████▋    | 229692/400000 [00:37<00:33, 5074.99it/s] 58%|█████▊    | 230213/400000 [00:37<00:34, 4912.28it/s] 58%|█████▊    | 230759/400000 [00:37<00:33, 5062.50it/s] 58%|█████▊    | 231412/400000 [00:37<00:31, 5426.89it/s] 58%|█████▊    | 231968/400000 [00:37<00:31, 5412.12it/s] 58%|█████▊    | 232538/400000 [00:38<00:30, 5494.42it/s] 58%|█████▊    | 233095/400000 [00:38<00:30, 5447.44it/s] 58%|█████▊    | 233712/400000 [00:38<00:29, 5643.13it/s] 59%|█████▊    | 234303/400000 [00:38<00:29, 5713.43it/s] 59%|█████▊    | 234879/400000 [00:38<00:30, 5452.11it/s] 59%|█████▉    | 235430/400000 [00:38<00:31, 5237.32it/s] 59%|█████▉    | 235960/400000 [00:38<00:31, 5169.89it/s] 59%|█████▉    | 236481/400000 [00:38<00:33, 4943.28it/s] 59%|█████▉    | 237015/400000 [00:38<00:32, 5055.79it/s] 59%|█████▉    | 237666/400000 [00:38<00:29, 5417.06it/s] 60%|█████▉    | 238393/400000 [00:39<00:27, 5863.93it/s] 60%|█████▉    | 238998/400000 [00:39<00:28, 5600.05it/s] 60%|█████▉    | 239608/400000 [00:39<00:27, 5740.20it/s] 60%|██████    | 240208/400000 [00:39<00:27, 5814.17it/s] 60%|██████    | 240798/400000 [00:39<00:28, 5654.33it/s] 60%|██████    | 241371/400000 [00:39<00:30, 5210.00it/s] 60%|██████    | 241904/400000 [00:39<00:30, 5241.88it/s] 61%|██████    | 242542/400000 [00:39<00:28, 5537.72it/s] 61%|██████    | 243106/400000 [00:39<00:28, 5424.33it/s] 61%|██████    | 243753/400000 [00:40<00:27, 5698.48it/s] 61%|██████    | 244332/400000 [00:40<00:28, 5380.51it/s] 61%|██████    | 244881/400000 [00:40<00:28, 5372.19it/s] 61%|██████▏   | 245453/400000 [00:40<00:28, 5470.14it/s] 62%|██████▏   | 246148/400000 [00:40<00:26, 5841.32it/s] 62%|██████▏   | 246879/400000 [00:40<00:24, 6214.42it/s] 62%|██████▏   | 247594/400000 [00:40<00:23, 6467.95it/s] 62%|██████▏   | 248254/400000 [00:40<00:24, 6085.82it/s] 62%|██████▏   | 248884/400000 [00:40<00:24, 6145.09it/s] 62%|██████▏   | 249509/400000 [00:41<00:27, 5545.97it/s] 63%|██████▎   | 250082/400000 [00:41<00:27, 5433.11it/s] 63%|██████▎   | 250639/400000 [00:41<00:27, 5398.75it/s] 63%|██████▎   | 251248/400000 [00:41<00:26, 5588.35it/s] 63%|██████▎   | 251815/400000 [00:41<00:27, 5426.48it/s] 63%|██████▎   | 252470/400000 [00:41<00:25, 5718.98it/s] 63%|██████▎   | 253051/400000 [00:41<00:25, 5698.19it/s] 63%|██████▎   | 253628/400000 [00:41<00:26, 5612.68it/s] 64%|██████▎   | 254194/400000 [00:41<00:27, 5306.24it/s] 64%|██████▎   | 254732/400000 [00:42<00:27, 5235.97it/s] 64%|██████▍   | 255362/400000 [00:42<00:26, 5515.16it/s] 64%|██████▍   | 256084/400000 [00:42<00:24, 5934.19it/s] 64%|██████▍   | 256692/400000 [00:42<00:25, 5626.41it/s] 64%|██████▍   | 257268/400000 [00:42<00:26, 5437.54it/s] 64%|██████▍   | 257971/400000 [00:42<00:24, 5833.28it/s] 65%|██████▍   | 258675/400000 [00:42<00:22, 6148.09it/s] 65%|██████▍   | 259379/400000 [00:42<00:22, 6390.66it/s] 65%|██████▌   | 260032/400000 [00:42<00:24, 5802.81it/s] 65%|██████▌   | 260632/400000 [00:43<00:25, 5490.45it/s] 65%|██████▌   | 261214/400000 [00:43<00:24, 5583.41it/s] 65%|██████▌   | 261834/400000 [00:43<00:24, 5754.34it/s] 66%|██████▌   | 262420/400000 [00:43<00:25, 5438.04it/s] 66%|██████▌   | 263069/400000 [00:43<00:23, 5714.77it/s] 66%|██████▌   | 263688/400000 [00:43<00:23, 5847.78it/s] 66%|██████▌   | 264282/400000 [00:43<00:24, 5429.54it/s] 66%|██████▌   | 264837/400000 [00:43<00:26, 5167.66it/s] 66%|██████▋   | 265424/400000 [00:43<00:25, 5357.14it/s] 66%|██████▋   | 265970/400000 [00:43<00:25, 5293.89it/s] 67%|██████▋   | 266507/400000 [00:44<00:26, 5018.54it/s] 67%|██████▋   | 267051/400000 [00:44<00:25, 5136.78it/s] 67%|██████▋   | 267781/400000 [00:44<00:23, 5636.33it/s] 67%|██████▋   | 268364/400000 [00:44<00:23, 5609.80it/s] 67%|██████▋   | 268939/400000 [00:44<00:23, 5638.37it/s] 67%|██████▋   | 269513/400000 [00:44<00:23, 5498.30it/s] 68%|██████▊   | 270144/400000 [00:44<00:22, 5717.69it/s] 68%|██████▊   | 270724/400000 [00:44<00:24, 5284.98it/s] 68%|██████▊   | 271265/400000 [00:44<00:25, 5105.41it/s] 68%|██████▊   | 271786/400000 [00:45<00:26, 4826.51it/s] 68%|██████▊   | 272419/400000 [00:45<00:24, 5195.81it/s] 68%|██████▊   | 273034/400000 [00:45<00:23, 5448.60it/s] 68%|██████▊   | 273674/400000 [00:45<00:22, 5701.90it/s] 69%|██████▊   | 274257/400000 [00:45<00:22, 5541.90it/s] 69%|██████▊   | 274822/400000 [00:45<00:22, 5520.54it/s] 69%|██████▉   | 275522/400000 [00:45<00:21, 5893.80it/s] 69%|██████▉   | 276124/400000 [00:45<00:20, 5902.91it/s] 69%|██████▉   | 276723/400000 [00:45<00:21, 5860.07it/s] 69%|██████▉   | 277315/400000 [00:46<00:21, 5766.43it/s] 69%|██████▉   | 277897/400000 [00:46<00:23, 5123.04it/s] 70%|██████▉   | 278538/400000 [00:46<00:22, 5451.01it/s] 70%|██████▉   | 279149/400000 [00:46<00:21, 5632.73it/s] 70%|██████▉   | 279726/400000 [00:46<00:22, 5420.03it/s] 70%|███████   | 280280/400000 [00:46<00:22, 5231.61it/s] 70%|███████   | 280837/400000 [00:46<00:22, 5326.93it/s] 70%|███████   | 281450/400000 [00:46<00:21, 5544.53it/s] 71%|███████   | 282102/400000 [00:46<00:20, 5804.74it/s] 71%|███████   | 282721/400000 [00:46<00:19, 5913.23it/s] 71%|███████   | 283319/400000 [00:47<00:21, 5494.09it/s] 71%|███████   | 283892/400000 [00:47<00:20, 5562.03it/s] 71%|███████   | 284599/400000 [00:47<00:19, 5939.95it/s] 71%|███████▏  | 285252/400000 [00:47<00:18, 6102.08it/s] 71%|███████▏  | 285872/400000 [00:47<00:20, 5611.19it/s] 72%|███████▏  | 286466/400000 [00:47<00:19, 5705.71it/s] 72%|███████▏  | 287100/400000 [00:47<00:19, 5877.60it/s] 72%|███████▏  | 287697/400000 [00:47<00:19, 5629.47it/s] 72%|███████▏  | 288356/400000 [00:47<00:18, 5886.84it/s] 72%|███████▏  | 288954/400000 [00:48<00:19, 5665.64it/s] 72%|███████▏  | 289529/400000 [00:48<00:20, 5379.72it/s] 73%|███████▎  | 290076/400000 [00:48<00:20, 5246.91it/s] 73%|███████▎  | 290753/400000 [00:48<00:19, 5625.94it/s] 73%|███████▎  | 291330/400000 [00:48<00:19, 5662.38it/s] 73%|███████▎  | 291971/400000 [00:48<00:18, 5867.13it/s] 73%|███████▎  | 292566/400000 [00:48<00:18, 5740.46it/s] 73%|███████▎  | 293147/400000 [00:48<00:18, 5709.29it/s] 73%|███████▎  | 293740/400000 [00:48<00:18, 5772.76it/s] 74%|███████▎  | 294321/400000 [00:49<00:18, 5584.61it/s] 74%|███████▎  | 294884/400000 [00:49<00:19, 5326.10it/s] 74%|███████▍  | 295422/400000 [00:49<00:19, 5290.60it/s] 74%|███████▍  | 295955/400000 [00:49<00:20, 5120.75it/s] 74%|███████▍  | 296471/400000 [00:49<00:20, 5025.77it/s] 74%|███████▍  | 297053/400000 [00:49<00:19, 5237.75it/s] 74%|███████▍  | 297731/400000 [00:49<00:18, 5620.95it/s] 75%|███████▍  | 298377/400000 [00:49<00:17, 5845.77it/s] 75%|███████▍  | 298972/400000 [00:49<00:17, 5826.26it/s] 75%|███████▍  | 299618/400000 [00:49<00:16, 6000.72it/s] 75%|███████▌  | 300225/400000 [00:50<00:17, 5613.26it/s] 75%|███████▌  | 300836/400000 [00:50<00:17, 5753.13it/s] 75%|███████▌  | 301439/400000 [00:50<00:16, 5825.92it/s] 76%|███████▌  | 302028/400000 [00:50<00:17, 5574.22it/s] 76%|███████▌  | 302632/400000 [00:50<00:17, 5704.17it/s] 76%|███████▌  | 303208/400000 [00:50<00:17, 5685.37it/s] 76%|███████▌  | 303781/400000 [00:50<00:17, 5579.60it/s] 76%|███████▌  | 304342/400000 [00:50<00:17, 5342.85it/s] 76%|███████▌  | 304978/400000 [00:50<00:16, 5610.10it/s] 76%|███████▋  | 305556/400000 [00:51<00:16, 5656.10it/s] 77%|███████▋  | 306177/400000 [00:51<00:16, 5810.91it/s] 77%|███████▋  | 306844/400000 [00:51<00:15, 6044.36it/s] 77%|███████▋  | 307454/400000 [00:51<00:15, 5945.22it/s] 77%|███████▋  | 308053/400000 [00:51<00:16, 5554.37it/s] 77%|███████▋  | 308617/400000 [00:51<00:17, 5351.88it/s] 77%|███████▋  | 309275/400000 [00:51<00:16, 5667.28it/s] 77%|███████▋  | 309852/400000 [00:51<00:16, 5351.19it/s] 78%|███████▊  | 310469/400000 [00:51<00:16, 5570.93it/s] 78%|███████▊  | 311051/400000 [00:52<00:15, 5640.74it/s] 78%|███████▊  | 311651/400000 [00:52<00:15, 5742.89it/s] 78%|███████▊  | 312298/400000 [00:52<00:14, 5941.92it/s] 78%|███████▊  | 312898/400000 [00:52<00:15, 5588.42it/s] 78%|███████▊  | 313466/400000 [00:52<00:16, 5363.47it/s] 79%|███████▊  | 314011/400000 [00:52<00:16, 5152.56it/s] 79%|███████▊  | 314584/400000 [00:52<00:16, 5312.85it/s] 79%|███████▉  | 315257/400000 [00:52<00:14, 5669.48it/s] 79%|███████▉  | 315835/400000 [00:52<00:14, 5617.34it/s] 79%|███████▉  | 316471/400000 [00:52<00:14, 5820.22it/s] 79%|███████▉  | 317122/400000 [00:53<00:13, 6007.79it/s] 79%|███████▉  | 317730/400000 [00:53<00:15, 5471.43it/s] 80%|███████▉  | 318440/400000 [00:53<00:13, 5875.48it/s] 80%|███████▉  | 319046/400000 [00:53<00:14, 5597.39it/s] 80%|███████▉  | 319688/400000 [00:53<00:13, 5819.41it/s] 80%|████████  | 320284/400000 [00:53<00:14, 5674.67it/s] 80%|████████  | 320959/400000 [00:53<00:13, 5958.94it/s] 80%|████████  | 321566/400000 [00:53<00:13, 5872.59it/s] 81%|████████  | 322162/400000 [00:53<00:13, 5887.09it/s] 81%|████████  | 322757/400000 [00:54<00:13, 5791.50it/s] 81%|████████  | 323341/400000 [00:54<00:13, 5671.51it/s] 81%|████████  | 323997/400000 [00:54<00:12, 5911.15it/s] 81%|████████  | 324594/400000 [00:54<00:12, 5808.28it/s] 81%|████████▏ | 325179/400000 [00:54<00:13, 5530.56it/s] 81%|████████▏ | 325738/400000 [00:54<00:13, 5481.88it/s] 82%|████████▏ | 326378/400000 [00:54<00:12, 5726.22it/s] 82%|████████▏ | 327052/400000 [00:54<00:12, 5996.67it/s] 82%|████████▏ | 327659/400000 [00:54<00:12, 5979.15it/s] 82%|████████▏ | 328263/400000 [00:54<00:12, 5766.09it/s] 82%|████████▏ | 328845/400000 [00:55<00:13, 5471.60it/s] 82%|████████▏ | 329399/400000 [00:55<00:13, 5406.90it/s] 82%|████████▏ | 329945/400000 [00:55<00:13, 5240.10it/s] 83%|████████▎ | 330548/400000 [00:55<00:12, 5450.06it/s] 83%|████████▎ | 331099/400000 [00:55<00:13, 4976.72it/s] 83%|████████▎ | 331609/400000 [00:55<00:14, 4735.37it/s] 83%|████████▎ | 332094/400000 [00:55<00:14, 4680.23it/s] 83%|████████▎ | 332570/400000 [00:55<00:14, 4626.48it/s] 83%|████████▎ | 333170/400000 [00:55<00:13, 4966.10it/s] 83%|████████▎ | 333678/400000 [00:56<00:13, 4890.18it/s] 84%|████████▎ | 334235/400000 [00:56<00:12, 5074.41it/s] 84%|████████▎ | 334954/400000 [00:56<00:11, 5564.72it/s] 84%|████████▍ | 335679/400000 [00:56<00:10, 5981.24it/s] 84%|████████▍ | 336300/400000 [00:56<00:11, 5783.24it/s] 84%|████████▍ | 336896/400000 [00:56<00:11, 5485.58it/s] 84%|████████▍ | 337462/400000 [00:56<00:11, 5531.11it/s] 85%|████████▍ | 338026/400000 [00:56<00:11, 5529.73it/s] 85%|████████▍ | 338643/400000 [00:56<00:10, 5705.74it/s] 85%|████████▍ | 339221/400000 [00:57<00:10, 5725.96it/s] 85%|████████▍ | 339799/400000 [00:57<00:11, 5366.33it/s] 85%|████████▌ | 340425/400000 [00:57<00:10, 5605.24it/s] 85%|████████▌ | 341047/400000 [00:57<00:10, 5773.40it/s] 85%|████████▌ | 341632/400000 [00:57<00:10, 5731.06it/s] 86%|████████▌ | 342211/400000 [00:57<00:10, 5486.92it/s] 86%|████████▌ | 342766/400000 [00:57<00:10, 5400.47it/s] 86%|████████▌ | 343311/400000 [00:57<00:10, 5251.16it/s] 86%|████████▌ | 343841/400000 [00:57<00:10, 5204.40it/s] 86%|████████▌ | 344381/400000 [00:58<00:10, 5258.83it/s] 86%|████████▌ | 344910/400000 [00:58<00:10, 5062.13it/s] 86%|████████▋ | 345450/400000 [00:58<00:10, 5158.49it/s] 87%|████████▋ | 346102/400000 [00:58<00:09, 5501.89it/s] 87%|████████▋ | 346705/400000 [00:58<00:09, 5648.62it/s] 87%|████████▋ | 347277/400000 [00:58<00:09, 5498.71it/s] 87%|████████▋ | 347846/400000 [00:58<00:09, 5552.80it/s] 87%|████████▋ | 348483/400000 [00:58<00:08, 5773.39it/s] 87%|████████▋ | 349153/400000 [00:58<00:08, 6021.53it/s] 87%|████████▋ | 349762/400000 [00:58<00:08, 5941.92it/s] 88%|████████▊ | 350401/400000 [00:59<00:08, 6068.12it/s] 88%|████████▊ | 351012/400000 [00:59<00:08, 5913.36it/s] 88%|████████▊ | 351607/400000 [00:59<00:08, 5696.44it/s] 88%|████████▊ | 352181/400000 [00:59<00:09, 5272.11it/s] 88%|████████▊ | 352718/400000 [00:59<00:09, 5153.35it/s] 88%|████████▊ | 353265/400000 [00:59<00:08, 5242.55it/s] 88%|████████▊ | 353795/400000 [00:59<00:08, 5252.87it/s] 89%|████████▊ | 354455/400000 [00:59<00:08, 5594.22it/s] 89%|████████▉ | 355126/400000 [00:59<00:07, 5885.97it/s] 89%|████████▉ | 355725/400000 [01:00<00:07, 5802.62it/s] 89%|████████▉ | 356329/400000 [01:00<00:07, 5868.11it/s] 89%|████████▉ | 356922/400000 [01:00<00:08, 5284.75it/s] 89%|████████▉ | 357466/400000 [01:00<00:08, 5051.59it/s] 90%|████████▉ | 358066/400000 [01:00<00:07, 5300.69it/s] 90%|████████▉ | 358685/400000 [01:00<00:07, 5538.32it/s] 90%|████████▉ | 359250/400000 [01:00<00:07, 5553.75it/s] 90%|████████▉ | 359867/400000 [01:00<00:07, 5722.50it/s] 90%|█████████ | 360502/400000 [01:00<00:06, 5895.78it/s] 90%|█████████ | 361098/400000 [01:00<00:07, 5439.07it/s] 90%|█████████ | 361654/400000 [01:01<00:07, 5429.69it/s] 91%|█████████ | 362228/400000 [01:01<00:06, 5518.18it/s] 91%|█████████ | 362841/400000 [01:01<00:06, 5686.44it/s] 91%|█████████ | 363416/400000 [01:01<00:06, 5430.46it/s] 91%|█████████ | 363993/400000 [01:01<00:06, 5526.61it/s] 91%|█████████ | 364600/400000 [01:01<00:06, 5678.85it/s] 91%|█████████▏| 365174/400000 [01:01<00:06, 5692.01it/s] 91%|█████████▏| 365747/400000 [01:01<00:06, 5201.14it/s] 92%|█████████▏| 366352/400000 [01:01<00:06, 5428.49it/s] 92%|█████████▏| 366936/400000 [01:02<00:05, 5543.87it/s] 92%|█████████▏| 367498/400000 [01:02<00:05, 5437.85it/s] 92%|█████████▏| 368048/400000 [01:02<00:05, 5336.31it/s] 92%|█████████▏| 368587/400000 [01:02<00:05, 5342.96it/s] 92%|█████████▏| 369172/400000 [01:02<00:05, 5485.37it/s] 92%|█████████▏| 369828/400000 [01:02<00:05, 5766.87it/s] 93%|█████████▎| 370411/400000 [01:02<00:05, 5739.12it/s] 93%|█████████▎| 371136/400000 [01:02<00:04, 6120.65it/s] 93%|█████████▎| 371818/400000 [01:02<00:04, 6313.27it/s] 93%|█████████▎| 372458/400000 [01:02<00:04, 6046.25it/s] 93%|█████████▎| 373071/400000 [01:03<00:04, 5698.76it/s] 93%|█████████▎| 373651/400000 [01:03<00:04, 5635.31it/s] 94%|█████████▎| 374373/400000 [01:03<00:04, 6030.94it/s] 94%|█████████▎| 374989/400000 [01:03<00:04, 5298.31it/s] 94%|█████████▍| 375693/400000 [01:03<00:04, 5722.15it/s] 94%|█████████▍| 376293/400000 [01:03<00:04, 5740.06it/s] 94%|█████████▍| 376898/400000 [01:03<00:03, 5827.39it/s] 94%|█████████▍| 377495/400000 [01:03<00:03, 5835.29it/s] 95%|█████████▍| 378138/400000 [01:03<00:03, 6001.36it/s] 95%|█████████▍| 378747/400000 [01:04<00:03, 5768.00it/s] 95%|█████████▍| 379403/400000 [01:04<00:03, 5983.03it/s] 95%|█████████▌| 380009/400000 [01:04<00:03, 5895.86it/s] 95%|█████████▌| 380604/400000 [01:04<00:03, 5677.02it/s] 95%|█████████▌| 381178/400000 [01:04<00:03, 5523.86it/s] 95%|█████████▌| 381834/400000 [01:04<00:03, 5797.06it/s] 96%|█████████▌| 382421/400000 [01:04<00:03, 5696.33it/s] 96%|█████████▌| 382996/400000 [01:04<00:03, 5590.84it/s] 96%|█████████▌| 383643/400000 [01:04<00:02, 5826.75it/s] 96%|█████████▌| 384267/400000 [01:05<00:02, 5943.61it/s] 96%|█████████▌| 384866/400000 [01:05<00:02, 5270.06it/s] 96%|█████████▋| 385411/400000 [01:05<00:02, 4972.84it/s] 96%|█████████▋| 385958/400000 [01:05<00:02, 5110.62it/s] 97%|█████████▋| 386488/400000 [01:05<00:02, 5164.59it/s] 97%|█████████▋| 387134/400000 [01:05<00:02, 5493.88it/s] 97%|█████████▋| 387750/400000 [01:05<00:02, 5647.53it/s] 97%|█████████▋| 388324/400000 [01:05<00:02, 5627.99it/s] 97%|█████████▋| 388947/400000 [01:05<00:01, 5793.90it/s] 97%|█████████▋| 389533/400000 [01:06<00:01, 5765.84it/s] 98%|█████████▊| 390147/400000 [01:06<00:01, 5871.48it/s] 98%|█████████▊| 390738/400000 [01:06<00:01, 5819.22it/s] 98%|█████████▊| 391323/400000 [01:06<00:01, 5807.84it/s] 98%|█████████▊| 391906/400000 [01:06<00:01, 5604.45it/s] 98%|█████████▊| 392470/400000 [01:06<00:01, 5313.82it/s] 98%|█████████▊| 393007/400000 [01:06<00:01, 5304.21it/s] 98%|█████████▊| 393541/400000 [01:06<00:01, 5259.84it/s] 99%|█████████▊| 394070/400000 [01:06<00:01, 5122.69it/s] 99%|█████████▊| 394629/400000 [01:06<00:01, 5253.60it/s] 99%|█████████▉| 395157/400000 [01:07<00:00, 5195.35it/s] 99%|█████████▉| 395775/400000 [01:07<00:00, 5455.94it/s] 99%|█████████▉| 396326/400000 [01:07<00:00, 5342.20it/s] 99%|█████████▉| 396865/400000 [01:07<00:00, 5043.14it/s] 99%|█████████▉| 397530/400000 [01:07<00:00, 5436.71it/s]100%|█████████▉| 398087/400000 [01:07<00:00, 5369.39it/s]100%|█████████▉| 398813/400000 [01:07<00:00, 5823.20it/s]100%|█████████▉| 399503/400000 [01:07<00:00, 6107.55it/s]100%|█████████▉| 399999/400000 [01:07<00:00, 5893.23it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fa2b1d69940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01179443647250938 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.011535751979087907 	 Accuracy: 50

  model saves at 50% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15782 out of table with 15704 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15782 out of table with 15704 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-14 13:27:20.986326: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 13:27:20.993111: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-14 13:27:20.993524: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5606e9380fa0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 13:27:20.993594: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fa2bd8daf98> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 15s - loss: 7.4213 - accuracy: 0.5160
 2000/25000 [=>............................] - ETA: 12s - loss: 7.4213 - accuracy: 0.5160
 3000/25000 [==>...........................] - ETA: 10s - loss: 7.5031 - accuracy: 0.5107
 4000/25000 [===>..........................] - ETA: 9s - loss: 7.5785 - accuracy: 0.5058 
 5000/25000 [=====>........................] - ETA: 8s - loss: 7.5961 - accuracy: 0.5046
 6000/25000 [======>.......................] - ETA: 8s - loss: 7.6462 - accuracy: 0.5013
 7000/25000 [=======>......................] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
 8000/25000 [========>.....................] - ETA: 7s - loss: 7.6647 - accuracy: 0.5001
 9000/25000 [=========>....................] - ETA: 6s - loss: 7.6564 - accuracy: 0.5007
10000/25000 [===========>..................] - ETA: 6s - loss: 7.7096 - accuracy: 0.4972
11000/25000 [============>.................] - ETA: 5s - loss: 7.6875 - accuracy: 0.4986
12000/25000 [=============>................] - ETA: 5s - loss: 7.6602 - accuracy: 0.5004
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6820 - accuracy: 0.4990
14000/25000 [===============>..............] - ETA: 4s - loss: 7.6874 - accuracy: 0.4986
15000/25000 [=================>............] - ETA: 3s - loss: 7.6554 - accuracy: 0.5007
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6513 - accuracy: 0.5010
17000/25000 [===================>..........] - ETA: 3s - loss: 7.6513 - accuracy: 0.5010
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6598 - accuracy: 0.5004
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6545 - accuracy: 0.5008
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6406 - accuracy: 0.5017
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6425 - accuracy: 0.5016
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6478 - accuracy: 0.5012
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6680 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 12s 468us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fa222791780> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fa223a31128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 2s 2s/step - loss: 1.7894 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.8020 - val_crf_viterbi_accuracy: 0.0133

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

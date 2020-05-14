
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fcd2e2bbfd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 14:12:44.200324
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 14:12:44.204409
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 14:12:44.207818
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 14:12:44.211102
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fcd3a2d3438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353611.6562
Epoch 2/10

1/1 [==============================] - 0s 113ms/step - loss: 260363.6406
Epoch 3/10

1/1 [==============================] - 0s 99ms/step - loss: 143864.6094
Epoch 4/10

1/1 [==============================] - 0s 97ms/step - loss: 75116.6094
Epoch 5/10

1/1 [==============================] - 0s 107ms/step - loss: 39069.3594
Epoch 6/10

1/1 [==============================] - 0s 105ms/step - loss: 21926.8203
Epoch 7/10

1/1 [==============================] - 0s 98ms/step - loss: 13598.9785
Epoch 8/10

1/1 [==============================] - 0s 95ms/step - loss: 9165.2021
Epoch 9/10

1/1 [==============================] - 0s 94ms/step - loss: 6656.1606
Epoch 10/10

1/1 [==============================] - 0s 96ms/step - loss: 5107.1631

  #### Inference Need return ypred, ytrue ######################### 
[[ 6.50514424e-01 -8.57108891e-01 -1.75278485e-02 -1.16565669e+00
  -6.55836761e-02 -2.26791382e-01  1.78663874e+00 -1.34198368e+00
   9.97042179e-01  1.63917267e+00  1.86508417e-01 -1.27309442e+00
  -1.96889484e+00  1.06113955e-01 -1.48443198e+00  9.77841020e-02
   5.41730762e-01  1.86491644e+00  1.09812483e-01 -3.76980722e-01
   3.09618771e-01  1.16289115e+00  8.56522024e-02 -7.88137853e-01
  -1.47232318e+00 -9.13769424e-01  2.18740940e+00  6.73294365e-02
  -5.86461902e-01  1.22061670e+00  8.01791668e-01  9.70725417e-02
   9.56932187e-01  1.44949174e+00 -3.64966691e-03 -7.64546245e-02
  -6.93675101e-01  1.66686273e+00  8.36239636e-01 -1.25697947e+00
  -1.01029241e+00  1.00918925e+00 -2.01731682e+00  6.22374356e-01
  -2.42642224e-01 -1.30913484e+00 -8.33517253e-01 -2.16596341e+00
   3.88695627e-01  1.25500631e+00  1.79628706e+00  8.13086510e-01
  -7.41240501e-01 -3.22552919e-02 -9.65298057e-01  8.36772919e-02
   2.15725735e-01 -3.17781329e-01  1.81672990e-01 -1.76371253e+00
   5.14714047e-02  1.00410843e+01  9.52354336e+00  1.00987282e+01
   9.04322529e+00  8.47139263e+00  8.70108986e+00  9.27182961e+00
   9.40441227e+00  1.00262880e+01  9.05703449e+00  9.27243900e+00
   1.06855354e+01  1.08509121e+01  7.12594557e+00  9.39066601e+00
   1.15284185e+01  9.72805214e+00  8.47767735e+00  1.09359465e+01
   8.35253716e+00  9.19606590e+00  7.74108887e+00  9.29073048e+00
   9.44018269e+00  9.66735840e+00  9.95017052e+00  9.92222214e+00
   9.80037785e+00  9.76383114e+00  8.67201424e+00  1.13996086e+01
   9.19380856e+00  1.09935360e+01  1.06161947e+01  8.64921093e+00
   1.04563894e+01  8.89102173e+00  1.07886305e+01  8.46292305e+00
   1.08634682e+01  9.07169628e+00  8.69546223e+00  9.65176296e+00
   1.05473337e+01  8.90205574e+00  9.42065525e+00  9.72955704e+00
   7.59754372e+00  1.22628212e+01  9.70601463e+00  1.07598991e+01
   8.16436195e+00  9.41302967e+00  1.18822470e+01  9.97091579e+00
   8.72417164e+00  8.85716343e+00  9.87448788e+00  1.02465057e+01
   1.18872643e+00  5.16047835e-01 -7.56252110e-01  1.49816835e+00
  -8.23911607e-01  3.49873930e-01 -4.62876737e-01  2.01057482e+00
  -1.15972757e-02  9.82580259e-02 -1.70798385e+00  1.45396292e+00
  -6.14573777e-01  3.34840775e-01  2.38921022e+00  1.63708079e+00
  -2.60785848e-01  4.93523061e-01 -2.86311805e-02  1.56980240e+00
  -9.41841424e-01  1.98088586e-02 -6.29728973e-01  1.83281338e+00
  -1.62787974e-01  3.84603620e-01 -1.62234381e-02 -2.86797076e-01
   2.73452342e-01 -5.16242504e-01 -1.59510881e-01 -1.81345010e+00
   1.69123769e-01 -1.15647042e+00 -2.63537019e-01 -3.76600325e-01
  -9.81822014e-01 -9.30413306e-01  8.57846200e-01  1.13627076e+00
  -1.66051590e+00 -3.41346532e-01 -4.56525207e-01 -7.54327953e-01
   5.43918312e-01 -1.84141517e-01 -1.28673339e+00  2.60774851e+00
   1.97964168e+00  2.14103296e-01 -5.44602215e-01 -2.66393960e-01
  -1.02722859e+00 -1.68431604e+00  1.26895785e-01 -8.74315679e-01
  -3.90513062e-01  9.44310188e-01  1.03624821e-01 -1.16082513e+00
   2.41384983e+00  5.74440539e-01  3.48185635e+00  3.59968901e-01
   9.99601364e-01  1.72456193e+00  2.00267315e+00  1.84287333e+00
   1.29050648e+00  1.12715840e-01  9.33197498e-01  2.04356384e+00
   7.98768938e-01  6.55180812e-01  1.75552416e+00  2.31044745e+00
   1.38618863e+00  2.13894844e+00  1.20075679e+00  9.18956816e-01
   6.96441174e-01  5.12247384e-01  1.90502048e-01  5.34618139e-01
   2.75740504e-01  1.02944255e+00  2.22905207e+00  3.98374796e-01
   2.71505833e-01  1.56876552e+00  2.44909430e+00  1.33661425e+00
   2.66534269e-01  1.99278903e+00  5.00777423e-01  2.06411076e+00
   1.07005382e+00  1.30741382e+00  9.35056984e-01  5.06952345e-01
   2.48564243e+00  1.92984557e+00  1.60442114e-01  4.41363275e-01
   5.79589307e-01  4.45301652e-01  1.51782680e+00  3.23084295e-01
   1.32279968e+00  1.73429012e+00  1.34837842e+00  1.90812683e+00
   2.10743999e+00  1.60648227e+00  4.81335521e-01  1.97280288e+00
   1.41275525e+00  1.88392401e-01  1.99747920e+00  1.12403119e+00
   1.85491800e-01  1.00031137e+01  8.57631874e+00  9.28340340e+00
   8.21576691e+00  9.92105865e+00  8.94592762e+00  9.84195328e+00
   1.12251501e+01  1.14755659e+01  9.47918129e+00  1.02759285e+01
   9.99487972e+00  1.12701426e+01  8.27291107e+00  7.65407419e+00
   9.30020046e+00  8.03262520e+00  7.74266815e+00  9.53199482e+00
   9.95479965e+00  9.50678730e+00  9.76470280e+00  1.07090273e+01
   1.10087919e+01  9.62829781e+00  1.10302238e+01  1.04715910e+01
   1.06733236e+01  1.00968781e+01  8.84661484e+00  8.99985409e+00
   1.02522144e+01  1.04185619e+01  9.93715286e+00  1.01524467e+01
   8.61625862e+00  7.73121881e+00  8.92708969e+00  9.28755951e+00
   9.31050205e+00  9.27334118e+00  9.27162170e+00  1.12256737e+01
   9.66265392e+00  9.63312054e+00  9.89532757e+00  1.11714640e+01
   9.34089375e+00  9.21405888e+00  1.03814459e+01  9.57640362e+00
   1.01869888e+01  9.35569954e+00  8.80075836e+00  9.68090534e+00
   9.98510647e+00  1.12022066e+01  9.65433979e+00  8.26096439e+00
   2.33615184e+00  2.30158865e-01  6.20895743e-01  2.60790896e+00
   3.36126661e+00  3.25253248e+00  2.05371571e+00  4.02601421e-01
   2.30220127e+00  7.69319177e-01  2.01637959e+00  2.78018057e-01
   1.82647300e+00  2.30413055e+00  1.12833524e+00  1.53916740e+00
   8.09050500e-01  2.52199745e+00  1.48261011e+00  2.32820988e+00
   1.06339562e+00  1.03303969e+00  2.53786993e+00  3.01065803e-01
   5.40294707e-01  2.82519150e+00  1.08705950e+00  3.57566118e-01
   4.62066710e-01  9.35012937e-01  5.77631176e-01  1.75137174e+00
   1.36655772e+00  7.88535476e-01  4.96409714e-01  1.25090718e+00
   7.67443061e-01  1.08826399e-01  2.13041735e+00  2.49224901e-01
   4.94289637e-01  2.42298007e-01  9.88234460e-01  5.18721759e-01
   2.97496676e-01  1.79309571e+00  9.63558078e-01  1.12125993e-01
   3.79165292e-01  8.25817227e-01  2.85253191e+00  7.50262976e-01
   5.37634730e-01  1.36531401e+00  5.28646529e-01  5.43206215e-01
   1.61453319e+00  1.20603621e-01  5.04627705e-01  1.25279355e+00
  -8.14367867e+00  1.25210629e+01 -1.48412857e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 14:12:53.336664
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.5247
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 14:12:53.340404
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8581.57
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 14:12:53.343751
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.7912
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 14:12:53.346895
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -767.541
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140518862881344
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140517921301112
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140517921301616
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140517921302120
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140517921302624
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140517921303128

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fcd36152f28> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.439503
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.416480
grad_step = 000002, loss = 0.399613
grad_step = 000003, loss = 0.383002
grad_step = 000004, loss = 0.364670
grad_step = 000005, loss = 0.347056
grad_step = 000006, loss = 0.334179
grad_step = 000007, loss = 0.327953
grad_step = 000008, loss = 0.317251
grad_step = 000009, loss = 0.305532
grad_step = 000010, loss = 0.295260
grad_step = 000011, loss = 0.285522
grad_step = 000012, loss = 0.276022
grad_step = 000013, loss = 0.266465
grad_step = 000014, loss = 0.256397
grad_step = 000015, loss = 0.245908
grad_step = 000016, loss = 0.235456
grad_step = 000017, loss = 0.225513
grad_step = 000018, loss = 0.215978
grad_step = 000019, loss = 0.206367
grad_step = 000020, loss = 0.196833
grad_step = 000021, loss = 0.187505
grad_step = 000022, loss = 0.178816
grad_step = 000023, loss = 0.170329
grad_step = 000024, loss = 0.161649
grad_step = 000025, loss = 0.152963
grad_step = 000026, loss = 0.144491
grad_step = 000027, loss = 0.136433
grad_step = 000028, loss = 0.128943
grad_step = 000029, loss = 0.121829
grad_step = 000030, loss = 0.114498
grad_step = 000031, loss = 0.107110
grad_step = 000032, loss = 0.100397
grad_step = 000033, loss = 0.094236
grad_step = 000034, loss = 0.088147
grad_step = 000035, loss = 0.082199
grad_step = 000036, loss = 0.076607
grad_step = 000037, loss = 0.071371
grad_step = 000038, loss = 0.066452
grad_step = 000039, loss = 0.061663
grad_step = 000040, loss = 0.057099
grad_step = 000041, loss = 0.052894
grad_step = 000042, loss = 0.048879
grad_step = 000043, loss = 0.045046
grad_step = 000044, loss = 0.041567
grad_step = 000045, loss = 0.038331
grad_step = 000046, loss = 0.035184
grad_step = 000047, loss = 0.032208
grad_step = 000048, loss = 0.029548
grad_step = 000049, loss = 0.027066
grad_step = 000050, loss = 0.024675
grad_step = 000051, loss = 0.022458
grad_step = 000052, loss = 0.020447
grad_step = 000053, loss = 0.018609
grad_step = 000054, loss = 0.016868
grad_step = 000055, loss = 0.015296
grad_step = 000056, loss = 0.013850
grad_step = 000057, loss = 0.012517
grad_step = 000058, loss = 0.011327
grad_step = 000059, loss = 0.010264
grad_step = 000060, loss = 0.009276
grad_step = 000061, loss = 0.008396
grad_step = 000062, loss = 0.007632
grad_step = 000063, loss = 0.006937
grad_step = 000064, loss = 0.006316
grad_step = 000065, loss = 0.005784
grad_step = 000066, loss = 0.005310
grad_step = 000067, loss = 0.004887
grad_step = 000068, loss = 0.004526
grad_step = 000069, loss = 0.004219
grad_step = 000070, loss = 0.003945
grad_step = 000071, loss = 0.003721
grad_step = 000072, loss = 0.003526
grad_step = 000073, loss = 0.003354
grad_step = 000074, loss = 0.003215
grad_step = 000075, loss = 0.003100
grad_step = 000076, loss = 0.002998
grad_step = 000077, loss = 0.002918
grad_step = 000078, loss = 0.002854
grad_step = 000079, loss = 0.002795
grad_step = 000080, loss = 0.002749
grad_step = 000081, loss = 0.002711
grad_step = 000082, loss = 0.002677
grad_step = 000083, loss = 0.002651
grad_step = 000084, loss = 0.002628
grad_step = 000085, loss = 0.002605
grad_step = 000086, loss = 0.002587
grad_step = 000087, loss = 0.002569
grad_step = 000088, loss = 0.002552
grad_step = 000089, loss = 0.002537
grad_step = 000090, loss = 0.002522
grad_step = 000091, loss = 0.002507
grad_step = 000092, loss = 0.002494
grad_step = 000093, loss = 0.002480
grad_step = 000094, loss = 0.002467
grad_step = 000095, loss = 0.002455
grad_step = 000096, loss = 0.002443
grad_step = 000097, loss = 0.002432
grad_step = 000098, loss = 0.002422
grad_step = 000099, loss = 0.002412
grad_step = 000100, loss = 0.002402
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002393
grad_step = 000102, loss = 0.002385
grad_step = 000103, loss = 0.002377
grad_step = 000104, loss = 0.002370
grad_step = 000105, loss = 0.002363
grad_step = 000106, loss = 0.002357
grad_step = 000107, loss = 0.002351
grad_step = 000108, loss = 0.002345
grad_step = 000109, loss = 0.002340
grad_step = 000110, loss = 0.002335
grad_step = 000111, loss = 0.002330
grad_step = 000112, loss = 0.002325
grad_step = 000113, loss = 0.002321
grad_step = 000114, loss = 0.002316
grad_step = 000115, loss = 0.002312
grad_step = 000116, loss = 0.002308
grad_step = 000117, loss = 0.002304
grad_step = 000118, loss = 0.002300
grad_step = 000119, loss = 0.002296
grad_step = 000120, loss = 0.002292
grad_step = 000121, loss = 0.002288
grad_step = 000122, loss = 0.002285
grad_step = 000123, loss = 0.002281
grad_step = 000124, loss = 0.002277
grad_step = 000125, loss = 0.002273
grad_step = 000126, loss = 0.002270
grad_step = 000127, loss = 0.002266
grad_step = 000128, loss = 0.002262
grad_step = 000129, loss = 0.002258
grad_step = 000130, loss = 0.002255
grad_step = 000131, loss = 0.002251
grad_step = 000132, loss = 0.002247
grad_step = 000133, loss = 0.002243
grad_step = 000134, loss = 0.002240
grad_step = 000135, loss = 0.002236
grad_step = 000136, loss = 0.002232
grad_step = 000137, loss = 0.002228
grad_step = 000138, loss = 0.002224
grad_step = 000139, loss = 0.002221
grad_step = 000140, loss = 0.002218
grad_step = 000141, loss = 0.002221
grad_step = 000142, loss = 0.002226
grad_step = 000143, loss = 0.002245
grad_step = 000144, loss = 0.002275
grad_step = 000145, loss = 0.002325
grad_step = 000146, loss = 0.002350
grad_step = 000147, loss = 0.002351
grad_step = 000148, loss = 0.002320
grad_step = 000149, loss = 0.002335
grad_step = 000150, loss = 0.002349
grad_step = 000151, loss = 0.002301
grad_step = 000152, loss = 0.002228
grad_step = 000153, loss = 0.002205
grad_step = 000154, loss = 0.002260
grad_step = 000155, loss = 0.002302
grad_step = 000156, loss = 0.002251
grad_step = 000157, loss = 0.002176
grad_step = 000158, loss = 0.002172
grad_step = 000159, loss = 0.002223
grad_step = 000160, loss = 0.002248
grad_step = 000161, loss = 0.002209
grad_step = 000162, loss = 0.002165
grad_step = 000163, loss = 0.002166
grad_step = 000164, loss = 0.002186
grad_step = 000165, loss = 0.002184
grad_step = 000166, loss = 0.002160
grad_step = 000167, loss = 0.002150
grad_step = 000168, loss = 0.002168
grad_step = 000169, loss = 0.002181
grad_step = 000170, loss = 0.002173
grad_step = 000171, loss = 0.002146
grad_step = 000172, loss = 0.002130
grad_step = 000173, loss = 0.002132
grad_step = 000174, loss = 0.002137
grad_step = 000175, loss = 0.002131
grad_step = 000176, loss = 0.002115
grad_step = 000177, loss = 0.002104
grad_step = 000178, loss = 0.002102
grad_step = 000179, loss = 0.002107
grad_step = 000180, loss = 0.002110
grad_step = 000181, loss = 0.002105
grad_step = 000182, loss = 0.002096
grad_step = 000183, loss = 0.002086
grad_step = 000184, loss = 0.002080
grad_step = 000185, loss = 0.002079
grad_step = 000186, loss = 0.002080
grad_step = 000187, loss = 0.002088
grad_step = 000188, loss = 0.002119
grad_step = 000189, loss = 0.002224
grad_step = 000190, loss = 0.002593
grad_step = 000191, loss = 0.002975
grad_step = 000192, loss = 0.003266
grad_step = 000193, loss = 0.002366
grad_step = 000194, loss = 0.002465
grad_step = 000195, loss = 0.002932
grad_step = 000196, loss = 0.002199
grad_step = 000197, loss = 0.002472
grad_step = 000198, loss = 0.002599
grad_step = 000199, loss = 0.002075
grad_step = 000200, loss = 0.002536
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002308
grad_step = 000202, loss = 0.002143
grad_step = 000203, loss = 0.002437
grad_step = 000204, loss = 0.002082
grad_step = 000205, loss = 0.002264
grad_step = 000206, loss = 0.002223
grad_step = 000207, loss = 0.002074
grad_step = 000208, loss = 0.002254
grad_step = 000209, loss = 0.002053
grad_step = 000210, loss = 0.002160
grad_step = 000211, loss = 0.002137
grad_step = 000212, loss = 0.002048
grad_step = 000213, loss = 0.002158
grad_step = 000214, loss = 0.002035
grad_step = 000215, loss = 0.002095
grad_step = 000216, loss = 0.002076
grad_step = 000217, loss = 0.002024
grad_step = 000218, loss = 0.002086
grad_step = 000219, loss = 0.002011
grad_step = 000220, loss = 0.002045
grad_step = 000221, loss = 0.002033
grad_step = 000222, loss = 0.001999
grad_step = 000223, loss = 0.002036
grad_step = 000224, loss = 0.001989
grad_step = 000225, loss = 0.002002
grad_step = 000226, loss = 0.002000
grad_step = 000227, loss = 0.001973
grad_step = 000228, loss = 0.001994
grad_step = 000229, loss = 0.001968
grad_step = 000230, loss = 0.001965
grad_step = 000231, loss = 0.001970
grad_step = 000232, loss = 0.001944
grad_step = 000233, loss = 0.001953
grad_step = 000234, loss = 0.001943
grad_step = 000235, loss = 0.001931
grad_step = 000236, loss = 0.001939
grad_step = 000237, loss = 0.001925
grad_step = 000238, loss = 0.001930
grad_step = 000239, loss = 0.001946
grad_step = 000240, loss = 0.001981
grad_step = 000241, loss = 0.002087
grad_step = 000242, loss = 0.002310
grad_step = 000243, loss = 0.002414
grad_step = 000244, loss = 0.002404
grad_step = 000245, loss = 0.002098
grad_step = 000246, loss = 0.001890
grad_step = 000247, loss = 0.001989
grad_step = 000248, loss = 0.002153
grad_step = 000249, loss = 0.002081
grad_step = 000250, loss = 0.001890
grad_step = 000251, loss = 0.001895
grad_step = 000252, loss = 0.002018
grad_step = 000253, loss = 0.002028
grad_step = 000254, loss = 0.001910
grad_step = 000255, loss = 0.001835
grad_step = 000256, loss = 0.001894
grad_step = 000257, loss = 0.001950
grad_step = 000258, loss = 0.001904
grad_step = 000259, loss = 0.001827
grad_step = 000260, loss = 0.001821
grad_step = 000261, loss = 0.001864
grad_step = 000262, loss = 0.001882
grad_step = 000263, loss = 0.001842
grad_step = 000264, loss = 0.001790
grad_step = 000265, loss = 0.001789
grad_step = 000266, loss = 0.001816
grad_step = 000267, loss = 0.001825
grad_step = 000268, loss = 0.001809
grad_step = 000269, loss = 0.001780
grad_step = 000270, loss = 0.001753
grad_step = 000271, loss = 0.001745
grad_step = 000272, loss = 0.001755
grad_step = 000273, loss = 0.001776
grad_step = 000274, loss = 0.001797
grad_step = 000275, loss = 0.001794
grad_step = 000276, loss = 0.001784
grad_step = 000277, loss = 0.001758
grad_step = 000278, loss = 0.001736
grad_step = 000279, loss = 0.001714
grad_step = 000280, loss = 0.001703
grad_step = 000281, loss = 0.001695
grad_step = 000282, loss = 0.001693
grad_step = 000283, loss = 0.001697
grad_step = 000284, loss = 0.001720
grad_step = 000285, loss = 0.001761
grad_step = 000286, loss = 0.001866
grad_step = 000287, loss = 0.001963
grad_step = 000288, loss = 0.002004
grad_step = 000289, loss = 0.001771
grad_step = 000290, loss = 0.001638
grad_step = 000291, loss = 0.001712
grad_step = 000292, loss = 0.001834
grad_step = 000293, loss = 0.001821
grad_step = 000294, loss = 0.001646
grad_step = 000295, loss = 0.001679
grad_step = 000296, loss = 0.001811
grad_step = 000297, loss = 0.001764
grad_step = 000298, loss = 0.001630
grad_step = 000299, loss = 0.001648
grad_step = 000300, loss = 0.001736
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001699
grad_step = 000302, loss = 0.001608
grad_step = 000303, loss = 0.001637
grad_step = 000304, loss = 0.001717
grad_step = 000305, loss = 0.001636
grad_step = 000306, loss = 0.001600
grad_step = 000307, loss = 0.001645
grad_step = 000308, loss = 0.001666
grad_step = 000309, loss = 0.001604
grad_step = 000310, loss = 0.001565
grad_step = 000311, loss = 0.001603
grad_step = 000312, loss = 0.001616
grad_step = 000313, loss = 0.001580
grad_step = 000314, loss = 0.001548
grad_step = 000315, loss = 0.001556
grad_step = 000316, loss = 0.001581
grad_step = 000317, loss = 0.001590
grad_step = 000318, loss = 0.001563
grad_step = 000319, loss = 0.001557
grad_step = 000320, loss = 0.001573
grad_step = 000321, loss = 0.001620
grad_step = 000322, loss = 0.001628
grad_step = 000323, loss = 0.001630
grad_step = 000324, loss = 0.001654
grad_step = 000325, loss = 0.001693
grad_step = 000326, loss = 0.001660
grad_step = 000327, loss = 0.001620
grad_step = 000328, loss = 0.001522
grad_step = 000329, loss = 0.001516
grad_step = 000330, loss = 0.001520
grad_step = 000331, loss = 0.001563
grad_step = 000332, loss = 0.001586
grad_step = 000333, loss = 0.001511
grad_step = 000334, loss = 0.001498
grad_step = 000335, loss = 0.001494
grad_step = 000336, loss = 0.001528
grad_step = 000337, loss = 0.001560
grad_step = 000338, loss = 0.001511
grad_step = 000339, loss = 0.001520
grad_step = 000340, loss = 0.001489
grad_step = 000341, loss = 0.001497
grad_step = 000342, loss = 0.001516
grad_step = 000343, loss = 0.001479
grad_step = 000344, loss = 0.001495
grad_step = 000345, loss = 0.001470
grad_step = 000346, loss = 0.001466
grad_step = 000347, loss = 0.001476
grad_step = 000348, loss = 0.001449
grad_step = 000349, loss = 0.001469
grad_step = 000350, loss = 0.001451
grad_step = 000351, loss = 0.001445
grad_step = 000352, loss = 0.001461
grad_step = 000353, loss = 0.001455
grad_step = 000354, loss = 0.001484
grad_step = 000355, loss = 0.001533
grad_step = 000356, loss = 0.001542
grad_step = 000357, loss = 0.001612
grad_step = 000358, loss = 0.001572
grad_step = 000359, loss = 0.001522
grad_step = 000360, loss = 0.001419
grad_step = 000361, loss = 0.001446
grad_step = 000362, loss = 0.001520
grad_step = 000363, loss = 0.001474
grad_step = 000364, loss = 0.001419
grad_step = 000365, loss = 0.001419
grad_step = 000366, loss = 0.001461
grad_step = 000367, loss = 0.001466
grad_step = 000368, loss = 0.001407
grad_step = 000369, loss = 0.001405
grad_step = 000370, loss = 0.001424
grad_step = 000371, loss = 0.001437
grad_step = 000372, loss = 0.001451
grad_step = 000373, loss = 0.001414
grad_step = 000374, loss = 0.001419
grad_step = 000375, loss = 0.001408
grad_step = 000376, loss = 0.001408
grad_step = 000377, loss = 0.001418
grad_step = 000378, loss = 0.001420
grad_step = 000379, loss = 0.001423
grad_step = 000380, loss = 0.001414
grad_step = 000381, loss = 0.001390
grad_step = 000382, loss = 0.001381
grad_step = 000383, loss = 0.001377
grad_step = 000384, loss = 0.001384
grad_step = 000385, loss = 0.001385
grad_step = 000386, loss = 0.001393
grad_step = 000387, loss = 0.001398
grad_step = 000388, loss = 0.001434
grad_step = 000389, loss = 0.001432
grad_step = 000390, loss = 0.001448
grad_step = 000391, loss = 0.001402
grad_step = 000392, loss = 0.001365
grad_step = 000393, loss = 0.001361
grad_step = 000394, loss = 0.001388
grad_step = 000395, loss = 0.001398
grad_step = 000396, loss = 0.001374
grad_step = 000397, loss = 0.001362
grad_step = 000398, loss = 0.001371
grad_step = 000399, loss = 0.001375
grad_step = 000400, loss = 0.001370
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001356
grad_step = 000402, loss = 0.001356
grad_step = 000403, loss = 0.001351
grad_step = 000404, loss = 0.001349
grad_step = 000405, loss = 0.001330
grad_step = 000406, loss = 0.001316
grad_step = 000407, loss = 0.001312
grad_step = 000408, loss = 0.001322
grad_step = 000409, loss = 0.001334
grad_step = 000410, loss = 0.001356
grad_step = 000411, loss = 0.001374
grad_step = 000412, loss = 0.001437
grad_step = 000413, loss = 0.001438
grad_step = 000414, loss = 0.001472
grad_step = 000415, loss = 0.001392
grad_step = 000416, loss = 0.001368
grad_step = 000417, loss = 0.001373
grad_step = 000418, loss = 0.001375
grad_step = 000419, loss = 0.001367
grad_step = 000420, loss = 0.001373
grad_step = 000421, loss = 0.001363
grad_step = 000422, loss = 0.001322
grad_step = 000423, loss = 0.001288
grad_step = 000424, loss = 0.001297
grad_step = 000425, loss = 0.001317
grad_step = 000426, loss = 0.001314
grad_step = 000427, loss = 0.001285
grad_step = 000428, loss = 0.001270
grad_step = 000429, loss = 0.001288
grad_step = 000430, loss = 0.001304
grad_step = 000431, loss = 0.001316
grad_step = 000432, loss = 0.001321
grad_step = 000433, loss = 0.001399
grad_step = 000434, loss = 0.001542
grad_step = 000435, loss = 0.001833
grad_step = 000436, loss = 0.001685
grad_step = 000437, loss = 0.001509
grad_step = 000438, loss = 0.001326
grad_step = 000439, loss = 0.001425
grad_step = 000440, loss = 0.001442
grad_step = 000441, loss = 0.001296
grad_step = 000442, loss = 0.001364
grad_step = 000443, loss = 0.001408
grad_step = 000444, loss = 0.001274
grad_step = 000445, loss = 0.001327
grad_step = 000446, loss = 0.001389
grad_step = 000447, loss = 0.001281
grad_step = 000448, loss = 0.001291
grad_step = 000449, loss = 0.001318
grad_step = 000450, loss = 0.001246
grad_step = 000451, loss = 0.001300
grad_step = 000452, loss = 0.001347
grad_step = 000453, loss = 0.001246
grad_step = 000454, loss = 0.001278
grad_step = 000455, loss = 0.001322
grad_step = 000456, loss = 0.001304
grad_step = 000457, loss = 0.001293
grad_step = 000458, loss = 0.001252
grad_step = 000459, loss = 0.001270
grad_step = 000460, loss = 0.001264
grad_step = 000461, loss = 0.001264
grad_step = 000462, loss = 0.001285
grad_step = 000463, loss = 0.001212
grad_step = 000464, loss = 0.001243
grad_step = 000465, loss = 0.001272
grad_step = 000466, loss = 0.001271
grad_step = 000467, loss = 0.001250
grad_step = 000468, loss = 0.001199
grad_step = 000469, loss = 0.001260
grad_step = 000470, loss = 0.001250
grad_step = 000471, loss = 0.001235
grad_step = 000472, loss = 0.001242
grad_step = 000473, loss = 0.001200
grad_step = 000474, loss = 0.001235
grad_step = 000475, loss = 0.001224
grad_step = 000476, loss = 0.001230
grad_step = 000477, loss = 0.001251
grad_step = 000478, loss = 0.001187
grad_step = 000479, loss = 0.001208
grad_step = 000480, loss = 0.001203
grad_step = 000481, loss = 0.001230
grad_step = 000482, loss = 0.001225
grad_step = 000483, loss = 0.001175
grad_step = 000484, loss = 0.001194
grad_step = 000485, loss = 0.001184
grad_step = 000486, loss = 0.001203
grad_step = 000487, loss = 0.001203
grad_step = 000488, loss = 0.001168
grad_step = 000489, loss = 0.001181
grad_step = 000490, loss = 0.001158
grad_step = 000491, loss = 0.001178
grad_step = 000492, loss = 0.001190
grad_step = 000493, loss = 0.001165
grad_step = 000494, loss = 0.001168
grad_step = 000495, loss = 0.001140
grad_step = 000496, loss = 0.001153
grad_step = 000497, loss = 0.001160
grad_step = 000498, loss = 0.001150
grad_step = 000499, loss = 0.001158
grad_step = 000500, loss = 0.001135
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001137
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

  date_run                              2020-05-14 14:13:12.443423
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.246471
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 14:13:12.449595
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.152726
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 14:13:12.457737
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.138959
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 14:13:12.463544
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.32072
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
0   2020-05-14 14:12:44.200324  ...    mean_absolute_error
1   2020-05-14 14:12:44.204409  ...     mean_squared_error
2   2020-05-14 14:12:44.207818  ...  median_absolute_error
3   2020-05-14 14:12:44.211102  ...               r2_score
4   2020-05-14 14:12:53.336664  ...    mean_absolute_error
5   2020-05-14 14:12:53.340404  ...     mean_squared_error
6   2020-05-14 14:12:53.343751  ...  median_absolute_error
7   2020-05-14 14:12:53.346895  ...               r2_score
8   2020-05-14 14:13:12.443423  ...    mean_absolute_error
9   2020-05-14 14:13:12.449595  ...     mean_squared_error
10  2020-05-14 14:13:12.457737  ...  median_absolute_error
11  2020-05-14 14:13:12.463544  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f375af89fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3645440/9912422 [00:00<00:00, 36446787.92it/s]9920512it [00:00, 35070384.25it/s]                             
0it [00:00, ?it/s]32768it [00:00, 619499.01it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:09, 163503.74it/s]1654784it [00:00, 11082795.45it/s]                         
0it [00:00, ?it/s]8192it [00:00, 104454.04it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f370d98ce80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f370cfbb0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f370d98ce80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f370cf100f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f370a74d4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f370a736748> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f370d98ce80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f370cece710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f370a74d4e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f375af94ef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f9f83c221d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=2c1656ec56569a1d6012a3f45699b6457b4a3f60b1531449afa2e7ae24e5f6f0
  Stored in directory: /tmp/pip-ephem-wheel-cache-r5m_xmw2/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f9f1ba1d748> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1703936/17464789 [=>............................] - ETA: 0s
 5406720/17464789 [========>.....................] - ETA: 0s
 9232384/17464789 [==============>...............] - ETA: 0s
12943360/17464789 [=====================>........] - ETA: 0s
16998400/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 14:14:39.544246: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 14:14:39.548763: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-14 14:14:39.548980: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fcd6380a00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 14:14:39.549018: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8660 - accuracy: 0.4870
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8430 - accuracy: 0.4885 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7126 - accuracy: 0.4970
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7816 - accuracy: 0.4925
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7188 - accuracy: 0.4966
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6896 - accuracy: 0.4985
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7148 - accuracy: 0.4969
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7030 - accuracy: 0.4976
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6871 - accuracy: 0.4987
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6559 - accuracy: 0.5007
11000/25000 [============>.................] - ETA: 3s - loss: 7.6304 - accuracy: 0.5024
12000/25000 [=============>................] - ETA: 3s - loss: 7.6296 - accuracy: 0.5024
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6489 - accuracy: 0.5012
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6447 - accuracy: 0.5014
15000/25000 [=================>............] - ETA: 2s - loss: 7.6380 - accuracy: 0.5019
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6168 - accuracy: 0.5033
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6197 - accuracy: 0.5031
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6394 - accuracy: 0.5018
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6497 - accuracy: 0.5011
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6574 - accuracy: 0.5006
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6535 - accuracy: 0.5009
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6534 - accuracy: 0.5009
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6520 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6609 - accuracy: 0.5004
25000/25000 [==============================] - 7s 283us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 14:14:53.261141
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 14:14:53.261141  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<42:56:38, 5.58kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<30:17:39, 7.91kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:01<21:14:59, 11.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 492k/862M [00:01<14:53:46, 16.1kB/s].vector_cache/glove.6B.zip:   0%|          | 999k/862M [00:01<10:26:08, 22.9kB/s].vector_cache/glove.6B.zip:   0%|          | 1.26M/862M [00:02<7:19:52, 32.6kB/s].vector_cache/glove.6B.zip:   0%|          | 1.53M/862M [00:02<5:09:24, 46.4kB/s].vector_cache/glove.6B.zip:   0%|          | 1.93M/862M [00:02<3:37:34, 65.9kB/s].vector_cache/glove.6B.zip:   0%|          | 2.46M/862M [00:02<2:33:03, 93.6kB/s].vector_cache/glove.6B.zip:   0%|          | 2.78M/862M [00:02<1:48:26, 132kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.13M/862M [00:02<1:17:08, 186kB/s].vector_cache/glove.6B.zip:   0%|          | 3.49M/862M [00:02<55:11, 259kB/s]  .vector_cache/glove.6B.zip:   0%|          | 3.82M/862M [00:02<39:58, 358kB/s].vector_cache/glove.6B.zip:   0%|          | 4.18M/862M [00:02<29:09, 490kB/s].vector_cache/glove.6B.zip:   1%|          | 4.95M/862M [00:02<20:57, 682kB/s].vector_cache/glove.6B.zip:   1%|          | 5.64M/862M [00:03<15:17, 934kB/s].vector_cache/glove.6B.zip:   1%|          | 6.54M/862M [00:03<11:09, 1.28MB/s].vector_cache/glove.6B.zip:   1%|          | 7.31M/862M [00:03<08:22, 1.70MB/s].vector_cache/glove.6B.zip:   1%|          | 8.35M/862M [00:03<06:16, 2.27MB/s].vector_cache/glove.6B.zip:   1%|          | 9.07M/862M [00:03<04:58, 2.85MB/s].vector_cache/glove.6B.zip:   1%|          | 10.3M/862M [00:03<03:50, 3.70MB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.3M/862M [00:03<03:06, 4.55MB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:03<02:32, 5.57MB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.5M/862M [00:03<02:14, 6.32MB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.7M/862M [00:03<01:54, 7.38MB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.2M/862M [00:04<01:37, 8.66MB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:04<01:25, 9.89MB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.4M/862M [00:04<01:14, 11.3MB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:04<01:06, 12.6MB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.6M/862M [00:04<01:04, 13.0MB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.1M/862M [00:04<01:02, 13.4MB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.2M/862M [00:04<00:55, 15.0MB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.2M/862M [00:04<00:51, 16.3MB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.8M/862M [00:04<00:45, 18.2MB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.1M/862M [00:04<00:42, 19.5MB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.0M/862M [00:05<00:38, 21.6MB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.7M/862M [00:05<00:36, 22.7MB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.0M/862M [00:05<00:32, 25.1MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.0M/862M [00:05<00:30, 26.4MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.8M/862M [00:05<00:28, 29.0MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:05<00:55, 14.7MB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.4M/862M [00:06<00:57, 14.1MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.2M/862M [00:06<00:53, 15.1MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:07<07:43, 1.74MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:08<13:55, 965kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:08<14:45, 911kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:08<12:48, 1.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:08<09:38, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:08<06:58, 1.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:09<08:45, 1.52MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:10<08:53, 1.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:10<06:53, 1.94MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:10<04:58, 2.67MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:11<1:52:06, 119kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.7M/862M [00:11<1:21:14, 164kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:12<57:30, 231kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:12<40:15, 329kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:13<43:13, 306kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:13<32:58, 401kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:14<23:38, 559kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.8M/862M [00:14<16:40, 790kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:15<19:07, 688kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:15<16:07, 815kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:16<11:57, 1.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:16<08:29, 1.54MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:17<17:29, 748kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:17<14:57, 875kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:18<11:07, 1.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:18<07:54, 1.65MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:19<1:52:49, 115kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:19<1:21:22, 160kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:19<58:19, 223kB/s]  .vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:20<41:03, 316kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:20<28:49, 449kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:21<1:24:45, 153kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:21<1:01:58, 209kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:22<44:00, 294kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 88.1M/862M [00:22<30:54, 417kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:23<27:22, 471kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:23<21:49, 590kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:23<15:46, 816kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:24<11:24, 1.13MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:25<10:56, 1.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:25<10:16, 1.25MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:25<07:45, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.3M/862M [00:25<05:46, 2.21MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:27<07:34, 1.68MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:28<10:31, 1.21MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:28<08:40, 1.47MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:28<06:21, 2.00MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<07:20, 1.73MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<07:43, 1.64MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:30<06:02, 2.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:30<04:26, 2.84MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<07:28, 1.69MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<07:55, 1.59MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<06:23, 1.97MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:32<04:46, 2.64MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<03:31, 3.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<39:06, 321kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<30:02, 417kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:34<21:41, 577kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:34<15:16, 817kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<22:12, 562kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<18:08, 688kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:36<13:14, 941kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:36<09:25, 1.32MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<11:40, 1.06MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<10:42, 1.16MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:37<08:00, 1.55MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:38<05:47, 2.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:39<08:54, 1.38MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:39<08:15, 1.49MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<07:10, 1.72MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:40<05:19, 2.31MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:40<03:52, 3.16MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:41<1:00:47, 202kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:41<45:04, 272kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:41<32:08, 381kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:42<22:34, 540kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:43<2:11:42, 92.6kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<1:34:42, 129kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<1:06:45, 182kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:43<46:48, 260kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:45<37:15, 326kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<28:35, 424kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<20:37, 587kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:45<14:30, 832kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<29:04, 415kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<22:47, 529kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:47<16:32, 728kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<11:40, 1.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<1:46:33, 113kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<1:16:54, 156kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:49<54:17, 221kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:49<38:01, 314kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<35:05, 340kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<27:01, 441kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:51<19:31, 610kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:51<13:45, 862kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:53<1:34:24, 126kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<1:08:28, 173kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:53<48:29, 244kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<33:56, 347kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:56<3:10:46, 61.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:56<2:25:12, 81.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:56<1:44:23, 113kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:56<1:13:41, 160kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:56<51:34, 228kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:58<41:41, 281kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:58<31:42, 369kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<22:42, 515kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:58<16:11, 721kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:58<11:29, 1.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:00<40:17, 289kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:00<30:41, 379kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<22:05, 527kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:00<15:36, 744kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:02<15:37, 741kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:02<13:28, 859kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:02<09:59, 1.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:02<07:07, 1.62MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:04<10:54, 1.06MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:04<09:35, 1.20MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:04<07:09, 1.61MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:04<05:10, 2.21MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<08:36, 1.33MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:06<07:53, 1.45MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:06<06:43, 1.70MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:06<04:56, 2.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<03:36, 3.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<2:44:47, 69.0kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:08<1:57:47, 96.5kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:08<1:22:57, 137kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:08<57:57, 195kB/s]  .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<54:52, 206kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<40:50, 277kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:10<29:48, 379kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:10<21:11, 532kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<14:55, 754kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<36:59, 304kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<28:20, 396kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:12<21:01, 534kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:12<15:11, 738kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:12<10:48, 1.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<11:28, 973kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<10:13, 1.09MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:14<08:25, 1.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:14<06:17, 1.77MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<04:31, 2.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:15<17:32, 633kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:15<14:45, 752kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<11:54, 931kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:16<09:00, 1.23MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:16<06:27, 1.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:17<08:12, 1.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:17<08:07, 1.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<06:24, 1.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:18<04:44, 2.32MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<03:29, 3.15MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:19<19:52, 552kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:19<16:15, 674kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<11:53, 921kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:20<08:33, 1.28MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:21<08:46, 1.24MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<08:24, 1.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<06:23, 1.70MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:22<04:50, 2.24MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:22<03:31, 3.07MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<34:13, 316kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<26:16, 412kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<18:52, 573kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<13:27, 802kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<12:00, 896kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<10:25, 1.03MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<08:26, 1.27MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<06:10, 1.74MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<04:27, 2.40MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:27<36:57, 289kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:28<31:44, 337kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:28<23:48, 449kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:28<17:01, 626kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:29<13:42, 774kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:30<11:36, 914kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:30<08:37, 1.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:31<07:47, 1.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:32<09:28, 1.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:32<07:37, 1.38MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:32<05:33, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:33<06:30, 1.61MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:34<06:28, 1.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:34<04:57, 2.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:34<03:34, 2.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:35<10:42, 972kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:36<09:25, 1.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:36<07:06, 1.46MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:36<05:12, 1.99MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:37<06:14, 1.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<06:58, 1.48MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:38<06:23, 1.62MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:38<05:06, 2.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:38<03:47, 2.72MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:39<05:12, 1.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:39<06:14, 1.64MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:40<06:15, 1.64MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:40<05:27, 1.88MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:40<04:19, 2.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:40<03:09, 3.24MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<08:41, 1.17MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<07:44, 1.32MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:42<06:50, 1.49MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:42<05:26, 1.87MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:42<04:08, 2.46MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:42<03:03, 3.32MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:43<11:21, 893kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:43<09:53, 1.02MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<07:19, 1.38MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:44<05:21, 1.88MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:45<06:19, 1.59MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:45<06:55, 1.45MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:45<05:45, 1.75MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:46<04:38, 2.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:46<03:38, 2.76MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:46<03:07, 3.21MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:46<02:47, 3.60MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:46<02:31, 3.98MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:46<02:11, 4.58MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:47<54:14, 184kB/s] .vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:47<39:38, 252kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:47<28:02, 356kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:48<19:54, 500kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:48<14:05, 705kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:49<20:41, 480kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:49<16:26, 604kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<11:51, 836kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:49<08:33, 1.16MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<06:08, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:51<33:10, 297kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<25:06, 393kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<18:01, 546kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:51<12:40, 774kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:53<20:18, 482kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<16:02, 610kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<11:47, 830kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:53<08:27, 1.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<08:21, 1.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<07:37, 1.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<05:47, 1.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:55<04:10, 2.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:57<09:36, 1.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:58<13:33, 712kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:58<11:29, 840kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:58<08:25, 1.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:58<06:02, 1.59MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:59<07:43, 1.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [02:00<07:08, 1.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:00<05:27, 1.75MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:00<04:02, 2.36MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:00<03:00, 3.17MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:01<14:42, 648kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:02<12:06, 786kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:02<08:54, 1.07MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:02<06:26, 1.47MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:03<06:54, 1.37MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:03<06:38, 1.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:04<05:00, 1.89MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:04<03:44, 2.52MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:05<04:58, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:05<05:17, 1.77MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:06<04:06, 2.28MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:06<03:05, 3.02MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:07<04:27, 2.09MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:07<04:52, 1.91MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:08<03:50, 2.42MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:09<04:13, 2.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:09<04:44, 1.95MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:10<03:44, 2.47MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:10<02:41, 3.40MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:11<33:28, 274kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:11<24:39, 372kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<17:28, 524kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:12<12:20, 739kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:13<14:13, 640kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:13<11:06, 819kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<08:01, 1.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:14<05:43, 1.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:15<12:16, 736kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:15<10:08, 891kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:15<07:28, 1.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:16<05:19, 1.69MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:17<10:49, 829kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:17<09:08, 980kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:17<06:46, 1.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:17<04:49, 1.85MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:19<13:14, 672kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:19<10:47, 825kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:19<07:55, 1.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:19<05:39, 1.57MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:21<09:16, 952kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:21<08:03, 1.10MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:21<06:01, 1.46MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:21<04:17, 2.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:23<24:04, 364kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:23<19:57, 439kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:23<14:43, 594kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:24<10:27, 835kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:25<09:36, 906kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:26<09:39, 900kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:26<07:37, 1.14MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:26<05:41, 1.53MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:26<04:08, 2.09MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:26<03:06, 2.78MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:29<29:18, 294kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:29<29:02, 297kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:29<22:21, 386kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:29<16:03, 536kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:29<11:21, 755kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:31<10:37, 805kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:31<10:27, 818kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:31<08:03, 1.06MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:31<05:47, 1.47MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:33<06:19, 1.34MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:33<06:41, 1.27MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:33<06:04, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:33<04:51, 1.75MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:33<03:39, 2.32MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:34<02:40, 3.16MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:35<12:32, 672kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:35<10:12, 824kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:35<07:26, 1.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:35<05:20, 1.57MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:37<06:47, 1.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:37<06:14, 1.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:37<04:40, 1.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:37<03:22, 2.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:39<07:12, 1.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:39<06:32, 1.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:39<04:54, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:41<04:49, 1.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:41<04:42, 1.75MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:41<04:02, 2.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:41<03:02, 2.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:43<03:45, 2.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:43<03:55, 2.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:43<03:30, 2.32MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:43<02:37, 3.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:46<04:22, 1.84MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:46<10:58, 735kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:46<09:34, 843kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:46<07:06, 1.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:46<05:03, 1.59MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:48<07:17, 1.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:48<06:30, 1.23MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:48<04:49, 1.66MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:48<03:33, 2.24MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:49<04:42, 1.68MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:50<04:35, 1.73MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:50<03:57, 2.00MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:50<03:09, 2.51MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:50<02:23, 3.31MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:50<01:51, 4.25MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:51<12:00, 656kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:52<09:50, 799kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:52<07:43, 1.02MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:52<05:34, 1.41MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:54<05:58, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:54<09:43, 802kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:54<08:13, 949kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:54<06:03, 1.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:54<04:22, 1.77MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:56<05:51, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:56<06:47, 1.14MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:56<05:26, 1.42MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:56<03:57, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:57<02:55, 2.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:58<07:14, 1.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:58<06:25, 1.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:58<04:46, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:58<03:29, 2.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [03:00<04:54, 1.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [03:00<05:35, 1.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [03:00<05:19, 1.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [03:00<04:21, 1.74MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [03:00<03:14, 2.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:01<02:21, 3.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:02<52:08, 144kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:02<37:51, 199kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:02<26:43, 281kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:02<18:48, 398kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:04<15:25, 483kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:04<12:06, 616kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:04<08:44, 851kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:04<06:13, 1.19MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:06<07:20, 1.01MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:06<06:27, 1.14MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:06<04:51, 1.52MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:06<03:34, 2.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:06<02:40, 2.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:08<06:23, 1.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:08<05:54, 1.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:08<04:28, 1.63MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:08<03:13, 2.25MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:10<05:42, 1.27MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:10<05:14, 1.38MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:10<03:56, 1.83MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:10<02:52, 2.51MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:12<04:55, 1.46MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:12<04:45, 1.51MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:12<03:35, 2.00MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:12<02:40, 2.67MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:14<03:49, 1.86MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:14<03:56, 1.81MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:14<03:03, 2.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:16<03:20, 2.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:16<03:31, 1.99MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:16<02:43, 2.57MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:16<02:01, 3.45MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:18<04:00, 1.74MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:18<04:01, 1.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:18<03:06, 2.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:20<03:21, 2.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:20<03:31, 1.96MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:20<02:43, 2.53MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:20<02:00, 3.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:22<04:25, 1.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:22<04:15, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:22<03:13, 2.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:22<02:22, 2.87MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:24<04:22, 1.55MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:24<03:56, 1.72MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:24<02:57, 2.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:24<02:10, 3.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:26<05:46, 1.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:26<09:35, 698kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:26<07:47, 860kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:26<05:51, 1.14MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:26<04:12, 1.58MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:28<04:32, 1.46MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:28<04:19, 1.53MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:28<03:18, 2.00MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:28<02:26, 2.69MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:28<01:53, 3.46MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:31<09:59, 656kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:31<11:37, 565kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:31<09:19, 703kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:31<06:46, 967kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:31<04:47, 1.36MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:33<09:02, 719kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:33<07:25, 874kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:33<05:27, 1.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:34<04:54, 1.31MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:35<04:32, 1.41MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:35<03:22, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:35<02:30, 2.54MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:36<03:35, 1.77MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:37<03:37, 1.75MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:37<02:48, 2.26MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:38<03:01, 2.07MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:39<03:13, 1.95MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:39<02:32, 2.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:39<01:54, 3.28MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:40<03:05, 2.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:41<03:13, 1.93MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:41<02:30, 2.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:42<02:48, 2.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:42<04:08, 1.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:43<03:25, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:43<02:31, 2.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:43<01:50, 3.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:44<18:56, 321kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:44<14:19, 424kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:45<10:15, 590kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:46<08:09, 736kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:46<06:45, 888kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:47<04:58, 1.20MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:48<04:28, 1.33MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:48<04:09, 1.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:49<03:09, 1.88MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:50<03:14, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:50<04:18, 1.36MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:51<03:30, 1.67MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:51<02:34, 2.27MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:52<03:23, 1.71MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:52<03:23, 1.71MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:53<02:37, 2.21MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:54<02:48, 2.04MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:54<02:58, 1.92MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:54<02:19, 2.46MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:56<02:35, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:56<02:46, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:56<02:11, 2.59MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:58<02:39, 2.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:59<05:50, 957kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:59<05:10, 1.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:59<04:07, 1.36MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:59<03:00, 1.85MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:59<02:12, 2.51MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [04:01<04:42, 1.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [04:01<04:14, 1.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:01<03:09, 1.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:01<02:18, 2.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:03<03:44, 1.46MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:03<04:32, 1.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:03<03:38, 1.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:03<02:39, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [04:05<03:00, 1.79MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [04:05<03:03, 1.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:05<02:21, 2.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:05<01:42, 3.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:07<06:51, 776kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:07<05:42, 932kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:07<04:10, 1.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:07<03:01, 1.74MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:07<02:13, 2.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:09<17:47, 295kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:09<13:19, 394kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:09<09:46, 537kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:09<06:54, 755kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:11<06:00, 863kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:11<05:04, 1.02MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:11<03:43, 1.39MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:11<02:42, 1.89MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:13<03:21, 1.53MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:13<03:14, 1.58MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:13<02:38, 1.93MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:13<02:09, 2.37MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:13<01:37, 3.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:15<02:27, 2.05MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:15<03:15, 1.55MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:15<02:57, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:15<02:19, 2.16MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:15<01:44, 2.87MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:15<01:19, 3.76MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:17<04:30, 1.11MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:17<04:00, 1.24MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:17<02:59, 1.66MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:17<02:10, 2.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:19<03:01, 1.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:19<02:58, 1.65MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:19<02:15, 2.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:19<01:38, 2.95MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:20<03:12, 1.51MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:21<02:57, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:21<02:16, 2.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:21<01:38, 2.91MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:22<03:47, 1.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:23<03:23, 1.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:23<02:49, 1.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:23<02:05, 2.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:24<02:25, 1.94MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:25<02:20, 2.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:25<02:05, 2.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:25<01:31, 3.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:26<02:18, 2.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:27<02:24, 1.92MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:27<01:51, 2.49MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:27<01:25, 3.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:28<02:04, 2.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:29<02:18, 1.97MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:29<01:48, 2.52MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:29<01:21, 3.33MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:30<02:14, 2.00MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:31<03:08, 1.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:31<02:31, 1.77MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:31<01:55, 2.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:32<02:07, 2.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:33<02:14, 1.97MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:33<01:47, 2.47MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:33<01:20, 3.28MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:34<02:09, 2.02MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:34<02:34, 1.69MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:35<02:14, 1.94MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:35<01:46, 2.43MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:35<01:18, 3.28MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:36<02:22, 1.80MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:36<02:24, 1.78MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:37<01:52, 2.28MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:37<01:23, 3.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:38<02:09, 1.96MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:38<02:13, 1.90MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:39<01:44, 2.42MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:41<02:02, 2.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:41<02:53, 1.44MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:41<02:23, 1.74MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:41<01:44, 2.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:43<02:20, 1.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:43<02:19, 1.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:43<01:47, 2.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:43<01:18, 3.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:45<03:08, 1.28MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:45<02:52, 1.39MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:45<02:10, 1.83MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:45<01:34, 2.51MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:46<02:54, 1.36MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:47<02:42, 1.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:47<02:03, 1.91MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:48<02:05, 1.85MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:49<02:06, 1.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:49<01:48, 2.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:49<01:19, 2.89MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:50<01:53, 2.01MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:51<01:48, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:51<01:22, 2.76MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:53<01:47, 2.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:53<02:03, 1.81MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:53<01:46, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:53<01:20, 2.77MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:53<01:03, 3.49MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:53<00:50, 4.41MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:55<03:40, 999kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:55<03:23, 1.08MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:55<02:33, 1.43MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:55<01:49, 1.98MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:57<02:42, 1.33MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:57<02:24, 1.49MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:57<02:00, 1.79MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:57<01:29, 2.40MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:57<01:05, 3.26MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:59<06:32, 539kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:59<05:12, 678kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:59<03:46, 929kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:59<02:39, 1.31MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [05:01<04:38, 746kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [05:01<03:50, 900kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [05:01<02:48, 1.23MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [05:01<02:01, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [05:03<02:22, 1.43MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [05:03<02:14, 1.52MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:03<01:40, 2.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:03<01:14, 2.70MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:04<01:52, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:05<01:48, 1.84MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:05<01:24, 2.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:05<01:02, 3.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:06<01:51, 1.75MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:07<01:53, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:07<01:27, 2.22MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:07<01:03, 3.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:08<02:23, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:09<02:12, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:09<01:40, 1.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:09<01:11, 2.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:10<10:57, 285kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:11<08:11, 380kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:11<05:50, 531kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:12<04:32, 671kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:12<03:41, 825kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:13<02:42, 1.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:14<02:22, 1.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:14<02:10, 1.37MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:15<01:38, 1.81MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:16<01:37, 1.79MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:16<01:38, 1.76MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:17<01:15, 2.31MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:17<00:55, 3.12MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:18<01:46, 1.61MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:18<01:43, 1.65MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:19<01:19, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:20<01:23, 2.00MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:20<01:56, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:20<01:34, 1.76MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:21<01:08, 2.39MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:22<01:31, 1.77MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:22<01:31, 1.76MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:22<01:10, 2.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:23<00:50, 3.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:24<03:48, 693kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:24<03:35, 733kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:25<02:42, 972kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:25<01:58, 1.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:26<01:51, 1.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:27<02:08, 1.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:27<01:43, 1.48MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:27<01:15, 2.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:27<00:54, 2.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:28<03:54, 638kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:28<03:09, 789kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:29<02:18, 1.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:29<01:37, 1.50MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:30<02:19, 1.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:30<02:02, 1.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:31<01:30, 1.60MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:31<01:04, 2.21MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:32<01:45, 1.35MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:32<01:32, 1.52MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:32<01:18, 1.80MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:33<00:58, 2.41MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:34<01:08, 2.00MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:34<01:11, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:35<00:55, 2.46MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:35<00:39, 3.39MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:36<15:41, 142kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:37<11:45, 189kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:37<08:22, 264kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:37<05:56, 371kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:37<04:09, 524kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:37<02:56, 734kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:38<03:56, 546kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:39<03:11, 675kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:39<02:18, 926kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:39<01:36, 1.30MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:40<02:19, 896kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:41<01:59, 1.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:41<01:27, 1.42MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:41<01:03, 1.93MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:42<01:18, 1.55MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:43<01:14, 1.61MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:43<00:57, 2.09MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:44<00:59, 1.97MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:45<01:01, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:45<00:46, 2.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:45<00:33, 3.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:46<01:42, 1.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:46<01:30, 1.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:47<01:07, 1.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:48<01:05, 1.67MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:48<01:04, 1.68MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:49<00:48, 2.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:49<00:35, 2.98MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:50<01:01, 1.69MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:50<01:01, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:51<00:47, 2.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:51<00:34, 2.98MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:52<01:02, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:52<01:01, 1.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:52<00:46, 2.12MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:53<00:34, 2.84MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:54<00:49, 1.96MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:54<00:48, 1.99MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:54<00:41, 2.29MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:55<00:32, 2.92MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:55<01:45, 900kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:55<01:13, 1.26MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:57<01:14, 1.22MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:57<01:56, 777kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:57<01:37, 923kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:57<01:11, 1.25MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:57<00:50, 1.74MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:59<01:09, 1.24MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:59<01:03, 1.35MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:59<00:49, 1.75MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:59<00:35, 2.37MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [06:01<00:45, 1.81MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [06:01<00:45, 1.79MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [06:01<00:34, 2.33MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [06:01<00:24, 3.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:03<00:57, 1.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:03<00:54, 1.44MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:03<00:40, 1.89MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:03<00:28, 2.62MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:05<03:34, 345kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:05<02:54, 424kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [06:05<02:07, 575kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:05<01:29, 808kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:05<01:01, 1.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:07<04:44, 245kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:07<03:30, 331kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:07<02:27, 465kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:07<01:42, 655kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:09<01:31, 722kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:09<01:13, 897kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:09<00:55, 1.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:09<00:38, 1.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:11<00:43, 1.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:11<00:41, 1.49MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:11<00:30, 1.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:13<00:30, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:13<00:31, 1.82MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:13<00:24, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:13<00:16, 3.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:15<05:02, 177kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:15<03:39, 242kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:15<02:33, 341kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:15<01:42, 484kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:17<02:09, 381kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:17<01:48, 452kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:17<01:18, 618kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:17<00:56, 853kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:18<00:38, 1.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:19<00:50, 887kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:19<00:43, 1.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:19<00:31, 1.39MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:19<00:21, 1.91MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:21<00:31, 1.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:21<00:29, 1.38MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:21<00:24, 1.63MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:21<00:19, 2.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:21<00:15, 2.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:22<00:12, 3.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:22<00:11, 3.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:22<00:09, 3.79MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:22<00:08, 4.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:23<01:16, 480kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:23<00:59, 618kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:23<00:41, 852kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:25<00:32, 1.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:25<00:28, 1.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:25<00:20, 1.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:25<00:14, 2.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:27<00:17, 1.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:27<00:20, 1.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:27<00:17, 1.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:27<00:12, 2.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:27<00:08, 2.93MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:29<00:19, 1.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:29<00:17, 1.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:29<00:12, 1.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:29<00:08, 2.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:31<00:11, 1.75MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:31<00:11, 1.75MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:31<00:08, 2.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:31<00:05, 3.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:33<00:08, 1.96MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:33<00:08, 1.89MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:33<00:06, 2.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:33<00:03, 3.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:35<00:08, 1.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:35<00:07, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:35<00:05, 1.98MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:35<00:03, 2.70MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:37<00:05, 1.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:37<00:05, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:37<00:03, 1.99MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:37<00:01, 2.74MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:39<00:03, 1.16MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:39<00:02, 1.25MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:39<00:02, 1.54MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:39<00:01, 1.97MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:39<00:00, 2.64MB/s].vector_cache/glove.6B.zip: 862MB [06:39, 2.16MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 848/400000 [00:00<00:47, 8479.44it/s]  0%|          | 1676/400000 [00:00<00:47, 8417.73it/s]  1%|          | 2527/400000 [00:00<00:47, 8443.57it/s]  1%|          | 3379/400000 [00:00<00:46, 8464.25it/s]  1%|          | 4232/400000 [00:00<00:46, 8482.71it/s]  1%|▏         | 5092/400000 [00:00<00:46, 8515.74it/s]  1%|▏         | 5943/400000 [00:00<00:46, 8511.17it/s]  2%|▏         | 6796/400000 [00:00<00:46, 8514.29it/s]  2%|▏         | 7604/400000 [00:00<00:46, 8376.68it/s]  2%|▏         | 8461/400000 [00:01<00:46, 8431.52it/s]  2%|▏         | 9322/400000 [00:01<00:46, 8482.44it/s]  3%|▎         | 10154/400000 [00:01<00:46, 8411.38it/s]  3%|▎         | 10997/400000 [00:01<00:46, 8416.60it/s]  3%|▎         | 11834/400000 [00:01<00:46, 8400.51it/s]  3%|▎         | 12669/400000 [00:01<00:46, 8345.28it/s]  3%|▎         | 13511/400000 [00:01<00:46, 8367.14it/s]  4%|▎         | 14368/400000 [00:01<00:45, 8424.32it/s]  4%|▍         | 15223/400000 [00:01<00:45, 8459.49it/s]  4%|▍         | 16068/400000 [00:01<00:46, 8344.41it/s]  4%|▍         | 16902/400000 [00:02<00:46, 8180.38it/s]  4%|▍         | 17757/400000 [00:02<00:46, 8286.32it/s]  5%|▍         | 18610/400000 [00:02<00:45, 8356.00it/s]  5%|▍         | 19467/400000 [00:02<00:45, 8416.71it/s]  5%|▌         | 20310/400000 [00:02<00:47, 8011.70it/s]  5%|▌         | 21154/400000 [00:02<00:46, 8133.69it/s]  5%|▌         | 21988/400000 [00:02<00:46, 8193.13it/s]  6%|▌         | 22845/400000 [00:02<00:45, 8299.97it/s]  6%|▌         | 23681/400000 [00:02<00:45, 8317.04it/s]  6%|▌         | 24520/400000 [00:02<00:45, 8338.05it/s]  6%|▋         | 25367/400000 [00:03<00:44, 8376.10it/s]  7%|▋         | 26223/400000 [00:03<00:44, 8429.44it/s]  7%|▋         | 27083/400000 [00:03<00:43, 8478.31it/s]  7%|▋         | 27937/400000 [00:03<00:43, 8495.88it/s]  7%|▋         | 28787/400000 [00:03<00:44, 8381.54it/s]  7%|▋         | 29644/400000 [00:03<00:43, 8436.68it/s]  8%|▊         | 30504/400000 [00:03<00:43, 8482.39it/s]  8%|▊         | 31363/400000 [00:03<00:43, 8513.75it/s]  8%|▊         | 32215/400000 [00:03<00:43, 8508.88it/s]  8%|▊         | 33067/400000 [00:03<00:43, 8498.88it/s]  8%|▊         | 33918/400000 [00:04<00:43, 8352.93it/s]  9%|▊         | 34766/400000 [00:04<00:43, 8388.40it/s]  9%|▉         | 35607/400000 [00:04<00:43, 8393.12it/s]  9%|▉         | 36469/400000 [00:04<00:42, 8459.44it/s]  9%|▉         | 37317/400000 [00:04<00:42, 8462.78it/s] 10%|▉         | 38176/400000 [00:04<00:42, 8499.50it/s] 10%|▉         | 39042/400000 [00:04<00:42, 8545.56it/s] 10%|▉         | 39902/400000 [00:04<00:42, 8561.42it/s] 10%|█         | 40762/400000 [00:04<00:41, 8572.83it/s] 10%|█         | 41620/400000 [00:04<00:42, 8502.67it/s] 11%|█         | 42478/400000 [00:05<00:41, 8523.11it/s] 11%|█         | 43338/400000 [00:05<00:41, 8543.40it/s] 11%|█         | 44196/400000 [00:05<00:41, 8552.18it/s] 11%|█▏        | 45057/400000 [00:05<00:41, 8567.27it/s] 11%|█▏        | 45914/400000 [00:05<00:41, 8535.64it/s] 12%|█▏        | 46776/400000 [00:05<00:41, 8559.07it/s] 12%|█▏        | 47632/400000 [00:05<00:41, 8507.96it/s] 12%|█▏        | 48483/400000 [00:05<00:43, 8080.35it/s] 12%|█▏        | 49345/400000 [00:05<00:42, 8233.08it/s] 13%|█▎        | 50174/400000 [00:05<00:42, 8249.54it/s] 13%|█▎        | 51036/400000 [00:06<00:41, 8356.29it/s] 13%|█▎        | 51874/400000 [00:06<00:41, 8340.16it/s] 13%|█▎        | 52734/400000 [00:06<00:41, 8413.97it/s] 13%|█▎        | 53599/400000 [00:06<00:40, 8480.67it/s] 14%|█▎        | 54449/400000 [00:06<00:40, 8480.17it/s] 14%|█▍        | 55308/400000 [00:06<00:40, 8512.71it/s] 14%|█▍        | 56160/400000 [00:06<00:40, 8501.53it/s] 14%|█▍        | 57023/400000 [00:06<00:40, 8536.95it/s] 14%|█▍        | 57882/400000 [00:06<00:40, 8552.07it/s] 15%|█▍        | 58738/400000 [00:06<00:40, 8499.24it/s] 15%|█▍        | 59600/400000 [00:07<00:39, 8534.83it/s] 15%|█▌        | 60455/400000 [00:07<00:39, 8539.17it/s] 15%|█▌        | 61310/400000 [00:07<00:39, 8507.12it/s] 16%|█▌        | 62161/400000 [00:07<00:39, 8503.85it/s] 16%|█▌        | 63012/400000 [00:07<00:40, 8395.22it/s] 16%|█▌        | 63858/400000 [00:07<00:39, 8413.70it/s] 16%|█▌        | 64700/400000 [00:07<00:39, 8404.34it/s] 16%|█▋        | 65565/400000 [00:07<00:39, 8475.97it/s] 17%|█▋        | 66424/400000 [00:07<00:39, 8509.13it/s] 17%|█▋        | 67284/400000 [00:07<00:38, 8534.72it/s] 17%|█▋        | 68147/400000 [00:08<00:38, 8560.87it/s] 17%|█▋        | 69012/400000 [00:08<00:38, 8585.55it/s] 17%|█▋        | 69871/400000 [00:08<00:38, 8586.85it/s] 18%|█▊        | 70734/400000 [00:08<00:38, 8598.92it/s] 18%|█▊        | 71594/400000 [00:08<00:39, 8362.53it/s] 18%|█▊        | 72432/400000 [00:08<00:41, 7916.69it/s] 18%|█▊        | 73288/400000 [00:08<00:40, 8098.87it/s] 19%|█▊        | 74146/400000 [00:08<00:39, 8235.16it/s] 19%|█▊        | 74984/400000 [00:08<00:39, 8276.42it/s] 19%|█▉        | 75821/400000 [00:09<00:39, 8302.55it/s] 19%|█▉        | 76674/400000 [00:09<00:38, 8368.72it/s] 19%|█▉        | 77537/400000 [00:09<00:38, 8444.98it/s] 20%|█▉        | 78400/400000 [00:09<00:37, 8496.76it/s] 20%|█▉        | 79261/400000 [00:09<00:37, 8528.45it/s] 20%|██        | 80115/400000 [00:09<00:39, 8196.28it/s] 20%|██        | 80972/400000 [00:09<00:38, 8303.27it/s] 20%|██        | 81814/400000 [00:09<00:38, 8337.76it/s] 21%|██        | 82673/400000 [00:09<00:37, 8410.98it/s] 21%|██        | 83525/400000 [00:09<00:37, 8441.60it/s] 21%|██        | 84371/400000 [00:10<00:37, 8427.85it/s] 21%|██▏       | 85231/400000 [00:10<00:37, 8477.63it/s] 22%|██▏       | 86080/400000 [00:10<00:37, 8356.42it/s] 22%|██▏       | 86938/400000 [00:10<00:37, 8420.60it/s] 22%|██▏       | 87798/400000 [00:10<00:36, 8472.68it/s] 22%|██▏       | 88646/400000 [00:10<00:37, 8389.03it/s] 22%|██▏       | 89501/400000 [00:10<00:36, 8436.09it/s] 23%|██▎       | 90355/400000 [00:10<00:36, 8464.35it/s] 23%|██▎       | 91210/400000 [00:10<00:36, 8489.46it/s] 23%|██▎       | 92071/400000 [00:10<00:36, 8523.85it/s] 23%|██▎       | 92924/400000 [00:11<00:36, 8353.92it/s] 23%|██▎       | 93779/400000 [00:11<00:36, 8411.36it/s] 24%|██▎       | 94631/400000 [00:11<00:36, 8443.12it/s] 24%|██▍       | 95491/400000 [00:11<00:35, 8489.48it/s] 24%|██▍       | 96350/400000 [00:11<00:35, 8519.32it/s] 24%|██▍       | 97203/400000 [00:11<00:35, 8462.17it/s] 25%|██▍       | 98060/400000 [00:11<00:35, 8492.87it/s] 25%|██▍       | 98910/400000 [00:11<00:36, 8246.79it/s] 25%|██▍       | 99771/400000 [00:11<00:35, 8351.42it/s] 25%|██▌       | 100628/400000 [00:11<00:35, 8413.74it/s] 25%|██▌       | 101475/400000 [00:12<00:35, 8429.35it/s] 26%|██▌       | 102322/400000 [00:12<00:35, 8439.76it/s] 26%|██▌       | 103176/400000 [00:12<00:35, 8466.85it/s] 26%|██▌       | 104035/400000 [00:12<00:34, 8501.01it/s] 26%|██▌       | 104889/400000 [00:12<00:34, 8511.43it/s] 26%|██▋       | 105741/400000 [00:12<00:34, 8502.10it/s] 27%|██▋       | 106604/400000 [00:12<00:34, 8539.76it/s] 27%|██▋       | 107459/400000 [00:12<00:34, 8411.07it/s] 27%|██▋       | 108318/400000 [00:12<00:34, 8462.32it/s] 27%|██▋       | 109177/400000 [00:12<00:34, 8498.52it/s] 28%|██▊       | 110028/400000 [00:13<00:34, 8482.61it/s] 28%|██▊       | 110886/400000 [00:13<00:33, 8510.45it/s] 28%|██▊       | 111748/400000 [00:13<00:33, 8542.27it/s] 28%|██▊       | 112603/400000 [00:13<00:33, 8520.72it/s] 28%|██▊       | 113465/400000 [00:13<00:33, 8547.40it/s] 29%|██▊       | 114320/400000 [00:13<00:33, 8413.77it/s] 29%|██▉       | 115162/400000 [00:13<00:34, 8350.21it/s] 29%|██▉       | 115998/400000 [00:13<00:34, 8352.18it/s] 29%|██▉       | 116849/400000 [00:13<00:33, 8398.34it/s] 29%|██▉       | 117708/400000 [00:13<00:33, 8454.24it/s] 30%|██▉       | 118554/400000 [00:14<00:33, 8444.47it/s] 30%|██▉       | 119412/400000 [00:14<00:33, 8482.21it/s] 30%|███       | 120261/400000 [00:14<00:33, 8461.26it/s] 30%|███       | 121108/400000 [00:14<00:33, 8273.09it/s] 30%|███       | 121972/400000 [00:14<00:33, 8378.01it/s] 31%|███       | 122811/400000 [00:14<00:33, 8341.62it/s] 31%|███       | 123670/400000 [00:14<00:32, 8414.56it/s] 31%|███       | 124526/400000 [00:14<00:32, 8457.32it/s] 31%|███▏      | 125383/400000 [00:14<00:32, 8489.76it/s] 32%|███▏      | 126238/400000 [00:14<00:32, 8507.31it/s] 32%|███▏      | 127090/400000 [00:15<00:32, 8486.22it/s] 32%|███▏      | 127953/400000 [00:15<00:31, 8528.61it/s] 32%|███▏      | 128812/400000 [00:15<00:31, 8544.78it/s] 32%|███▏      | 129667/400000 [00:15<00:31, 8507.09it/s] 33%|███▎      | 130518/400000 [00:15<00:32, 8418.02it/s] 33%|███▎      | 131365/400000 [00:15<00:31, 8432.78it/s] 33%|███▎      | 132227/400000 [00:15<00:31, 8487.52it/s] 33%|███▎      | 133079/400000 [00:15<00:31, 8494.73it/s] 33%|███▎      | 133935/400000 [00:15<00:31, 8512.91it/s] 34%|███▎      | 134787/400000 [00:15<00:31, 8487.78it/s] 34%|███▍      | 135638/400000 [00:16<00:31, 8493.06it/s] 34%|███▍      | 136500/400000 [00:16<00:30, 8528.26it/s] 34%|███▍      | 137355/400000 [00:16<00:30, 8533.07it/s] 35%|███▍      | 138219/400000 [00:16<00:30, 8564.04it/s] 35%|███▍      | 139076/400000 [00:16<00:30, 8452.35it/s] 35%|███▍      | 139931/400000 [00:16<00:30, 8481.13it/s] 35%|███▌      | 140780/400000 [00:16<00:31, 8350.62it/s] 35%|███▌      | 141643/400000 [00:16<00:30, 8429.99it/s] 36%|███▌      | 142492/400000 [00:16<00:30, 8447.04it/s] 36%|███▌      | 143343/400000 [00:16<00:30, 8463.52it/s] 36%|███▌      | 144197/400000 [00:17<00:30, 8484.16it/s] 36%|███▋      | 145056/400000 [00:17<00:29, 8514.53it/s] 36%|███▋      | 145917/400000 [00:17<00:29, 8540.23it/s] 37%|███▋      | 146773/400000 [00:17<00:29, 8545.46it/s] 37%|███▋      | 147628/400000 [00:17<00:30, 8310.30it/s] 37%|███▋      | 148477/400000 [00:17<00:30, 8362.60it/s] 37%|███▋      | 149315/400000 [00:17<00:29, 8360.96it/s] 38%|███▊      | 150168/400000 [00:17<00:29, 8410.87it/s] 38%|███▊      | 151026/400000 [00:17<00:29, 8458.01it/s] 38%|███▊      | 151890/400000 [00:18<00:29, 8509.19it/s] 38%|███▊      | 152745/400000 [00:18<00:29, 8519.77it/s] 38%|███▊      | 153599/400000 [00:18<00:28, 8525.17it/s] 39%|███▊      | 154452/400000 [00:18<00:29, 8202.08it/s] 39%|███▉      | 155297/400000 [00:18<00:29, 8273.47it/s] 39%|███▉      | 156157/400000 [00:18<00:29, 8368.65it/s] 39%|███▉      | 157007/400000 [00:18<00:28, 8406.51it/s] 39%|███▉      | 157849/400000 [00:18<00:29, 8279.71it/s] 40%|███▉      | 158679/400000 [00:18<00:29, 8267.40it/s] 40%|███▉      | 159522/400000 [00:18<00:28, 8314.44it/s] 40%|████      | 160385/400000 [00:19<00:28, 8406.61it/s] 40%|████      | 161235/400000 [00:19<00:28, 8432.00it/s] 41%|████      | 162091/400000 [00:19<00:28, 8469.99it/s] 41%|████      | 162943/400000 [00:19<00:27, 8482.82it/s] 41%|████      | 163805/400000 [00:19<00:27, 8522.64it/s] 41%|████      | 164658/400000 [00:19<00:28, 8404.26it/s] 41%|████▏     | 165499/400000 [00:19<00:27, 8378.92it/s] 42%|████▏     | 166352/400000 [00:19<00:27, 8422.42it/s] 42%|████▏     | 167195/400000 [00:19<00:27, 8339.78it/s] 42%|████▏     | 168037/400000 [00:19<00:27, 8363.32it/s] 42%|████▏     | 168894/400000 [00:20<00:27, 8423.04it/s] 42%|████▏     | 169741/400000 [00:20<00:27, 8435.21it/s] 43%|████▎     | 170585/400000 [00:20<00:27, 8340.92it/s] 43%|████▎     | 171442/400000 [00:20<00:27, 8408.33it/s] 43%|████▎     | 172303/400000 [00:20<00:26, 8467.49it/s] 43%|████▎     | 173168/400000 [00:20<00:26, 8519.62it/s] 44%|████▎     | 174021/400000 [00:20<00:27, 8332.94it/s] 44%|████▎     | 174871/400000 [00:20<00:26, 8380.73it/s] 44%|████▍     | 175730/400000 [00:20<00:26, 8441.52it/s] 44%|████▍     | 176575/400000 [00:20<00:26, 8345.96it/s] 44%|████▍     | 177411/400000 [00:21<00:26, 8257.70it/s] 45%|████▍     | 178257/400000 [00:21<00:26, 8315.14it/s] 45%|████▍     | 179114/400000 [00:21<00:26, 8388.52it/s] 45%|████▍     | 179975/400000 [00:21<00:26, 8452.30it/s] 45%|████▌     | 180832/400000 [00:21<00:25, 8486.87it/s] 45%|████▌     | 181682/400000 [00:21<00:25, 8403.09it/s] 46%|████▌     | 182523/400000 [00:21<00:26, 8236.01it/s] 46%|████▌     | 183377/400000 [00:21<00:26, 8324.39it/s] 46%|████▌     | 184227/400000 [00:21<00:25, 8373.52it/s] 46%|████▋     | 185076/400000 [00:21<00:25, 8406.49it/s] 46%|████▋     | 185934/400000 [00:22<00:25, 8455.60it/s] 47%|████▋     | 186791/400000 [00:22<00:25, 8487.59it/s] 47%|████▋     | 187649/400000 [00:22<00:24, 8513.45it/s] 47%|████▋     | 188513/400000 [00:22<00:24, 8549.66it/s] 47%|████▋     | 189369/400000 [00:22<00:24, 8547.32it/s] 48%|████▊     | 190224/400000 [00:22<00:24, 8521.10it/s] 48%|████▊     | 191077/400000 [00:22<00:24, 8486.53it/s] 48%|████▊     | 191926/400000 [00:22<00:24, 8345.14it/s] 48%|████▊     | 192762/400000 [00:22<00:25, 8272.11it/s] 48%|████▊     | 193621/400000 [00:22<00:24, 8362.94it/s] 49%|████▊     | 194479/400000 [00:23<00:24, 8426.58it/s] 49%|████▉     | 195332/400000 [00:23<00:24, 8455.15it/s] 49%|████▉     | 196178/400000 [00:23<00:24, 8406.15it/s] 49%|████▉     | 197035/400000 [00:23<00:24, 8452.33it/s] 49%|████▉     | 197892/400000 [00:23<00:23, 8485.62it/s] 50%|████▉     | 198741/400000 [00:23<00:23, 8434.04it/s] 50%|████▉     | 199589/400000 [00:23<00:23, 8447.44it/s] 50%|█████     | 200434/400000 [00:23<00:23, 8379.18it/s] 50%|█████     | 201288/400000 [00:23<00:23, 8424.24it/s] 51%|█████     | 202138/400000 [00:23<00:23, 8446.77it/s] 51%|█████     | 202998/400000 [00:24<00:23, 8491.93it/s] 51%|█████     | 203848/400000 [00:24<00:23, 8429.26it/s] 51%|█████     | 204701/400000 [00:24<00:23, 8458.84it/s] 51%|█████▏    | 205554/400000 [00:24<00:22, 8478.25it/s] 52%|█████▏    | 206406/400000 [00:24<00:22, 8489.36it/s] 52%|█████▏    | 207269/400000 [00:24<00:22, 8528.77it/s] 52%|█████▏    | 208122/400000 [00:24<00:22, 8510.92it/s] 52%|█████▏    | 208974/400000 [00:24<00:22, 8454.19it/s] 52%|█████▏    | 209834/400000 [00:24<00:22, 8494.59it/s] 53%|█████▎    | 210694/400000 [00:24<00:22, 8523.33it/s] 53%|█████▎    | 211553/400000 [00:25<00:22, 8541.11it/s] 53%|█████▎    | 212408/400000 [00:25<00:21, 8535.27it/s] 53%|█████▎    | 213264/400000 [00:25<00:21, 8542.63it/s] 54%|█████▎    | 214124/400000 [00:25<00:21, 8557.59it/s] 54%|█████▎    | 214980/400000 [00:25<00:21, 8471.84it/s] 54%|█████▍    | 215828/400000 [00:25<00:21, 8474.29it/s] 54%|█████▍    | 216676/400000 [00:25<00:21, 8402.60it/s] 54%|█████▍    | 217517/400000 [00:25<00:21, 8356.02it/s] 55%|█████▍    | 218353/400000 [00:25<00:22, 8149.07it/s] 55%|█████▍    | 219217/400000 [00:26<00:21, 8288.28it/s] 55%|█████▌    | 220079/400000 [00:26<00:21, 8384.50it/s] 55%|█████▌    | 220933/400000 [00:26<00:21, 8428.51it/s] 55%|█████▌    | 221788/400000 [00:26<00:21, 8463.75it/s] 56%|█████▌    | 222646/400000 [00:26<00:20, 8496.73it/s] 56%|█████▌    | 223510/400000 [00:26<00:20, 8536.49it/s] 56%|█████▌    | 224369/400000 [00:26<00:20, 8549.98it/s] 56%|█████▋    | 225227/400000 [00:26<00:20, 8557.14it/s] 57%|█████▋    | 226084/400000 [00:26<00:20, 8560.35it/s] 57%|█████▋    | 226941/400000 [00:26<00:20, 8338.18it/s] 57%|█████▋    | 227801/400000 [00:27<00:20, 8414.45it/s] 57%|█████▋    | 228657/400000 [00:27<00:20, 8455.92it/s] 57%|█████▋    | 229504/400000 [00:27<00:20, 8422.78it/s] 58%|█████▊    | 230348/400000 [00:27<00:20, 8426.02it/s] 58%|█████▊    | 231211/400000 [00:27<00:19, 8485.58it/s] 58%|█████▊    | 232068/400000 [00:27<00:19, 8510.67it/s] 58%|█████▊    | 232924/400000 [00:27<00:19, 8524.97it/s] 58%|█████▊    | 233778/400000 [00:27<00:19, 8526.77it/s] 59%|█████▊    | 234631/400000 [00:27<00:20, 8224.82it/s] 59%|█████▉    | 235487/400000 [00:27<00:19, 8321.79it/s] 59%|█████▉    | 236350/400000 [00:28<00:19, 8410.42it/s] 59%|█████▉    | 237211/400000 [00:28<00:19, 8468.59it/s] 60%|█████▉    | 238071/400000 [00:28<00:19, 8506.88it/s] 60%|█████▉    | 238923/400000 [00:28<00:19, 8476.61it/s] 60%|█████▉    | 239780/400000 [00:28<00:18, 8501.79it/s] 60%|██████    | 240637/400000 [00:28<00:18, 8521.21it/s] 60%|██████    | 241490/400000 [00:28<00:18, 8478.72it/s] 61%|██████    | 242339/400000 [00:28<00:18, 8406.35it/s] 61%|██████    | 243180/400000 [00:28<00:18, 8388.07it/s] 61%|██████    | 244020/400000 [00:28<00:18, 8347.97it/s] 61%|██████    | 244883/400000 [00:29<00:18, 8429.87it/s] 61%|██████▏   | 245742/400000 [00:29<00:18, 8475.95it/s] 62%|██████▏   | 246601/400000 [00:29<00:18, 8507.50it/s] 62%|██████▏   | 247458/400000 [00:29<00:17, 8523.58it/s] 62%|██████▏   | 248314/400000 [00:29<00:17, 8533.40it/s] 62%|██████▏   | 249173/400000 [00:29<00:17, 8549.13it/s] 63%|██████▎   | 250029/400000 [00:29<00:17, 8548.05it/s] 63%|██████▎   | 250893/400000 [00:29<00:17, 8574.15it/s] 63%|██████▎   | 251751/400000 [00:29<00:17, 8452.69it/s] 63%|██████▎   | 252597/400000 [00:29<00:17, 8365.93it/s] 63%|██████▎   | 253435/400000 [00:30<00:17, 8341.66it/s] 64%|██████▎   | 254271/400000 [00:30<00:17, 8345.11it/s] 64%|██████▍   | 255128/400000 [00:30<00:17, 8411.28it/s] 64%|██████▍   | 255973/400000 [00:30<00:17, 8420.36it/s] 64%|██████▍   | 256830/400000 [00:30<00:16, 8462.59it/s] 64%|██████▍   | 257691/400000 [00:30<00:16, 8504.77it/s] 65%|██████▍   | 258545/400000 [00:30<00:16, 8512.27it/s] 65%|██████▍   | 259402/400000 [00:30<00:16, 8529.01it/s] 65%|██████▌   | 260256/400000 [00:30<00:16, 8519.22it/s] 65%|██████▌   | 261113/400000 [00:30<00:16, 8532.17it/s] 65%|██████▌   | 261973/400000 [00:31<00:16, 8551.78it/s] 66%|██████▌   | 262829/400000 [00:31<00:16, 8462.70it/s] 66%|██████▌   | 263681/400000 [00:31<00:16, 8479.62it/s] 66%|██████▌   | 264530/400000 [00:31<00:16, 8433.70it/s] 66%|██████▋   | 265392/400000 [00:31<00:15, 8486.59it/s] 67%|██████▋   | 266249/400000 [00:31<00:15, 8511.06it/s] 67%|██████▋   | 267110/400000 [00:31<00:15, 8539.46it/s] 67%|██████▋   | 267977/400000 [00:31<00:15, 8577.40it/s] 67%|██████▋   | 268835/400000 [00:31<00:15, 8562.27it/s] 67%|██████▋   | 269698/400000 [00:31<00:15, 8582.46it/s] 68%|██████▊   | 270562/400000 [00:32<00:15, 8599.45it/s] 68%|██████▊   | 271425/400000 [00:32<00:14, 8608.39it/s] 68%|██████▊   | 272286/400000 [00:32<00:14, 8579.09it/s] 68%|██████▊   | 273144/400000 [00:32<00:15, 8376.21it/s] 68%|██████▊   | 273999/400000 [00:32<00:14, 8425.53it/s] 69%|██████▊   | 274858/400000 [00:32<00:14, 8471.52it/s] 69%|██████▉   | 275717/400000 [00:32<00:14, 8506.49it/s] 69%|██████▉   | 276570/400000 [00:32<00:14, 8511.49it/s] 69%|██████▉   | 277422/400000 [00:32<00:14, 8495.28it/s] 70%|██████▉   | 278281/400000 [00:32<00:14, 8520.73it/s] 70%|██████▉   | 279141/400000 [00:33<00:14, 8541.66it/s] 70%|███████   | 280006/400000 [00:33<00:13, 8571.58it/s] 70%|███████   | 280864/400000 [00:33<00:14, 8435.95it/s] 70%|███████   | 281709/400000 [00:33<00:14, 8437.57it/s] 71%|███████   | 282554/400000 [00:33<00:13, 8439.26it/s] 71%|███████   | 283399/400000 [00:33<00:13, 8437.32it/s] 71%|███████   | 284253/400000 [00:33<00:13, 8465.59it/s] 71%|███████▏  | 285107/400000 [00:33<00:13, 8487.60it/s] 71%|███████▏  | 285956/400000 [00:33<00:13, 8473.70it/s] 72%|███████▏  | 286816/400000 [00:33<00:13, 8511.15it/s] 72%|███████▏  | 287670/400000 [00:34<00:13, 8518.72it/s] 72%|███████▏  | 288528/400000 [00:34<00:13, 8536.57it/s] 72%|███████▏  | 289382/400000 [00:34<00:13, 8491.36it/s] 73%|███████▎  | 290234/400000 [00:34<00:12, 8498.13it/s] 73%|███████▎  | 291098/400000 [00:34<00:12, 8537.50it/s] 73%|███████▎  | 291957/400000 [00:34<00:12, 8550.76it/s] 73%|███████▎  | 292813/400000 [00:34<00:12, 8477.77it/s] 73%|███████▎  | 293670/400000 [00:34<00:12, 8504.14it/s] 74%|███████▎  | 294521/400000 [00:34<00:12, 8385.57it/s] 74%|███████▍  | 295382/400000 [00:34<00:12, 8448.99it/s] 74%|███████▍  | 296246/400000 [00:35<00:12, 8502.98it/s] 74%|███████▍  | 297105/400000 [00:35<00:12, 8527.88it/s] 74%|███████▍  | 297961/400000 [00:35<00:11, 8535.96it/s] 75%|███████▍  | 298815/400000 [00:35<00:11, 8534.48it/s] 75%|███████▍  | 299669/400000 [00:35<00:11, 8506.92it/s] 75%|███████▌  | 300520/400000 [00:35<00:11, 8483.53it/s] 75%|███████▌  | 301380/400000 [00:35<00:11, 8518.07it/s] 76%|███████▌  | 302240/400000 [00:35<00:11, 8541.27it/s] 76%|███████▌  | 303095/400000 [00:35<00:11, 8438.13it/s] 76%|███████▌  | 303954/400000 [00:35<00:11, 8483.05it/s] 76%|███████▌  | 304818/400000 [00:36<00:11, 8526.67it/s] 76%|███████▋  | 305680/400000 [00:36<00:11, 8551.99it/s] 77%|███████▋  | 306538/400000 [00:36<00:10, 8558.97it/s] 77%|███████▋  | 307395/400000 [00:36<00:10, 8538.10it/s] 77%|███████▋  | 308249/400000 [00:36<00:10, 8532.79it/s] 77%|███████▋  | 309110/400000 [00:36<00:10, 8553.29it/s] 77%|███████▋  | 309968/400000 [00:36<00:10, 8560.33it/s] 78%|███████▊  | 310825/400000 [00:36<00:10, 8537.03it/s] 78%|███████▊  | 311679/400000 [00:36<00:10, 8531.28it/s] 78%|███████▊  | 312535/400000 [00:36<00:10, 8536.92it/s] 78%|███████▊  | 313396/400000 [00:37<00:10, 8557.23it/s] 79%|███████▊  | 314252/400000 [00:37<00:10, 8430.54it/s] 79%|███████▉  | 315096/400000 [00:37<00:10, 8401.92it/s] 79%|███████▉  | 315951/400000 [00:37<00:09, 8444.19it/s] 79%|███████▉  | 316796/400000 [00:37<00:10, 8208.21it/s] 79%|███████▉  | 317619/400000 [00:37<00:10, 8149.86it/s] 80%|███████▉  | 318474/400000 [00:37<00:09, 8263.20it/s] 80%|███████▉  | 319319/400000 [00:37<00:09, 8316.16it/s] 80%|████████  | 320152/400000 [00:37<00:09, 8122.54it/s] 80%|████████  | 321008/400000 [00:38<00:09, 8248.03it/s] 80%|████████  | 321868/400000 [00:38<00:09, 8348.51it/s] 81%|████████  | 322729/400000 [00:38<00:09, 8424.21it/s] 81%|████████  | 323573/400000 [00:38<00:09, 8426.84it/s] 81%|████████  | 324424/400000 [00:38<00:08, 8450.42it/s] 81%|████████▏ | 325278/400000 [00:38<00:08, 8475.35it/s] 82%|████████▏ | 326140/400000 [00:38<00:08, 8516.45it/s] 82%|████████▏ | 326998/400000 [00:38<00:08, 8534.90it/s] 82%|████████▏ | 327852/400000 [00:38<00:08, 8532.97it/s] 82%|████████▏ | 328706/400000 [00:38<00:08, 8523.09it/s] 82%|████████▏ | 329565/400000 [00:39<00:08, 8540.87it/s] 83%|████████▎ | 330420/400000 [00:39<00:08, 8514.28it/s] 83%|████████▎ | 331280/400000 [00:39<00:08, 8538.64it/s] 83%|████████▎ | 332134/400000 [00:39<00:07, 8537.65it/s] 83%|████████▎ | 332988/400000 [00:39<00:07, 8384.58it/s] 83%|████████▎ | 333840/400000 [00:39<00:07, 8423.96it/s] 84%|████████▎ | 334704/400000 [00:39<00:07, 8486.65it/s] 84%|████████▍ | 335563/400000 [00:39<00:07, 8517.31it/s] 84%|████████▍ | 336416/400000 [00:39<00:07, 8500.02it/s] 84%|████████▍ | 337267/400000 [00:39<00:07, 8491.49it/s] 85%|████████▍ | 338117/400000 [00:40<00:07, 8436.04it/s] 85%|████████▍ | 338961/400000 [00:40<00:07, 8370.03it/s] 85%|████████▍ | 339802/400000 [00:40<00:07, 8380.46it/s] 85%|████████▌ | 340660/400000 [00:40<00:07, 8438.59it/s] 85%|████████▌ | 341510/400000 [00:40<00:06, 8456.41it/s] 86%|████████▌ | 342363/400000 [00:40<00:06, 8476.19it/s] 86%|████████▌ | 343211/400000 [00:40<00:06, 8318.87it/s] 86%|████████▌ | 344070/400000 [00:40<00:06, 8396.08it/s] 86%|████████▌ | 344921/400000 [00:40<00:06, 8428.96it/s] 86%|████████▋ | 345772/400000 [00:40<00:06, 8450.24it/s] 87%|████████▋ | 346618/400000 [00:41<00:06, 8428.40it/s] 87%|████████▋ | 347475/400000 [00:41<00:06, 8468.30it/s] 87%|████████▋ | 348336/400000 [00:41<00:06, 8509.79it/s] 87%|████████▋ | 349189/400000 [00:41<00:05, 8514.54it/s] 88%|████████▊ | 350049/400000 [00:41<00:05, 8539.35it/s] 88%|████████▊ | 350904/400000 [00:41<00:05, 8350.00it/s] 88%|████████▊ | 351763/400000 [00:41<00:05, 8414.78it/s] 88%|████████▊ | 352606/400000 [00:41<00:05, 8318.42it/s] 88%|████████▊ | 353451/400000 [00:41<00:05, 8356.94it/s] 89%|████████▊ | 354310/400000 [00:41<00:05, 8423.60it/s] 89%|████████▉ | 355166/400000 [00:42<00:05, 8461.87it/s] 89%|████████▉ | 356025/400000 [00:42<00:05, 8498.54it/s] 89%|████████▉ | 356876/400000 [00:42<00:05, 8499.09it/s] 89%|████████▉ | 357735/400000 [00:42<00:04, 8525.99it/s] 90%|████████▉ | 358588/400000 [00:42<00:04, 8477.14it/s] 90%|████████▉ | 359436/400000 [00:42<00:04, 8471.51it/s] 90%|█████████ | 360293/400000 [00:42<00:04, 8499.31it/s] 90%|█████████ | 361144/400000 [00:42<00:04, 8377.08it/s] 90%|█████████ | 361995/400000 [00:42<00:04, 8415.11it/s] 91%|█████████ | 362854/400000 [00:42<00:04, 8466.16it/s] 91%|█████████ | 363701/400000 [00:43<00:04, 8433.45it/s] 91%|█████████ | 364562/400000 [00:43<00:04, 8484.87it/s] 91%|█████████▏| 365422/400000 [00:43<00:04, 8518.24it/s] 92%|█████████▏| 366275/400000 [00:43<00:03, 8498.73it/s] 92%|█████████▏| 367126/400000 [00:43<00:03, 8458.58it/s] 92%|█████████▏| 367973/400000 [00:43<00:03, 8434.52it/s] 92%|█████████▏| 368822/400000 [00:43<00:03, 8449.36it/s] 92%|█████████▏| 369681/400000 [00:43<00:03, 8490.16it/s] 93%|█████████▎| 370531/400000 [00:43<00:03, 8491.67it/s] 93%|█████████▎| 371389/400000 [00:43<00:03, 8515.08it/s] 93%|█████████▎| 372241/400000 [00:44<00:03, 8505.81it/s] 93%|█████████▎| 373103/400000 [00:44<00:03, 8538.18it/s] 93%|█████████▎| 373969/400000 [00:44<00:03, 8571.78it/s] 94%|█████████▎| 374829/400000 [00:44<00:02, 8578.64it/s] 94%|█████████▍| 375687/400000 [00:44<00:02, 8424.42it/s] 94%|█████████▍| 376531/400000 [00:44<00:02, 8402.83it/s] 94%|█████████▍| 377372/400000 [00:44<00:02, 8324.94it/s] 95%|█████████▍| 378228/400000 [00:44<00:02, 8391.93it/s] 95%|█████████▍| 379084/400000 [00:44<00:02, 8439.05it/s] 95%|█████████▍| 379940/400000 [00:44<00:02, 8472.66it/s] 95%|█████████▌| 380788/400000 [00:45<00:02, 8474.49it/s] 95%|█████████▌| 381643/400000 [00:45<00:02, 8496.13it/s] 96%|█████████▌| 382502/400000 [00:45<00:02, 8523.05it/s] 96%|█████████▌| 383358/400000 [00:45<00:01, 8532.14it/s] 96%|█████████▌| 384212/400000 [00:45<00:01, 8517.06it/s] 96%|█████████▋| 385064/400000 [00:45<00:01, 8510.97it/s] 96%|█████████▋| 385916/400000 [00:45<00:01, 8485.12it/s] 97%|█████████▋| 386778/400000 [00:45<00:01, 8522.49it/s] 97%|█████████▋| 387634/400000 [00:45<00:01, 8531.88it/s] 97%|█████████▋| 388488/400000 [00:45<00:01, 8534.02it/s] 97%|█████████▋| 389342/400000 [00:46<00:01, 8496.49it/s] 98%|█████████▊| 390192/400000 [00:46<00:01, 8439.45it/s] 98%|█████████▊| 391050/400000 [00:46<00:01, 8478.59it/s] 98%|█████████▊| 391898/400000 [00:46<00:00, 8437.25it/s] 98%|█████████▊| 392742/400000 [00:46<00:00, 8316.78it/s] 98%|█████████▊| 393593/400000 [00:46<00:00, 8372.58it/s] 99%|█████████▊| 394431/400000 [00:46<00:00, 8322.25it/s] 99%|█████████▉| 395291/400000 [00:46<00:00, 8400.47it/s] 99%|█████████▉| 396132/400000 [00:46<00:00, 8381.64it/s] 99%|█████████▉| 396992/400000 [00:46<00:00, 8445.50it/s] 99%|█████████▉| 397837/400000 [00:47<00:00, 8335.29it/s]100%|█████████▉| 398697/400000 [00:47<00:00, 8412.03it/s]100%|█████████▉| 399555/400000 [00:47<00:00, 8460.64it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8446.93it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fdce878ac88> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011225400905581844 	 Accuracy: 55
Train Epoch: 1 	 Loss: 0.011391976206597676 	 Accuracy: 51

  model saves at 51% accuracy 

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
2020-05-14 14:24:16.436390: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 14:24:16.440869: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-14 14:24:16.441032: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55880aef4be0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 14:24:16.441047: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fdcf4306f28> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7586 - accuracy: 0.4940
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6666 - accuracy: 0.5000 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6104 - accuracy: 0.5037
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6935 - accuracy: 0.4983
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6482 - accuracy: 0.5012
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5848 - accuracy: 0.5053
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6141 - accuracy: 0.5034
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5689 - accuracy: 0.5064
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6513 - accuracy: 0.5010
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6789 - accuracy: 0.4992
11000/25000 [============>.................] - ETA: 3s - loss: 7.6485 - accuracy: 0.5012
12000/25000 [=============>................] - ETA: 3s - loss: 7.6411 - accuracy: 0.5017
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6336 - accuracy: 0.5022
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6524 - accuracy: 0.5009
15000/25000 [=================>............] - ETA: 2s - loss: 7.6554 - accuracy: 0.5007
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6446 - accuracy: 0.5014
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6558 - accuracy: 0.5007
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6308 - accuracy: 0.5023
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6392 - accuracy: 0.5018
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6452 - accuracy: 0.5014
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6506 - accuracy: 0.5010
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6624 - accuracy: 0.5003
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6740 - accuracy: 0.4995
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6800 - accuracy: 0.4991
25000/25000 [==============================] - 7s 286us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fdc4967a668> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fdc49649d68> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.9145 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.8488 - val_crf_viterbi_accuracy: 0.0133

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

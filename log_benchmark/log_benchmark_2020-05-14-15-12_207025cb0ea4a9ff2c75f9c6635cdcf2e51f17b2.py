
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f5811cc5fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 15:12:55.567121
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 15:12:55.570551
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 15:12:55.573775
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 15:12:55.576670
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f581dcdd470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352542.5000
Epoch 2/10

1/1 [==============================] - 0s 102ms/step - loss: 203432.4375
Epoch 3/10

1/1 [==============================] - 0s 87ms/step - loss: 97387.6719
Epoch 4/10

1/1 [==============================] - 0s 90ms/step - loss: 43606.8438
Epoch 5/10

1/1 [==============================] - 0s 86ms/step - loss: 22027.7090
Epoch 6/10

1/1 [==============================] - 0s 89ms/step - loss: 13045.8896
Epoch 7/10

1/1 [==============================] - 0s 86ms/step - loss: 8258.7051
Epoch 8/10

1/1 [==============================] - 0s 89ms/step - loss: 6102.3408
Epoch 9/10

1/1 [==============================] - 0s 89ms/step - loss: 4513.4141
Epoch 10/10

1/1 [==============================] - 0s 89ms/step - loss: 3578.4651

  #### Inference Need return ypred, ytrue ######################### 
[[  0.1504947    1.9513637    0.09863204  -0.29211706   1.1532665
    0.1500724   -0.51466674   0.6822753    1.579932     0.18976909
    1.3902303   -0.56740344   3.1410427    1.1448196    0.540055
    1.3549557    1.7665641   -0.09218898   0.7709905    1.7401581
   -2.1584148    2.3785808   -1.5132024   -1.9779755    1.3481145
   -1.9438146   -1.8206084   -1.5999913    1.8324862   -1.4572396
    0.7915473    1.3113493    1.180577     1.2733192    2.1286669
    1.5013059   -0.040295     2.3830895   -1.2151527   -2.9653344
    2.1420934    1.9270833    0.84042174  -1.1269648   -1.1157721
    1.4792132   -1.2032683   -2.0725648   -0.43714863  -0.48113
    1.3512836    1.1500092    0.9959706   -1.0806887   -0.35218528
   -0.36214998  -1.3349317    1.4775407    1.1862769    0.60418844
    0.23781532   1.4826589    0.68874794   0.48264253  -0.08690542
    0.70317346  -0.42365485  -3.7288642   -1.4226997   -0.0778349
   -0.8427342   -1.0732937    2.6000807   -0.9612381    1.8683286
    0.6578828    0.9731867    0.3207829    2.4647608    1.262481
   -0.12366322  -1.4442694   -1.0860423    1.833507    -1.2763741
    0.62581336   0.57752424   1.7272433    1.1546825    1.8907619
   -0.02300227   0.34763026  -4.2414546   -2.200739     0.4225339
   -0.04106911  -1.1404532   -1.0381699   -1.1302737    2.138943
    0.6231301   -0.30647492  -0.26129693  -2.525107    -0.56944096
    0.5189297   -2.3533573   -1.5344847   -1.2574304   -2.0091734
   -1.57709     -0.685555    -2.6971703   -0.15107548  -1.9048226
   -0.02012426  -0.81215304   2.9922519    0.52132493   1.5485215
    0.07630881  12.994188    12.518211    13.123352    14.213162
   14.052925    12.313634    10.346816    11.39497     11.239344
   11.969701    12.0935335   12.119351    12.3454485   14.502012
   11.767446    14.641234    11.698956    11.381186    13.526351
   13.942731     8.994418    13.6970415   11.42278     13.370754
   12.999816    13.550191    13.385477    10.895926    12.605441
   11.895684    15.194407    15.061448    10.747045    13.712541
   12.406002    13.91793     10.327123    15.332522    12.651438
   10.66456     11.442662    12.2622795   13.741651    10.933711
   11.128542    13.643151     9.527861    11.196806    13.29841
    9.648489    14.292339    11.924202    11.568948    14.116498
   11.556104    13.314759    12.862649    12.53392     10.973468
    2.7825098    2.002407     0.48425525   1.8462911    1.6719683
    1.6497214    1.8535745    0.37785923   1.0376729    0.63316
    0.6585866    0.05081505   0.5260254    0.04202712   0.5845861
    1.9800502    3.0020614    0.48203743   1.2963372    0.52137816
    0.25752735   1.4814042    0.07152069   3.1934395    1.7776957
    0.06867129   1.5599209    1.13076      1.849576     0.93512046
    1.3453255    1.3307027    0.5747906    2.0063257    0.1661697
    0.43564004   1.964151     0.26230687   2.6856294    1.4440322
    2.3773108    1.9367735    0.29822463   2.0564709    0.3154614
    1.1874082    0.9901016    0.08475357   0.8862288    0.23126924
    2.3774858    0.49888158   2.1212382    2.9121943    0.42451477
    1.5565051    0.75676876   2.2671847    2.2273846    0.8472451
    1.8804269    0.9594038    0.41034085   0.780372     1.1960387
    0.24124557   0.12455767   0.36720777   0.7478303    0.39671874
    0.35563332   0.44610524   1.0395584    1.6260296    2.3034244
    1.4541074    0.64452714   0.8955625    1.1407641    0.19545609
    1.240993     1.1157739    0.35333532   3.1748137    1.6823905
    1.9395694    0.8993217    3.873798     0.41424274   3.2734222
    0.08874089   1.7107465    1.1418179    0.26969516   0.20882905
    2.09703      1.2597908    0.3140309    2.7999148    1.1598929
    0.8995645    1.8308407    2.5204606    0.18326223   0.3327394
    0.27968687   0.7405356    0.7725166    0.08724618   0.8872669
    0.04395652   1.4323848    0.20315051   1.0018848    0.8016821
    2.1099844    1.1072156    0.31892455   2.2376678    2.4561176
    0.18130672  10.096003    10.702936    13.175578    12.122427
   15.133232    14.220671     9.201598    10.792035    13.64894
   13.565777    11.937896     9.257536    10.223786    13.544906
   12.987762    11.327945    10.05504      8.513339    12.047937
   14.116639    11.987332    14.081297    10.225662    10.756417
   12.664331    13.251224    12.694872    11.361001    12.743839
   12.783989    13.886641    12.251231    14.072402    12.4896145
    9.087194    10.776956    10.356849    13.074578    13.602006
   12.426514    10.269253    14.177303     8.820091    12.992487
    8.470475    13.19338     10.538874    12.299769    14.33041
   11.879405    13.015062    12.1732645   11.007048    12.398665
    9.13222      9.174944     9.484187    13.643689    11.165945
   -8.801497   -14.0488825   15.511656  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 15:13:03.781393
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   90.1407
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 15:13:03.785009
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8158.42
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 15:13:03.788063
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   90.3239
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 15:13:03.791083
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -729.645
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140015892951560
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140014934311488
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140014934311992
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140014934312496
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140014934313000
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140014934313504

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f5811c9ccc0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.505426
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.479804
grad_step = 000002, loss = 0.458693
grad_step = 000003, loss = 0.436128
grad_step = 000004, loss = 0.411959
grad_step = 000005, loss = 0.390202
grad_step = 000006, loss = 0.376937
grad_step = 000007, loss = 0.368132
grad_step = 000008, loss = 0.356707
grad_step = 000009, loss = 0.341878
grad_step = 000010, loss = 0.329472
grad_step = 000011, loss = 0.320075
grad_step = 000012, loss = 0.311912
grad_step = 000013, loss = 0.303302
grad_step = 000014, loss = 0.293634
grad_step = 000015, loss = 0.283226
grad_step = 000016, loss = 0.272743
grad_step = 000017, loss = 0.262709
grad_step = 000018, loss = 0.253333
grad_step = 000019, loss = 0.244220
grad_step = 000020, loss = 0.234946
grad_step = 000021, loss = 0.225576
grad_step = 000022, loss = 0.216587
grad_step = 000023, loss = 0.208315
grad_step = 000024, loss = 0.200422
grad_step = 000025, loss = 0.192026
grad_step = 000026, loss = 0.183266
grad_step = 000027, loss = 0.174444
grad_step = 000028, loss = 0.165908
grad_step = 000029, loss = 0.157939
grad_step = 000030, loss = 0.150440
grad_step = 000031, loss = 0.143273
grad_step = 000032, loss = 0.136389
grad_step = 000033, loss = 0.129889
grad_step = 000034, loss = 0.123699
grad_step = 000035, loss = 0.117480
grad_step = 000036, loss = 0.111375
grad_step = 000037, loss = 0.105678
grad_step = 000038, loss = 0.100303
grad_step = 000039, loss = 0.095031
grad_step = 000040, loss = 0.089841
grad_step = 000041, loss = 0.084735
grad_step = 000042, loss = 0.079933
grad_step = 000043, loss = 0.075547
grad_step = 000044, loss = 0.071300
grad_step = 000045, loss = 0.067112
grad_step = 000046, loss = 0.063151
grad_step = 000047, loss = 0.059508
grad_step = 000048, loss = 0.056030
grad_step = 000049, loss = 0.052573
grad_step = 000050, loss = 0.049263
grad_step = 000051, loss = 0.046174
grad_step = 000052, loss = 0.043255
grad_step = 000053, loss = 0.040405
grad_step = 000054, loss = 0.037739
grad_step = 000055, loss = 0.035261
grad_step = 000056, loss = 0.032900
grad_step = 000057, loss = 0.030642
grad_step = 000058, loss = 0.028543
grad_step = 000059, loss = 0.026556
grad_step = 000060, loss = 0.024679
grad_step = 000061, loss = 0.022944
grad_step = 000062, loss = 0.021310
grad_step = 000063, loss = 0.019746
grad_step = 000064, loss = 0.018322
grad_step = 000065, loss = 0.017013
grad_step = 000066, loss = 0.015782
grad_step = 000067, loss = 0.014667
grad_step = 000068, loss = 0.013660
grad_step = 000069, loss = 0.012709
grad_step = 000070, loss = 0.011834
grad_step = 000071, loss = 0.011042
grad_step = 000072, loss = 0.010295
grad_step = 000073, loss = 0.009610
grad_step = 000074, loss = 0.008989
grad_step = 000075, loss = 0.008414
grad_step = 000076, loss = 0.007888
grad_step = 000077, loss = 0.007399
grad_step = 000078, loss = 0.006943
grad_step = 000079, loss = 0.006528
grad_step = 000080, loss = 0.006141
grad_step = 000081, loss = 0.005782
grad_step = 000082, loss = 0.005456
grad_step = 000083, loss = 0.005153
grad_step = 000084, loss = 0.004873
grad_step = 000085, loss = 0.004620
grad_step = 000086, loss = 0.004384
grad_step = 000087, loss = 0.004164
grad_step = 000088, loss = 0.003964
grad_step = 000089, loss = 0.003778
grad_step = 000090, loss = 0.003612
grad_step = 000091, loss = 0.003470
grad_step = 000092, loss = 0.003362
grad_step = 000093, loss = 0.003302
grad_step = 000094, loss = 0.003227
grad_step = 000095, loss = 0.003091
grad_step = 000096, loss = 0.002881
grad_step = 000097, loss = 0.002728
grad_step = 000098, loss = 0.002699
grad_step = 000099, loss = 0.002685
grad_step = 000100, loss = 0.002572
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002439
grad_step = 000102, loss = 0.002395
grad_step = 000103, loss = 0.002396
grad_step = 000104, loss = 0.002348
grad_step = 000105, loss = 0.002261
grad_step = 000106, loss = 0.002210
grad_step = 000107, loss = 0.002214
grad_step = 000108, loss = 0.002206
grad_step = 000109, loss = 0.002151
grad_step = 000110, loss = 0.002105
grad_step = 000111, loss = 0.002101
grad_step = 000112, loss = 0.002106
grad_step = 000113, loss = 0.002087
grad_step = 000114, loss = 0.002052
grad_step = 000115, loss = 0.002033
grad_step = 000116, loss = 0.002038
grad_step = 000117, loss = 0.002043
grad_step = 000118, loss = 0.002030
grad_step = 000119, loss = 0.002007
grad_step = 000120, loss = 0.001994
grad_step = 000121, loss = 0.001993
grad_step = 000122, loss = 0.001998
grad_step = 000123, loss = 0.001999
grad_step = 000124, loss = 0.001992
grad_step = 000125, loss = 0.001979
grad_step = 000126, loss = 0.001965
grad_step = 000127, loss = 0.001957
grad_step = 000128, loss = 0.001954
grad_step = 000129, loss = 0.001954
grad_step = 000130, loss = 0.001957
grad_step = 000131, loss = 0.001961
grad_step = 000132, loss = 0.001968
grad_step = 000133, loss = 0.001976
grad_step = 000134, loss = 0.001990
grad_step = 000135, loss = 0.001999
grad_step = 000136, loss = 0.002008
grad_step = 000137, loss = 0.001996
grad_step = 000138, loss = 0.001974
grad_step = 000139, loss = 0.001939
grad_step = 000140, loss = 0.001911
grad_step = 000141, loss = 0.001897
grad_step = 000142, loss = 0.001899
grad_step = 000143, loss = 0.001913
grad_step = 000144, loss = 0.001931
grad_step = 000145, loss = 0.001951
grad_step = 000146, loss = 0.001968
grad_step = 000147, loss = 0.001981
grad_step = 000148, loss = 0.001968
grad_step = 000149, loss = 0.001939
grad_step = 000150, loss = 0.001896
grad_step = 000151, loss = 0.001865
grad_step = 000152, loss = 0.001854
grad_step = 000153, loss = 0.001859
grad_step = 000154, loss = 0.001875
grad_step = 000155, loss = 0.001890
grad_step = 000156, loss = 0.001903
grad_step = 000157, loss = 0.001903
grad_step = 000158, loss = 0.001892
grad_step = 000159, loss = 0.001868
grad_step = 000160, loss = 0.001844
grad_step = 000161, loss = 0.001824
grad_step = 000162, loss = 0.001813
grad_step = 000163, loss = 0.001809
grad_step = 000164, loss = 0.001810
grad_step = 000165, loss = 0.001815
grad_step = 000166, loss = 0.001824
grad_step = 000167, loss = 0.001840
grad_step = 000168, loss = 0.001865
grad_step = 000169, loss = 0.001904
grad_step = 000170, loss = 0.001928
grad_step = 000171, loss = 0.001934
grad_step = 000172, loss = 0.001886
grad_step = 000173, loss = 0.001825
grad_step = 000174, loss = 0.001788
grad_step = 000175, loss = 0.001785
grad_step = 000176, loss = 0.001793
grad_step = 000177, loss = 0.001790
grad_step = 000178, loss = 0.001783
grad_step = 000179, loss = 0.001780
grad_step = 000180, loss = 0.001781
grad_step = 000181, loss = 0.001779
grad_step = 000182, loss = 0.001761
grad_step = 000183, loss = 0.001740
grad_step = 000184, loss = 0.001728
grad_step = 000185, loss = 0.001730
grad_step = 000186, loss = 0.001741
grad_step = 000187, loss = 0.001749
grad_step = 000188, loss = 0.001748
grad_step = 000189, loss = 0.001734
grad_step = 000190, loss = 0.001717
grad_step = 000191, loss = 0.001702
grad_step = 000192, loss = 0.001696
grad_step = 000193, loss = 0.001704
grad_step = 000194, loss = 0.001732
grad_step = 000195, loss = 0.001768
grad_step = 000196, loss = 0.001811
grad_step = 000197, loss = 0.001850
grad_step = 000198, loss = 0.001841
grad_step = 000199, loss = 0.001795
grad_step = 000200, loss = 0.001722
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001714
grad_step = 000202, loss = 0.001742
grad_step = 000203, loss = 0.001751
grad_step = 000204, loss = 0.001742
grad_step = 000205, loss = 0.001709
grad_step = 000206, loss = 0.001645
grad_step = 000207, loss = 0.001673
grad_step = 000208, loss = 0.001725
grad_step = 000209, loss = 0.001680
grad_step = 000210, loss = 0.001706
grad_step = 000211, loss = 0.001695
grad_step = 000212, loss = 0.001655
grad_step = 000213, loss = 0.001742
grad_step = 000214, loss = 0.001723
grad_step = 000215, loss = 0.001661
grad_step = 000216, loss = 0.001730
grad_step = 000217, loss = 0.001700
grad_step = 000218, loss = 0.001647
grad_step = 000219, loss = 0.001726
grad_step = 000220, loss = 0.001666
grad_step = 000221, loss = 0.001672
grad_step = 000222, loss = 0.001691
grad_step = 000223, loss = 0.001624
grad_step = 000224, loss = 0.001665
grad_step = 000225, loss = 0.001615
grad_step = 000226, loss = 0.001626
grad_step = 000227, loss = 0.001664
grad_step = 000228, loss = 0.001619
grad_step = 000229, loss = 0.001645
grad_step = 000230, loss = 0.001657
grad_step = 000231, loss = 0.001614
grad_step = 000232, loss = 0.001643
grad_step = 000233, loss = 0.001650
grad_step = 000234, loss = 0.001632
grad_step = 000235, loss = 0.001695
grad_step = 000236, loss = 0.001705
grad_step = 000237, loss = 0.001795
grad_step = 000238, loss = 0.001889
grad_step = 000239, loss = 0.001930
grad_step = 000240, loss = 0.001871
grad_step = 000241, loss = 0.001734
grad_step = 000242, loss = 0.001639
grad_step = 000243, loss = 0.001669
grad_step = 000244, loss = 0.001701
grad_step = 000245, loss = 0.001697
grad_step = 000246, loss = 0.001655
grad_step = 000247, loss = 0.001612
grad_step = 000248, loss = 0.001632
grad_step = 000249, loss = 0.001673
grad_step = 000250, loss = 0.001656
grad_step = 000251, loss = 0.001607
grad_step = 000252, loss = 0.001573
grad_step = 000253, loss = 0.001591
grad_step = 000254, loss = 0.001629
grad_step = 000255, loss = 0.001637
grad_step = 000256, loss = 0.001603
grad_step = 000257, loss = 0.001584
grad_step = 000258, loss = 0.001587
grad_step = 000259, loss = 0.001597
grad_step = 000260, loss = 0.001595
grad_step = 000261, loss = 0.001576
grad_step = 000262, loss = 0.001565
grad_step = 000263, loss = 0.001573
grad_step = 000264, loss = 0.001585
grad_step = 000265, loss = 0.001586
grad_step = 000266, loss = 0.001577
grad_step = 000267, loss = 0.001561
grad_step = 000268, loss = 0.001557
grad_step = 000269, loss = 0.001561
grad_step = 000270, loss = 0.001564
grad_step = 000271, loss = 0.001561
grad_step = 000272, loss = 0.001555
grad_step = 000273, loss = 0.001548
grad_step = 000274, loss = 0.001548
grad_step = 000275, loss = 0.001552
grad_step = 000276, loss = 0.001554
grad_step = 000277, loss = 0.001555
grad_step = 000278, loss = 0.001554
grad_step = 000279, loss = 0.001554
grad_step = 000280, loss = 0.001560
grad_step = 000281, loss = 0.001573
grad_step = 000282, loss = 0.001596
grad_step = 000283, loss = 0.001624
grad_step = 000284, loss = 0.001668
grad_step = 000285, loss = 0.001695
grad_step = 000286, loss = 0.001712
grad_step = 000287, loss = 0.001664
grad_step = 000288, loss = 0.001601
grad_step = 000289, loss = 0.001545
grad_step = 000290, loss = 0.001538
grad_step = 000291, loss = 0.001572
grad_step = 000292, loss = 0.001603
grad_step = 000293, loss = 0.001609
grad_step = 000294, loss = 0.001577
grad_step = 000295, loss = 0.001538
grad_step = 000296, loss = 0.001525
grad_step = 000297, loss = 0.001541
grad_step = 000298, loss = 0.001562
grad_step = 000299, loss = 0.001564
grad_step = 000300, loss = 0.001550
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001529
grad_step = 000302, loss = 0.001520
grad_step = 000303, loss = 0.001526
grad_step = 000304, loss = 0.001536
grad_step = 000305, loss = 0.001542
grad_step = 000306, loss = 0.001539
grad_step = 000307, loss = 0.001529
grad_step = 000308, loss = 0.001519
grad_step = 000309, loss = 0.001511
grad_step = 000310, loss = 0.001511
grad_step = 000311, loss = 0.001515
grad_step = 000312, loss = 0.001518
grad_step = 000313, loss = 0.001518
grad_step = 000314, loss = 0.001517
grad_step = 000315, loss = 0.001513
grad_step = 000316, loss = 0.001508
grad_step = 000317, loss = 0.001505
grad_step = 000318, loss = 0.001505
grad_step = 000319, loss = 0.001506
grad_step = 000320, loss = 0.001508
grad_step = 000321, loss = 0.001511
grad_step = 000322, loss = 0.001517
grad_step = 000323, loss = 0.001525
grad_step = 000324, loss = 0.001540
grad_step = 000325, loss = 0.001562
grad_step = 000326, loss = 0.001601
grad_step = 000327, loss = 0.001650
grad_step = 000328, loss = 0.001726
grad_step = 000329, loss = 0.001777
grad_step = 000330, loss = 0.001820
grad_step = 000331, loss = 0.001740
grad_step = 000332, loss = 0.001618
grad_step = 000333, loss = 0.001527
grad_step = 000334, loss = 0.001550
grad_step = 000335, loss = 0.001653
grad_step = 000336, loss = 0.001709
grad_step = 000337, loss = 0.001669
grad_step = 000338, loss = 0.001555
grad_step = 000339, loss = 0.001512
grad_step = 000340, loss = 0.001553
grad_step = 000341, loss = 0.001587
grad_step = 000342, loss = 0.001569
grad_step = 000343, loss = 0.001521
grad_step = 000344, loss = 0.001506
grad_step = 000345, loss = 0.001538
grad_step = 000346, loss = 0.001577
grad_step = 000347, loss = 0.001574
grad_step = 000348, loss = 0.001591
grad_step = 000349, loss = 0.001547
grad_step = 000350, loss = 0.001567
grad_step = 000351, loss = 0.001587
grad_step = 000352, loss = 0.001536
grad_step = 000353, loss = 0.001481
grad_step = 000354, loss = 0.001496
grad_step = 000355, loss = 0.001523
grad_step = 000356, loss = 0.001518
grad_step = 000357, loss = 0.001516
grad_step = 000358, loss = 0.001552
grad_step = 000359, loss = 0.001679
grad_step = 000360, loss = 0.001615
grad_step = 000361, loss = 0.001585
grad_step = 000362, loss = 0.001572
grad_step = 000363, loss = 0.001568
grad_step = 000364, loss = 0.001495
grad_step = 000365, loss = 0.001478
grad_step = 000366, loss = 0.001516
grad_step = 000367, loss = 0.001513
grad_step = 000368, loss = 0.001509
grad_step = 000369, loss = 0.001514
grad_step = 000370, loss = 0.001525
grad_step = 000371, loss = 0.001485
grad_step = 000372, loss = 0.001461
grad_step = 000373, loss = 0.001457
grad_step = 000374, loss = 0.001465
grad_step = 000375, loss = 0.001469
grad_step = 000376, loss = 0.001470
grad_step = 000377, loss = 0.001482
grad_step = 000378, loss = 0.001499
grad_step = 000379, loss = 0.001510
grad_step = 000380, loss = 0.001512
grad_step = 000381, loss = 0.001506
grad_step = 000382, loss = 0.001494
grad_step = 000383, loss = 0.001481
grad_step = 000384, loss = 0.001461
grad_step = 000385, loss = 0.001446
grad_step = 000386, loss = 0.001442
grad_step = 000387, loss = 0.001443
grad_step = 000388, loss = 0.001448
grad_step = 000389, loss = 0.001451
grad_step = 000390, loss = 0.001454
grad_step = 000391, loss = 0.001454
grad_step = 000392, loss = 0.001450
grad_step = 000393, loss = 0.001444
grad_step = 000394, loss = 0.001437
grad_step = 000395, loss = 0.001432
grad_step = 000396, loss = 0.001429
grad_step = 000397, loss = 0.001427
grad_step = 000398, loss = 0.001427
grad_step = 000399, loss = 0.001428
grad_step = 000400, loss = 0.001429
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001430
grad_step = 000402, loss = 0.001431
grad_step = 000403, loss = 0.001432
grad_step = 000404, loss = 0.001432
grad_step = 000405, loss = 0.001431
grad_step = 000406, loss = 0.001429
grad_step = 000407, loss = 0.001427
grad_step = 000408, loss = 0.001426
grad_step = 000409, loss = 0.001425
grad_step = 000410, loss = 0.001423
grad_step = 000411, loss = 0.001422
grad_step = 000412, loss = 0.001422
grad_step = 000413, loss = 0.001423
grad_step = 000414, loss = 0.001424
grad_step = 000415, loss = 0.001427
grad_step = 000416, loss = 0.001433
grad_step = 000417, loss = 0.001441
grad_step = 000418, loss = 0.001454
grad_step = 000419, loss = 0.001470
grad_step = 000420, loss = 0.001483
grad_step = 000421, loss = 0.001497
grad_step = 000422, loss = 0.001508
grad_step = 000423, loss = 0.001508
grad_step = 000424, loss = 0.001508
grad_step = 000425, loss = 0.001514
grad_step = 000426, loss = 0.001536
grad_step = 000427, loss = 0.001548
grad_step = 000428, loss = 0.001555
grad_step = 000429, loss = 0.001536
grad_step = 000430, loss = 0.001529
grad_step = 000431, loss = 0.001481
grad_step = 000432, loss = 0.001447
grad_step = 000433, loss = 0.001407
grad_step = 000434, loss = 0.001419
grad_step = 000435, loss = 0.001439
grad_step = 000436, loss = 0.001436
grad_step = 000437, loss = 0.001451
grad_step = 000438, loss = 0.001444
grad_step = 000439, loss = 0.001411
grad_step = 000440, loss = 0.001394
grad_step = 000441, loss = 0.001405
grad_step = 000442, loss = 0.001416
grad_step = 000443, loss = 0.001425
grad_step = 000444, loss = 0.001422
grad_step = 000445, loss = 0.001443
grad_step = 000446, loss = 0.001488
grad_step = 000447, loss = 0.001528
grad_step = 000448, loss = 0.001564
grad_step = 000449, loss = 0.001556
grad_step = 000450, loss = 0.001471
grad_step = 000451, loss = 0.001390
grad_step = 000452, loss = 0.001383
grad_step = 000453, loss = 0.001424
grad_step = 000454, loss = 0.001439
grad_step = 000455, loss = 0.001422
grad_step = 000456, loss = 0.001386
grad_step = 000457, loss = 0.001370
grad_step = 000458, loss = 0.001393
grad_step = 000459, loss = 0.001413
grad_step = 000460, loss = 0.001402
grad_step = 000461, loss = 0.001386
grad_step = 000462, loss = 0.001401
grad_step = 000463, loss = 0.001445
grad_step = 000464, loss = 0.001457
grad_step = 000465, loss = 0.001476
grad_step = 000466, loss = 0.001488
grad_step = 000467, loss = 0.001538
grad_step = 000468, loss = 0.001573
grad_step = 000469, loss = 0.001569
grad_step = 000470, loss = 0.001513
grad_step = 000471, loss = 0.001456
grad_step = 000472, loss = 0.001413
grad_step = 000473, loss = 0.001427
grad_step = 000474, loss = 0.001448
grad_step = 000475, loss = 0.001420
grad_step = 000476, loss = 0.001374
grad_step = 000477, loss = 0.001351
grad_step = 000478, loss = 0.001357
grad_step = 000479, loss = 0.001381
grad_step = 000480, loss = 0.001379
grad_step = 000481, loss = 0.001355
grad_step = 000482, loss = 0.001357
grad_step = 000483, loss = 0.001375
grad_step = 000484, loss = 0.001375
grad_step = 000485, loss = 0.001360
grad_step = 000486, loss = 0.001358
grad_step = 000487, loss = 0.001366
grad_step = 000488, loss = 0.001361
grad_step = 000489, loss = 0.001344
grad_step = 000490, loss = 0.001336
grad_step = 000491, loss = 0.001341
grad_step = 000492, loss = 0.001341
grad_step = 000493, loss = 0.001334
grad_step = 000494, loss = 0.001328
grad_step = 000495, loss = 0.001331
grad_step = 000496, loss = 0.001335
grad_step = 000497, loss = 0.001331
grad_step = 000498, loss = 0.001326
grad_step = 000499, loss = 0.001324
grad_step = 000500, loss = 0.001327
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001328
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

  date_run                              2020-05-14 15:13:21.372384
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.269522
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 15:13:21.378944
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.211262
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 15:13:21.385062
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.139347
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 15:13:21.390109
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -2.2102
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
0   2020-05-14 15:12:55.567121  ...    mean_absolute_error
1   2020-05-14 15:12:55.570551  ...     mean_squared_error
2   2020-05-14 15:12:55.573775  ...  median_absolute_error
3   2020-05-14 15:12:55.576670  ...               r2_score
4   2020-05-14 15:13:03.781393  ...    mean_absolute_error
5   2020-05-14 15:13:03.785009  ...     mean_squared_error
6   2020-05-14 15:13:03.788063  ...  median_absolute_error
7   2020-05-14 15:13:03.791083  ...               r2_score
8   2020-05-14 15:13:21.372384  ...    mean_absolute_error
9   2020-05-14 15:13:21.378944  ...     mean_squared_error
10  2020-05-14 15:13:21.385062  ...  median_absolute_error
11  2020-05-14 15:13:21.390109  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f20077b8898> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 315388.03it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 406405.13it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 563587.20it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 796161.96it/s] 75%|███████▌  | 7446528/9912422 [00:00<00:02, 1125278.61it/s]9920512it [00:00, 10571620.14it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 140665.40it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 318553.26it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 411815.64it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 570119.14it/s]1654784it [00:00, 2796044.96it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 50096.36it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1fba168dd8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1fb6fb4048> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1fba168dd8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1fb6fb4048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1fb6f28438> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1fb6fb4048> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1fba168dd8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1fb6fb4048> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1fb6f28438> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2007770e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fd30a15a1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=b11d4c9353cb2d6fe7ef1ccdb08c02191fb5633cd625730c9efc4216b9c059e4
  Stored in directory: /tmp/pip-ephem-wheel-cache-bp7ect17/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd3004e0048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 44s
   57344/17464789 [..............................] - ETA: 38s
   90112/17464789 [..............................] - ETA: 36s
  196608/17464789 [..............................] - ETA: 22s
  368640/17464789 [..............................] - ETA: 14s
  737280/17464789 [>.............................] - ETA: 8s 
 1466368/17464789 [=>............................] - ETA: 4s
 2932736/17464789 [====>.........................] - ETA: 2s
 5816320/17464789 [========>.....................] - ETA: 1s
 8847360/17464789 [==============>...............] - ETA: 0s
11943936/17464789 [===================>..........] - ETA: 0s
14974976/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 15:14:50.672682: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 15:14:50.676681: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-14 15:14:50.676833: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55eb6489ee30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 15:14:50.676847: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8046 - accuracy: 0.4910
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8966 - accuracy: 0.4850 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.8762 - accuracy: 0.4863
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8775 - accuracy: 0.4863
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.8598 - accuracy: 0.4874
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.8200 - accuracy: 0.4900
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7630 - accuracy: 0.4937
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6877 - accuracy: 0.4986
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6615 - accuracy: 0.5003
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6682 - accuracy: 0.4999
11000/25000 [============>.................] - ETA: 3s - loss: 7.6457 - accuracy: 0.5014
12000/25000 [=============>................] - ETA: 3s - loss: 7.6820 - accuracy: 0.4990
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6749 - accuracy: 0.4995
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6841 - accuracy: 0.4989
15000/25000 [=================>............] - ETA: 2s - loss: 7.6922 - accuracy: 0.4983
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6791 - accuracy: 0.4992
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6820 - accuracy: 0.4990
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6785 - accuracy: 0.4992
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6828 - accuracy: 0.4989
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6643 - accuracy: 0.5002
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6630 - accuracy: 0.5002
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6471 - accuracy: 0.5013
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6580 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6711 - accuracy: 0.4997
25000/25000 [==============================] - 7s 275us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 15:15:04.070420
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 15:15:04.070420  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<10:04:54, 23.8kB/s].vector_cache/glove.6B.zip:   0%|          | 147k/862M [00:00<7:06:30, 33.7kB/s]  .vector_cache/glove.6B.zip:   0%|          | 1.47M/862M [00:00<4:58:25, 48.1kB/s].vector_cache/glove.6B.zip:   1%|          | 5.62M/862M [00:00<3:27:59, 68.6kB/s].vector_cache/glove.6B.zip:   1%|          | 9.90M/862M [00:00<2:24:57, 98.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.8M/862M [00:00<1:41:07, 140kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.9M/862M [00:00<1:10:32, 199kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.7M/862M [00:01<49:02, 285kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.8M/862M [00:01<34:05, 406kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.3M/862M [00:01<23:40, 579kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.9M/862M [00:01<16:30, 823kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:01<11:40, 1.16MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:01<08:29, 1.58MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<12:18:26, 18.2kB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:03<8:37:08, 26.0kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<6:02:34, 36.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:05<4:15:11, 52.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.4M/862M [00:05<2:57:56, 74.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:07<2:31:25, 87.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:07<1:47:26, 124kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 67.1M/862M [00:07<1:15:03, 177kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:09<1:01:48, 214kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<45:01, 294kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 71.3M/862M [00:09<31:30, 418kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:11<33:27, 394kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:11<25:11, 523kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:11<17:46, 739kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:13<16:16, 805kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:13<12:13, 1.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:13<08:41, 1.50MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:15<11:27, 1.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:15<12:06, 1.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:15<09:07, 1.43MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.8M/862M [00:15<06:34, 1.97MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:17<08:53, 1.46MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:17<09:23, 1.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:17<06:47, 1.91MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:19<07:44, 1.67MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:19<10:12, 1.26MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:19<07:33, 1.71MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.3M/862M [00:19<05:24, 2.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:21<26:06, 491kB/s] .vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:21<20:30, 625kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:21<14:33, 879kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:23<13:04, 976kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:23<12:18, 1.04MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:23<08:49, 1.44MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<09:12, 1.38MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<08:18, 1.53MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:25<05:58, 2.12MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<08:00, 1.58MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<07:32, 1.67MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:27<05:26, 2.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:27<04:01, 3.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<3:13:45, 64.8kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<2:17:26, 91.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:29<1:36:04, 130kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<1:11:10, 175kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<51:36, 242kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:31<36:10, 344kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<29:27, 421kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<22:24, 554kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:33<15:47, 784kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<15:29, 797kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<12:48, 963kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<09:14, 1.33MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:35<06:42, 1.83MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<10:08, 1.21MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<08:48, 1.39MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<06:23, 1.92MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:37<04:37, 2.64MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<58:41, 208kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<44:54, 272kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<32:08, 379kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<22:47, 534kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<16:10, 751kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<13:53, 874kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:41<09:52, 1.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<10:36, 1.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<08:28, 1.42MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<06:03, 1.98MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<08:38, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<07:43, 1.55MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:34, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<07:10, 1.66MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<06:49, 1.75MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:03, 2.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<03:49, 3.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:51<09:48, 1.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:51<08:42, 1.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<06:18, 1.88MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<04:38, 2.55MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:53<10:22, 1.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<08:42, 1.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<06:28, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<04:51, 2.42MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<03:44, 3.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:55<09:41, 1.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:55<13:20, 878kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:55<12:59, 903kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:55<09:56, 1.18MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<07:30, 1.56MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<05:53, 1.98MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<04:45, 2.46MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<03:34, 3.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<14:46, 789kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:57<11:16, 1.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<07:57, 1.46MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<24:49, 467kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:59<19:01, 609kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<13:28, 858kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<12:18, 937kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:01<09:55, 1.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<07:03, 1.63MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<09:54, 1.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<08:32, 1.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<06:05, 1.87MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:04<08:56, 1.27MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:04<07:45, 1.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<05:32, 2.04MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:06<08:15, 1.37MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<07:22, 1.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:07<05:16, 2.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:08<08:33, 1.31MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<07:37, 1.48MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:26, 2.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<08:37, 1.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<07:35, 1.47MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:11<05:26, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:13<09:27, 1.18MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:13<08:20, 1.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<06:04, 1.82MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:15<06:08, 1.80MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:15<07:58, 1.38MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:15<07:18, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<05:46, 1.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<04:51, 2.27MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<04:16, 2.58MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:16<03:09, 3.47MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<13:02, 841kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<10:41, 1.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:18<07:34, 1.44MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:19<08:03, 1.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:19<06:58, 1.56MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<06:12, 1.75MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<05:16, 2.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:20<04:16, 2.54MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:20<03:33, 3.05MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:20<02:56, 3.69MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:20<02:22, 4.55MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:23<37:54, 286kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:23<28:06, 385kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<19:55, 542kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<14:12, 760kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:25<13:21, 805kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:25<16:23, 657kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:25<17:13, 625kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<16:52, 638kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<15:11, 708kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:26<11:37, 926kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:26<08:37, 1.24MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:26<06:27, 1.66MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:26<04:40, 2.29MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:27<26:30, 404kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:27<18:55, 564kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:29<14:59, 709kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:29<15:46, 673kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:29<14:26, 736kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:29<13:38, 778kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:29<11:28, 925kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:30<08:27, 1.25MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:30<06:14, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:32<07:55, 1.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:32<06:30, 1.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:32<04:39, 2.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:34<09:03, 1.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:34<07:48, 1.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:34<05:34, 1.87MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:36<07:40, 1.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:36<07:00, 1.49MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:36<05:07, 2.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:36<03:50, 2.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:36<02:58, 3.49MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:39<25:25, 407kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:39<19:19, 535kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:39<13:43, 752kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:39<09:53, 1.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:39<07:08, 1.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:41<34:13, 300kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:41<29:40, 346kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:41<25:07, 409kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:41<20:21, 505kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:41<17:23, 591kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:41<15:51, 648kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:41<14:31, 708kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:41<13:06, 784kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:41<11:37, 883kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:42<10:53, 943kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:42<08:55, 1.15MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:42<06:45, 1.52MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:42<04:59, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:43<05:16, 1.94MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:43<03:55, 2.60MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:43<02:58, 3.42MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:45<05:39, 1.79MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:45<05:29, 1.85MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:45<03:56, 2.57MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:46<07:21, 1.37MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:47<06:35, 1.53MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:47<04:42, 2.13MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:48<07:43, 1.30MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:49<06:50, 1.46MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:49<04:53, 2.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:50<06:55, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:51<06:06, 1.63MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:51<04:21, 2.27MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:52<07:20, 1.34MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:52<07:43, 1.28MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:53<06:21, 1.55MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:53<05:03, 1.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:53<04:01, 2.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:53<03:17, 2.98MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:53<02:32, 3.85MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:55<09:43, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:55<08:11, 1.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:55<05:49, 1.67MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:57<08:02, 1.21MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:57<06:33, 1.48MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:57<04:42, 2.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:59<06:19, 1.53MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:59<05:45, 1.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:59<04:10, 2.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [02:01<05:26, 1.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [02:01<05:11, 1.85MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:01<03:47, 2.52MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:03<05:22, 1.77MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:03<05:12, 1.83MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:03<03:44, 2.53MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:05<05:48, 1.63MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:05<05:15, 1.80MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:05<03:47, 2.48MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:07<06:00, 1.56MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:07<05:39, 1.66MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:07<04:07, 2.27MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:09<05:13, 1.78MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:09<04:59, 1.87MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:09<03:37, 2.56MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:11<05:05, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:11<04:52, 1.90MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:11<03:33, 2.59MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:11<02:42, 3.40MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:13<07:54, 1.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:13<06:50, 1.34MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:13<04:52, 1.87MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:15<06:30, 1.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:15<06:49, 1.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:15<05:03, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:15<03:42, 2.45MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:17<06:34, 1.37MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:17<05:27, 1.66MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:17<03:55, 2.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:19<05:41, 1.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:19<05:23, 1.67MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:19<03:54, 2.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:20<02:55, 3.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:21<08:03, 1.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:21<06:56, 1.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:21<04:56, 1.79MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:24<07:12, 1.23MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:24<06:51, 1.29MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:24<05:09, 1.71MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:24<03:47, 2.32MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:24<02:51, 3.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:25<09:34, 915kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:26<08:04, 1.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:26<05:57, 1.47MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:26<04:22, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:26<03:12, 2.71MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:27<28:15, 308kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:28<21:00, 414kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:28<14:44, 587kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:29<13:40, 631kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:30<10:39, 809kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:30<07:32, 1.14MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:31<08:24, 1.02MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:32<07:05, 1.21MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:32<05:04, 1.68MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:32<03:43, 2.28MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:33<10:39, 797kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:33<08:12, 1.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:34<05:49, 1.45MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:35<07:21, 1.15MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:35<06:00, 1.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:36<04:16, 1.96MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:37<06:49, 1.22MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:37<06:01, 1.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:38<04:24, 1.89MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:38<03:12, 2.58MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:39<07:03, 1.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:39<06:09, 1.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:40<04:25, 1.87MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:41<05:18, 1.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:41<04:55, 1.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:41<03:32, 2.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:43<05:14, 1.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:43<04:51, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:43<03:30, 2.31MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:44<02:36, 3.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:45<11:39, 693kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:45<09:19, 866kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:45<06:35, 1.22MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:47<08:04, 992kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:47<06:51, 1.17MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:47<04:52, 1.63MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:49<06:28, 1.23MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:49<05:36, 1.42MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:49<03:59, 1.98MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:51<06:12, 1.27MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:51<05:27, 1.44MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:51<03:53, 2.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:53<06:09, 1.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:53<05:26, 1.43MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:53<03:52, 2.00MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:55<05:51, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:55<05:11, 1.49MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:55<03:44, 2.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:57<04:32, 1.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:57<03:52, 1.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:57<02:48, 2.71MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:59<04:09, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:59<04:08, 1.84MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:59<03:01, 2.50MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:01<03:48, 1.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:01<03:43, 2.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:01<02:42, 2.77MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:03<04:03, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:03<03:56, 1.89MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:03<02:51, 2.59MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:05<03:59, 1.85MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:05<03:40, 2.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:05<02:38, 2.78MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:07<04:52, 1.50MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:07<04:27, 1.64MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:07<03:11, 2.28MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:09<05:34, 1.30MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:09<04:53, 1.48MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:09<03:28, 2.07MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:11<05:30, 1.30MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:11<04:54, 1.46MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:11<03:32, 2.02MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:13<04:13, 1.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:13<03:40, 1.93MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:13<02:38, 2.68MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:15<04:53, 1.44MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:15<03:56, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:15<02:50, 2.46MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:17<04:17, 1.63MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:17<03:59, 1.74MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:17<02:52, 2.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:19<04:26, 1.56MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:19<04:05, 1.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:19<02:56, 2.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:21<04:19, 1.58MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:21<03:41, 1.85MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:21<02:38, 2.57MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:23<04:54, 1.38MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:23<04:19, 1.57MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:23<03:04, 2.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:25<04:57, 1.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:25<04:17, 1.56MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:25<03:03, 2.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:27<05:00, 1.32MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:27<04:26, 1.49MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:27<03:10, 2.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:29<04:20, 1.51MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:29<03:42, 1.77MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:29<02:39, 2.45MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:30<04:22, 1.49MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:31<03:52, 1.67MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:31<02:46, 2.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:32<04:50, 1.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:33<03:56, 1.63MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:33<02:49, 2.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:34<04:43, 1.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:35<04:10, 1.52MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:35<02:58, 2.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:36<04:59, 1.26MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:36<03:55, 1.60MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:37<02:48, 2.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:38<04:59, 1.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:38<04:00, 1.55MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:39<02:51, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:40<05:09, 1.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:40<04:11, 1.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:41<02:59, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:42<04:31, 1.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:42<04:03, 1.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:42<02:53, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:44<04:37, 1.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:44<03:49, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:44<02:43, 2.18MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:46<04:28, 1.33MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:46<04:02, 1.47MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:46<02:52, 2.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:48<04:38, 1.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:48<04:05, 1.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:48<02:54, 2.00MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:50<05:01, 1.15MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:50<04:22, 1.33MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:50<03:06, 1.85MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:52<04:28, 1.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:52<03:54, 1.47MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:52<02:47, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:54<03:47, 1.50MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:54<03:28, 1.63MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:54<02:29, 2.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:56<03:36, 1.55MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:56<03:21, 1.67MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:56<02:24, 2.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:58<03:29, 1.58MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:58<03:15, 1.69MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:58<02:21, 2.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:00<03:11, 1.72MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:00<02:42, 2.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:00<01:57, 2.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [04:02<02:56, 1.84MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [04:02<02:39, 2.03MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:02<01:54, 2.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:04<03:09, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:04<03:01, 1.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:04<02:10, 2.44MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:06<03:17, 1.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:06<03:07, 1.68MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:06<02:14, 2.32MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:08<03:11, 1.63MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:08<03:00, 1.72MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:08<02:10, 2.37MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:10<02:53, 1.77MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:10<02:50, 1.80MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:10<02:04, 2.45MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:12<02:39, 1.90MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:12<02:37, 1.92MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:12<01:56, 2.59MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:12<01:27, 3.44MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:14<03:21, 1.48MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:14<03:31, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:14<02:37, 1.89MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:14<01:56, 2.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:14<01:29, 3.29MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:16<04:32, 1.08MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:16<03:56, 1.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:16<02:54, 1.68MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:16<02:10, 2.24MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:16<01:36, 3.02MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:18<08:01, 603kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:18<06:20, 764kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:18<04:28, 1.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:20<04:25, 1.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:20<03:49, 1.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:20<02:46, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:20<02:01, 2.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:22<04:58, 946kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:22<03:54, 1.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:22<02:47, 1.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:24<03:11, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:24<02:55, 1.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:24<02:05, 2.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:25<01:33, 2.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:26<05:49, 784kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:26<04:43, 966kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:26<03:20, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:27<02:25, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:28<07:32, 597kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:28<05:54, 761kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:28<04:09, 1.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:30<04:24, 1.01MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:30<03:42, 1.19MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:30<02:37, 1.67MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:32<03:23, 1.28MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:32<02:59, 1.46MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:32<02:09, 2.01MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:32<01:33, 2.75MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:34<12:00, 357kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:34<09:03, 474kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:34<06:19, 671kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:36<05:51, 721kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:36<04:26, 949kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:36<03:09, 1.32MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:38<03:14, 1.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:38<02:51, 1.45MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:38<02:02, 2.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:40<02:56, 1.39MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:40<02:38, 1.55MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:40<01:53, 2.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:42<02:32, 1.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:42<02:22, 1.69MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:42<01:42, 2.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:44<02:32, 1.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:44<02:20, 1.68MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:44<01:41, 2.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:44<01:15, 3.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:46<04:58, 781kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:46<04:01, 963kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:46<02:50, 1.35MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:48<03:17, 1.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:48<03:43, 1.02MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:48<02:56, 1.29MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:48<02:17, 1.66MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:48<01:46, 2.12MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:48<01:25, 2.65MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:49<01:04, 3.49MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:50<03:21, 1.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:50<02:43, 1.37MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:50<02:00, 1.85MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:50<01:27, 2.53MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:52<02:58, 1.23MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:52<02:36, 1.41MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:52<01:51, 1.96MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:54<02:35, 1.39MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:54<02:20, 1.53MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:54<01:40, 2.13MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:56<02:29, 1.42MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:56<02:14, 1.58MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:56<01:35, 2.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:58<02:36, 1.33MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:58<02:12, 1.57MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:58<01:35, 2.17MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [05:00<02:00, 1.69MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [05:00<01:41, 2.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:00<01:13, 2.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:02<02:01, 1.65MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:02<01:51, 1.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:02<01:19, 2.47MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:04<02:08, 1.52MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:04<02:00, 1.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:04<01:26, 2.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:06<01:52, 1.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:06<01:45, 1.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:06<01:15, 2.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:08<01:51, 1.68MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:08<01:45, 1.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:08<01:16, 2.43MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:10<01:38, 1.86MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:10<01:35, 1.91MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:10<01:09, 2.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:12<01:39, 1.79MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:12<01:36, 1.85MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:12<01:14, 2.38MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:12<00:52, 3.31MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:15<05:22, 538kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:15<04:11, 687kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:15<02:56, 970kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:17<02:58, 945kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:17<02:27, 1.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:17<01:43, 1.61MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:19<02:12, 1.25MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:19<01:53, 1.45MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:19<01:20, 2.02MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:21<01:52, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:21<01:35, 1.69MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:21<01:07, 2.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:23<01:46, 1.47MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:23<01:37, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:23<01:09, 2.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:25<01:35, 1.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:25<01:26, 1.77MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:25<01:01, 2.45MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:27<01:37, 1.52MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:27<01:29, 1.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:27<01:03, 2.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:28<01:38, 1.47MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:29<01:24, 1.70MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:29<00:59, 2.37MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:30<01:37, 1.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:31<01:28, 1.59MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:31<01:02, 2.21MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:32<01:36, 1.42MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:33<01:27, 1.56MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:33<01:01, 2.16MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:34<01:24, 1.56MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:34<01:18, 1.68MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:35<00:55, 2.33MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:36<01:26, 1.47MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:36<01:10, 1.80MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:37<00:51, 2.46MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:38<01:09, 1.77MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:38<01:01, 2.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:39<00:43, 2.79MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:40<01:27, 1.37MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:40<01:15, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:40<00:52, 2.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:42<01:29, 1.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:42<01:14, 1.55MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:42<00:52, 2.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:44<01:16, 1.45MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:44<01:06, 1.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:44<00:46, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:46<01:24, 1.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:46<01:12, 1.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:46<00:50, 2.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:48<01:32, 1.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:48<01:19, 1.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:48<00:55, 1.80MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:50<01:23, 1.19MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:50<01:12, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:50<00:50, 1.90MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:52<01:17, 1.22MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:52<01:04, 1.47MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:52<00:44, 2.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:54<01:08, 1.32MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:54<00:58, 1.55MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:54<00:40, 2.17MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:56<01:13, 1.17MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:56<01:04, 1.34MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:56<00:44, 1.87MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:59<01:31, 900kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [06:00<01:15, 1.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [06:00<00:52, 1.52MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:01<00:48, 1.61MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:01<00:34, 2.23MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:02<00:44, 1.66MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:03<00:41, 1.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:03<00:28, 2.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:04<00:51, 1.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:05<00:46, 1.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:05<00:31, 2.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:06<00:50, 1.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:07<00:43, 1.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:07<00:29, 2.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:08<00:53, 1.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:08<00:42, 1.46MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:09<00:28, 2.03MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:10<00:46, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:10<00:39, 1.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:11<00:26, 2.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:12<00:41, 1.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:12<00:34, 1.57MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:13<00:23, 2.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:14<00:39, 1.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:14<00:33, 1.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:14<00:23, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:16<00:25, 1.81MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:16<00:24, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:16<00:16, 2.54MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:18<00:27, 1.52MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:18<00:25, 1.61MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:18<00:17, 2.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:20<00:26, 1.38MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:20<00:24, 1.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:20<00:16, 2.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:22<00:22, 1.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:22<00:20, 1.60MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:22<00:13, 2.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:24<00:18, 1.54MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:24<00:17, 1.67MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:24<00:11, 2.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:26<00:14, 1.73MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:26<00:14, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:26<00:09, 2.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:28<00:10, 2.00MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:28<00:10, 1.98MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:28<00:07, 2.67MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:30<00:07, 2.08MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:30<00:07, 2.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:30<00:05, 2.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:32<00:05, 2.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:32<00:05, 2.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:32<00:03, 3.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:34<00:04, 2.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:34<00:03, 2.03MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:34<00:02, 2.76MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:36<00:02, 2.05MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:36<00:01, 2.06MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:36<00:00, 2.84MB/s].vector_cache/glove.6B.zip: 862MB [06:36, 2.17MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 861/400000 [00:00<00:46, 8607.46it/s]  0%|          | 1623/400000 [00:00<00:48, 8285.09it/s]  1%|          | 2463/400000 [00:00<00:47, 8319.08it/s]  1%|          | 3315/400000 [00:00<00:47, 8377.34it/s]  1%|          | 4209/400000 [00:00<00:46, 8536.97it/s]  1%|▏         | 5044/400000 [00:00<00:46, 8479.39it/s]  1%|▏         | 5913/400000 [00:00<00:46, 8539.06it/s]  2%|▏         | 6779/400000 [00:00<00:45, 8574.78it/s]  2%|▏         | 7657/400000 [00:00<00:45, 8633.66it/s]  2%|▏         | 8561/400000 [00:01<00:44, 8751.38it/s]  2%|▏         | 9441/400000 [00:01<00:44, 8763.78it/s]  3%|▎         | 10356/400000 [00:01<00:43, 8876.14it/s]  3%|▎         | 11255/400000 [00:01<00:43, 8908.03it/s]  3%|▎         | 12139/400000 [00:01<00:44, 8780.56it/s]  3%|▎         | 13036/400000 [00:01<00:43, 8834.58it/s]  3%|▎         | 13942/400000 [00:01<00:43, 8898.53it/s]  4%|▎         | 14857/400000 [00:01<00:42, 8972.35it/s]  4%|▍         | 15785/400000 [00:01<00:42, 9059.92it/s]  4%|▍         | 16691/400000 [00:01<00:42, 8968.06it/s]  4%|▍         | 17588/400000 [00:02<00:43, 8856.59it/s]  5%|▍         | 18524/400000 [00:02<00:42, 9001.08it/s]  5%|▍         | 19454/400000 [00:02<00:41, 9087.69it/s]  5%|▌         | 20366/400000 [00:02<00:41, 9095.38it/s]  5%|▌         | 21276/400000 [00:02<00:41, 9074.09it/s]  6%|▌         | 22185/400000 [00:02<00:41, 9077.91it/s]  6%|▌         | 23109/400000 [00:02<00:41, 9124.49it/s]  6%|▌         | 24025/400000 [00:02<00:41, 9133.94it/s]  6%|▌         | 24948/400000 [00:02<00:40, 9160.19it/s]  6%|▋         | 25865/400000 [00:02<00:41, 9082.78it/s]  7%|▋         | 26779/400000 [00:03<00:41, 9099.31it/s]  7%|▋         | 27690/400000 [00:03<00:41, 9017.90it/s]  7%|▋         | 28593/400000 [00:03<00:41, 8870.09it/s]  7%|▋         | 29482/400000 [00:03<00:41, 8873.57it/s]  8%|▊         | 30370/400000 [00:03<00:41, 8869.18it/s]  8%|▊         | 31273/400000 [00:03<00:41, 8916.30it/s]  8%|▊         | 32195/400000 [00:03<00:40, 9003.71it/s]  8%|▊         | 33096/400000 [00:03<00:41, 8891.56it/s]  9%|▊         | 34015/400000 [00:03<00:40, 8978.80it/s]  9%|▊         | 34928/400000 [00:03<00:40, 9020.92it/s]  9%|▉         | 35831/400000 [00:04<00:40, 8934.56it/s]  9%|▉         | 36735/400000 [00:04<00:40, 8962.90it/s]  9%|▉         | 37670/400000 [00:04<00:39, 9074.78it/s] 10%|▉         | 38609/400000 [00:04<00:39, 9166.79it/s] 10%|▉         | 39527/400000 [00:04<00:39, 9113.10it/s] 10%|█         | 40439/400000 [00:04<00:39, 9028.59it/s] 10%|█         | 41369/400000 [00:04<00:39, 9107.45it/s] 11%|█         | 42287/400000 [00:04<00:39, 9127.81it/s] 11%|█         | 43201/400000 [00:04<00:39, 9073.33it/s] 11%|█         | 44109/400000 [00:04<00:39, 9029.00it/s] 11%|█▏        | 45013/400000 [00:05<00:39, 8996.38it/s] 11%|█▏        | 45921/400000 [00:05<00:39, 9018.45it/s] 12%|█▏        | 46824/400000 [00:05<00:40, 8759.68it/s] 12%|█▏        | 47718/400000 [00:05<00:39, 8810.45it/s] 12%|█▏        | 48619/400000 [00:05<00:39, 8864.31it/s] 12%|█▏        | 49521/400000 [00:05<00:39, 8909.19it/s] 13%|█▎        | 50413/400000 [00:05<00:39, 8873.67it/s] 13%|█▎        | 51302/400000 [00:05<00:39, 8877.05it/s] 13%|█▎        | 52191/400000 [00:05<00:39, 8837.20it/s] 13%|█▎        | 53076/400000 [00:05<00:39, 8796.87it/s] 13%|█▎        | 53956/400000 [00:06<00:39, 8779.05it/s] 14%|█▎        | 54835/400000 [00:06<00:39, 8763.27it/s] 14%|█▍        | 55712/400000 [00:06<00:39, 8737.26it/s] 14%|█▍        | 56586/400000 [00:06<00:39, 8737.54it/s] 14%|█▍        | 57460/400000 [00:06<00:39, 8725.99it/s] 15%|█▍        | 58333/400000 [00:06<00:39, 8726.94it/s] 15%|█▍        | 59206/400000 [00:06<00:39, 8684.21it/s] 15%|█▌        | 60122/400000 [00:06<00:38, 8819.08it/s] 15%|█▌        | 61020/400000 [00:06<00:38, 8866.22it/s] 15%|█▌        | 61927/400000 [00:06<00:37, 8924.87it/s] 16%|█▌        | 62820/400000 [00:07<00:37, 8880.32it/s] 16%|█▌        | 63709/400000 [00:07<00:37, 8852.65it/s] 16%|█▌        | 64607/400000 [00:07<00:37, 8887.35it/s] 16%|█▋        | 65500/400000 [00:07<00:37, 8900.01it/s] 17%|█▋        | 66417/400000 [00:07<00:37, 8977.46it/s] 17%|█▋        | 67316/400000 [00:07<00:37, 8967.55it/s] 17%|█▋        | 68232/400000 [00:07<00:36, 9021.86it/s] 17%|█▋        | 69142/400000 [00:07<00:36, 9042.30it/s] 18%|█▊        | 70047/400000 [00:07<00:36, 9036.23it/s] 18%|█▊        | 70951/400000 [00:07<00:36, 9013.07it/s] 18%|█▊        | 71853/400000 [00:08<00:36, 8975.78it/s] 18%|█▊        | 72772/400000 [00:08<00:36, 9038.64it/s] 18%|█▊        | 73677/400000 [00:08<00:36, 8944.85it/s] 19%|█▊        | 74580/400000 [00:08<00:36, 8970.03it/s] 19%|█▉        | 75495/400000 [00:08<00:35, 9020.48it/s] 19%|█▉        | 76398/400000 [00:08<00:35, 9018.77it/s] 19%|█▉        | 77302/400000 [00:08<00:35, 9024.45it/s] 20%|█▉        | 78224/400000 [00:08<00:35, 9079.56it/s] 20%|█▉        | 79149/400000 [00:08<00:35, 9127.58it/s] 20%|██        | 80084/400000 [00:08<00:34, 9191.71it/s] 20%|██        | 81004/400000 [00:09<00:34, 9144.66it/s] 20%|██        | 81919/400000 [00:09<00:34, 9141.84it/s] 21%|██        | 82834/400000 [00:09<00:35, 8847.17it/s] 21%|██        | 83721/400000 [00:09<00:36, 8720.38it/s] 21%|██        | 84617/400000 [00:09<00:35, 8788.63it/s] 21%|██▏       | 85506/400000 [00:09<00:35, 8816.24it/s] 22%|██▏       | 86404/400000 [00:09<00:35, 8863.39it/s] 22%|██▏       | 87310/400000 [00:09<00:35, 8919.32it/s] 22%|██▏       | 88226/400000 [00:09<00:34, 8987.40it/s] 22%|██▏       | 89126/400000 [00:09<00:34, 8924.75it/s] 23%|██▎       | 90019/400000 [00:10<00:34, 8904.11it/s] 23%|██▎       | 90922/400000 [00:10<00:34, 8941.34it/s] 23%|██▎       | 91817/400000 [00:10<00:35, 8767.07it/s] 23%|██▎       | 92732/400000 [00:10<00:34, 8876.75it/s] 23%|██▎       | 93641/400000 [00:10<00:34, 8938.79it/s] 24%|██▎       | 94536/400000 [00:10<00:34, 8929.88it/s] 24%|██▍       | 95438/400000 [00:10<00:34, 8956.47it/s] 24%|██▍       | 96335/400000 [00:10<00:33, 8936.00it/s] 24%|██▍       | 97240/400000 [00:10<00:33, 8969.59it/s] 25%|██▍       | 98156/400000 [00:11<00:33, 9025.75it/s] 25%|██▍       | 99062/400000 [00:11<00:33, 9034.84it/s] 25%|██▍       | 99986/400000 [00:11<00:32, 9094.16it/s] 25%|██▌       | 100901/400000 [00:11<00:32, 9109.47it/s] 25%|██▌       | 101818/400000 [00:11<00:32, 9127.14it/s] 26%|██▌       | 102731/400000 [00:11<00:32, 9108.14it/s] 26%|██▌       | 103642/400000 [00:11<00:32, 9066.09it/s] 26%|██▌       | 104549/400000 [00:11<00:32, 9027.63it/s] 26%|██▋       | 105452/400000 [00:11<00:32, 8997.84it/s] 27%|██▋       | 106358/400000 [00:11<00:32, 9015.82it/s] 27%|██▋       | 107260/400000 [00:12<00:32, 9004.94it/s] 27%|██▋       | 108161/400000 [00:12<00:32, 8982.85it/s] 27%|██▋       | 109060/400000 [00:12<00:32, 8945.86it/s] 27%|██▋       | 109955/400000 [00:12<00:33, 8725.74it/s] 28%|██▊       | 110848/400000 [00:12<00:32, 8784.76it/s] 28%|██▊       | 111728/400000 [00:12<00:33, 8728.14it/s] 28%|██▊       | 112628/400000 [00:12<00:32, 8806.96it/s] 28%|██▊       | 113526/400000 [00:12<00:32, 8856.45it/s] 29%|██▊       | 114442/400000 [00:12<00:31, 8944.03it/s] 29%|██▉       | 115370/400000 [00:12<00:31, 9040.34it/s] 29%|██▉       | 116278/400000 [00:13<00:31, 9049.55it/s] 29%|██▉       | 117189/400000 [00:13<00:31, 9065.86it/s] 30%|██▉       | 118097/400000 [00:13<00:31, 9067.52it/s] 30%|██▉       | 119004/400000 [00:13<00:31, 8960.90it/s] 30%|██▉       | 119905/400000 [00:13<00:31, 8972.84it/s] 30%|███       | 120811/400000 [00:13<00:31, 8996.39it/s] 30%|███       | 121711/400000 [00:13<00:31, 8972.77it/s] 31%|███       | 122609/400000 [00:13<00:30, 8949.11it/s] 31%|███       | 123535/400000 [00:13<00:30, 9037.28it/s] 31%|███       | 124447/400000 [00:13<00:30, 9060.47it/s] 31%|███▏      | 125354/400000 [00:14<00:30, 8980.87it/s] 32%|███▏      | 126253/400000 [00:14<00:30, 8952.44it/s] 32%|███▏      | 127151/400000 [00:14<00:30, 8960.30it/s] 32%|███▏      | 128048/400000 [00:14<00:30, 8907.75it/s] 32%|███▏      | 128939/400000 [00:14<00:30, 8898.77it/s] 32%|███▏      | 129836/400000 [00:14<00:30, 8917.12it/s] 33%|███▎      | 130728/400000 [00:14<00:31, 8659.87it/s] 33%|███▎      | 131617/400000 [00:14<00:30, 8726.42it/s] 33%|███▎      | 132517/400000 [00:14<00:30, 8804.50it/s] 33%|███▎      | 133445/400000 [00:14<00:29, 8939.62it/s] 34%|███▎      | 134377/400000 [00:15<00:29, 9047.88it/s] 34%|███▍      | 135283/400000 [00:15<00:29, 8915.05it/s] 34%|███▍      | 136184/400000 [00:15<00:29, 8851.17it/s] 34%|███▍      | 137096/400000 [00:15<00:29, 8928.84it/s] 34%|███▍      | 137995/400000 [00:15<00:29, 8945.73it/s] 35%|███▍      | 138903/400000 [00:15<00:29, 8985.51it/s] 35%|███▍      | 139803/400000 [00:15<00:29, 8864.15it/s] 35%|███▌      | 140691/400000 [00:15<00:29, 8784.77it/s] 35%|███▌      | 141571/400000 [00:15<00:29, 8753.67it/s] 36%|███▌      | 142447/400000 [00:15<00:30, 8508.04it/s] 36%|███▌      | 143323/400000 [00:16<00:29, 8579.98it/s] 36%|███▌      | 144205/400000 [00:16<00:29, 8648.19it/s] 36%|███▋      | 145081/400000 [00:16<00:29, 8678.98it/s] 36%|███▋      | 145979/400000 [00:16<00:28, 8765.30it/s] 37%|███▋      | 146859/400000 [00:16<00:28, 8773.70it/s] 37%|███▋      | 147759/400000 [00:16<00:28, 8839.91it/s] 37%|███▋      | 148644/400000 [00:16<00:28, 8813.62it/s] 37%|███▋      | 149526/400000 [00:16<00:28, 8800.82it/s] 38%|███▊      | 150410/400000 [00:16<00:28, 8811.12it/s] 38%|███▊      | 151305/400000 [00:16<00:28, 8849.52it/s] 38%|███▊      | 152209/400000 [00:17<00:27, 8903.20it/s] 38%|███▊      | 153100/400000 [00:17<00:27, 8897.28it/s] 38%|███▊      | 153990/400000 [00:17<00:27, 8889.39it/s] 39%|███▊      | 154905/400000 [00:17<00:27, 8965.50it/s] 39%|███▉      | 155802/400000 [00:17<00:27, 8946.30it/s] 39%|███▉      | 156718/400000 [00:17<00:27, 9007.59it/s] 39%|███▉      | 157619/400000 [00:17<00:27, 8802.38it/s] 40%|███▉      | 158553/400000 [00:17<00:26, 8956.65it/s] 40%|███▉      | 159484/400000 [00:17<00:26, 9058.83it/s] 40%|████      | 160392/400000 [00:17<00:26, 9042.02it/s] 40%|████      | 161307/400000 [00:18<00:26, 9071.60it/s] 41%|████      | 162215/400000 [00:18<00:26, 9017.03it/s] 41%|████      | 163118/400000 [00:18<00:26, 9011.46it/s] 41%|████      | 164024/400000 [00:18<00:26, 9024.57it/s] 41%|████      | 164927/400000 [00:18<00:26, 8976.41it/s] 41%|████▏     | 165830/400000 [00:18<00:26, 8991.09it/s] 42%|████▏     | 166730/400000 [00:18<00:26, 8862.69it/s] 42%|████▏     | 167647/400000 [00:18<00:25, 8952.59it/s] 42%|████▏     | 168583/400000 [00:18<00:25, 9070.82it/s] 42%|████▏     | 169491/400000 [00:18<00:25, 9059.00it/s] 43%|████▎     | 170405/400000 [00:19<00:25, 9080.90it/s] 43%|████▎     | 171325/400000 [00:19<00:25, 9114.17it/s] 43%|████▎     | 172237/400000 [00:19<00:25, 9108.72it/s] 43%|████▎     | 173151/400000 [00:19<00:24, 9115.96it/s] 44%|████▎     | 174063/400000 [00:19<00:24, 9113.62it/s] 44%|████▎     | 174975/400000 [00:19<00:24, 9088.06it/s] 44%|████▍     | 175884/400000 [00:19<00:24, 9001.70it/s] 44%|████▍     | 176785/400000 [00:19<00:24, 8992.20it/s] 44%|████▍     | 177726/400000 [00:19<00:24, 9111.24it/s] 45%|████▍     | 178638/400000 [00:20<00:24, 9042.17it/s] 45%|████▍     | 179547/400000 [00:20<00:24, 9054.78it/s] 45%|████▌     | 180453/400000 [00:20<00:24, 9035.62it/s] 45%|████▌     | 181377/400000 [00:20<00:24, 9093.54it/s] 46%|████▌     | 182295/400000 [00:20<00:23, 9117.75it/s] 46%|████▌     | 183209/400000 [00:20<00:23, 9122.60it/s] 46%|████▌     | 184124/400000 [00:20<00:23, 9128.65it/s] 46%|████▋     | 185037/400000 [00:20<00:23, 9032.82it/s] 46%|████▋     | 185941/400000 [00:20<00:23, 9007.07it/s] 47%|████▋     | 186861/400000 [00:20<00:23, 9063.30it/s] 47%|████▋     | 187775/400000 [00:21<00:23, 9084.88it/s] 47%|████▋     | 188684/400000 [00:21<00:23, 9050.34it/s] 47%|████▋     | 189590/400000 [00:21<00:23, 9013.21it/s] 48%|████▊     | 190492/400000 [00:21<00:23, 8908.25it/s] 48%|████▊     | 191385/400000 [00:21<00:23, 8913.56it/s] 48%|████▊     | 192277/400000 [00:21<00:23, 8852.73it/s] 48%|████▊     | 193199/400000 [00:21<00:23, 8954.37it/s] 49%|████▊     | 194095/400000 [00:21<00:23, 8828.58it/s] 49%|████▊     | 194979/400000 [00:21<00:23, 8831.01it/s] 49%|████▉     | 195891/400000 [00:21<00:22, 8913.74it/s] 49%|████▉     | 196791/400000 [00:22<00:22, 8938.33it/s] 49%|████▉     | 197712/400000 [00:22<00:22, 9015.16it/s] 50%|████▉     | 198642/400000 [00:22<00:22, 9097.77it/s] 50%|████▉     | 199553/400000 [00:22<00:22, 9098.44it/s] 50%|█████     | 200482/400000 [00:22<00:21, 9154.48it/s] 50%|█████     | 201398/400000 [00:22<00:21, 9148.16it/s] 51%|█████     | 202314/400000 [00:22<00:21, 9118.59it/s] 51%|█████     | 203243/400000 [00:22<00:21, 9167.46it/s] 51%|█████     | 204160/400000 [00:22<00:21, 9086.58it/s] 51%|█████▏    | 205069/400000 [00:22<00:21, 8996.40it/s] 51%|█████▏    | 205997/400000 [00:23<00:21, 9078.38it/s] 52%|█████▏    | 206918/400000 [00:23<00:21, 9115.28it/s] 52%|█████▏    | 207834/400000 [00:23<00:21, 9128.46it/s] 52%|█████▏    | 208748/400000 [00:23<00:21, 9104.11it/s] 52%|█████▏    | 209659/400000 [00:23<00:20, 9101.84it/s] 53%|█████▎    | 210570/400000 [00:23<00:20, 9043.32it/s] 53%|█████▎    | 211506/400000 [00:23<00:20, 9134.65it/s] 53%|█████▎    | 212459/400000 [00:23<00:20, 9247.37it/s] 53%|█████▎    | 213385/400000 [00:23<00:20, 9175.64it/s] 54%|█████▎    | 214324/400000 [00:23<00:20, 9237.69it/s] 54%|█████▍    | 215249/400000 [00:24<00:20, 9189.89it/s] 54%|█████▍    | 216169/400000 [00:24<00:20, 9147.78it/s] 54%|█████▍    | 217085/400000 [00:24<00:20, 9082.18it/s] 54%|█████▍    | 217994/400000 [00:24<00:20, 9008.56it/s] 55%|█████▍    | 218896/400000 [00:24<00:20, 9010.04it/s] 55%|█████▍    | 219798/400000 [00:24<00:20, 8993.53it/s] 55%|█████▌    | 220699/400000 [00:24<00:19, 8996.81it/s] 55%|█████▌    | 221599/400000 [00:24<00:20, 8845.51it/s] 56%|█████▌    | 222485/400000 [00:24<00:20, 8835.09it/s] 56%|█████▌    | 223377/400000 [00:24<00:19, 8859.18it/s] 56%|█████▌    | 224276/400000 [00:25<00:19, 8897.70it/s] 56%|█████▋    | 225167/400000 [00:25<00:19, 8896.74it/s] 57%|█████▋    | 226057/400000 [00:25<00:19, 8762.55it/s] 57%|█████▋    | 226938/400000 [00:25<00:19, 8776.05it/s] 57%|█████▋    | 227834/400000 [00:25<00:19, 8830.31it/s] 57%|█████▋    | 228738/400000 [00:25<00:19, 8891.81it/s] 57%|█████▋    | 229628/400000 [00:25<00:19, 8843.49it/s] 58%|█████▊    | 230513/400000 [00:25<00:19, 8816.45it/s] 58%|█████▊    | 231395/400000 [00:25<00:19, 8751.76it/s] 58%|█████▊    | 232274/400000 [00:25<00:19, 8763.11it/s] 58%|█████▊    | 233151/400000 [00:26<00:19, 8751.75it/s] 59%|█████▊    | 234036/400000 [00:26<00:18, 8778.05it/s] 59%|█████▊    | 234916/400000 [00:26<00:18, 8779.47it/s] 59%|█████▉    | 235795/400000 [00:26<00:18, 8767.41it/s] 59%|█████▉    | 236675/400000 [00:26<00:18, 8775.71it/s] 59%|█████▉    | 237555/400000 [00:26<00:18, 8780.49it/s] 60%|█████▉    | 238455/400000 [00:26<00:18, 8842.56it/s] 60%|█████▉    | 239348/400000 [00:26<00:18, 8866.23it/s] 60%|██████    | 240235/400000 [00:26<00:18, 8805.87it/s] 60%|██████    | 241118/400000 [00:26<00:18, 8812.71it/s] 61%|██████    | 242025/400000 [00:27<00:17, 8886.30it/s] 61%|██████    | 242940/400000 [00:27<00:17, 8962.35it/s] 61%|██████    | 243858/400000 [00:27<00:17, 9026.14it/s] 61%|██████    | 244775/400000 [00:27<00:17, 9068.74it/s] 61%|██████▏   | 245705/400000 [00:27<00:16, 9136.16it/s] 62%|██████▏   | 246619/400000 [00:27<00:16, 9089.92it/s] 62%|██████▏   | 247531/400000 [00:27<00:16, 9096.56it/s] 62%|██████▏   | 248441/400000 [00:27<00:16, 9016.29it/s] 62%|██████▏   | 249343/400000 [00:27<00:16, 8957.20it/s] 63%|██████▎   | 250258/400000 [00:27<00:16, 9013.86it/s] 63%|██████▎   | 251160/400000 [00:28<00:16, 8911.47it/s] 63%|██████▎   | 252065/400000 [00:28<00:16, 8949.54it/s] 63%|██████▎   | 252967/400000 [00:28<00:16, 8968.91it/s] 63%|██████▎   | 253876/400000 [00:28<00:16, 9004.39it/s] 64%|██████▎   | 254777/400000 [00:28<00:16, 8997.30it/s] 64%|██████▍   | 255684/400000 [00:28<00:16, 9018.24it/s] 64%|██████▍   | 256586/400000 [00:28<00:15, 8980.15it/s] 64%|██████▍   | 257485/400000 [00:28<00:15, 8940.76it/s] 65%|██████▍   | 258390/400000 [00:28<00:15, 8970.88it/s] 65%|██████▍   | 259293/400000 [00:28<00:15, 8987.75it/s] 65%|██████▌   | 260237/400000 [00:29<00:15, 9117.82it/s] 65%|██████▌   | 261166/400000 [00:29<00:15, 9167.00it/s] 66%|██████▌   | 262088/400000 [00:29<00:15, 9180.15it/s] 66%|██████▌   | 263007/400000 [00:29<00:14, 9153.21it/s] 66%|██████▌   | 263923/400000 [00:29<00:14, 9134.82it/s] 66%|██████▌   | 264837/400000 [00:29<00:14, 9075.80it/s] 66%|██████▋   | 265745/400000 [00:29<00:14, 9055.76it/s] 67%|██████▋   | 266651/400000 [00:29<00:14, 9051.02it/s] 67%|██████▋   | 267557/400000 [00:29<00:14, 9030.17it/s] 67%|██████▋   | 268461/400000 [00:29<00:14, 9013.39it/s] 67%|██████▋   | 269385/400000 [00:30<00:14, 9077.76it/s] 68%|██████▊   | 270301/400000 [00:30<00:14, 9102.03it/s] 68%|██████▊   | 271212/400000 [00:30<00:14, 9075.59it/s] 68%|██████▊   | 272120/400000 [00:30<00:14, 8960.42it/s] 68%|██████▊   | 273017/400000 [00:30<00:14, 8913.46it/s] 68%|██████▊   | 273924/400000 [00:30<00:14, 8958.05it/s] 69%|██████▊   | 274821/400000 [00:30<00:14, 8929.36it/s] 69%|██████▉   | 275732/400000 [00:30<00:13, 8981.23it/s] 69%|██████▉   | 276631/400000 [00:30<00:13, 8951.77it/s] 69%|██████▉   | 277527/400000 [00:31<00:13, 8770.70it/s] 70%|██████▉   | 278417/400000 [00:31<00:13, 8809.03it/s] 70%|██████▉   | 279311/400000 [00:31<00:13, 8845.66it/s] 70%|███████   | 280242/400000 [00:31<00:13, 8978.99it/s] 70%|███████   | 281141/400000 [00:31<00:13, 8860.90it/s] 71%|███████   | 282077/400000 [00:31<00:13, 9003.58it/s] 71%|███████   | 282979/400000 [00:31<00:13, 8987.42it/s] 71%|███████   | 283879/400000 [00:31<00:12, 8958.79it/s] 71%|███████   | 284803/400000 [00:31<00:12, 9039.85it/s] 71%|███████▏  | 285708/400000 [00:31<00:12, 8941.41it/s] 72%|███████▏  | 286603/400000 [00:32<00:12, 8867.97it/s] 72%|███████▏  | 287494/400000 [00:32<00:12, 8879.77it/s] 72%|███████▏  | 288418/400000 [00:32<00:12, 8983.66it/s] 72%|███████▏  | 289317/400000 [00:32<00:12, 8948.07it/s] 73%|███████▎  | 290234/400000 [00:32<00:12, 9011.75it/s] 73%|███████▎  | 291136/400000 [00:32<00:12, 8883.84it/s] 73%|███████▎  | 292026/400000 [00:32<00:12, 8849.65it/s] 73%|███████▎  | 292912/400000 [00:32<00:12, 8807.08it/s] 73%|███████▎  | 293798/400000 [00:32<00:12, 8820.97it/s] 74%|███████▎  | 294708/400000 [00:32<00:11, 8900.92it/s] 74%|███████▍  | 295607/400000 [00:33<00:11, 8924.88it/s] 74%|███████▍  | 296500/400000 [00:33<00:11, 8887.26it/s] 74%|███████▍  | 297389/400000 [00:33<00:11, 8846.09it/s] 75%|███████▍  | 298298/400000 [00:33<00:11, 8916.20it/s] 75%|███████▍  | 299190/400000 [00:33<00:11, 8851.59it/s] 75%|███████▌  | 300091/400000 [00:33<00:11, 8898.12it/s] 75%|███████▌  | 300982/400000 [00:33<00:11, 8884.89it/s] 75%|███████▌  | 301871/400000 [00:33<00:11, 8818.01it/s] 76%|███████▌  | 302754/400000 [00:33<00:11, 8660.99it/s] 76%|███████▌  | 303665/400000 [00:33<00:10, 8790.38it/s] 76%|███████▌  | 304571/400000 [00:34<00:10, 8867.26it/s] 76%|███████▋  | 305525/400000 [00:34<00:10, 9056.48it/s] 77%|███████▋  | 306447/400000 [00:34<00:10, 9102.88it/s] 77%|███████▋  | 307372/400000 [00:34<00:10, 9145.76it/s] 77%|███████▋  | 308316/400000 [00:34<00:09, 9231.09it/s] 77%|███████▋  | 309240/400000 [00:34<00:09, 9139.08it/s] 78%|███████▊  | 310169/400000 [00:34<00:09, 9183.56it/s] 78%|███████▊  | 311088/400000 [00:34<00:09, 9133.54it/s] 78%|███████▊  | 312043/400000 [00:34<00:09, 9252.74it/s] 78%|███████▊  | 312969/400000 [00:34<00:09, 9211.77it/s] 78%|███████▊  | 313891/400000 [00:35<00:09, 9074.37it/s] 79%|███████▊  | 314800/400000 [00:35<00:09, 8969.01it/s] 79%|███████▉  | 315709/400000 [00:35<00:09, 9003.19it/s] 79%|███████▉  | 316610/400000 [00:35<00:09, 8958.77it/s] 79%|███████▉  | 317523/400000 [00:35<00:09, 9008.62it/s] 80%|███████▉  | 318425/400000 [00:35<00:09, 8958.56it/s] 80%|███████▉  | 319322/400000 [00:35<00:09, 8894.19it/s] 80%|████████  | 320220/400000 [00:35<00:08, 8918.14it/s] 80%|████████  | 321113/400000 [00:35<00:08, 8841.39it/s] 80%|████████  | 321998/400000 [00:35<00:08, 8826.52it/s] 81%|████████  | 322881/400000 [00:36<00:08, 8804.78it/s] 81%|████████  | 323763/400000 [00:36<00:08, 8807.98it/s] 81%|████████  | 324644/400000 [00:36<00:08, 8803.69it/s] 81%|████████▏ | 325526/400000 [00:36<00:08, 8807.42it/s] 82%|████████▏ | 326411/400000 [00:36<00:08, 8818.22it/s] 82%|████████▏ | 327293/400000 [00:36<00:08, 8805.63it/s] 82%|████████▏ | 328191/400000 [00:36<00:08, 8855.71it/s] 82%|████████▏ | 329107/400000 [00:36<00:07, 8944.83it/s] 83%|████████▎ | 330050/400000 [00:36<00:07, 9084.21it/s] 83%|████████▎ | 330960/400000 [00:36<00:07, 9043.63it/s] 83%|████████▎ | 331873/400000 [00:37<00:07, 9068.37it/s] 83%|████████▎ | 332797/400000 [00:37<00:07, 9117.66it/s] 83%|████████▎ | 333738/400000 [00:37<00:07, 9200.99it/s] 84%|████████▎ | 334671/400000 [00:37<00:07, 9236.38it/s] 84%|████████▍ | 335603/400000 [00:37<00:06, 9259.27it/s] 84%|████████▍ | 336530/400000 [00:37<00:06, 9173.62it/s] 84%|████████▍ | 337448/400000 [00:37<00:06, 9112.54it/s] 85%|████████▍ | 338360/400000 [00:37<00:06, 9113.20it/s] 85%|████████▍ | 339272/400000 [00:37<00:06, 9098.78it/s] 85%|████████▌ | 340183/400000 [00:37<00:06, 9010.77it/s] 85%|████████▌ | 341085/400000 [00:38<00:06, 8947.12it/s] 85%|████████▌ | 341995/400000 [00:38<00:06, 8990.40it/s] 86%|████████▌ | 342895/400000 [00:38<00:06, 8966.24it/s] 86%|████████▌ | 343808/400000 [00:38<00:06, 9013.50it/s] 86%|████████▌ | 344737/400000 [00:38<00:06, 9092.93it/s] 86%|████████▋ | 345652/400000 [00:38<00:05, 9108.32it/s] 87%|████████▋ | 346574/400000 [00:38<00:05, 9140.23it/s] 87%|████████▋ | 347489/400000 [00:38<00:05, 9047.23it/s] 87%|████████▋ | 348404/400000 [00:38<00:05, 9075.85it/s] 87%|████████▋ | 349312/400000 [00:38<00:05, 9059.03it/s] 88%|████████▊ | 350219/400000 [00:39<00:05, 9021.85it/s] 88%|████████▊ | 351133/400000 [00:39<00:05, 9055.85it/s] 88%|████████▊ | 352062/400000 [00:39<00:05, 9123.67it/s] 88%|████████▊ | 352975/400000 [00:39<00:05, 9081.41it/s] 88%|████████▊ | 353884/400000 [00:39<00:05, 9078.70it/s] 89%|████████▊ | 354793/400000 [00:39<00:05, 9028.97it/s] 89%|████████▉ | 355710/400000 [00:39<00:04, 9068.48it/s] 89%|████████▉ | 356618/400000 [00:39<00:04, 9061.78it/s] 89%|████████▉ | 357525/400000 [00:39<00:04, 9032.47it/s] 90%|████████▉ | 358429/400000 [00:40<00:04, 8998.06it/s] 90%|████████▉ | 359329/400000 [00:40<00:04, 8991.71it/s] 90%|█████████ | 360247/400000 [00:40<00:04, 9044.65it/s] 90%|█████████ | 361152/400000 [00:40<00:04, 8925.47it/s] 91%|█████████ | 362045/400000 [00:40<00:04, 8906.98it/s] 91%|█████████ | 362946/400000 [00:40<00:04, 8934.37it/s] 91%|█████████ | 363853/400000 [00:40<00:04, 8972.50it/s] 91%|█████████ | 364762/400000 [00:40<00:03, 9006.94it/s] 91%|█████████▏| 365680/400000 [00:40<00:03, 9055.53it/s] 92%|█████████▏| 366586/400000 [00:40<00:03, 9033.07it/s] 92%|█████████▏| 367490/400000 [00:41<00:03, 8990.09it/s] 92%|█████████▏| 368390/400000 [00:41<00:03, 8991.66it/s] 92%|█████████▏| 369310/400000 [00:41<00:03, 9052.27it/s] 93%|█████████▎| 370216/400000 [00:41<00:03, 8877.78it/s] 93%|█████████▎| 371142/400000 [00:41<00:03, 8986.86it/s] 93%|█████████▎| 372067/400000 [00:41<00:03, 9061.59it/s] 93%|█████████▎| 372978/400000 [00:41<00:02, 9073.47it/s] 93%|█████████▎| 373906/400000 [00:41<00:02, 9132.92it/s] 94%|█████████▎| 374820/400000 [00:41<00:02, 9134.64it/s] 94%|█████████▍| 375739/400000 [00:41<00:02, 9148.53it/s] 94%|█████████▍| 376655/400000 [00:42<00:02, 9064.40it/s] 94%|█████████▍| 377562/400000 [00:42<00:02, 8982.06it/s] 95%|█████████▍| 378504/400000 [00:42<00:02, 9108.26it/s] 95%|█████████▍| 379432/400000 [00:42<00:02, 9156.82it/s] 95%|█████████▌| 380349/400000 [00:42<00:02, 9045.81it/s] 95%|█████████▌| 381263/400000 [00:42<00:02, 9072.34it/s] 96%|█████████▌| 382171/400000 [00:42<00:01, 9034.15it/s] 96%|█████████▌| 383085/400000 [00:42<00:01, 9063.61it/s] 96%|█████████▌| 383992/400000 [00:42<00:01, 9033.80it/s] 96%|█████████▌| 384903/400000 [00:42<00:01, 9056.17it/s] 96%|█████████▋| 385809/400000 [00:43<00:01, 8994.78it/s] 97%|█████████▋| 386709/400000 [00:43<00:01, 8984.59it/s] 97%|█████████▋| 387632/400000 [00:43<00:01, 9054.84it/s] 97%|█████████▋| 388538/400000 [00:43<00:01, 9040.36it/s] 97%|█████████▋| 389463/400000 [00:43<00:01, 9100.10it/s] 98%|█████████▊| 390374/400000 [00:43<00:01, 9019.08it/s] 98%|█████████▊| 391277/400000 [00:43<00:00, 8847.03it/s] 98%|█████████▊| 392163/400000 [00:43<00:00, 8790.48it/s] 98%|█████████▊| 393067/400000 [00:43<00:00, 8863.45it/s] 98%|█████████▊| 393995/400000 [00:43<00:00, 8984.17it/s] 99%|█████████▊| 394931/400000 [00:44<00:00, 9092.82it/s] 99%|█████████▉| 395856/400000 [00:44<00:00, 9138.32it/s] 99%|█████████▉| 396771/400000 [00:44<00:00, 9113.14it/s] 99%|█████████▉| 397683/400000 [00:44<00:00, 9092.94it/s]100%|█████████▉| 398606/400000 [00:44<00:00, 9132.27it/s]100%|█████████▉| 399520/400000 [00:44<00:00, 9080.77it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8967.13it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f2d2c40e940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011444732610759817 	 Accuracy: 50
Train Epoch: 1 	 Loss: 0.011301228035253824 	 Accuracy: 51

  model saves at 51% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15893 out of table with 15798 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15893 out of table with 15798 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-14 15:24:13.051698: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 15:24:13.056264: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-14 15:24:13.056438: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d93f4502e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 15:24:13.056453: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f2cd8de3e10> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7740 - accuracy: 0.4930
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8276 - accuracy: 0.4895 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6513 - accuracy: 0.5010
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6015 - accuracy: 0.5042
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5624 - accuracy: 0.5068
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5721 - accuracy: 0.5062
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5593 - accuracy: 0.5070
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5401 - accuracy: 0.5082
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5371 - accuracy: 0.5084
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5593 - accuracy: 0.5070
11000/25000 [============>.................] - ETA: 3s - loss: 7.5314 - accuracy: 0.5088
12000/25000 [=============>................] - ETA: 3s - loss: 7.5197 - accuracy: 0.5096
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5215 - accuracy: 0.5095
14000/25000 [===============>..............] - ETA: 2s - loss: 7.5352 - accuracy: 0.5086
15000/25000 [=================>............] - ETA: 2s - loss: 7.5542 - accuracy: 0.5073
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5612 - accuracy: 0.5069
17000/25000 [===================>..........] - ETA: 1s - loss: 7.5818 - accuracy: 0.5055
18000/25000 [====================>.........] - ETA: 1s - loss: 7.5840 - accuracy: 0.5054
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6198 - accuracy: 0.5031
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6444 - accuracy: 0.5015
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6476 - accuracy: 0.5012
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6569 - accuracy: 0.5006
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6540 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6545 - accuracy: 0.5008
25000/25000 [==============================] - 7s 284us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f2c8c1b2860> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f2ca95290f0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.5533 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.4138 - val_crf_viterbi_accuracy: 0.0267

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

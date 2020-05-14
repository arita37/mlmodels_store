
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f39140f1fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 16:14:27.437276
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 16:14:27.440769
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 16:14:27.443783
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 16:14:27.446887
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f3920109438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356601.1562
Epoch 2/10

1/1 [==============================] - 0s 95ms/step - loss: 279753.6562
Epoch 3/10

1/1 [==============================] - 0s 91ms/step - loss: 187478.4219
Epoch 4/10

1/1 [==============================] - 0s 93ms/step - loss: 115600.5234
Epoch 5/10

1/1 [==============================] - 0s 83ms/step - loss: 68641.6094
Epoch 6/10

1/1 [==============================] - 0s 84ms/step - loss: 41442.6094
Epoch 7/10

1/1 [==============================] - 0s 99ms/step - loss: 26308.6816
Epoch 8/10

1/1 [==============================] - 0s 84ms/step - loss: 17734.1562
Epoch 9/10

1/1 [==============================] - 0s 88ms/step - loss: 12668.3076
Epoch 10/10

1/1 [==============================] - 0s 91ms/step - loss: 9510.6865

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.2127022e+00  8.2827556e-01  3.6872607e-01 -1.9118014e-01
   1.7785002e+00  6.8522280e-01 -2.8553230e-01 -5.1449794e-01
  -7.8554118e-01 -1.5800030e+00 -6.2053725e-02  1.1241832e+00
   1.5758618e+00 -2.1044002e-01  1.7464238e-01  4.6224803e-01
  -8.3338872e-02  2.6541445e-01 -1.0848721e+00  1.5332240e-01
  -2.2468872e-01 -3.2439852e-01  3.1619561e-01  4.1711292e-01
  -2.7533552e-01  1.0571604e+00 -5.8329082e-01  1.4820348e+00
  -8.2765073e-01 -4.9397312e-02  5.5160826e-01  1.2540469e-01
   3.7570804e-01  1.1140945e+00  1.3657349e+00  1.0355886e+00
  -2.8253826e-01 -1.6713376e-01 -8.6088860e-01  7.5985187e-01
   9.5647538e-01  2.8789377e-01 -1.5388708e+00 -1.0153238e+00
   6.8776792e-01  6.0747969e-01  6.0186452e-01  1.0325453e+00
  -2.9515868e-01 -8.0502272e-01 -7.8215897e-01  6.8733716e-01
   1.4397426e+00  1.0953097e+00 -5.4205626e-01 -1.1957251e+00
   4.1861239e-01  4.4154745e-01  3.9052960e-01  1.6177641e-01
   2.5506279e-01  7.6867394e+00  5.6128421e+00  6.0289211e+00
   6.7873306e+00  7.1522188e+00  6.3409195e+00  5.8584690e+00
   5.1292663e+00  6.4897590e+00  5.8654857e+00  6.4114666e+00
   5.7304816e+00  6.1692576e+00  7.5835910e+00  7.5003309e+00
   6.2882042e+00  4.3700981e+00  5.7607498e+00  5.2148046e+00
   6.3558626e+00  6.2875867e+00  7.7375422e+00  5.6427269e+00
   7.6698799e+00  7.2838306e+00  6.8420825e+00  6.0216398e+00
   7.0660639e+00  7.1551876e+00  5.6616430e+00  5.5500154e+00
   7.2462368e+00  7.8486772e+00  8.2992706e+00  5.7048082e+00
   4.8901339e+00  6.7768455e+00  6.6102247e+00  7.9580588e+00
   7.5711350e+00  6.3770318e+00  6.8033552e+00  7.6318698e+00
   5.8806119e+00  7.2725763e+00  6.7427201e+00  5.7102075e+00
   4.6041546e+00  6.0383248e+00  5.2557011e+00  5.1564593e+00
   5.8813214e+00  6.9026289e+00  6.4082928e+00  7.3680525e+00
   5.0974097e+00  5.6322508e+00  6.2261066e+00  7.5555782e+00
  -5.7376492e-01  6.0593283e-01 -1.6857910e+00  2.0198229e-01
  -8.0731744e-01 -9.9680096e-02 -2.6496816e-01  1.3804728e-01
   2.6189119e-02  2.6422942e-01 -1.2208345e+00  3.7059394e-01
  -8.3257592e-01  8.4711391e-01  1.2922069e+00  4.6032339e-01
   1.4086421e+00  2.0125541e-01 -2.8452411e-02  2.4049391e-01
  -7.2666514e-01  2.5845003e-01  5.9971493e-01 -1.4247339e+00
   6.2679279e-01  1.2203665e+00 -8.4854260e-02 -5.4325521e-01
   5.2111053e-01  7.7854669e-01 -7.5029576e-01 -3.8454682e-01
   1.5800185e+00 -5.9597498e-01  3.6634597e-01 -1.4247993e+00
   2.8423446e-01  7.7737793e-02  5.4543507e-01 -1.0867333e+00
  -2.4716631e-01 -4.1257155e-01  7.1606207e-01  1.1859999e+00
  -1.4841813e-01 -5.3862798e-01  3.2962549e-01  1.5421543e-01
   7.0437849e-01 -5.1091182e-01  1.0276960e+00 -2.6059136e-01
  -3.3299765e-01  6.0327947e-03  1.2647003e+00 -7.4043655e-01
  -8.6874992e-01 -5.9876561e-01 -4.6758497e-01  9.2776835e-02
   3.9086556e-01  1.8722506e+00  9.9340224e-01  1.4560227e+00
   1.1874915e+00  1.0466071e+00  3.3365178e-01  8.3350909e-01
   7.7849090e-01  1.4100552e-01  2.1516438e+00  5.5313981e-01
   4.7076452e-01  7.9789305e-01  1.1063348e+00  3.3974218e-01
   4.1706383e-01  1.3155638e+00  7.2999477e-01  4.9096727e-01
   1.8009579e-01  3.0167443e-01  1.7754909e+00  1.0021619e+00
   2.9079251e+00  6.3770020e-01  1.3107517e+00  7.2534490e-01
   3.2842779e+00  4.9484110e-01  1.8771448e+00  3.4159988e-01
   5.9492427e-01  1.4051697e+00  6.2052530e-01  3.2625973e-01
   1.8200053e+00  4.4603944e-01  7.1386814e-01  4.9310571e-01
   2.4651389e+00  4.2113167e-01  2.8487272e+00  1.3745248e+00
   3.1440592e-01  7.3715007e-01  7.7719402e-01  1.9782531e-01
   5.3162295e-01  2.2919643e-01  7.2562575e-01  5.7706565e-01
   2.1409123e+00  7.9834431e-01  9.7389179e-01  8.0958688e-01
   4.8365647e-01  9.3994862e-01  7.0115554e-01  1.3503571e+00
   8.0675244e-02  7.4909053e+00  7.0001826e+00  6.8096752e+00
   6.0081029e+00  6.5000176e+00  6.2285686e+00  7.6406307e+00
   6.9823322e+00  6.7557950e+00  6.7270908e+00  8.0105495e+00
   7.7003956e+00  6.2961712e+00  7.0570841e+00  7.2930760e+00
   7.5799971e+00  7.3956246e+00  7.4738564e+00  8.0000057e+00
   6.4225612e+00  7.1996717e+00  7.1299491e+00  7.2263331e+00
   7.0719647e+00  6.5521212e+00  6.5278635e+00  7.3541589e+00
   7.4305544e+00  5.8101053e+00  6.8084650e+00  6.9272985e+00
   7.2119489e+00  6.2576756e+00  6.8365974e+00  5.6500416e+00
   6.5989580e+00  6.8402429e+00  6.1666875e+00  7.4431696e+00
   6.8184495e+00  8.4216166e+00  7.1754675e+00  6.7729683e+00
   7.6660008e+00  6.8206172e+00  7.9909019e+00  7.5188632e+00
   5.5788965e+00  6.2174916e+00  7.1709394e+00  7.3846135e+00
   7.5959344e+00  6.4997978e+00  7.7841702e+00  7.4979796e+00
   6.8010039e+00  7.0132847e+00  7.1472597e+00  7.8403435e+00
   1.4377253e+00  1.0196977e+00  7.8155535e-01  1.8704388e+00
   1.7736022e+00  1.3289667e+00  2.4069471e+00  9.3140477e-01
   1.0180304e+00  1.9490336e+00  1.9367683e+00  7.8013229e-01
   4.5054704e-01  6.4567697e-01  7.1389169e-01  1.2418612e+00
   2.7110081e+00  6.8929082e-01  1.2912172e+00  1.6388006e+00
   7.2612309e-01  1.3328273e+00  1.9868205e+00  1.6314797e+00
   5.0134861e-01  6.8803108e-01  3.6786371e-01  3.9041585e-01
   1.3627298e+00  2.3236418e+00  2.8036141e-01  7.2845042e-01
   1.3998594e+00  1.8181944e-01  2.0168033e+00  1.3844484e+00
   4.0859181e-01  8.9311951e-01  1.1239104e+00  1.9627024e+00
   3.1859648e-01  6.9130474e-01  5.0649667e-01  2.5153995e-01
   2.4406085e+00  1.6619803e+00  8.5358185e-01  5.9170920e-01
   6.4048409e-01  1.5291963e+00  2.5286794e-01  2.2198861e+00
   5.4680294e-01  7.0808280e-01  3.7775224e-01  6.3138044e-01
   1.3355110e+00  1.2342073e+00  2.9035923e+00  2.1915495e-01
  -7.8047452e+00  1.4630194e+00 -3.4151945e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 16:14:35.554545
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.6664
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 16:14:35.558178
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9171.28
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 16:14:35.561095
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.4197
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 16:14:35.564118
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -820.353
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139882769638960
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139880257286776
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139880257287280
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139880257287784
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139880257288288
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139880257288792

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f390074e4a8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.426325
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.384337
grad_step = 000002, loss = 0.350972
grad_step = 000003, loss = 0.316456
grad_step = 000004, loss = 0.283939
grad_step = 000005, loss = 0.265300
grad_step = 000006, loss = 0.268244
grad_step = 000007, loss = 0.259601
grad_step = 000008, loss = 0.239659
grad_step = 000009, loss = 0.225361
grad_step = 000010, loss = 0.218884
grad_step = 000011, loss = 0.214304
grad_step = 000012, loss = 0.208476
grad_step = 000013, loss = 0.200474
grad_step = 000014, loss = 0.190938
grad_step = 000015, loss = 0.181380
grad_step = 000016, loss = 0.173391
grad_step = 000017, loss = 0.167112
grad_step = 000018, loss = 0.160779
grad_step = 000019, loss = 0.153429
grad_step = 000020, loss = 0.145056
grad_step = 000021, loss = 0.137386
grad_step = 000022, loss = 0.130682
grad_step = 000023, loss = 0.124291
grad_step = 000024, loss = 0.117612
grad_step = 000025, loss = 0.110596
grad_step = 000026, loss = 0.103727
grad_step = 000027, loss = 0.097564
grad_step = 000028, loss = 0.092170
grad_step = 000029, loss = 0.086957
grad_step = 000030, loss = 0.081684
grad_step = 000031, loss = 0.076713
grad_step = 000032, loss = 0.072419
grad_step = 000033, loss = 0.068659
grad_step = 000034, loss = 0.065006
grad_step = 000035, loss = 0.061203
grad_step = 000036, loss = 0.057316
grad_step = 000037, loss = 0.053618
grad_step = 000038, loss = 0.050278
grad_step = 000039, loss = 0.047163
grad_step = 000040, loss = 0.044038
grad_step = 000041, loss = 0.040922
grad_step = 000042, loss = 0.038045
grad_step = 000043, loss = 0.035489
grad_step = 000044, loss = 0.033138
grad_step = 000045, loss = 0.030850
grad_step = 000046, loss = 0.028595
grad_step = 000047, loss = 0.026432
grad_step = 000048, loss = 0.024411
grad_step = 000049, loss = 0.022524
grad_step = 000050, loss = 0.020741
grad_step = 000051, loss = 0.019081
grad_step = 000052, loss = 0.017572
grad_step = 000053, loss = 0.016200
grad_step = 000054, loss = 0.014930
grad_step = 000055, loss = 0.013738
grad_step = 000056, loss = 0.012611
grad_step = 000057, loss = 0.011554
grad_step = 000058, loss = 0.010579
grad_step = 000059, loss = 0.009696
grad_step = 000060, loss = 0.008910
grad_step = 000061, loss = 0.008212
grad_step = 000062, loss = 0.007585
grad_step = 000063, loss = 0.007022
grad_step = 000064, loss = 0.006514
grad_step = 000065, loss = 0.006046
grad_step = 000066, loss = 0.005618
grad_step = 000067, loss = 0.005240
grad_step = 000068, loss = 0.004912
grad_step = 000069, loss = 0.004620
grad_step = 000070, loss = 0.004355
grad_step = 000071, loss = 0.004122
grad_step = 000072, loss = 0.003921
grad_step = 000073, loss = 0.003745
grad_step = 000074, loss = 0.003583
grad_step = 000075, loss = 0.003436
grad_step = 000076, loss = 0.003306
grad_step = 000077, loss = 0.003195
grad_step = 000078, loss = 0.003095
grad_step = 000079, loss = 0.003006
grad_step = 000080, loss = 0.002925
grad_step = 000081, loss = 0.002854
grad_step = 000082, loss = 0.002790
grad_step = 000083, loss = 0.002731
grad_step = 000084, loss = 0.002676
grad_step = 000085, loss = 0.002628
grad_step = 000086, loss = 0.002587
grad_step = 000087, loss = 0.002551
grad_step = 000088, loss = 0.002516
grad_step = 000089, loss = 0.002483
grad_step = 000090, loss = 0.002453
grad_step = 000091, loss = 0.002426
grad_step = 000092, loss = 0.002402
grad_step = 000093, loss = 0.002380
grad_step = 000094, loss = 0.002361
grad_step = 000095, loss = 0.002343
grad_step = 000096, loss = 0.002328
grad_step = 000097, loss = 0.002313
grad_step = 000098, loss = 0.002300
grad_step = 000099, loss = 0.002288
grad_step = 000100, loss = 0.002279
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002270
grad_step = 000102, loss = 0.002261
grad_step = 000103, loss = 0.002254
grad_step = 000104, loss = 0.002247
grad_step = 000105, loss = 0.002241
grad_step = 000106, loss = 0.002236
grad_step = 000107, loss = 0.002230
grad_step = 000108, loss = 0.002225
grad_step = 000109, loss = 0.002221
grad_step = 000110, loss = 0.002216
grad_step = 000111, loss = 0.002212
grad_step = 000112, loss = 0.002208
grad_step = 000113, loss = 0.002204
grad_step = 000114, loss = 0.002200
grad_step = 000115, loss = 0.002196
grad_step = 000116, loss = 0.002192
grad_step = 000117, loss = 0.002187
grad_step = 000118, loss = 0.002184
grad_step = 000119, loss = 0.002179
grad_step = 000120, loss = 0.002175
grad_step = 000121, loss = 0.002171
grad_step = 000122, loss = 0.002167
grad_step = 000123, loss = 0.002163
grad_step = 000124, loss = 0.002158
grad_step = 000125, loss = 0.002154
grad_step = 000126, loss = 0.002150
grad_step = 000127, loss = 0.002145
grad_step = 000128, loss = 0.002141
grad_step = 000129, loss = 0.002136
grad_step = 000130, loss = 0.002132
grad_step = 000131, loss = 0.002127
grad_step = 000132, loss = 0.002122
grad_step = 000133, loss = 0.002117
grad_step = 000134, loss = 0.002112
grad_step = 000135, loss = 0.002107
grad_step = 000136, loss = 0.002102
grad_step = 000137, loss = 0.002097
grad_step = 000138, loss = 0.002092
grad_step = 000139, loss = 0.002086
grad_step = 000140, loss = 0.002081
grad_step = 000141, loss = 0.002075
grad_step = 000142, loss = 0.002070
grad_step = 000143, loss = 0.002066
grad_step = 000144, loss = 0.002062
grad_step = 000145, loss = 0.002058
grad_step = 000146, loss = 0.002053
grad_step = 000147, loss = 0.002045
grad_step = 000148, loss = 0.002036
grad_step = 000149, loss = 0.002028
grad_step = 000150, loss = 0.002023
grad_step = 000151, loss = 0.002018
grad_step = 000152, loss = 0.002014
grad_step = 000153, loss = 0.002007
grad_step = 000154, loss = 0.001999
grad_step = 000155, loss = 0.001990
grad_step = 000156, loss = 0.001982
grad_step = 000157, loss = 0.001975
grad_step = 000158, loss = 0.001969
grad_step = 000159, loss = 0.001962
grad_step = 000160, loss = 0.001956
grad_step = 000161, loss = 0.001948
grad_step = 000162, loss = 0.001941
grad_step = 000163, loss = 0.001932
grad_step = 000164, loss = 0.001923
grad_step = 000165, loss = 0.001914
grad_step = 000166, loss = 0.001905
grad_step = 000167, loss = 0.001896
grad_step = 000168, loss = 0.001888
grad_step = 000169, loss = 0.001882
grad_step = 000170, loss = 0.001878
grad_step = 000171, loss = 0.001881
grad_step = 000172, loss = 0.001890
grad_step = 000173, loss = 0.001895
grad_step = 000174, loss = 0.001880
grad_step = 000175, loss = 0.001853
grad_step = 000176, loss = 0.001833
grad_step = 000177, loss = 0.001811
grad_step = 000178, loss = 0.001813
grad_step = 000179, loss = 0.001840
grad_step = 000180, loss = 0.001836
grad_step = 000181, loss = 0.001802
grad_step = 000182, loss = 0.001773
grad_step = 000183, loss = 0.001757
grad_step = 000184, loss = 0.001743
grad_step = 000185, loss = 0.001735
grad_step = 000186, loss = 0.001753
grad_step = 000187, loss = 0.001782
grad_step = 000188, loss = 0.001800
grad_step = 000189, loss = 0.001816
grad_step = 000190, loss = 0.001806
grad_step = 000191, loss = 0.001731
grad_step = 000192, loss = 0.001673
grad_step = 000193, loss = 0.001671
grad_step = 000194, loss = 0.001691
grad_step = 000195, loss = 0.001709
grad_step = 000196, loss = 0.001708
grad_step = 000197, loss = 0.001718
grad_step = 000198, loss = 0.001735
grad_step = 000199, loss = 0.001651
grad_step = 000200, loss = 0.001635
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001627
grad_step = 000202, loss = 0.001593
grad_step = 000203, loss = 0.001634
grad_step = 000204, loss = 0.001614
grad_step = 000205, loss = 0.001646
grad_step = 000206, loss = 0.001746
grad_step = 000207, loss = 0.001884
grad_step = 000208, loss = 0.002093
grad_step = 000209, loss = 0.001812
grad_step = 000210, loss = 0.001554
grad_step = 000211, loss = 0.001663
grad_step = 000212, loss = 0.001730
grad_step = 000213, loss = 0.001617
grad_step = 000214, loss = 0.001575
grad_step = 000215, loss = 0.001636
grad_step = 000216, loss = 0.001651
grad_step = 000217, loss = 0.001521
grad_step = 000218, loss = 0.001571
grad_step = 000219, loss = 0.001602
grad_step = 000220, loss = 0.001533
grad_step = 000221, loss = 0.001496
grad_step = 000222, loss = 0.001538
grad_step = 000223, loss = 0.001542
grad_step = 000224, loss = 0.001494
grad_step = 000225, loss = 0.001475
grad_step = 000226, loss = 0.001471
grad_step = 000227, loss = 0.001505
grad_step = 000228, loss = 0.001509
grad_step = 000229, loss = 0.001497
grad_step = 000230, loss = 0.001471
grad_step = 000231, loss = 0.001437
grad_step = 000232, loss = 0.001431
grad_step = 000233, loss = 0.001422
grad_step = 000234, loss = 0.001438
grad_step = 000235, loss = 0.001465
grad_step = 000236, loss = 0.001563
grad_step = 000237, loss = 0.001735
grad_step = 000238, loss = 0.002151
grad_step = 000239, loss = 0.002139
grad_step = 000240, loss = 0.001806
grad_step = 000241, loss = 0.001432
grad_step = 000242, loss = 0.001782
grad_step = 000243, loss = 0.001857
grad_step = 000244, loss = 0.001440
grad_step = 000245, loss = 0.001784
grad_step = 000246, loss = 0.001777
grad_step = 000247, loss = 0.001447
grad_step = 000248, loss = 0.001766
grad_step = 000249, loss = 0.001559
grad_step = 000250, loss = 0.001494
grad_step = 000251, loss = 0.001675
grad_step = 000252, loss = 0.001430
grad_step = 000253, loss = 0.001528
grad_step = 000254, loss = 0.001522
grad_step = 000255, loss = 0.001400
grad_step = 000256, loss = 0.001508
grad_step = 000257, loss = 0.001419
grad_step = 000258, loss = 0.001418
grad_step = 000259, loss = 0.001469
grad_step = 000260, loss = 0.001373
grad_step = 000261, loss = 0.001432
grad_step = 000262, loss = 0.001412
grad_step = 000263, loss = 0.001368
grad_step = 000264, loss = 0.001411
grad_step = 000265, loss = 0.001382
grad_step = 000266, loss = 0.001356
grad_step = 000267, loss = 0.001390
grad_step = 000268, loss = 0.001360
grad_step = 000269, loss = 0.001340
grad_step = 000270, loss = 0.001370
grad_step = 000271, loss = 0.001337
grad_step = 000272, loss = 0.001332
grad_step = 000273, loss = 0.001346
grad_step = 000274, loss = 0.001335
grad_step = 000275, loss = 0.001310
grad_step = 000276, loss = 0.001329
grad_step = 000277, loss = 0.001322
grad_step = 000278, loss = 0.001304
grad_step = 000279, loss = 0.001301
grad_step = 000280, loss = 0.001310
grad_step = 000281, loss = 0.001303
grad_step = 000282, loss = 0.001285
grad_step = 000283, loss = 0.001282
grad_step = 000284, loss = 0.001286
grad_step = 000285, loss = 0.001284
grad_step = 000286, loss = 0.001271
grad_step = 000287, loss = 0.001262
grad_step = 000288, loss = 0.001261
grad_step = 000289, loss = 0.001262
grad_step = 000290, loss = 0.001263
grad_step = 000291, loss = 0.001254
grad_step = 000292, loss = 0.001244
grad_step = 000293, loss = 0.001233
grad_step = 000294, loss = 0.001226
grad_step = 000295, loss = 0.001225
grad_step = 000296, loss = 0.001226
grad_step = 000297, loss = 0.001229
grad_step = 000298, loss = 0.001236
grad_step = 000299, loss = 0.001253
grad_step = 000300, loss = 0.001278
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001331
grad_step = 000302, loss = 0.001387
grad_step = 000303, loss = 0.001450
grad_step = 000304, loss = 0.001395
grad_step = 000305, loss = 0.001331
grad_step = 000306, loss = 0.001212
grad_step = 000307, loss = 0.001179
grad_step = 000308, loss = 0.001229
grad_step = 000309, loss = 0.001284
grad_step = 000310, loss = 0.001283
grad_step = 000311, loss = 0.001212
grad_step = 000312, loss = 0.001167
grad_step = 000313, loss = 0.001177
grad_step = 000314, loss = 0.001207
grad_step = 000315, loss = 0.001231
grad_step = 000316, loss = 0.001194
grad_step = 000317, loss = 0.001165
grad_step = 000318, loss = 0.001148
grad_step = 000319, loss = 0.001159
grad_step = 000320, loss = 0.001176
grad_step = 000321, loss = 0.001181
grad_step = 000322, loss = 0.001180
grad_step = 000323, loss = 0.001161
grad_step = 000324, loss = 0.001142
grad_step = 000325, loss = 0.001135
grad_step = 000326, loss = 0.001138
grad_step = 000327, loss = 0.001153
grad_step = 000328, loss = 0.001161
grad_step = 000329, loss = 0.001173
grad_step = 000330, loss = 0.001172
grad_step = 000331, loss = 0.001171
grad_step = 000332, loss = 0.001150
grad_step = 000333, loss = 0.001130
grad_step = 000334, loss = 0.001117
grad_step = 000335, loss = 0.001117
grad_step = 000336, loss = 0.001126
grad_step = 000337, loss = 0.001137
grad_step = 000338, loss = 0.001155
grad_step = 000339, loss = 0.001166
grad_step = 000340, loss = 0.001178
grad_step = 000341, loss = 0.001165
grad_step = 000342, loss = 0.001148
grad_step = 000343, loss = 0.001117
grad_step = 000344, loss = 0.001098
grad_step = 000345, loss = 0.001098
grad_step = 000346, loss = 0.001110
grad_step = 000347, loss = 0.001132
grad_step = 000348, loss = 0.001146
grad_step = 000349, loss = 0.001166
grad_step = 000350, loss = 0.001157
grad_step = 000351, loss = 0.001144
grad_step = 000352, loss = 0.001109
grad_step = 000353, loss = 0.001085
grad_step = 000354, loss = 0.001079
grad_step = 000355, loss = 0.001090
grad_step = 000356, loss = 0.001115
grad_step = 000357, loss = 0.001134
grad_step = 000358, loss = 0.001165
grad_step = 000359, loss = 0.001153
grad_step = 000360, loss = 0.001141
grad_step = 000361, loss = 0.001102
grad_step = 000362, loss = 0.001071
grad_step = 000363, loss = 0.001061
grad_step = 000364, loss = 0.001075
grad_step = 000365, loss = 0.001105
grad_step = 000366, loss = 0.001122
grad_step = 000367, loss = 0.001144
grad_step = 000368, loss = 0.001136
grad_step = 000369, loss = 0.001129
grad_step = 000370, loss = 0.001086
grad_step = 000371, loss = 0.001052
grad_step = 000372, loss = 0.001048
grad_step = 000373, loss = 0.001069
grad_step = 000374, loss = 0.001099
grad_step = 000375, loss = 0.001114
grad_step = 000376, loss = 0.001131
grad_step = 000377, loss = 0.001105
grad_step = 000378, loss = 0.001079
grad_step = 000379, loss = 0.001043
grad_step = 000380, loss = 0.001031
grad_step = 000381, loss = 0.001044
grad_step = 000382, loss = 0.001062
grad_step = 000383, loss = 0.001077
grad_step = 000384, loss = 0.001069
grad_step = 000385, loss = 0.001060
grad_step = 000386, loss = 0.001038
grad_step = 000387, loss = 0.001021
grad_step = 000388, loss = 0.001015
grad_step = 000389, loss = 0.001019
grad_step = 000390, loss = 0.001033
grad_step = 000391, loss = 0.001042
grad_step = 000392, loss = 0.001052
grad_step = 000393, loss = 0.001050
grad_step = 000394, loss = 0.001053
grad_step = 000395, loss = 0.001046
grad_step = 000396, loss = 0.001042
grad_step = 000397, loss = 0.001027
grad_step = 000398, loss = 0.001013
grad_step = 000399, loss = 0.001000
grad_step = 000400, loss = 0.000993
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000993
grad_step = 000402, loss = 0.000997
grad_step = 000403, loss = 0.001005
grad_step = 000404, loss = 0.001014
grad_step = 000405, loss = 0.001033
grad_step = 000406, loss = 0.001046
grad_step = 000407, loss = 0.001082
grad_step = 000408, loss = 0.001105
grad_step = 000409, loss = 0.001175
grad_step = 000410, loss = 0.001174
grad_step = 000411, loss = 0.001211
grad_step = 000412, loss = 0.001098
grad_step = 000413, loss = 0.001009
grad_step = 000414, loss = 0.000978
grad_step = 000415, loss = 0.001025
grad_step = 000416, loss = 0.001092
grad_step = 000417, loss = 0.001080
grad_step = 000418, loss = 0.001035
grad_step = 000419, loss = 0.000976
grad_step = 000420, loss = 0.000978
grad_step = 000421, loss = 0.001019
grad_step = 000422, loss = 0.001031
grad_step = 000423, loss = 0.001014
grad_step = 000424, loss = 0.000972
grad_step = 000425, loss = 0.000961
grad_step = 000426, loss = 0.000982
grad_step = 000427, loss = 0.000996
grad_step = 000428, loss = 0.001001
grad_step = 000429, loss = 0.000976
grad_step = 000430, loss = 0.000956
grad_step = 000431, loss = 0.000949
grad_step = 000432, loss = 0.000955
grad_step = 000433, loss = 0.000968
grad_step = 000434, loss = 0.000976
grad_step = 000435, loss = 0.000986
grad_step = 000436, loss = 0.000980
grad_step = 000437, loss = 0.000975
grad_step = 000438, loss = 0.000960
grad_step = 000439, loss = 0.000949
grad_step = 000440, loss = 0.000938
grad_step = 000441, loss = 0.000933
grad_step = 000442, loss = 0.000931
grad_step = 000443, loss = 0.000932
grad_step = 000444, loss = 0.000936
grad_step = 000445, loss = 0.000941
grad_step = 000446, loss = 0.000951
grad_step = 000447, loss = 0.000962
grad_step = 000448, loss = 0.000983
grad_step = 000449, loss = 0.000995
grad_step = 000450, loss = 0.001028
grad_step = 000451, loss = 0.001038
grad_step = 000452, loss = 0.001071
grad_step = 000453, loss = 0.001038
grad_step = 000454, loss = 0.001014
grad_step = 000455, loss = 0.000956
grad_step = 000456, loss = 0.000919
grad_step = 000457, loss = 0.000913
grad_step = 000458, loss = 0.000933
grad_step = 000459, loss = 0.000960
grad_step = 000460, loss = 0.000971
grad_step = 000461, loss = 0.000969
grad_step = 000462, loss = 0.000937
grad_step = 000463, loss = 0.000910
grad_step = 000464, loss = 0.000901
grad_step = 000465, loss = 0.000911
grad_step = 000466, loss = 0.000928
grad_step = 000467, loss = 0.000934
grad_step = 000468, loss = 0.000933
grad_step = 000469, loss = 0.000919
grad_step = 000470, loss = 0.000905
grad_step = 000471, loss = 0.000893
grad_step = 000472, loss = 0.000888
grad_step = 000473, loss = 0.000889
grad_step = 000474, loss = 0.000894
grad_step = 000475, loss = 0.000903
grad_step = 000476, loss = 0.000910
grad_step = 000477, loss = 0.000923
grad_step = 000478, loss = 0.000935
grad_step = 000479, loss = 0.000966
grad_step = 000480, loss = 0.000982
grad_step = 000481, loss = 0.001022
grad_step = 000482, loss = 0.001023
grad_step = 000483, loss = 0.001053
grad_step = 000484, loss = 0.001001
grad_step = 000485, loss = 0.000952
grad_step = 000486, loss = 0.000885
grad_step = 000487, loss = 0.000875
grad_step = 000488, loss = 0.000915
grad_step = 000489, loss = 0.000950
grad_step = 000490, loss = 0.000958
grad_step = 000491, loss = 0.000913
grad_step = 000492, loss = 0.000872
grad_step = 000493, loss = 0.000863
grad_step = 000494, loss = 0.000885
grad_step = 000495, loss = 0.000908
grad_step = 000496, loss = 0.000900
grad_step = 000497, loss = 0.000880
grad_step = 000498, loss = 0.000858
grad_step = 000499, loss = 0.000852
grad_step = 000500, loss = 0.000861
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000874
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

  date_run                              2020-05-14 16:14:52.659897
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.239184
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 16:14:52.665491
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.150667
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 16:14:52.671914
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.121879
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 16:14:52.676560
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.28945
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
0   2020-05-14 16:14:27.437276  ...    mean_absolute_error
1   2020-05-14 16:14:27.440769  ...     mean_squared_error
2   2020-05-14 16:14:27.443783  ...  median_absolute_error
3   2020-05-14 16:14:27.446887  ...               r2_score
4   2020-05-14 16:14:35.554545  ...    mean_absolute_error
5   2020-05-14 16:14:35.558178  ...     mean_squared_error
6   2020-05-14 16:14:35.561095  ...  median_absolute_error
7   2020-05-14 16:14:35.564118  ...               r2_score
8   2020-05-14 16:14:52.659897  ...    mean_absolute_error
9   2020-05-14 16:14:52.665491  ...     mean_squared_error
10  2020-05-14 16:14:52.671914  ...  median_absolute_error
11  2020-05-14 16:14:52.676560  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9cd4c86fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:22, 119234.89it/s] 62%|██████▏   | 6152192/9912422 [00:00<00:22, 170193.79it/s]9920512it [00:00, 36047199.77it/s]                           
0it [00:00, ?it/s]32768it [00:00, 529338.18it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 158780.30it/s]1654784it [00:00, 11108035.51it/s]                         
0it [00:00, ?it/s]8192it [00:00, 180499.68it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9c87689e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9c86cb70b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9c87689e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9c86c0e0f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9c844494e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9c84434c50> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9c87689e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9c86bcc710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9c844494e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9c86cb7128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f0b09395208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=f18b26aec9becb9da2ce8e9dfab564ac34a45b40b3b66d12fac40cac690649d7
  Stored in directory: /tmp/pip-ephem-wheel-cache-kq9zb1yk/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f0aa11906d8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
  860160/17464789 [>.............................] - ETA: 0s
 1867776/17464789 [==>...........................] - ETA: 0s
 2924544/17464789 [====>.........................] - ETA: 0s
 4022272/17464789 [=====>........................] - ETA: 0s
 5193728/17464789 [=======>......................] - ETA: 0s
 6373376/17464789 [=========>....................] - ETA: 0s
 7643136/17464789 [============>.................] - ETA: 0s
 9076736/17464789 [==============>...............] - ETA: 0s
10608640/17464789 [=================>............] - ETA: 0s
12034048/17464789 [===================>..........] - ETA: 0s
13533184/17464789 [======================>.......] - ETA: 0s
15065088/17464789 [========================>.....] - ETA: 0s
16678912/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 16:16:18.332215: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 16:16:18.336054: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-14 16:16:18.336181: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562d77293160 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 16:16:18.336195: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 8.0500 - accuracy: 0.4750
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8890 - accuracy: 0.4855 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.8148 - accuracy: 0.4903
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7280 - accuracy: 0.4960
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6636 - accuracy: 0.5002
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6308 - accuracy: 0.5023
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6425 - accuracy: 0.5016
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6896 - accuracy: 0.4985
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6871 - accuracy: 0.4987
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7126 - accuracy: 0.4970
11000/25000 [============>.................] - ETA: 3s - loss: 7.7321 - accuracy: 0.4957
12000/25000 [=============>................] - ETA: 3s - loss: 7.7152 - accuracy: 0.4968
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6949 - accuracy: 0.4982
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6710 - accuracy: 0.4997
15000/25000 [=================>............] - ETA: 2s - loss: 7.6411 - accuracy: 0.5017
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6494 - accuracy: 0.5011
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6711 - accuracy: 0.4997
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6598 - accuracy: 0.5004
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6489 - accuracy: 0.5012
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6544 - accuracy: 0.5008
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6630 - accuracy: 0.5002
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6548 - accuracy: 0.5008
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6406 - accuracy: 0.5017
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6526 - accuracy: 0.5009
25000/25000 [==============================] - 7s 269us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 16:16:31.372640
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 16:16:31.372640  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<40:34:02, 5.90kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<28:37:44, 8.36kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<20:05:37, 11.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<14:04:07, 17.0kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<9:49:12, 24.3kB/s].vector_cache/glove.6B.zip:   1%|          | 9.24M/862M [00:02<6:49:50, 34.7kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.3M/862M [00:02<4:46:00, 49.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.8M/862M [00:02<3:18:59, 70.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:02<2:18:51, 101kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 25.9M/862M [00:02<1:36:45, 144kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.7M/862M [00:02<1:07:31, 205kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.5M/862M [00:02<47:04, 293kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:02<32:55, 417kB/s].vector_cache/glove.6B.zip:   5%|▍         | 43.0M/862M [00:02<22:59, 594kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.8M/862M [00:02<16:08, 842kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:03<11:18, 1.19MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:03<11:40, 1.16MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:03<08:15, 1.63MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<14:14, 943kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<12:09, 1.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<08:57, 1.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:05<06:27, 2.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:07<11:29, 1.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:07<09:33, 1.40MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<07:01, 1.90MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<07:45, 1.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:09<06:49, 1.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.9M/862M [00:09<05:06, 2.59MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:11<06:40, 1.98MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<07:21, 1.80MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:11<05:43, 2.30MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:11<04:12, 3.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:13<08:07, 1.62MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:13<07:01, 1.87MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.1M/862M [00:13<05:14, 2.50MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:15<06:44, 1.94MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:15<07:22, 1.77MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:15<05:45, 2.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:15<04:13, 3.09MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<10:19, 1.26MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:16<07:53, 1.65MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:18<12:08:14, 17.8kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:18<8:30:45, 25.4kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<5:57:02, 36.3kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:20<4:12:30, 51.2kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.0M/862M [00:20<2:59:20, 72.0kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.7M/862M [00:20<2:06:00, 102kB/s] .vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:20<1:28:04, 146kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:21<1:11:19, 180kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:22<51:12, 251kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:22<36:06, 355kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:23<28:12, 453kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:24<22:20, 572kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.0M/862M [00:24<16:12, 788kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<11:27, 1.11MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:25<17:38, 721kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:26<13:40, 930kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<09:50, 1.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<09:50, 1.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<09:30, 1.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:28<07:13, 1.75MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:12, 2.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<10:57, 1.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:29<08:58, 1.40MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<06:32, 1.92MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<07:31, 1.66MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<07:55, 1.58MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:32<06:11, 2.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:28, 2.78MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<1:31:55, 135kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<1:05:36, 190kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<46:09, 269kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<35:05, 353kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<25:51, 478kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<18:22, 672kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<15:42, 783kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<13:29, 911kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<10:03, 1.22MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<07:08, 1.72MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<1:34:19, 130kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<1:07:14, 182kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<47:14, 258kB/s]  .vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<35:50, 340kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<26:17, 463kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<18:40, 650kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<15:53, 761kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<13:36, 889kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<10:03, 1.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<07:09, 1.68MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:45<13:33, 887kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<10:42, 1.12MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<07:47, 1.54MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<08:14, 1.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<08:12, 1.46MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<06:17, 1.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<04:31, 2.63MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<14:24, 826kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<11:19, 1.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<08:12, 1.44MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<08:29, 1.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<07:08, 1.65MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:15, 2.24MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<06:28, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<06:55, 1.70MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<05:22, 2.18MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<03:55, 2.98MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<07:48, 1.50MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<06:39, 1.76MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<04:54, 2.37MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<06:09, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<05:29, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<04:07, 2.81MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<05:37, 2.06MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<06:17, 1.83MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<04:55, 2.34MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<03:33, 3.22MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<12:36, 911kB/s] .vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<10:01, 1.15MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<07:17, 1.57MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<07:46, 1.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<07:51, 1.45MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<06:00, 1.90MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:21, 2.61MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<07:50, 1.45MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<06:38, 1.71MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<04:55, 2.29MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:07<05:46, 1.95MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<08:17, 1.36MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<06:55, 1.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:04, 2.21MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<06:11, 1.81MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<05:31, 2.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<04:09, 2.69MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<05:23, 2.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<06:03, 1.83MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<04:42, 2.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:27, 3.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<07:18, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<06:14, 1.77MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:38, 2.38MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<05:49, 1.89MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<05:12, 2.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<03:54, 2.80MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:17<05:17, 2.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:17<05:55, 1.84MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<04:37, 2.36MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<03:20, 3.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:19<12:21, 878kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<09:45, 1.11MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:05, 1.52MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:21<07:28, 1.44MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<06:20, 1.70MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:40, 2.30MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:23<05:48, 1.84MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:23<05:09, 2.08MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<03:52, 2.76MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<05:13, 2.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<05:50, 1.82MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:34, 2.32MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<03:18, 3.20MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<12:11, 868kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<09:37, 1.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:59, 1.51MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<07:20, 1.43MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<06:01, 1.74MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:31, 2.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<03:16, 3.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<41:45, 250kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<31:20, 333kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<22:26, 464kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<15:45, 658kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<22:02, 471kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<16:30, 628kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<11:44, 880kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<10:36, 971kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<08:28, 1.22MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<06:08, 1.67MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:37<06:43, 1.52MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:37<06:47, 1.51MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<05:12, 1.96MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:48, 2.68MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<06:35, 1.54MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<05:39, 1.79MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:12, 2.40MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<05:18, 1.90MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<05:46, 1.75MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<04:33, 2.21MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:18, 3.03MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<1:13:52, 136kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:43<52:42, 190kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<37:03, 270kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<28:10, 353kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<21:44, 458kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<15:39, 635kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<11:03, 896kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<12:35, 786kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<09:50, 1.00MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<07:07, 1.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<07:15, 1.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<07:00, 1.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:49<06:00, 1.63MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:26, 2.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:16, 1.85MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<05:32, 1.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<04:17, 2.27MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<03:06, 3.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<08:40, 1.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<07:05, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:10, 1.86MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<05:46, 1.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<05:59, 1.60MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:37, 2.07MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<03:19, 2.87MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<16:44, 570kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<12:42, 750kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<09:06, 1.04MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<08:34, 1.10MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<06:58, 1.36MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<05:04, 1.86MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:46, 1.63MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:00, 1.88MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<03:43, 2.51MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<04:48, 1.94MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:19, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:13, 2.89MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<04:27, 2.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<05:00, 1.85MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<03:58, 2.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<04:15, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<03:54, 2.35MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<02:56, 3.11MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<04:11, 2.18MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:08<04:47, 1.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<03:45, 2.42MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<02:44, 3.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<07:06, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<05:54, 1.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<04:21, 2.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<05:09, 1.75MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<04:30, 1.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<03:22, 2.65MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<04:28, 1.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<04:59, 1.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<03:51, 2.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<02:52, 3.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<04:32, 1.95MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<04:05, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:03, 2.89MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<04:11, 2.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<04:43, 1.86MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<03:42, 2.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<02:40, 3.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<16:32, 527kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<12:29, 698kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<08:55, 974kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<08:14, 1.05MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<07:33, 1.14MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<05:43, 1.51MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<04:04, 2.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<20:40, 415kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<15:20, 559kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<10:55, 782kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<09:36, 886kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<07:35, 1.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<05:28, 1.55MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:28<05:50, 1.45MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<04:56, 1.71MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<03:37, 2.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<04:32, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<04:53, 1.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:46, 2.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<02:47, 2.98MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<04:31, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<04:01, 2.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:01, 2.73MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<04:01, 2.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<04:29, 1.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<03:33, 2.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<02:34, 3.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<7:52:28, 17.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<5:31:19, 24.6kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<3:51:23, 35.2kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<2:43:09, 49.6kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<1:55:49, 69.9kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<1:21:22, 99.4kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<56:42, 142kB/s]   .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<1:36:57, 82.8kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<1:08:37, 117kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<48:05, 166kB/s]  .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<35:22, 225kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<25:34, 311kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<18:02, 439kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<14:26, 547kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<10:54, 723kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<07:47, 1.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<07:17, 1.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:45, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:11, 1.86MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<03:01, 2.57MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<33:26, 232kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<24:58, 311kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<17:47, 435kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<12:28, 617kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<17:36, 437kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<13:07, 586kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<09:19, 822kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<08:17, 920kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<07:25, 1.03MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:35, 1.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:59, 1.90MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<56:39, 133kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<40:25, 187kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<28:21, 265kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<21:31, 348kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<15:48, 473kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<11:11, 666kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<09:34, 775kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<08:12, 904kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<06:07, 1.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:20, 1.69MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<31:12, 235kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<22:34, 325kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<15:56, 459kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<12:48, 568kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<09:41, 750kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<06:55, 1.05MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<06:32, 1.10MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<06:02, 1.19MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:35, 1.57MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<03:16, 2.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<6:55:20, 17.2kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<4:51:12, 24.5kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<3:23:17, 35.0kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<2:23:14, 49.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<1:41:39, 69.5kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<1:11:23, 98.8kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<50:44, 138kB/s]   .vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<36:06, 194kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<25:23, 275kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<17:45, 391kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<25:30, 272kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<18:32, 374kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<13:06, 526kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<10:45, 638kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<08:13, 834kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<05:54, 1.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<05:43, 1.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<05:23, 1.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<04:03, 1.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:56, 2.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<04:39, 1.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<03:57, 1.70MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:54, 2.30MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<03:35, 1.85MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<03:52, 1.72MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<02:59, 2.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:10, 3.03MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<04:32, 1.45MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<03:51, 1.71MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:51, 2.30MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<03:31, 1.85MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<03:07, 2.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:20, 2.77MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<03:09, 2.04MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<02:52, 2.25MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:09, 2.97MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<03:01, 2.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<03:24, 1.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<02:42, 2.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<01:56, 3.24MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<6:06:53, 17.2kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<4:17:12, 24.5kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<2:59:26, 35.0kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<2:06:21, 49.5kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:31<1:29:40, 69.6kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<1:02:54, 99.1kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<43:53, 141kB/s]   .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<33:20, 185kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<23:57, 258kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<16:49, 365kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<13:09, 465kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<09:49, 622kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<06:58, 871kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<06:17, 959kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<05:38, 1.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:14, 1.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<03:00, 1.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<51:23, 116kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<36:32, 163kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<25:36, 232kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<19:12, 308kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<14:01, 421kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<09:54, 593kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<08:17, 705kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<06:23, 913kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<04:35, 1.26MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<04:34, 1.26MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:46, 1.52MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:46, 2.06MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:17, 1.73MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:27, 1.65MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:40, 2.13MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<01:55, 2.93MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<04:29, 1.26MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:42, 1.51MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:43, 2.05MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<03:12, 1.73MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:48, 1.97MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<02:06, 2.63MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:45, 2.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:28, 2.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<01:50, 2.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:34, 2.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:54, 1.86MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:18, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:28, 2.17MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:17, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<01:43, 3.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:26, 2.17MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:47, 1.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:10, 2.42MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<01:34, 3.31MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<04:15, 1.23MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<03:30, 1.49MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:34, 2.01MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<03:00, 1.72MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<03:08, 1.63MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:27, 2.09MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<02:31, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:13, 2.29MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<01:39, 3.05MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<01:13, 4.10MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<13:27, 372kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<10:26, 480kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<07:31, 664kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<05:16, 940kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<12:15, 404kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<09:04, 544kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<06:26, 762kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<05:37, 867kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<04:25, 1.10MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:12, 1.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<03:22, 1.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:50, 1.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:05, 2.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:34, 1.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:17, 2.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:41, 2.78MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:16<02:17, 2.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<02:04, 2.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<01:33, 2.97MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<02:10, 2.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<02:27, 1.87MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<01:55, 2.38MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:23, 3.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<05:12, 870kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:20<04:06, 1.10MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:57, 1.52MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:06, 1.44MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:37, 1.70MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<01:56, 2.28MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:23, 1.84MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<02:34, 1.71MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<01:58, 2.21MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:25, 3.04MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<03:52, 1.12MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<03:09, 1.37MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:18, 1.86MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:35, 1.64MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:15, 1.89MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:40, 2.53MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:09, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:21, 1.77MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<01:51, 2.25MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:57, 2.11MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:47, 2.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:20, 3.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<01:53, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<01:44, 2.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:17, 3.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:50, 2.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:05, 1.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:38, 2.42MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<01:13, 3.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:54, 2.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:43, 2.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:17, 3.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<01:48, 2.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<02:03, 1.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:37, 2.35MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<01:44, 2.18MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<01:35, 2.36MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:11, 3.13MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<01:42, 2.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:56, 1.91MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:31, 2.42MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:05, 3.32MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<03:40, 993kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:56, 1.24MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:08, 1.69MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:19, 1.54MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:55, 1.85MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:26, 2.47MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<01:49, 1.93MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<01:37, 2.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:12, 2.88MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:39, 2.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:30, 2.27MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:08, 2.99MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:35, 2.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:23, 2.41MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:03, 3.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<01:31, 2.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:23, 2.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:03, 3.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:57<01:29, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:57<01:42, 1.90MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:20, 2.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<00:57, 3.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<02:54, 1.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:21, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:42, 1.83MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:01<01:54, 1.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:01<01:57, 1.57MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:31, 2.02MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:32, 1.96MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:23, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:02, 2.88MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:24, 2.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:36, 1.83MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:15, 2.34MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<00:54, 3.20MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<02:18, 1.25MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<01:54, 1.51MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:23, 2.06MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<01:37, 1.74MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<01:25, 1.98MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:03, 2.64MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<01:22, 1.99MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<01:14, 2.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<00:55, 2.92MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:16, 2.10MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:09, 2.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<00:52, 3.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:13, 2.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:07, 2.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<00:50, 3.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:11, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:21, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:03, 2.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:45, 3.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<03:08, 790kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:26, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:45, 1.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:46, 1.36MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:28, 1.62MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:05, 2.18MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:18, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:09, 2.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<00:51, 2.70MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:07, 2.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<00:58, 2.32MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:44, 3.05MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:01, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:56, 2.33MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:41, 3.10MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<00:59, 2.15MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:07, 1.89MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:52, 2.42MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:38, 3.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:22, 1.50MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:10, 1.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:51, 2.36MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:03, 1.88MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:56, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:42, 2.80MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<00:56, 2.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:51, 2.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:37, 3.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:52, 2.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:48, 2.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:35, 3.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:50, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:45, 2.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:33, 3.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:47, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:54, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:43, 2.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<00:45, 2.18MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:41, 2.35MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:31, 3.09MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:43, 2.18MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:40, 2.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:30, 3.10MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<00:42, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<00:47, 1.90MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:37, 2.42MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:26, 3.31MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<01:07, 1.29MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:55, 1.55MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:40, 2.09MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:50<00:47, 1.76MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:41, 2.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:30, 2.68MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:52<00:39, 2.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:43, 1.81MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:33, 2.29MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:34, 2.14MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:32, 2.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:23, 3.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:32, 2.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:30, 2.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:22, 3.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:30, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:34, 1.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:27, 2.38MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<00:28, 2.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<00:26, 2.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:19, 3.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<00:26, 2.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<00:24, 2.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:17, 3.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:24, 2.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:04<00:28, 1.90MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:21, 2.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:15, 3.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<01:11, 694kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:54, 900kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:38, 1.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:36, 1.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:34, 1.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:25, 1.74MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:17, 2.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:40, 1.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:32, 1.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:22, 1.73MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:23, 1.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:24, 1.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:18, 1.94MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:12, 2.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<04:07, 135kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<02:54, 189kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<01:57, 268kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<01:23, 352kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<01:00, 477kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:40, 672kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:32, 783kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:27, 911kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:19, 1.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:15, 1.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:12, 1.62MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:09, 1.80MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:09, 1.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:04, 3.00MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<12:23, 17.3kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<08:26, 24.6kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<05:10, 35.2kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<02:56, 49.6kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<02:02, 69.9kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<01:18, 99.4kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:38, 142kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:25, 180kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:16, 250kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:07, 354kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:01, 453kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 571kB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 880/400000 [00:00<00:45, 8792.52it/s]  0%|          | 1845/400000 [00:00<00:44, 9030.84it/s]  1%|          | 2814/400000 [00:00<00:43, 9217.18it/s]  1%|          | 3740/400000 [00:00<00:42, 9228.57it/s]  1%|          | 4651/400000 [00:00<00:43, 9189.78it/s]  1%|▏         | 5580/400000 [00:00<00:42, 9216.44it/s]  2%|▏         | 6540/400000 [00:00<00:42, 9325.60it/s]  2%|▏         | 7509/400000 [00:00<00:41, 9429.84it/s]  2%|▏         | 8446/400000 [00:00<00:41, 9410.97it/s]  2%|▏         | 9353/400000 [00:01<00:42, 9182.50it/s]  3%|▎         | 10249/400000 [00:01<00:42, 9090.04it/s]  3%|▎         | 11145/400000 [00:01<00:42, 9049.59it/s]  3%|▎         | 12058/400000 [00:01<00:42, 9072.94it/s]  3%|▎         | 13020/400000 [00:01<00:41, 9228.51it/s]  3%|▎         | 13953/400000 [00:01<00:41, 9256.76it/s]  4%|▎         | 14876/400000 [00:01<00:41, 9226.65it/s]  4%|▍         | 15797/400000 [00:01<00:41, 9154.58it/s]  4%|▍         | 16730/400000 [00:01<00:41, 9205.16it/s]  4%|▍         | 17695/400000 [00:01<00:40, 9333.59it/s]  5%|▍         | 18634/400000 [00:02<00:40, 9349.70it/s]  5%|▍         | 19577/400000 [00:02<00:40, 9372.44it/s]  5%|▌         | 20532/400000 [00:02<00:40, 9422.55it/s]  5%|▌         | 21475/400000 [00:02<00:40, 9256.10it/s]  6%|▌         | 22402/400000 [00:02<00:40, 9218.57it/s]  6%|▌         | 23325/400000 [00:02<00:41, 9089.57it/s]  6%|▌         | 24270/400000 [00:02<00:40, 9194.07it/s]  6%|▋         | 25205/400000 [00:02<00:40, 9237.76it/s]  7%|▋         | 26179/400000 [00:02<00:39, 9381.40it/s]  7%|▋         | 27138/400000 [00:02<00:39, 9442.32it/s]  7%|▋         | 28084/400000 [00:03<00:39, 9398.77it/s]  7%|▋         | 29030/400000 [00:03<00:39, 9414.64it/s]  7%|▋         | 29982/400000 [00:03<00:39, 9444.11it/s]  8%|▊         | 30927/400000 [00:03<00:39, 9402.00it/s]  8%|▊         | 31868/400000 [00:03<00:39, 9340.02it/s]  8%|▊         | 32803/400000 [00:03<00:39, 9317.65it/s]  8%|▊         | 33735/400000 [00:03<00:39, 9237.57it/s]  9%|▊         | 34660/400000 [00:03<00:39, 9145.15it/s]  9%|▉         | 35575/400000 [00:03<00:39, 9111.87it/s]  9%|▉         | 36490/400000 [00:03<00:39, 9121.45it/s]  9%|▉         | 37415/400000 [00:04<00:39, 9158.65it/s] 10%|▉         | 38381/400000 [00:04<00:38, 9301.93it/s] 10%|▉         | 39329/400000 [00:04<00:38, 9353.35it/s] 10%|█         | 40289/400000 [00:04<00:38, 9423.82it/s] 10%|█         | 41247/400000 [00:04<00:37, 9467.37it/s] 11%|█         | 42195/400000 [00:04<00:37, 9432.88it/s] 11%|█         | 43153/400000 [00:04<00:37, 9474.27it/s] 11%|█         | 44107/400000 [00:04<00:37, 9490.27it/s] 11%|█▏        | 45062/400000 [00:04<00:37, 9505.42it/s] 12%|█▏        | 46013/400000 [00:04<00:37, 9480.21it/s] 12%|█▏        | 46962/400000 [00:05<00:38, 9212.63it/s] 12%|█▏        | 47885/400000 [00:05<00:38, 9129.74it/s] 12%|█▏        | 48800/400000 [00:05<00:38, 9098.10it/s] 12%|█▏        | 49714/400000 [00:05<00:38, 9108.75it/s] 13%|█▎        | 50643/400000 [00:05<00:38, 9160.36it/s] 13%|█▎        | 51560/400000 [00:05<00:38, 9148.07it/s] 13%|█▎        | 52518/400000 [00:05<00:37, 9270.96it/s] 13%|█▎        | 53469/400000 [00:05<00:37, 9339.71it/s] 14%|█▎        | 54419/400000 [00:05<00:36, 9385.02it/s] 14%|█▍        | 55377/400000 [00:05<00:36, 9441.95it/s] 14%|█▍        | 56322/400000 [00:06<00:36, 9355.98it/s] 14%|█▍        | 57259/400000 [00:06<00:36, 9336.85it/s] 15%|█▍        | 58204/400000 [00:06<00:36, 9368.09it/s] 15%|█▍        | 59142/400000 [00:06<00:36, 9262.88it/s] 15%|█▌        | 60069/400000 [00:06<00:36, 9189.46it/s] 15%|█▌        | 60989/400000 [00:06<00:37, 9094.25it/s] 15%|█▌        | 61899/400000 [00:06<00:37, 8964.64it/s] 16%|█▌        | 62816/400000 [00:06<00:37, 9024.86it/s] 16%|█▌        | 63761/400000 [00:06<00:36, 9146.42it/s] 16%|█▌        | 64723/400000 [00:06<00:36, 9281.14it/s] 16%|█▋        | 65656/400000 [00:07<00:35, 9293.85it/s] 17%|█▋        | 66629/400000 [00:07<00:35, 9418.06it/s] 17%|█▋        | 67576/400000 [00:07<00:35, 9432.24it/s] 17%|█▋        | 68520/400000 [00:07<00:35, 9319.78it/s] 17%|█▋        | 69483/400000 [00:07<00:35, 9409.30it/s] 18%|█▊        | 70426/400000 [00:07<00:35, 9413.09it/s] 18%|█▊        | 71368/400000 [00:07<00:35, 9377.28it/s] 18%|█▊        | 72307/400000 [00:07<00:35, 9277.66it/s] 18%|█▊        | 73236/400000 [00:07<00:35, 9212.73it/s] 19%|█▊        | 74158/400000 [00:07<00:35, 9096.07it/s] 19%|█▉        | 75069/400000 [00:08<00:35, 9048.10it/s] 19%|█▉        | 75979/400000 [00:08<00:35, 9063.59it/s] 19%|█▉        | 76886/400000 [00:08<00:35, 9055.96it/s] 19%|█▉        | 77792/400000 [00:08<00:35, 9015.28it/s] 20%|█▉        | 78737/400000 [00:08<00:35, 9139.78it/s] 20%|█▉        | 79659/400000 [00:08<00:34, 9162.23it/s] 20%|██        | 80590/400000 [00:08<00:34, 9203.90it/s] 20%|██        | 81561/400000 [00:08<00:34, 9348.15it/s] 21%|██        | 82517/400000 [00:08<00:33, 9410.28it/s] 21%|██        | 83468/400000 [00:08<00:33, 9437.96it/s] 21%|██        | 84413/400000 [00:09<00:33, 9339.13it/s] 21%|██▏       | 85348/400000 [00:09<00:33, 9283.33it/s] 22%|██▏       | 86277/400000 [00:09<00:33, 9227.39it/s] 22%|██▏       | 87201/400000 [00:09<00:33, 9207.71it/s] 22%|██▏       | 88164/400000 [00:09<00:33, 9329.52it/s] 22%|██▏       | 89098/400000 [00:09<00:33, 9282.76it/s] 23%|██▎       | 90059/400000 [00:09<00:33, 9378.33it/s] 23%|██▎       | 91014/400000 [00:09<00:32, 9427.30it/s] 23%|██▎       | 91977/400000 [00:09<00:32, 9485.04it/s] 23%|██▎       | 92926/400000 [00:10<00:33, 9252.17it/s] 23%|██▎       | 93853/400000 [00:10<00:33, 9125.29it/s] 24%|██▎       | 94808/400000 [00:10<00:33, 9247.69it/s] 24%|██▍       | 95764/400000 [00:10<00:32, 9338.93it/s] 24%|██▍       | 96700/400000 [00:10<00:32, 9326.67it/s] 24%|██▍       | 97634/400000 [00:10<00:32, 9278.76it/s] 25%|██▍       | 98563/400000 [00:10<00:32, 9169.52it/s] 25%|██▍       | 99492/400000 [00:10<00:32, 9203.67it/s] 25%|██▌       | 100446/400000 [00:10<00:32, 9301.20it/s] 25%|██▌       | 101411/400000 [00:10<00:31, 9401.91it/s] 26%|██▌       | 102352/400000 [00:11<00:31, 9387.13it/s] 26%|██▌       | 103292/400000 [00:11<00:32, 9129.02it/s] 26%|██▌       | 104223/400000 [00:11<00:32, 9180.17it/s] 26%|██▋       | 105179/400000 [00:11<00:31, 9289.69it/s] 27%|██▋       | 106118/400000 [00:11<00:31, 9317.89it/s] 27%|██▋       | 107051/400000 [00:11<00:31, 9242.86it/s] 27%|██▋       | 107977/400000 [00:11<00:31, 9212.89it/s] 27%|██▋       | 108899/400000 [00:11<00:31, 9105.43it/s] 27%|██▋       | 109811/400000 [00:11<00:32, 9023.06it/s] 28%|██▊       | 110714/400000 [00:11<00:32, 8987.94it/s] 28%|██▊       | 111623/400000 [00:12<00:31, 9017.59it/s] 28%|██▊       | 112531/400000 [00:12<00:31, 9033.66it/s] 28%|██▊       | 113459/400000 [00:12<00:31, 9106.12it/s] 29%|██▊       | 114395/400000 [00:12<00:31, 9178.25it/s] 29%|██▉       | 115349/400000 [00:12<00:30, 9283.53it/s] 29%|██▉       | 116330/400000 [00:12<00:30, 9433.29it/s] 29%|██▉       | 117275/400000 [00:12<00:29, 9429.64it/s] 30%|██▉       | 118219/400000 [00:12<00:29, 9406.08it/s] 30%|██▉       | 119191/400000 [00:12<00:29, 9495.79it/s] 30%|███       | 120142/400000 [00:12<00:29, 9481.29it/s] 30%|███       | 121112/400000 [00:13<00:29, 9543.03it/s] 31%|███       | 122067/400000 [00:13<00:29, 9434.07it/s] 31%|███       | 123011/400000 [00:13<00:29, 9344.46it/s] 31%|███       | 123947/400000 [00:13<00:30, 9045.01it/s] 31%|███       | 124854/400000 [00:13<00:30, 8938.48it/s] 31%|███▏      | 125750/400000 [00:13<00:30, 8911.25it/s] 32%|███▏      | 126677/400000 [00:13<00:30, 9013.83it/s] 32%|███▏      | 127624/400000 [00:13<00:29, 9144.24it/s] 32%|███▏      | 128580/400000 [00:13<00:29, 9263.67it/s] 32%|███▏      | 129542/400000 [00:13<00:28, 9367.46it/s] 33%|███▎      | 130499/400000 [00:14<00:28, 9425.34it/s] 33%|███▎      | 131443/400000 [00:14<00:29, 9240.72it/s] 33%|███▎      | 132384/400000 [00:14<00:28, 9289.12it/s] 33%|███▎      | 133314/400000 [00:14<00:29, 9181.13it/s] 34%|███▎      | 134268/400000 [00:14<00:28, 9284.68it/s] 34%|███▍      | 135198/400000 [00:14<00:28, 9218.95it/s] 34%|███▍      | 136121/400000 [00:14<00:28, 9119.33it/s] 34%|███▍      | 137056/400000 [00:14<00:28, 9184.90it/s] 35%|███▍      | 138017/400000 [00:14<00:28, 9306.57it/s] 35%|███▍      | 138951/400000 [00:14<00:28, 9315.49it/s] 35%|███▍      | 139884/400000 [00:15<00:27, 9311.83it/s] 35%|███▌      | 140816/400000 [00:15<00:28, 9235.54it/s] 35%|███▌      | 141792/400000 [00:15<00:27, 9385.85it/s] 36%|███▌      | 142732/400000 [00:15<00:27, 9332.55it/s] 36%|███▌      | 143682/400000 [00:15<00:27, 9379.83it/s] 36%|███▌      | 144621/400000 [00:15<00:27, 9380.29it/s] 36%|███▋      | 145560/400000 [00:15<00:27, 9338.43it/s] 37%|███▋      | 146508/400000 [00:15<00:27, 9379.15it/s] 37%|███▋      | 147449/400000 [00:15<00:26, 9387.71it/s] 37%|███▋      | 148388/400000 [00:16<00:27, 9267.89it/s] 37%|███▋      | 149316/400000 [00:16<00:27, 9170.43it/s] 38%|███▊      | 150234/400000 [00:16<00:27, 9057.70it/s] 38%|███▊      | 151153/400000 [00:16<00:27, 9094.28it/s] 38%|███▊      | 152063/400000 [00:16<00:27, 9064.63it/s] 38%|███▊      | 152970/400000 [00:16<00:27, 8926.71it/s] 38%|███▊      | 153868/400000 [00:16<00:27, 8940.48it/s] 39%|███▊      | 154763/400000 [00:16<00:27, 8914.72it/s] 39%|███▉      | 155655/400000 [00:16<00:27, 8814.21it/s] 39%|███▉      | 156586/400000 [00:16<00:27, 8956.85it/s] 39%|███▉      | 157541/400000 [00:17<00:26, 9125.31it/s] 40%|███▉      | 158466/400000 [00:17<00:26, 9161.44it/s] 40%|███▉      | 159384/400000 [00:17<00:26, 9152.81it/s] 40%|████      | 160301/400000 [00:17<00:26, 9098.88it/s] 40%|████      | 161212/400000 [00:17<00:26, 9039.96it/s] 41%|████      | 162117/400000 [00:17<00:26, 8982.19it/s] 41%|████      | 163024/400000 [00:17<00:26, 9006.95it/s] 41%|████      | 163926/400000 [00:17<00:26, 8938.52it/s] 41%|████      | 164824/400000 [00:17<00:26, 8950.12it/s] 41%|████▏     | 165720/400000 [00:17<00:26, 8937.37it/s] 42%|████▏     | 166628/400000 [00:18<00:25, 8977.64it/s] 42%|████▏     | 167532/400000 [00:18<00:25, 8993.95it/s] 42%|████▏     | 168432/400000 [00:18<00:25, 8947.24it/s] 42%|████▏     | 169327/400000 [00:18<00:25, 8933.77it/s] 43%|████▎     | 170221/400000 [00:18<00:26, 8815.26it/s] 43%|████▎     | 171119/400000 [00:18<00:25, 8863.90it/s] 43%|████▎     | 172039/400000 [00:18<00:25, 8960.65it/s] 43%|████▎     | 172936/400000 [00:18<00:25, 8951.07it/s] 43%|████▎     | 173869/400000 [00:18<00:24, 9060.54it/s] 44%|████▎     | 174816/400000 [00:18<00:24, 9176.93it/s] 44%|████▍     | 175735/400000 [00:19<00:24, 9080.00it/s] 44%|████▍     | 176673/400000 [00:19<00:24, 9165.46it/s] 44%|████▍     | 177591/400000 [00:19<00:24, 9018.30it/s] 45%|████▍     | 178494/400000 [00:19<00:24, 8973.62it/s] 45%|████▍     | 179449/400000 [00:19<00:24, 9137.93it/s] 45%|████▌     | 180376/400000 [00:19<00:23, 9176.87it/s] 45%|████▌     | 181295/400000 [00:19<00:24, 9093.10it/s] 46%|████▌     | 182227/400000 [00:19<00:23, 9158.20it/s] 46%|████▌     | 183179/400000 [00:19<00:23, 9262.26it/s] 46%|████▌     | 184106/400000 [00:19<00:23, 9233.19it/s] 46%|████▋     | 185030/400000 [00:20<00:23, 9038.55it/s] 46%|████▋     | 185936/400000 [00:20<00:23, 8991.23it/s] 47%|████▋     | 186837/400000 [00:20<00:24, 8824.76it/s] 47%|████▋     | 187732/400000 [00:20<00:23, 8859.90it/s] 47%|████▋     | 188619/400000 [00:20<00:24, 8773.64it/s] 47%|████▋     | 189550/400000 [00:20<00:23, 8925.76it/s] 48%|████▊     | 190497/400000 [00:20<00:23, 9081.16it/s] 48%|████▊     | 191436/400000 [00:20<00:22, 9169.03it/s] 48%|████▊     | 192355/400000 [00:20<00:24, 8621.84it/s] 48%|████▊     | 193239/400000 [00:20<00:23, 8685.29it/s] 49%|████▊     | 194166/400000 [00:21<00:23, 8844.56it/s] 49%|████▉     | 195100/400000 [00:21<00:22, 8985.21it/s] 49%|████▉     | 196016/400000 [00:21<00:22, 9035.44it/s] 49%|████▉     | 196923/400000 [00:21<00:22, 8992.29it/s] 49%|████▉     | 197825/400000 [00:21<00:22, 8974.01it/s] 50%|████▉     | 198736/400000 [00:21<00:22, 9012.05it/s] 50%|████▉     | 199639/400000 [00:21<00:22, 8864.67it/s] 50%|█████     | 200527/400000 [00:21<00:22, 8843.22it/s] 50%|█████     | 201457/400000 [00:21<00:22, 8975.41it/s] 51%|█████     | 202357/400000 [00:22<00:22, 8982.00it/s] 51%|█████     | 203289/400000 [00:22<00:21, 9078.29it/s] 51%|█████     | 204252/400000 [00:22<00:21, 9236.39it/s] 51%|█████▏    | 205177/400000 [00:22<00:21, 9067.84it/s] 52%|█████▏    | 206127/400000 [00:22<00:21, 9191.63it/s] 52%|█████▏    | 207100/400000 [00:22<00:20, 9345.53it/s] 52%|█████▏    | 208037/400000 [00:22<00:20, 9209.74it/s] 52%|█████▏    | 208960/400000 [00:22<00:21, 9088.19it/s] 52%|█████▏    | 209871/400000 [00:22<00:21, 9038.12it/s] 53%|█████▎    | 210780/400000 [00:22<00:20, 9052.38it/s] 53%|█████▎    | 211716/400000 [00:23<00:20, 9140.75it/s] 53%|█████▎    | 212660/400000 [00:23<00:20, 9226.77it/s] 53%|█████▎    | 213584/400000 [00:23<00:20, 9078.38it/s] 54%|█████▎    | 214493/400000 [00:23<00:20, 9056.80it/s] 54%|█████▍    | 215400/400000 [00:23<00:20, 9016.21it/s] 54%|█████▍    | 216344/400000 [00:23<00:20, 9136.92it/s] 54%|█████▍    | 217296/400000 [00:23<00:19, 9246.94it/s] 55%|█████▍    | 218222/400000 [00:23<00:19, 9247.08it/s] 55%|█████▍    | 219148/400000 [00:23<00:19, 9082.81it/s] 55%|█████▌    | 220099/400000 [00:23<00:19, 9205.33it/s] 55%|█████▌    | 221044/400000 [00:24<00:19, 9276.34it/s] 56%|█████▌    | 222011/400000 [00:24<00:18, 9390.25it/s] 56%|█████▌    | 222952/400000 [00:24<00:18, 9390.29it/s] 56%|█████▌    | 223892/400000 [00:24<00:19, 9209.05it/s] 56%|█████▌    | 224815/400000 [00:24<00:19, 9101.74it/s] 56%|█████▋    | 225746/400000 [00:24<00:19, 9161.82it/s] 57%|█████▋    | 226726/400000 [00:24<00:18, 9344.34it/s] 57%|█████▋    | 227691/400000 [00:24<00:18, 9432.26it/s] 57%|█████▋    | 228636/400000 [00:24<00:18, 9351.77it/s] 57%|█████▋    | 229573/400000 [00:24<00:18, 9280.23it/s] 58%|█████▊    | 230502/400000 [00:25<00:18, 9167.69it/s] 58%|█████▊    | 231424/400000 [00:25<00:18, 9182.00it/s] 58%|█████▊    | 232382/400000 [00:25<00:18, 9295.86it/s] 58%|█████▊    | 233313/400000 [00:25<00:17, 9280.59it/s] 59%|█████▊    | 234242/400000 [00:25<00:18, 9125.18it/s] 59%|█████▉    | 235156/400000 [00:25<00:18, 9002.87it/s] 59%|█████▉    | 236109/400000 [00:25<00:17, 9152.95it/s] 59%|█████▉    | 237062/400000 [00:25<00:17, 9261.86it/s] 60%|█████▉    | 238012/400000 [00:25<00:17, 9331.94it/s] 60%|█████▉    | 238969/400000 [00:25<00:17, 9400.79it/s] 60%|█████▉    | 239910/400000 [00:26<00:17, 9330.61it/s] 60%|██████    | 240844/400000 [00:26<00:17, 8886.22it/s] 60%|██████    | 241771/400000 [00:26<00:17, 8996.88it/s] 61%|██████    | 242681/400000 [00:26<00:17, 9026.08it/s] 61%|██████    | 243643/400000 [00:26<00:17, 9195.42it/s] 61%|██████    | 244580/400000 [00:26<00:16, 9245.25it/s] 61%|██████▏   | 245507/400000 [00:26<00:16, 9152.91it/s] 62%|██████▏   | 246424/400000 [00:26<00:16, 9079.95it/s] 62%|██████▏   | 247334/400000 [00:26<00:17, 8937.12it/s] 62%|██████▏   | 248232/400000 [00:27<00:16, 8949.61it/s] 62%|██████▏   | 249128/400000 [00:27<00:16, 8940.22it/s] 63%|██████▎   | 250041/400000 [00:27<00:16, 8996.14it/s] 63%|██████▎   | 250942/400000 [00:27<00:16, 8880.33it/s] 63%|██████▎   | 251893/400000 [00:27<00:16, 9059.09it/s] 63%|██████▎   | 252864/400000 [00:27<00:15, 9244.83it/s] 63%|██████▎   | 253831/400000 [00:27<00:15, 9366.14it/s] 64%|██████▎   | 254770/400000 [00:27<00:15, 9331.70it/s] 64%|██████▍   | 255727/400000 [00:27<00:15, 9401.33it/s] 64%|██████▍   | 256669/400000 [00:27<00:15, 9379.68it/s] 64%|██████▍   | 257618/400000 [00:28<00:15, 9410.93it/s] 65%|██████▍   | 258577/400000 [00:28<00:14, 9463.15it/s] 65%|██████▍   | 259524/400000 [00:28<00:15, 9287.76it/s] 65%|██████▌   | 260454/400000 [00:28<00:15, 9179.67it/s] 65%|██████▌   | 261373/400000 [00:28<00:15, 9082.10it/s] 66%|██████▌   | 262283/400000 [00:28<00:15, 8965.12it/s] 66%|██████▌   | 263194/400000 [00:28<00:15, 9006.22it/s] 66%|██████▌   | 264152/400000 [00:28<00:14, 9169.75it/s] 66%|██████▋   | 265093/400000 [00:28<00:14, 9238.33it/s] 67%|██████▋   | 266024/400000 [00:28<00:14, 9259.26it/s] 67%|██████▋   | 266983/400000 [00:29<00:14, 9355.65it/s] 67%|██████▋   | 267955/400000 [00:29<00:13, 9461.86it/s] 67%|██████▋   | 268902/400000 [00:29<00:13, 9460.66it/s] 67%|██████▋   | 269860/400000 [00:29<00:13, 9496.06it/s] 68%|██████▊   | 270811/400000 [00:29<00:13, 9392.29it/s] 68%|██████▊   | 271751/400000 [00:29<00:13, 9345.12it/s] 68%|██████▊   | 272686/400000 [00:29<00:13, 9175.77it/s] 68%|██████▊   | 273605/400000 [00:29<00:13, 9099.39it/s] 69%|██████▊   | 274557/400000 [00:29<00:13, 9218.67it/s] 69%|██████▉   | 275483/400000 [00:29<00:13, 9228.74it/s] 69%|██████▉   | 276432/400000 [00:30<00:13, 9305.31it/s] 69%|██████▉   | 277364/400000 [00:30<00:13, 9099.28it/s] 70%|██████▉   | 278276/400000 [00:30<00:13, 8976.28it/s] 70%|██████▉   | 279213/400000 [00:30<00:13, 9089.87it/s] 70%|███████   | 280132/400000 [00:30<00:13, 9118.20it/s] 70%|███████   | 281053/400000 [00:30<00:13, 9144.44it/s] 70%|███████   | 281990/400000 [00:30<00:12, 9210.13it/s] 71%|███████   | 282959/400000 [00:30<00:12, 9346.81it/s] 71%|███████   | 283908/400000 [00:30<00:12, 9389.01it/s] 71%|███████   | 284848/400000 [00:30<00:12, 9195.49it/s] 71%|███████▏  | 285769/400000 [00:31<00:12, 9074.72it/s] 72%|███████▏  | 286678/400000 [00:31<00:12, 9066.97it/s] 72%|███████▏  | 287629/400000 [00:31<00:12, 9194.13it/s] 72%|███████▏  | 288578/400000 [00:31<00:12, 9279.22it/s] 72%|███████▏  | 289507/400000 [00:31<00:11, 9227.50it/s] 73%|███████▎  | 290472/400000 [00:31<00:11, 9348.21it/s] 73%|███████▎  | 291430/400000 [00:31<00:11, 9415.21it/s] 73%|███████▎  | 292397/400000 [00:31<00:11, 9488.27it/s] 73%|███████▎  | 293347/400000 [00:31<00:11, 9383.13it/s] 74%|███████▎  | 294287/400000 [00:31<00:11, 9257.56it/s] 74%|███████▍  | 295251/400000 [00:32<00:11, 9368.37it/s] 74%|███████▍  | 296210/400000 [00:32<00:11, 9432.15it/s] 74%|███████▍  | 297154/400000 [00:32<00:11, 9186.55it/s] 75%|███████▍  | 298075/400000 [00:32<00:11, 9069.52it/s] 75%|███████▍  | 298984/400000 [00:32<00:11, 9017.93it/s] 75%|███████▍  | 299942/400000 [00:32<00:10, 9178.04it/s] 75%|███████▌  | 300888/400000 [00:32<00:10, 9258.72it/s] 75%|███████▌  | 301855/400000 [00:32<00:10, 9376.62it/s] 76%|███████▌  | 302821/400000 [00:32<00:10, 9457.83it/s] 76%|███████▌  | 303788/400000 [00:33<00:10, 9519.53it/s] 76%|███████▌  | 304754/400000 [00:33<00:09, 9558.27it/s] 76%|███████▋  | 305711/400000 [00:33<00:09, 9523.94it/s] 77%|███████▋  | 306664/400000 [00:33<00:09, 9514.58it/s] 77%|███████▋  | 307630/400000 [00:33<00:09, 9557.62it/s] 77%|███████▋  | 308587/400000 [00:33<00:09, 9419.29it/s] 77%|███████▋  | 309530/400000 [00:33<00:09, 9282.28it/s] 78%|███████▊  | 310460/400000 [00:33<00:09, 9174.48it/s] 78%|███████▊  | 311391/400000 [00:33<00:09, 9212.57it/s] 78%|███████▊  | 312348/400000 [00:33<00:09, 9316.42it/s] 78%|███████▊  | 313281/400000 [00:34<00:09, 9208.67it/s] 79%|███████▊  | 314219/400000 [00:34<00:09, 9257.49it/s] 79%|███████▉  | 315165/400000 [00:34<00:09, 9314.80it/s] 79%|███████▉  | 316117/400000 [00:34<00:08, 9375.42it/s] 79%|███████▉  | 317067/400000 [00:34<00:08, 9411.69it/s] 80%|███████▉  | 318009/400000 [00:34<00:08, 9377.82it/s] 80%|███████▉  | 318967/400000 [00:34<00:08, 9435.11it/s] 80%|███████▉  | 319924/400000 [00:34<00:08, 9473.73it/s] 80%|████████  | 320898/400000 [00:34<00:08, 9549.81it/s] 80%|████████  | 321856/400000 [00:34<00:08, 9556.28it/s] 81%|████████  | 322812/400000 [00:35<00:08, 9397.21it/s] 81%|████████  | 323753/400000 [00:35<00:08, 9347.67it/s] 81%|████████  | 324689/400000 [00:35<00:08, 9164.06it/s] 81%|████████▏ | 325607/400000 [00:35<00:08, 9089.29it/s] 82%|████████▏ | 326550/400000 [00:35<00:07, 9187.86it/s] 82%|████████▏ | 327470/400000 [00:35<00:07, 9191.04it/s] 82%|████████▏ | 328422/400000 [00:35<00:07, 9286.75it/s] 82%|████████▏ | 329374/400000 [00:35<00:07, 9354.60it/s] 83%|████████▎ | 330341/400000 [00:35<00:07, 9444.25it/s] 83%|████████▎ | 331294/400000 [00:35<00:07, 9468.58it/s] 83%|████████▎ | 332242/400000 [00:36<00:07, 9379.71it/s] 83%|████████▎ | 333181/400000 [00:36<00:07, 9266.77it/s] 84%|████████▎ | 334109/400000 [00:36<00:07, 9268.88it/s] 84%|████████▍ | 335037/400000 [00:36<00:07, 9214.18it/s] 84%|████████▍ | 335959/400000 [00:36<00:06, 9179.87it/s] 84%|████████▍ | 336878/400000 [00:36<00:06, 9071.08it/s] 84%|████████▍ | 337789/400000 [00:36<00:06, 9079.94it/s] 85%|████████▍ | 338698/400000 [00:36<00:06, 9073.66it/s] 85%|████████▍ | 339660/400000 [00:36<00:06, 9229.75it/s] 85%|████████▌ | 340584/400000 [00:36<00:06, 9038.36it/s] 85%|████████▌ | 341490/400000 [00:37<00:06, 8989.05it/s] 86%|████████▌ | 342452/400000 [00:37<00:06, 9169.27it/s] 86%|████████▌ | 343429/400000 [00:37<00:06, 9339.31it/s] 86%|████████▌ | 344365/400000 [00:37<00:06, 9062.47it/s] 86%|████████▋ | 345282/400000 [00:37<00:06, 9094.11it/s] 87%|████████▋ | 346222/400000 [00:37<00:05, 9183.48it/s] 87%|████████▋ | 347158/400000 [00:37<00:05, 9234.44it/s] 87%|████████▋ | 348083/400000 [00:37<00:05, 9187.49it/s] 87%|████████▋ | 349003/400000 [00:37<00:05, 9148.18it/s] 87%|████████▋ | 349919/400000 [00:37<00:05, 8991.46it/s] 88%|████████▊ | 350820/400000 [00:38<00:05, 8930.84it/s] 88%|████████▊ | 351715/400000 [00:38<00:05, 8935.07it/s] 88%|████████▊ | 352610/400000 [00:38<00:05, 8870.23it/s] 88%|████████▊ | 353510/400000 [00:38<00:05, 8906.33it/s] 89%|████████▊ | 354432/400000 [00:38<00:05, 8997.73it/s] 89%|████████▉ | 355333/400000 [00:38<00:05, 8904.80it/s] 89%|████████▉ | 356225/400000 [00:38<00:04, 8905.55it/s] 89%|████████▉ | 357170/400000 [00:38<00:04, 9060.55it/s] 90%|████████▉ | 358145/400000 [00:38<00:04, 9255.45it/s] 90%|████████▉ | 359092/400000 [00:38<00:04, 9317.59it/s] 90%|█████████ | 360026/400000 [00:39<00:04, 9287.71it/s] 90%|█████████ | 360988/400000 [00:39<00:04, 9383.97it/s] 90%|█████████ | 361928/400000 [00:39<00:04, 9383.03it/s] 91%|█████████ | 362867/400000 [00:39<00:03, 9352.91it/s] 91%|█████████ | 363803/400000 [00:39<00:03, 9214.44it/s] 91%|█████████ | 364726/400000 [00:39<00:03, 9192.29it/s] 91%|█████████▏| 365670/400000 [00:39<00:03, 9262.77it/s] 92%|█████████▏| 366635/400000 [00:39<00:03, 9374.53it/s] 92%|█████████▏| 367603/400000 [00:39<00:03, 9462.41it/s] 92%|█████████▏| 368565/400000 [00:40<00:03, 9506.96it/s] 92%|█████████▏| 369517/400000 [00:40<00:03, 9391.12it/s] 93%|█████████▎| 370476/400000 [00:40<00:03, 9432.85it/s] 93%|█████████▎| 371420/400000 [00:40<00:03, 9286.13it/s] 93%|█████████▎| 372350/400000 [00:40<00:02, 9261.40it/s] 93%|█████████▎| 373277/400000 [00:40<00:02, 9159.31it/s] 94%|█████████▎| 374194/400000 [00:40<00:02, 9145.96it/s] 94%|█████████▍| 375148/400000 [00:40<00:02, 9259.03it/s] 94%|█████████▍| 376118/400000 [00:40<00:02, 9386.77it/s] 94%|█████████▍| 377088/400000 [00:40<00:02, 9476.47it/s] 95%|█████████▍| 378045/400000 [00:41<00:02, 9503.39it/s] 95%|█████████▍| 378996/400000 [00:41<00:02, 9348.21it/s] 95%|█████████▍| 379932/400000 [00:41<00:02, 9226.52it/s] 95%|█████████▌| 380857/400000 [00:41<00:02, 9232.35it/s] 95%|█████████▌| 381802/400000 [00:41<00:01, 9294.14it/s] 96%|█████████▌| 382754/400000 [00:41<00:01, 9357.86it/s] 96%|█████████▌| 383691/400000 [00:41<00:01, 9025.19it/s] 96%|█████████▌| 384602/400000 [00:41<00:01, 9049.23it/s] 96%|█████████▋| 385509/400000 [00:41<00:01, 8974.05it/s] 97%|█████████▋| 386408/400000 [00:41<00:01, 8762.62it/s] 97%|█████████▋| 387287/400000 [00:42<00:01, 8587.02it/s] 97%|█████████▋| 388192/400000 [00:42<00:01, 8714.51it/s] 97%|█████████▋| 389157/400000 [00:42<00:01, 8974.02it/s] 98%|█████████▊| 390058/400000 [00:42<00:01, 8915.46it/s] 98%|█████████▊| 391006/400000 [00:42<00:00, 9075.86it/s] 98%|█████████▊| 391966/400000 [00:42<00:00, 9224.90it/s] 98%|█████████▊| 392891/400000 [00:42<00:00, 9228.45it/s] 98%|█████████▊| 393837/400000 [00:42<00:00, 9294.30it/s] 99%|█████████▊| 394808/400000 [00:42<00:00, 9412.25it/s] 99%|█████████▉| 395772/400000 [00:42<00:00, 9478.80it/s] 99%|█████████▉| 396734/400000 [00:43<00:00, 9519.10it/s] 99%|█████████▉| 397687/400000 [00:43<00:00, 9379.53it/s]100%|█████████▉| 398626/400000 [00:43<00:00, 9249.80it/s]100%|█████████▉| 399553/400000 [00:43<00:00, 9231.27it/s]100%|█████████▉| 399999/400000 [00:43<00:00, 9211.71it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fb7ba464f98> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011025493833981189 	 Accuracy: 54
Train Epoch: 1 	 Loss: 0.011509424866641246 	 Accuracy: 50

  model saves at 50% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15652 out of table with 15568 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15652 out of table with 15568 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-14 16:25:28.323118: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 16:25:28.326840: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-14 16:25:28.327837: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56289100b150 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 16:25:28.328093: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fb7c5fd5f98> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 8.0346 - accuracy: 0.4760
 2000/25000 [=>............................] - ETA: 8s - loss: 8.0500 - accuracy: 0.4750 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7791 - accuracy: 0.4927
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7433 - accuracy: 0.4950
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7065 - accuracy: 0.4974
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6845 - accuracy: 0.4988
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6403 - accuracy: 0.5017
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5900 - accuracy: 0.5050
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6343 - accuracy: 0.5021
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6007 - accuracy: 0.5043
11000/25000 [============>.................] - ETA: 3s - loss: 7.6095 - accuracy: 0.5037
12000/25000 [=============>................] - ETA: 3s - loss: 7.5938 - accuracy: 0.5048
13000/25000 [==============>...............] - ETA: 2s - loss: 7.5770 - accuracy: 0.5058
14000/25000 [===============>..............] - ETA: 2s - loss: 7.5856 - accuracy: 0.5053
15000/25000 [=================>............] - ETA: 2s - loss: 7.6012 - accuracy: 0.5043
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6139 - accuracy: 0.5034
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6351 - accuracy: 0.5021
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6649 - accuracy: 0.5001
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6690 - accuracy: 0.4998
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6712 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6805 - accuracy: 0.4991
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6799 - accuracy: 0.4991
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6593 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6609 - accuracy: 0.5004
25000/25000 [==============================] - 7s 262us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fb72af0ab38> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fb72c0ee128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 998ms/step - loss: 1.5625 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.5005 - val_crf_viterbi_accuracy: 0.0000e+00

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

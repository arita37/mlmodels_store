
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f2669008f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 17:13:20.565829
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 17:13:20.570440
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 17:13:20.574265
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 17:13:20.577998
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f2674dd2400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354760.7188
Epoch 2/10

1/1 [==============================] - 0s 105ms/step - loss: 278213.5000
Epoch 3/10

1/1 [==============================] - 0s 132ms/step - loss: 175342.5938
Epoch 4/10

1/1 [==============================] - 0s 117ms/step - loss: 99458.5469
Epoch 5/10

1/1 [==============================] - 0s 102ms/step - loss: 53748.4531
Epoch 6/10

1/1 [==============================] - 0s 106ms/step - loss: 29371.6777
Epoch 7/10

1/1 [==============================] - 0s 105ms/step - loss: 16919.3398
Epoch 8/10

1/1 [==============================] - 0s 110ms/step - loss: 10797.4453
Epoch 9/10

1/1 [==============================] - 0s 105ms/step - loss: 7443.4214
Epoch 10/10

1/1 [==============================] - 0s 105ms/step - loss: 5540.0439

  #### Inference Need return ypred, ytrue ######################### 
[[ 5.63024879e-01 -2.71677792e-01 -2.19684049e-01 -1.05965948e+00
   1.17512274e+00 -7.07265258e-01  7.39162385e-01 -1.43926024e-01
   8.34589005e-01 -5.70529699e-01  1.11814424e-01 -1.39123392e+00
   1.15745127e+00 -9.28345084e-01  2.05264941e-01 -5.17383039e-01
   9.17912364e-01 -4.82267737e-02 -8.41852248e-01 -5.97909451e-01
  -9.20238137e-01 -1.08856022e-01  8.44733357e-01  1.09879184e+00
   1.35579836e+00  1.40759617e-01  1.43107963e+00 -1.76167011e+00
  -6.82004333e-01  2.02102214e-03  7.92145014e-01 -7.88648963e-01
  -1.54985905e+00  1.28515363e+00 -1.00768387e-01  6.67177200e-01
  -6.76515758e-01 -5.12214124e-01 -1.39948606e+00  1.12218380e-01
   5.91881424e-02  4.96255994e-01 -1.10812879e+00 -9.87313509e-01
  -2.75630474e-01  4.89038140e-01 -1.23695090e-01 -1.20394301e+00
  -8.41260791e-01  3.86831343e-01  4.73966628e-01 -4.84512866e-01
  -1.32974339e+00  1.13659811e+00 -7.09392354e-02 -2.57947946e+00
   3.75341326e-01  1.81205606e+00 -4.86452073e-01 -1.17542124e+00
   5.88120401e-01  7.90834475e+00  9.66492462e+00  9.28403378e+00
   9.57756996e+00  8.83397007e+00  8.80445099e+00  1.01799669e+01
   8.08246803e+00  8.63936043e+00  1.05038452e+01  1.06502476e+01
   9.34445286e+00  9.51635742e+00  1.05185108e+01  8.37292767e+00
   7.13807440e+00  9.08706760e+00  9.29476452e+00  7.46012640e+00
   9.20978928e+00  8.31417942e+00  7.20089912e+00  8.81676579e+00
   1.02969465e+01  1.05137072e+01  8.82944584e+00  8.60613441e+00
   8.22780800e+00  1.08234606e+01  9.46709061e+00  8.59993076e+00
   8.48742962e+00  9.36739635e+00  1.14158869e+01  9.06697464e+00
   9.02923775e+00  8.30772781e+00  8.97680855e+00  9.07080650e+00
   7.29849482e+00  9.15067101e+00  8.11950302e+00  9.18920326e+00
   8.55603123e+00  9.17749691e+00  7.80223036e+00  8.58836079e+00
   9.12241077e+00  7.45068598e+00  1.05797615e+01  9.55108261e+00
   9.60787868e+00  8.51107883e+00  8.62617493e+00  9.61909294e+00
   7.70941830e+00  1.05400858e+01  9.80563927e+00  7.74674606e+00
  -4.56127465e-01  1.35018694e+00  3.43844503e-01 -4.28109691e-02
  -1.90943539e+00 -7.59774566e-01 -2.74490714e-02  3.61916780e-01
  -5.05802631e-02  1.59567222e-01 -2.99680859e-01  3.81460160e-01
  -1.11148810e+00  1.21710360e-01 -1.18906581e+00  3.04683208e-01
  -6.05810165e-01  4.42012250e-01  1.79760361e+00  1.17648780e+00
  -1.42437541e+00  1.05382705e+00  4.11725044e-01  2.34132457e+00
  -6.26976967e-01 -1.96228886e+00  6.25192583e-01  8.12974751e-01
   2.16770363e+00  2.20667124e+00  6.52166963e-01 -1.49678946e+00
  -1.10937846e+00 -1.77740860e+00  2.10291147e+00  8.81778955e-01
   7.06742764e-01 -6.53138578e-01  9.78604436e-01 -8.79080594e-01
  -2.46885872e+00  5.11198223e-01 -6.60965294e-02 -1.49498820e+00
   2.94471323e-01  8.00298750e-01 -1.99095905e-01 -1.63373268e+00
   3.89766872e-01 -2.08568048e+00 -5.23725510e-01 -1.29314616e-01
  -5.72991073e-01  5.96294761e-01  2.17729926e+00  9.58974957e-01
  -1.90721834e+00  8.30959320e-01  6.16895258e-01 -1.96653306e-01
   8.62267852e-01  1.04180777e+00  6.79111242e-01  2.27892375e+00
   2.20837927e+00  1.58312583e+00  1.34546471e+00  1.61203086e-01
   8.04097414e-01  3.42576861e-01  1.70347810e-01  2.32270241e+00
   5.98741055e-01  1.47297883e+00  3.35545242e-01  1.05615294e+00
   5.15641034e-01  2.93467069e+00  3.67202044e-01  1.34000754e+00
   4.96240854e-01  2.31844902e+00  1.74682033e+00  6.02723837e-01
   4.30905819e-01  4.80823994e-01  1.22523618e+00  1.19255209e+00
   1.29082978e+00  2.08078432e+00  3.52092385e-01  1.67315841e-01
   1.34512711e+00  2.64464974e-01  8.95492375e-01  1.52439404e+00
   9.66763377e-01  2.01389980e+00  1.45471203e+00  1.42192173e+00
   1.81465960e+00  4.75213945e-01  4.59431291e-01  1.78286004e+00
   1.72921991e+00  2.26587152e+00  2.59047270e+00  1.07478106e+00
   8.05222988e-01  9.86634195e-01  2.25113893e+00  1.22074771e+00
   2.13339746e-01  7.95183778e-02  1.40729141e+00  2.68698549e+00
   2.29887629e+00  7.50453591e-01  6.27973974e-01  4.94954705e-01
   1.12945795e-01  9.11006069e+00  9.70996571e+00  9.23468685e+00
   9.92622852e+00  1.02366419e+01  9.45234108e+00  8.92817497e+00
   1.03061123e+01  7.55069780e+00  1.02181320e+01  8.24352455e+00
   9.16446495e+00  1.04140882e+01  8.60995102e+00  1.00428801e+01
   9.51679420e+00  9.75645256e+00  9.23148251e+00  1.00299282e+01
   8.68839359e+00  8.04112434e+00  9.31914139e+00  8.91972351e+00
   9.51923656e+00  9.48190880e+00  8.60628986e+00  1.00477867e+01
   8.65708351e+00  8.89317989e+00  8.95870590e+00  9.38865757e+00
   8.96506119e+00  8.59844398e+00  1.04287233e+01  1.03759842e+01
   9.29704475e+00  9.40899754e+00  7.98269272e+00  9.46886063e+00
   8.53267670e+00  8.14915371e+00  9.15469646e+00  9.52099323e+00
   1.04169397e+01  9.57489491e+00  9.59185410e+00  8.78460789e+00
   9.51622772e+00  9.62015533e+00  9.99666214e+00  9.10325813e+00
   9.70593739e+00  9.98172951e+00  9.53607845e+00  1.01519575e+01
   9.02865410e+00  9.10705090e+00  8.29857922e+00  8.45100594e+00
   1.26239800e+00  9.85698700e-01  5.99690914e-01  6.86941266e-01
   2.47962046e+00  2.50907373e+00  1.57107329e+00  5.73065400e-01
   1.36439633e+00  7.94952452e-01  9.82218802e-01  1.28519011e+00
   1.17894781e+00  1.06779313e+00  1.68841577e+00  4.25385833e-01
   2.48249197e+00  2.43784487e-01  6.86017871e-01  6.18702352e-01
   1.75865412e+00  1.14728475e+00  4.90681648e-01  2.53720403e+00
   2.58344030e+00  3.30420434e-01  1.61698806e+00  1.91545987e+00
   9.89735425e-01  1.17447710e+00  2.06355333e+00  1.83132291e-01
   5.14183581e-01  4.29529548e-01  3.47044051e-01  1.18577290e+00
   6.25199795e-01  1.71888435e+00  1.40431023e+00  2.63996243e-01
   8.93818855e-01  7.63135016e-01  2.15383387e+00  1.19476759e+00
   1.64806104e+00  2.91378617e+00  2.23988008e+00  9.39394891e-01
   9.06809926e-01  6.77654982e-01  3.44111681e-01  6.42693460e-01
   7.80129969e-01  3.66082907e-01  1.38147306e+00  7.84255803e-01
   2.40658760e-01  1.03756392e+00  2.16201544e+00  1.68570662e+00
  -5.37145376e+00  1.26546659e+01 -1.36728907e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 17:13:30.577197
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    92.628
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 17:13:30.581501
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8608.02
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 17:13:30.585428
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.6545
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 17:13:30.589343
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -769.909
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139802588078488
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139801377919608
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139801377920112
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139801377920616
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139801377921120
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139801377921624

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f26704e10f0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.614537
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.567760
grad_step = 000002, loss = 0.538721
grad_step = 000003, loss = 0.507956
grad_step = 000004, loss = 0.473417
grad_step = 000005, loss = 0.438834
grad_step = 000006, loss = 0.415656
grad_step = 000007, loss = 0.411332
grad_step = 000008, loss = 0.398428
grad_step = 000009, loss = 0.375589
grad_step = 000010, loss = 0.357889
grad_step = 000011, loss = 0.346027
grad_step = 000012, loss = 0.337197
grad_step = 000013, loss = 0.327989
grad_step = 000014, loss = 0.316763
grad_step = 000015, loss = 0.303864
grad_step = 000016, loss = 0.290878
grad_step = 000017, loss = 0.279530
grad_step = 000018, loss = 0.270138
grad_step = 000019, loss = 0.260532
grad_step = 000020, loss = 0.249133
grad_step = 000021, loss = 0.237659
grad_step = 000022, loss = 0.227852
grad_step = 000023, loss = 0.219427
grad_step = 000024, loss = 0.211039
grad_step = 000025, loss = 0.201908
grad_step = 000026, loss = 0.192303
grad_step = 000027, loss = 0.183120
grad_step = 000028, loss = 0.175076
grad_step = 000029, loss = 0.167763
grad_step = 000030, loss = 0.160031
grad_step = 000031, loss = 0.151924
grad_step = 000032, loss = 0.144421
grad_step = 000033, loss = 0.137722
grad_step = 000034, loss = 0.131263
grad_step = 000035, loss = 0.124664
grad_step = 000036, loss = 0.118076
grad_step = 000037, loss = 0.111946
grad_step = 000038, loss = 0.106354
grad_step = 000039, loss = 0.100883
grad_step = 000040, loss = 0.095382
grad_step = 000041, loss = 0.090145
grad_step = 000042, loss = 0.085323
grad_step = 000043, loss = 0.080795
grad_step = 000044, loss = 0.076366
grad_step = 000045, loss = 0.072012
grad_step = 000046, loss = 0.067895
grad_step = 000047, loss = 0.064094
grad_step = 000048, loss = 0.060489
grad_step = 000049, loss = 0.056958
grad_step = 000050, loss = 0.053568
grad_step = 000051, loss = 0.050452
grad_step = 000052, loss = 0.047517
grad_step = 000053, loss = 0.044631
grad_step = 000054, loss = 0.041872
grad_step = 000055, loss = 0.039332
grad_step = 000056, loss = 0.036943
grad_step = 000057, loss = 0.034623
grad_step = 000058, loss = 0.032426
grad_step = 000059, loss = 0.030384
grad_step = 000060, loss = 0.028445
grad_step = 000061, loss = 0.026593
grad_step = 000062, loss = 0.024839
grad_step = 000063, loss = 0.023207
grad_step = 000064, loss = 0.021679
grad_step = 000065, loss = 0.020214
grad_step = 000066, loss = 0.018829
grad_step = 000067, loss = 0.017556
grad_step = 000068, loss = 0.016354
grad_step = 000069, loss = 0.015202
grad_step = 000070, loss = 0.014138
grad_step = 000071, loss = 0.013154
grad_step = 000072, loss = 0.012226
grad_step = 000073, loss = 0.011357
grad_step = 000074, loss = 0.010552
grad_step = 000075, loss = 0.009811
grad_step = 000076, loss = 0.009115
grad_step = 000077, loss = 0.008466
grad_step = 000078, loss = 0.007878
grad_step = 000079, loss = 0.007334
grad_step = 000080, loss = 0.006827
grad_step = 000081, loss = 0.006366
grad_step = 000082, loss = 0.005943
grad_step = 000083, loss = 0.005554
grad_step = 000084, loss = 0.005197
grad_step = 000085, loss = 0.004875
grad_step = 000086, loss = 0.004580
grad_step = 000087, loss = 0.004309
grad_step = 000088, loss = 0.004068
grad_step = 000089, loss = 0.003848
grad_step = 000090, loss = 0.003646
grad_step = 000091, loss = 0.003466
grad_step = 000092, loss = 0.003303
grad_step = 000093, loss = 0.003156
grad_step = 000094, loss = 0.003023
grad_step = 000095, loss = 0.002905
grad_step = 000096, loss = 0.002796
grad_step = 000097, loss = 0.002699
grad_step = 000098, loss = 0.002612
grad_step = 000099, loss = 0.002532
grad_step = 000100, loss = 0.002460
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002397
grad_step = 000102, loss = 0.002342
grad_step = 000103, loss = 0.002296
grad_step = 000104, loss = 0.002263
grad_step = 000105, loss = 0.002238
grad_step = 000106, loss = 0.002197
grad_step = 000107, loss = 0.002144
grad_step = 000108, loss = 0.002108
grad_step = 000109, loss = 0.002095
grad_step = 000110, loss = 0.002081
grad_step = 000111, loss = 0.002050
grad_step = 000112, loss = 0.002016
grad_step = 000113, loss = 0.001995
grad_step = 000114, loss = 0.001986
grad_step = 000115, loss = 0.001977
grad_step = 000116, loss = 0.001961
grad_step = 000117, loss = 0.001939
grad_step = 000118, loss = 0.001919
grad_step = 000119, loss = 0.001905
grad_step = 000120, loss = 0.001897
grad_step = 000121, loss = 0.001892
grad_step = 000122, loss = 0.001890
grad_step = 000123, loss = 0.001888
grad_step = 000124, loss = 0.001889
grad_step = 000125, loss = 0.001889
grad_step = 000126, loss = 0.001880
grad_step = 000127, loss = 0.001860
grad_step = 000128, loss = 0.001833
grad_step = 000129, loss = 0.001812
grad_step = 000130, loss = 0.001798
grad_step = 000131, loss = 0.001794
grad_step = 000132, loss = 0.001797
grad_step = 000133, loss = 0.001806
grad_step = 000134, loss = 0.001819
grad_step = 000135, loss = 0.001842
grad_step = 000136, loss = 0.001863
grad_step = 000137, loss = 0.001867
grad_step = 000138, loss = 0.001828
grad_step = 000139, loss = 0.001773
grad_step = 000140, loss = 0.001734
grad_step = 000141, loss = 0.001731
grad_step = 000142, loss = 0.001756
grad_step = 000143, loss = 0.001780
grad_step = 000144, loss = 0.001787
grad_step = 000145, loss = 0.001762
grad_step = 000146, loss = 0.001726
grad_step = 000147, loss = 0.001696
grad_step = 000148, loss = 0.001686
grad_step = 000149, loss = 0.001694
grad_step = 000150, loss = 0.001711
grad_step = 000151, loss = 0.001729
grad_step = 000152, loss = 0.001740
grad_step = 000153, loss = 0.001743
grad_step = 000154, loss = 0.001726
grad_step = 000155, loss = 0.001700
grad_step = 000156, loss = 0.001668
grad_step = 000157, loss = 0.001645
grad_step = 000158, loss = 0.001634
grad_step = 000159, loss = 0.001634
grad_step = 000160, loss = 0.001642
grad_step = 000161, loss = 0.001657
grad_step = 000162, loss = 0.001685
grad_step = 000163, loss = 0.001728
grad_step = 000164, loss = 0.001799
grad_step = 000165, loss = 0.001852
grad_step = 000166, loss = 0.001880
grad_step = 000167, loss = 0.001780
grad_step = 000168, loss = 0.001655
grad_step = 000169, loss = 0.001589
grad_step = 000170, loss = 0.001621
grad_step = 000171, loss = 0.001695
grad_step = 000172, loss = 0.001703
grad_step = 000173, loss = 0.001653
grad_step = 000174, loss = 0.001585
grad_step = 000175, loss = 0.001572
grad_step = 000176, loss = 0.001607
grad_step = 000177, loss = 0.001633
grad_step = 000178, loss = 0.001631
grad_step = 000179, loss = 0.001586
grad_step = 000180, loss = 0.001557
grad_step = 000181, loss = 0.001554
grad_step = 000182, loss = 0.001571
grad_step = 000183, loss = 0.001592
grad_step = 000184, loss = 0.001595
grad_step = 000185, loss = 0.001585
grad_step = 000186, loss = 0.001562
grad_step = 000187, loss = 0.001542
grad_step = 000188, loss = 0.001530
grad_step = 000189, loss = 0.001525
grad_step = 000190, loss = 0.001524
grad_step = 000191, loss = 0.001526
grad_step = 000192, loss = 0.001531
grad_step = 000193, loss = 0.001542
grad_step = 000194, loss = 0.001563
grad_step = 000195, loss = 0.001589
grad_step = 000196, loss = 0.001635
grad_step = 000197, loss = 0.001657
grad_step = 000198, loss = 0.001689
grad_step = 000199, loss = 0.001639
grad_step = 000200, loss = 0.001584
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001515
grad_step = 000202, loss = 0.001498
grad_step = 000203, loss = 0.001525
grad_step = 000204, loss = 0.001555
grad_step = 000205, loss = 0.001562
grad_step = 000206, loss = 0.001527
grad_step = 000207, loss = 0.001492
grad_step = 000208, loss = 0.001474
grad_step = 000209, loss = 0.001482
grad_step = 000210, loss = 0.001502
grad_step = 000211, loss = 0.001519
grad_step = 000212, loss = 0.001543
grad_step = 000213, loss = 0.001550
grad_step = 000214, loss = 0.001555
grad_step = 000215, loss = 0.001518
grad_step = 000216, loss = 0.001483
grad_step = 000217, loss = 0.001459
grad_step = 000218, loss = 0.001458
grad_step = 000219, loss = 0.001467
grad_step = 000220, loss = 0.001474
grad_step = 000221, loss = 0.001478
grad_step = 000222, loss = 0.001472
grad_step = 000223, loss = 0.001463
grad_step = 000224, loss = 0.001449
grad_step = 000225, loss = 0.001438
grad_step = 000226, loss = 0.001433
grad_step = 000227, loss = 0.001435
grad_step = 000228, loss = 0.001438
grad_step = 000229, loss = 0.001442
grad_step = 000230, loss = 0.001452
grad_step = 000231, loss = 0.001475
grad_step = 000232, loss = 0.001528
grad_step = 000233, loss = 0.001604
grad_step = 000234, loss = 0.001747
grad_step = 000235, loss = 0.001775
grad_step = 000236, loss = 0.001775
grad_step = 000237, loss = 0.001550
grad_step = 000238, loss = 0.001432
grad_step = 000239, loss = 0.001480
grad_step = 000240, loss = 0.001556
grad_step = 000241, loss = 0.001521
grad_step = 000242, loss = 0.001432
grad_step = 000243, loss = 0.001447
grad_step = 000244, loss = 0.001501
grad_step = 000245, loss = 0.001488
grad_step = 000246, loss = 0.001451
grad_step = 000247, loss = 0.001422
grad_step = 000248, loss = 0.001425
grad_step = 000249, loss = 0.001444
grad_step = 000250, loss = 0.001453
grad_step = 000251, loss = 0.001453
grad_step = 000252, loss = 0.001440
grad_step = 000253, loss = 0.001420
grad_step = 000254, loss = 0.001397
grad_step = 000255, loss = 0.001398
grad_step = 000256, loss = 0.001414
grad_step = 000257, loss = 0.001426
grad_step = 000258, loss = 0.001423
grad_step = 000259, loss = 0.001403
grad_step = 000260, loss = 0.001389
grad_step = 000261, loss = 0.001387
grad_step = 000262, loss = 0.001389
grad_step = 000263, loss = 0.001391
grad_step = 000264, loss = 0.001393
grad_step = 000265, loss = 0.001396
grad_step = 000266, loss = 0.001395
grad_step = 000267, loss = 0.001390
grad_step = 000268, loss = 0.001381
grad_step = 000269, loss = 0.001374
grad_step = 000270, loss = 0.001372
grad_step = 000271, loss = 0.001373
grad_step = 000272, loss = 0.001372
grad_step = 000273, loss = 0.001371
grad_step = 000274, loss = 0.001372
grad_step = 000275, loss = 0.001375
grad_step = 000276, loss = 0.001380
grad_step = 000277, loss = 0.001385
grad_step = 000278, loss = 0.001394
grad_step = 000279, loss = 0.001407
grad_step = 000280, loss = 0.001434
grad_step = 000281, loss = 0.001473
grad_step = 000282, loss = 0.001542
grad_step = 000283, loss = 0.001603
grad_step = 000284, loss = 0.001686
grad_step = 000285, loss = 0.001642
grad_step = 000286, loss = 0.001550
grad_step = 000287, loss = 0.001406
grad_step = 000288, loss = 0.001356
grad_step = 000289, loss = 0.001414
grad_step = 000290, loss = 0.001482
grad_step = 000291, loss = 0.001481
grad_step = 000292, loss = 0.001399
grad_step = 000293, loss = 0.001350
grad_step = 000294, loss = 0.001377
grad_step = 000295, loss = 0.001422
grad_step = 000296, loss = 0.001424
grad_step = 000297, loss = 0.001376
grad_step = 000298, loss = 0.001345
grad_step = 000299, loss = 0.001357
grad_step = 000300, loss = 0.001385
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001394
grad_step = 000302, loss = 0.001370
grad_step = 000303, loss = 0.001345
grad_step = 000304, loss = 0.001340
grad_step = 000305, loss = 0.001352
grad_step = 000306, loss = 0.001366
grad_step = 000307, loss = 0.001363
grad_step = 000308, loss = 0.001350
grad_step = 000309, loss = 0.001337
grad_step = 000310, loss = 0.001333
grad_step = 000311, loss = 0.001339
grad_step = 000312, loss = 0.001345
grad_step = 000313, loss = 0.001347
grad_step = 000314, loss = 0.001341
grad_step = 000315, loss = 0.001332
grad_step = 000316, loss = 0.001326
grad_step = 000317, loss = 0.001326
grad_step = 000318, loss = 0.001328
grad_step = 000319, loss = 0.001331
grad_step = 000320, loss = 0.001333
grad_step = 000321, loss = 0.001331
grad_step = 000322, loss = 0.001327
grad_step = 000323, loss = 0.001322
grad_step = 000324, loss = 0.001319
grad_step = 000325, loss = 0.001316
grad_step = 000326, loss = 0.001316
grad_step = 000327, loss = 0.001316
grad_step = 000328, loss = 0.001317
grad_step = 000329, loss = 0.001318
grad_step = 000330, loss = 0.001319
grad_step = 000331, loss = 0.001320
grad_step = 000332, loss = 0.001321
grad_step = 000333, loss = 0.001324
grad_step = 000334, loss = 0.001327
grad_step = 000335, loss = 0.001334
grad_step = 000336, loss = 0.001343
grad_step = 000337, loss = 0.001359
grad_step = 000338, loss = 0.001378
grad_step = 000339, loss = 0.001413
grad_step = 000340, loss = 0.001443
grad_step = 000341, loss = 0.001497
grad_step = 000342, loss = 0.001514
grad_step = 000343, loss = 0.001535
grad_step = 000344, loss = 0.001475
grad_step = 000345, loss = 0.001405
grad_step = 000346, loss = 0.001335
grad_step = 000347, loss = 0.001316
grad_step = 000348, loss = 0.001343
grad_step = 000349, loss = 0.001372
grad_step = 000350, loss = 0.001380
grad_step = 000351, loss = 0.001346
grad_step = 000352, loss = 0.001313
grad_step = 000353, loss = 0.001304
grad_step = 000354, loss = 0.001320
grad_step = 000355, loss = 0.001338
grad_step = 000356, loss = 0.001333
grad_step = 000357, loss = 0.001317
grad_step = 000358, loss = 0.001300
grad_step = 000359, loss = 0.001299
grad_step = 000360, loss = 0.001308
grad_step = 000361, loss = 0.001313
grad_step = 000362, loss = 0.001311
grad_step = 000363, loss = 0.001298
grad_step = 000364, loss = 0.001287
grad_step = 000365, loss = 0.001283
grad_step = 000366, loss = 0.001288
grad_step = 000367, loss = 0.001295
grad_step = 000368, loss = 0.001296
grad_step = 000369, loss = 0.001291
grad_step = 000370, loss = 0.001284
grad_step = 000371, loss = 0.001281
grad_step = 000372, loss = 0.001282
grad_step = 000373, loss = 0.001287
grad_step = 000374, loss = 0.001292
grad_step = 000375, loss = 0.001296
grad_step = 000376, loss = 0.001301
grad_step = 000377, loss = 0.001306
grad_step = 000378, loss = 0.001320
grad_step = 000379, loss = 0.001341
grad_step = 000380, loss = 0.001379
grad_step = 000381, loss = 0.001422
grad_step = 000382, loss = 0.001487
grad_step = 000383, loss = 0.001530
grad_step = 000384, loss = 0.001573
grad_step = 000385, loss = 0.001536
grad_step = 000386, loss = 0.001475
grad_step = 000387, loss = 0.001379
grad_step = 000388, loss = 0.001303
grad_step = 000389, loss = 0.001275
grad_step = 000390, loss = 0.001294
grad_step = 000391, loss = 0.001332
grad_step = 000392, loss = 0.001347
grad_step = 000393, loss = 0.001333
grad_step = 000394, loss = 0.001302
grad_step = 000395, loss = 0.001281
grad_step = 000396, loss = 0.001282
grad_step = 000397, loss = 0.001294
grad_step = 000398, loss = 0.001306
grad_step = 000399, loss = 0.001302
grad_step = 000400, loss = 0.001291
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001275
grad_step = 000402, loss = 0.001263
grad_step = 000403, loss = 0.001259
grad_step = 000404, loss = 0.001262
grad_step = 000405, loss = 0.001271
grad_step = 000406, loss = 0.001278
grad_step = 000407, loss = 0.001283
grad_step = 000408, loss = 0.001279
grad_step = 000409, loss = 0.001272
grad_step = 000410, loss = 0.001262
grad_step = 000411, loss = 0.001253
grad_step = 000412, loss = 0.001248
grad_step = 000413, loss = 0.001246
grad_step = 000414, loss = 0.001247
grad_step = 000415, loss = 0.001250
grad_step = 000416, loss = 0.001253
grad_step = 000417, loss = 0.001254
grad_step = 000418, loss = 0.001255
grad_step = 000419, loss = 0.001254
grad_step = 000420, loss = 0.001254
grad_step = 000421, loss = 0.001254
grad_step = 000422, loss = 0.001256
grad_step = 000423, loss = 0.001257
grad_step = 000424, loss = 0.001261
grad_step = 000425, loss = 0.001265
grad_step = 000426, loss = 0.001273
grad_step = 000427, loss = 0.001279
grad_step = 000428, loss = 0.001292
grad_step = 000429, loss = 0.001302
grad_step = 000430, loss = 0.001322
grad_step = 000431, loss = 0.001335
grad_step = 000432, loss = 0.001362
grad_step = 000433, loss = 0.001372
grad_step = 000434, loss = 0.001392
grad_step = 000435, loss = 0.001385
grad_step = 000436, loss = 0.001385
grad_step = 000437, loss = 0.001354
grad_step = 000438, loss = 0.001320
grad_step = 000439, loss = 0.001287
grad_step = 000440, loss = 0.001268
grad_step = 000441, loss = 0.001268
grad_step = 000442, loss = 0.001268
grad_step = 000443, loss = 0.001267
grad_step = 000444, loss = 0.001258
grad_step = 000445, loss = 0.001251
grad_step = 000446, loss = 0.001251
grad_step = 000447, loss = 0.001256
grad_step = 000448, loss = 0.001262
grad_step = 000449, loss = 0.001257
grad_step = 000450, loss = 0.001246
grad_step = 000451, loss = 0.001229
grad_step = 000452, loss = 0.001219
grad_step = 000453, loss = 0.001219
grad_step = 000454, loss = 0.001226
grad_step = 000455, loss = 0.001233
grad_step = 000456, loss = 0.001236
grad_step = 000457, loss = 0.001235
grad_step = 000458, loss = 0.001231
grad_step = 000459, loss = 0.001227
grad_step = 000460, loss = 0.001226
grad_step = 000461, loss = 0.001229
grad_step = 000462, loss = 0.001239
grad_step = 000463, loss = 0.001254
grad_step = 000464, loss = 0.001280
grad_step = 000465, loss = 0.001308
grad_step = 000466, loss = 0.001354
grad_step = 000467, loss = 0.001388
grad_step = 000468, loss = 0.001442
grad_step = 000469, loss = 0.001436
grad_step = 000470, loss = 0.001426
grad_step = 000471, loss = 0.001364
grad_step = 000472, loss = 0.001305
grad_step = 000473, loss = 0.001268
grad_step = 000474, loss = 0.001259
grad_step = 000475, loss = 0.001274
grad_step = 000476, loss = 0.001274
grad_step = 000477, loss = 0.001266
grad_step = 000478, loss = 0.001241
grad_step = 000479, loss = 0.001223
grad_step = 000480, loss = 0.001222
grad_step = 000481, loss = 0.001233
grad_step = 000482, loss = 0.001246
grad_step = 000483, loss = 0.001239
grad_step = 000484, loss = 0.001222
grad_step = 000485, loss = 0.001202
grad_step = 000486, loss = 0.001197
grad_step = 000487, loss = 0.001209
grad_step = 000488, loss = 0.001220
grad_step = 000489, loss = 0.001222
grad_step = 000490, loss = 0.001208
grad_step = 000491, loss = 0.001196
grad_step = 000492, loss = 0.001192
grad_step = 000493, loss = 0.001197
grad_step = 000494, loss = 0.001203
grad_step = 000495, loss = 0.001203
grad_step = 000496, loss = 0.001199
grad_step = 000497, loss = 0.001196
grad_step = 000498, loss = 0.001195
grad_step = 000499, loss = 0.001195
grad_step = 000500, loss = 0.001197
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001202
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

  date_run                              2020-05-14 17:13:55.529305
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.252242
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 17:13:55.535725
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.156671
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 17:13:55.543786
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.151792
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 17:13:55.550202
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.38067
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
0   2020-05-14 17:13:20.565829  ...    mean_absolute_error
1   2020-05-14 17:13:20.570440  ...     mean_squared_error
2   2020-05-14 17:13:20.574265  ...  median_absolute_error
3   2020-05-14 17:13:20.577998  ...               r2_score
4   2020-05-14 17:13:30.577197  ...    mean_absolute_error
5   2020-05-14 17:13:30.581501  ...     mean_squared_error
6   2020-05-14 17:13:30.585428  ...  median_absolute_error
7   2020-05-14 17:13:30.589343  ...               r2_score
8   2020-05-14 17:13:55.529305  ...    mean_absolute_error
9   2020-05-14 17:13:55.535725  ...     mean_squared_error
10  2020-05-14 17:13:55.543786  ...  median_absolute_error
11  2020-05-14 17:13:55.550202  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc97cb5dfd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:14, 132152.84it/s] 35%|███▍      | 3465216/9912422 [00:00<00:34, 188471.61it/s] 68%|██████▊   | 6782976/9912422 [00:00<00:11, 268587.81it/s]9920512it [00:00, 24348816.12it/s]                           
0it [00:00, ?it/s]32768it [00:00, 489097.89it/s]
0it [00:00, ?it/s]  5%|▌         | 90112/1648877 [00:00<00:01, 893424.83it/s]1654784it [00:00, 10524007.52it/s]                         
0it [00:00, ?it/s]8192it [00:00, 195624.82it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc92f560e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc92eb8c0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc92f560e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc92eae40f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc92c3224e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc92c309748> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc92f560e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc92eaa2710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc92c3224e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc97cb67ef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fdcef4d2208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=0e1e8a22406562da18eca9a4724ace6f14a86cbd9100c7b2693371611fd2377f
  Stored in directory: /tmp/pip-ephem-wheel-cache-1_4g3f3_/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fdce5858080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1900544/17464789 [==>...........................] - ETA: 0s
 6414336/17464789 [==========>...................] - ETA: 0s
11403264/17464789 [==================>...........] - ETA: 0s
14311424/17464789 [=======================>......] - ETA: 0s
17047552/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 17:15:25.509005: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 17:15:25.514123: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-14 17:15:25.514265: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56057538e4a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 17:15:25.514282: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.6360 - accuracy: 0.5020
 2000/25000 [=>............................] - ETA: 10s - loss: 7.5746 - accuracy: 0.5060
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.6564 - accuracy: 0.5007 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.5631 - accuracy: 0.5067
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6360 - accuracy: 0.5020
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6538 - accuracy: 0.5008
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6688 - accuracy: 0.4999
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6915 - accuracy: 0.4984
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7160 - accuracy: 0.4968
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7188 - accuracy: 0.4966
11000/25000 [============>.................] - ETA: 4s - loss: 7.6945 - accuracy: 0.4982
12000/25000 [=============>................] - ETA: 4s - loss: 7.6896 - accuracy: 0.4985
13000/25000 [==============>...............] - ETA: 4s - loss: 7.7079 - accuracy: 0.4973
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7104 - accuracy: 0.4971
15000/25000 [=================>............] - ETA: 3s - loss: 7.7004 - accuracy: 0.4978
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6982 - accuracy: 0.4979
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6720 - accuracy: 0.4996
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6871 - accuracy: 0.4987
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6949 - accuracy: 0.4982
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6965 - accuracy: 0.4981
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6790 - accuracy: 0.4992
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6778 - accuracy: 0.4993
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6786 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6737 - accuracy: 0.4995
25000/25000 [==============================] - 10s 403us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 17:15:43.506166
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 17:15:43.506166  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<45:20:43, 5.28kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<31:58:33, 7.49kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<22:26:05, 10.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<15:42:22, 15.2kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:02<10:57:46, 21.8kB/s].vector_cache/glove.6B.zip:   1%|          | 9.21M/862M [00:02<7:37:32, 31.1kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.9M/862M [00:02<5:19:00, 44.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.6M/862M [00:02<3:41:53, 63.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.7M/862M [00:02<2:34:54, 90.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.2M/862M [00:02<1:47:47, 129kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 30.3M/862M [00:02<1:15:19, 184kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.8M/862M [00:02<52:31, 263kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 38.7M/862M [00:02<36:42, 374kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.4M/862M [00:03<25:38, 532kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.2M/862M [00:03<17:58, 756kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:03<12:49, 1.05MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:05<10:51, 1.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:05<09:28, 1.42MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.4M/862M [00:05<07:02, 1.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:07<07:28, 1.79MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:07<08:06, 1.65MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<06:16, 2.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.9M/862M [00:07<04:35, 2.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:09<08:16, 1.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:09<06:57, 1.91MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:09<05:20, 2.48MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:09<03:52, 3.41MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:11<34:40, 382kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:11<25:23, 521kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:11<18:04, 731kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:11<12:46, 1.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:13<1:01:58, 212kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:13<44:43, 294kB/s]  .vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:13<31:35, 416kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:15<25:08, 521kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:15<18:56, 691kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:15<13:33, 963kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:17<12:33, 1.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:17<10:06, 1.29MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:17<07:23, 1.76MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:19<08:14, 1.57MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:19<07:05, 1.83MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:19<05:16, 2.45MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:21<06:44, 1.91MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:21<06:02, 2.13MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:21<04:33, 2.82MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:23<06:12, 2.06MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:23<05:39, 2.26MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:23<04:16, 2.99MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:25<06:00, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:25<05:29, 2.32MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:25<04:09, 3.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<05:54, 2.15MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<06:43, 1.89MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<05:17, 2.39MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<03:50, 3.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:29<14:27, 873kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<11:24, 1.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<08:17, 1.52MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<08:43, 1.44MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<08:40, 1.45MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<06:43, 1.86MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<04:51, 2.57MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<1:32:21, 135kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<1:05:54, 189kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<46:21, 268kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<35:16, 352kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<25:55, 478kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<18:23, 673kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<15:46, 782kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<12:17, 1.00MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<08:54, 1.38MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<09:07, 1.35MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:38, 1.60MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<05:36, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:49, 1.79MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:01, 2.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<04:31, 2.69MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<06:01, 2.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:27, 2.22MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<04:07, 2.94MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<05:44, 2.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<06:29, 1.86MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:03, 2.38MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<03:41, 3.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:46<08:15, 1.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:02, 1.70MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:13, 2.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:25, 1.86MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:43, 2.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<04:15, 2.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<05:47, 2.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<06:29, 1.83MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:08, 2.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:29, 2.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:02, 2.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<03:49, 3.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<05:25, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<06:11, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<04:52, 2.40MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<03:33, 3.28MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<09:13, 1.26MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<07:27, 1.56MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:28, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<03:58, 2.91MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<33:11, 349kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<25:30, 454kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<18:22, 630kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<13:02, 885kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<12:38, 911kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<10:03, 1.14MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<07:19, 1.57MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<07:47, 1.47MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:37, 1.73MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<04:55, 2.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:04<06:07, 1.86MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:26, 2.09MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<04:06, 2.76MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<05:32, 2.04MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<06:12, 1.82MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<04:54, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<03:33, 3.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<10:48:33, 17.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<7:34:54, 24.7kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<5:17:59, 35.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<3:44:29, 49.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<2:38:12, 70.6kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<1:50:46, 101kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<1:19:53, 139kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<57:01, 195kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<40:06, 276kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<30:35, 361kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<22:32, 489kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<16:01, 687kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<13:46, 796kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:16<10:45, 1.02MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:47, 1.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<08:01, 1.36MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<07:51, 1.39MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<05:58, 1.82MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<04:19, 2.51MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:20<07:44, 1.40MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<06:33, 1.65MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<04:51, 2.22MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<05:54, 1.82MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<05:04, 2.12MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<03:50, 2.80MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<02:48, 3.81MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<48:02, 223kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<34:42, 308kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<24:30, 435kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<19:36, 542kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<15:54, 668kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<11:34, 916kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<08:14, 1.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<10:00, 1.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<08:06, 1.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:56, 1.77MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<06:36, 1.59MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:30, 1.90MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<04:06, 2.54MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<03:00, 3.47MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<44:05, 236kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<32:59, 316kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<23:32, 442kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<16:34, 626kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<16:20, 634kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<12:30, 827kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<08:58, 1.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<08:39, 1.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<07:06, 1.44MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:13, 1.96MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<06:03, 1.68MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:17, 1.93MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:57, 2.58MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<05:10, 1.96MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<05:41, 1.78MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<04:26, 2.28MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<03:13, 3.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<08:04, 1.25MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<06:42, 1.50MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:56, 2.03MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<05:47, 1.73MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<05:06, 1.96MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<03:46, 2.64MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<04:58, 2.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<04:29, 2.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:23, 2.92MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<04:42, 2.09MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<05:19, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:13, 2.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:04, 3.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<9:20:44, 17.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<6:33:17, 24.9kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<4:34:49, 35.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<3:13:55, 50.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<2:16:30, 71.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<1:35:36, 102kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<1:06:46, 145kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<1:23:12, 116kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<59:11, 163kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<41:34, 232kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:55<31:16, 307kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<23:51, 402kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<17:10, 558kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<12:02, 791kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<9:23:53, 16.9kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<6:35:26, 24.1kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<4:36:18, 34.4kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<3:14:52, 48.5kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<2:17:17, 68.9kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<1:36:04, 98.1kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<1:09:13, 136kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<49:22, 190kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<34:40, 270kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<26:24, 353kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<19:26, 479kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<13:48, 673kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<11:48, 783kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<10:09, 911kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<07:30, 1.23MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:23, 1.71MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<07:23, 1.24MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<06:06, 1.50MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:29, 2.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:09<05:15, 1.73MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<04:36, 1.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<03:25, 2.66MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:11<04:30, 2.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<04:05, 2.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<03:05, 2.92MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:13<04:16, 2.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<03:54, 2.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<02:57, 3.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:10, 2.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<03:50, 2.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<02:54, 3.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<04:06, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:17<04:41, 1.89MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<03:43, 2.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<02:42, 3.24MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<8:23:43, 17.4kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<5:53:15, 24.8kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<4:06:46, 35.4kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<2:54:03, 50.0kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<2:03:34, 70.4kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<1:26:45, 100kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<1:00:35, 143kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<46:31, 186kB/s]  .vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<33:25, 258kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<23:32, 365kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<18:25, 465kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<13:45, 622kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<09:49, 869kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<08:51, 959kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<07:03, 1.20MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<05:08, 1.65MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<05:35, 1.51MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:45, 1.77MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<03:32, 2.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<04:27, 1.88MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:45, 1.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:45, 2.22MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<03:57, 2.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<03:38, 2.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<02:43, 3.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<03:49, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<03:31, 2.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<02:39, 3.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<03:47, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<04:19, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:21, 2.42MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<02:31, 3.22MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<04:04, 1.99MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<03:40, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<02:44, 2.95MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<03:47, 2.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<04:18, 1.86MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:25, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<03:39, 2.17MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:23, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<02:34, 3.07MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<03:37, 2.17MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<03:20, 2.36MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<02:32, 3.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<03:36, 2.16MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:07, 1.89MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<03:16, 2.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<03:32, 2.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<03:16, 2.36MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<02:29, 3.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<03:31, 2.18MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<03:16, 2.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<02:29, 3.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<03:31, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<04:01, 1.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:12, 2.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<03:26, 2.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<03:12, 2.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<02:25, 3.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<03:25, 2.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<03:55, 1.90MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:04, 2.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<02:15, 3.30MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:58<04:54, 1.51MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:58<04:12, 1.76MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:07, 2.36MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<03:53, 1.88MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<04:17, 1.71MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<03:22, 2.17MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<02:25, 2.99MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:02<27:59, 260kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:02<20:20, 357kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<14:22, 503kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:04<11:42, 615kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<09:39, 745kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<07:06, 1.01MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<05:01, 1.42MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<6:53:15, 17.2kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<4:49:44, 24.6kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<3:22:16, 35.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<2:22:31, 49.5kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<1:40:24, 70.2kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<1:10:12, 100kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<50:31, 138kB/s]  .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<36:46, 190kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<26:01, 268kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<18:09, 381kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<25:50, 268kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<18:48, 368kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<13:17, 518kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<10:51, 631kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<09:00, 761kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<06:35, 1.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:41, 1.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<05:57, 1.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<04:51, 1.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:33, 1.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<04:03, 1.66MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:31, 1.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:38, 2.54MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<03:23, 1.96MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:43, 1.78MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<02:54, 2.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:06, 3.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<06:04, 1.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:56, 1.33MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:36, 1.81MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:02, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:30, 1.86MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:36, 2.48MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:19, 1.94MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:59, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:14, 2.86MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:04, 2.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:27, 1.85MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:41, 2.37MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<01:59, 3.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<03:16, 1.92MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<02:57, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:13, 2.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<03:00, 2.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<02:46, 2.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:04, 2.99MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<02:50, 2.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<03:18, 1.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:38, 2.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<01:55, 3.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<20:38, 295kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<15:05, 404kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<10:39, 569kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<08:46, 687kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<06:47, 887kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:52, 1.23MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<04:44, 1.26MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<04:35, 1.29MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:28, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:33, 2.32MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<03:24, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<03:01, 1.95MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:15, 2.59MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:43<02:52, 2.02MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<03:17, 1.77MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:33, 2.26MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<01:52, 3.07MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<03:24, 1.68MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:45<03:00, 1.90MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:14, 2.56MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<02:48, 2.03MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<03:11, 1.78MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:28, 2.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<01:49, 3.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<03:25, 1.64MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<02:59, 1.88MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:12, 2.53MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<02:47, 1.99MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<03:08, 1.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<02:29, 2.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<01:48, 3.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<18:34, 295kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<13:34, 403kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<09:36, 567kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<07:53, 685kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<06:05, 886kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:22, 1.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<04:22, 1.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<04:12, 1.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:11, 1.67MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:17, 2.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<03:38, 1.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<03:06, 1.69MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:17, 2.29MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<02:45, 1.89MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<03:02, 1.71MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:22, 2.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<01:44, 2.97MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<03:01, 1.70MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:39, 1.92MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<01:58, 2.59MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<02:31, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<02:50, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<02:12, 2.28MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<01:39, 3.04MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<02:31, 1.98MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<02:17, 2.17MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<01:44, 2.86MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<02:18, 2.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<02:40, 1.84MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:05, 2.34MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<01:31, 3.20MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:11<03:34, 1.36MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:11<03:01, 1.61MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:13, 2.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<02:37, 1.82MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<02:15, 2.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<01:46, 2.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<01:17, 3.68MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:15<07:03, 667kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:15<05:57, 792kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<04:24, 1.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:06, 1.50MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:17<13:18, 349kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<09:48, 473kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<06:56, 665kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<05:51, 782kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<04:29, 1.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<03:15, 1.40MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:19, 1.94MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<06:45, 668kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<05:11, 867kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:43, 1.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<03:36, 1.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<02:59, 1.48MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:11, 2.01MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<02:30, 1.74MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<02:41, 1.62MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:05, 2.09MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:30, 2.87MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<03:30, 1.23MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:55, 1.47MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<02:07, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<02:26, 1.74MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:29<02:36, 1.62MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<02:03, 2.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:28, 2.82MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<13:30, 308kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<09:53, 420kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<06:59, 591kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<05:46, 709kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<04:55, 831kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<03:37, 1.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:34, 1.57MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<03:45, 1.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<03:03, 1.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:12, 1.80MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:26, 1.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:37<02:32, 1.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<01:57, 2.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:26, 2.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:03, 1.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:51, 2.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<01:23, 2.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<01:50, 2.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:06, 1.81MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<01:39, 2.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:11, 3.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<04:13, 888kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<03:20, 1.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<02:25, 1.54MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<02:31, 1.46MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:09, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:35, 2.29MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<01:55, 1.88MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:44, 2.08MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:18, 2.75MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:42, 2.08MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:33, 2.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:10, 3.01MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:35, 2.18MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:51, 1.87MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:28, 2.33MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:03, 3.23MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<08:04, 421kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<05:56, 571kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:52<04:13, 799kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<02:58, 1.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<06:25, 519kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<05:10, 643kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<03:45, 882kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<02:38, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<03:47, 861kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:59, 1.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:09, 1.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<02:13, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:53, 1.68MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:23, 2.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:40, 1.87MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:50, 1.70MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:25, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:01, 3.00MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:02<02:17, 1.34MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<01:55, 1.59MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:24, 2.15MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:38, 1.81MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:47, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:24, 2.11MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:00, 2.91MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<07:58, 366kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<05:53, 495kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<04:09, 696kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<03:30, 811kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<03:04, 929kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:16, 1.25MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:36, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<02:16, 1.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:52, 1.48MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:22, 2.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<01:34, 1.73MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:23, 1.95MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:01, 2.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<01:18, 2.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:28, 1.79MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:09, 2.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:49, 3.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<12:58, 199kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<09:20, 275kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<06:31, 390kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<05:05, 493kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<03:46, 662kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<02:42, 918kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:53, 1.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<04:30, 541kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<03:40, 662kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<02:40, 907kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:53, 1.27MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<02:04, 1.14MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<01:39, 1.42MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:12, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<01:22, 1.68MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<01:26, 1.59MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<01:07, 2.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:48, 2.80MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<05:41, 392kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<04:10, 533kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:57, 747kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<02:04, 1.05MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<04:43, 458kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:28<03:45, 575kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:28<02:42, 791kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:53, 1.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:30<02:19, 901kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:30<01:48, 1.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:18, 1.58MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:55, 2.20MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<04:35, 442kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<03:38, 556kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<02:38, 762kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:49, 1.07MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<07:09, 273kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<05:12, 375kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<03:38, 529kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<02:55, 645kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:36<02:27, 768kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<01:48, 1.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:15, 1.45MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<13:51, 131kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:38<09:52, 184kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<06:51, 261kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<05:05, 344kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:40<03:54, 447kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<02:47, 621kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:58, 869kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:48, 934kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:26, 1.17MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:02, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<01:04, 1.51MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<01:03, 1.52MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<00:49, 1.94MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:35, 2.67MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:08, 1.35MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:57, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<00:41, 2.17MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:48, 1.82MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:52, 1.67MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<00:41, 2.12MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:29, 2.91MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<10:22, 136kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<07:22, 190kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<05:06, 270kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<03:46, 354kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<02:55, 457kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:52<02:05, 630kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:25, 890kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<04:48, 264kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<03:27, 365kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<02:27, 510kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:40, 723kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<02:30, 479kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:52, 638kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:18, 890kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<01:08, 988kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:54, 1.23MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:39, 1.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:41, 1.54MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:42, 1.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:32, 1.92MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:22, 2.65MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<02:44, 363kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<02:00, 492kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<01:23, 690kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:09, 803kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:00, 919kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:44, 1.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:30, 1.72MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<02:22, 360kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<01:44, 487kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:12, 683kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:59, 799kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:51, 917kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:37, 1.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:25, 1.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:09<02:01, 356kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:09<01:28, 482kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<01:00, 676kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:49, 793kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:38, 1.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:26, 1.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:25, 1.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:21, 1.61MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:15, 2.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:16, 1.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:18, 1.67MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:13, 2.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:09, 2.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:27, 970kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:21, 1.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:14, 1.65MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:14, 1.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:14, 1.49MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:11, 1.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:06, 2.64MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:46, 390kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:34, 526kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:22, 737kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:16, 847kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:12, 1.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:08, 1.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:07, 1.42MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:05, 1.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:03, 1.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:03, 1.69MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:02, 2.17MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.99MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:02, 789kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:01, 1.01MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 754/400000 [00:00<00:52, 7537.08it/s]  0%|          | 1424/400000 [00:00<00:54, 7263.43it/s]  1%|          | 2092/400000 [00:00<00:56, 7075.09it/s]  1%|          | 2782/400000 [00:00<00:56, 7019.14it/s]  1%|          | 3486/400000 [00:00<00:56, 7023.54it/s]  1%|          | 4181/400000 [00:00<00:56, 6999.13it/s]  1%|          | 4941/400000 [00:00<00:55, 7167.93it/s]  1%|▏         | 5638/400000 [00:00<00:55, 7104.83it/s]  2%|▏         | 6387/400000 [00:00<00:54, 7215.89it/s]  2%|▏         | 7104/400000 [00:01<00:54, 7199.44it/s]  2%|▏         | 7818/400000 [00:01<00:54, 7179.18it/s]  2%|▏         | 8563/400000 [00:01<00:53, 7255.26it/s]  2%|▏         | 9279/400000 [00:01<00:54, 7218.68it/s]  2%|▏         | 9995/400000 [00:01<00:54, 7183.55it/s]  3%|▎         | 10709/400000 [00:01<00:54, 7132.47it/s]  3%|▎         | 11443/400000 [00:01<00:54, 7191.95it/s]  3%|▎         | 12183/400000 [00:01<00:53, 7252.38it/s]  3%|▎         | 12942/400000 [00:01<00:52, 7349.70it/s]  3%|▎         | 13677/400000 [00:01<00:52, 7336.57it/s]  4%|▎         | 14413/400000 [00:02<00:52, 7342.91it/s]  4%|▍         | 15177/400000 [00:02<00:51, 7426.43it/s]  4%|▍         | 15935/400000 [00:02<00:51, 7469.09it/s]  4%|▍         | 16690/400000 [00:02<00:51, 7490.69it/s]  4%|▍         | 17440/400000 [00:02<00:51, 7450.66it/s]  5%|▍         | 18186/400000 [00:02<00:52, 7333.14it/s]  5%|▍         | 18941/400000 [00:02<00:51, 7396.75it/s]  5%|▍         | 19688/400000 [00:02<00:51, 7417.98it/s]  5%|▌         | 20440/400000 [00:02<00:50, 7447.25it/s]  5%|▌         | 21201/400000 [00:02<00:50, 7492.57it/s]  5%|▌         | 21951/400000 [00:03<00:50, 7440.07it/s]  6%|▌         | 22696/400000 [00:03<00:50, 7399.27it/s]  6%|▌         | 23437/400000 [00:03<00:51, 7345.32it/s]  6%|▌         | 24172/400000 [00:03<00:51, 7340.16it/s]  6%|▌         | 24907/400000 [00:03<00:51, 7342.18it/s]  6%|▋         | 25667/400000 [00:03<00:50, 7416.87it/s]  7%|▋         | 26411/400000 [00:03<00:50, 7420.16it/s]  7%|▋         | 27172/400000 [00:03<00:49, 7474.15it/s]  7%|▋         | 27920/400000 [00:03<00:49, 7473.51it/s]  7%|▋         | 28668/400000 [00:03<00:50, 7405.49it/s]  7%|▋         | 29425/400000 [00:04<00:49, 7451.29it/s]  8%|▊         | 30171/400000 [00:04<00:49, 7423.02it/s]  8%|▊         | 30936/400000 [00:04<00:49, 7488.46it/s]  8%|▊         | 31686/400000 [00:04<00:49, 7476.07it/s]  8%|▊         | 32434/400000 [00:04<00:49, 7455.13it/s]  8%|▊         | 33192/400000 [00:04<00:48, 7490.42it/s]  8%|▊         | 33946/400000 [00:04<00:48, 7502.39it/s]  9%|▊         | 34705/400000 [00:04<00:48, 7526.50it/s]  9%|▉         | 35458/400000 [00:04<00:48, 7511.92it/s]  9%|▉         | 36210/400000 [00:04<00:49, 7423.96it/s]  9%|▉         | 36957/400000 [00:05<00:48, 7435.12it/s]  9%|▉         | 37726/400000 [00:05<00:48, 7507.65it/s] 10%|▉         | 38499/400000 [00:05<00:47, 7571.22it/s] 10%|▉         | 39257/400000 [00:05<00:47, 7546.08it/s] 10%|█         | 40012/400000 [00:05<00:48, 7482.42it/s] 10%|█         | 40761/400000 [00:05<00:48, 7468.96it/s] 10%|█         | 41509/400000 [00:05<00:48, 7459.48it/s] 11%|█         | 42266/400000 [00:05<00:47, 7490.24it/s] 11%|█         | 43031/400000 [00:05<00:47, 7534.87it/s] 11%|█         | 43785/400000 [00:05<00:47, 7431.40it/s] 11%|█         | 44529/400000 [00:06<00:47, 7425.42it/s] 11%|█▏        | 45301/400000 [00:06<00:47, 7509.75it/s] 12%|█▏        | 46065/400000 [00:06<00:46, 7545.46it/s] 12%|█▏        | 46827/400000 [00:06<00:46, 7566.51it/s] 12%|█▏        | 47584/400000 [00:06<00:47, 7447.04it/s] 12%|█▏        | 48330/400000 [00:06<00:47, 7375.69it/s] 12%|█▏        | 49073/400000 [00:06<00:47, 7389.61it/s] 12%|█▏        | 49832/400000 [00:06<00:47, 7448.00it/s] 13%|█▎        | 50578/400000 [00:06<00:46, 7435.95it/s] 13%|█▎        | 51322/400000 [00:06<00:46, 7419.43it/s] 13%|█▎        | 52065/400000 [00:07<00:46, 7415.69it/s] 13%|█▎        | 52833/400000 [00:07<00:46, 7490.46it/s] 13%|█▎        | 53583/400000 [00:07<00:46, 7471.65it/s] 14%|█▎        | 54331/400000 [00:07<00:46, 7451.00it/s] 14%|█▍        | 55077/400000 [00:07<00:46, 7441.80it/s] 14%|█▍        | 55841/400000 [00:07<00:45, 7497.60it/s] 14%|█▍        | 56597/400000 [00:07<00:45, 7516.10it/s] 14%|█▍        | 57349/400000 [00:07<00:45, 7515.15it/s] 15%|█▍        | 58103/400000 [00:07<00:45, 7520.35it/s] 15%|█▍        | 58856/400000 [00:07<00:45, 7453.70it/s] 15%|█▍        | 59606/400000 [00:08<00:45, 7467.23it/s] 15%|█▌        | 60353/400000 [00:08<00:46, 7287.21it/s] 15%|█▌        | 61083/400000 [00:08<00:48, 6976.09it/s] 15%|█▌        | 61825/400000 [00:08<00:47, 7103.13it/s] 16%|█▌        | 62575/400000 [00:08<00:46, 7215.25it/s] 16%|█▌        | 63300/400000 [00:08<00:46, 7214.09it/s] 16%|█▌        | 64051/400000 [00:08<00:46, 7298.72it/s] 16%|█▌        | 64814/400000 [00:08<00:45, 7393.87it/s] 16%|█▋        | 65571/400000 [00:08<00:44, 7444.97it/s] 17%|█▋        | 66317/400000 [00:08<00:44, 7423.15it/s] 17%|█▋        | 67079/400000 [00:09<00:44, 7480.36it/s] 17%|█▋        | 67828/400000 [00:09<00:44, 7469.91it/s] 17%|█▋        | 68593/400000 [00:09<00:44, 7521.59it/s] 17%|█▋        | 69346/400000 [00:09<00:43, 7516.04it/s] 18%|█▊        | 70098/400000 [00:09<00:44, 7460.44it/s] 18%|█▊        | 70845/400000 [00:09<00:44, 7419.17it/s] 18%|█▊        | 71597/400000 [00:09<00:44, 7449.05it/s] 18%|█▊        | 72371/400000 [00:09<00:43, 7533.47it/s] 18%|█▊        | 73125/400000 [00:09<00:43, 7524.18it/s] 18%|█▊        | 73878/400000 [00:09<00:43, 7413.86it/s] 19%|█▊        | 74641/400000 [00:10<00:43, 7476.38it/s] 19%|█▉        | 75416/400000 [00:10<00:42, 7554.84it/s] 19%|█▉        | 76184/400000 [00:10<00:42, 7591.55it/s] 19%|█▉        | 76944/400000 [00:10<00:42, 7559.30it/s] 19%|█▉        | 77701/400000 [00:10<00:42, 7511.60it/s] 20%|█▉        | 78453/400000 [00:10<00:42, 7509.79it/s] 20%|█▉        | 79209/400000 [00:10<00:42, 7523.57it/s] 20%|█▉        | 79962/400000 [00:10<00:42, 7518.19it/s] 20%|██        | 80726/400000 [00:10<00:42, 7552.34it/s] 20%|██        | 81482/400000 [00:11<00:42, 7548.95it/s] 21%|██        | 82249/400000 [00:11<00:41, 7584.24it/s] 21%|██        | 83008/400000 [00:11<00:41, 7562.30it/s] 21%|██        | 83765/400000 [00:11<00:42, 7493.30it/s] 21%|██        | 84518/400000 [00:11<00:42, 7501.89it/s] 21%|██▏       | 85269/400000 [00:11<00:42, 7443.77it/s] 22%|██▏       | 86014/400000 [00:11<00:42, 7356.44it/s] 22%|██▏       | 86778/400000 [00:11<00:42, 7438.27it/s] 22%|██▏       | 87534/400000 [00:11<00:41, 7471.68it/s] 22%|██▏       | 88297/400000 [00:11<00:41, 7515.58it/s] 22%|██▏       | 89049/400000 [00:12<00:41, 7439.70it/s] 22%|██▏       | 89816/400000 [00:12<00:41, 7504.60it/s] 23%|██▎       | 90584/400000 [00:12<00:40, 7554.44it/s] 23%|██▎       | 91346/400000 [00:12<00:40, 7573.43it/s] 23%|██▎       | 92104/400000 [00:12<00:41, 7494.85it/s] 23%|██▎       | 92854/400000 [00:12<00:41, 7400.43it/s] 23%|██▎       | 93595/400000 [00:12<00:41, 7346.23it/s] 24%|██▎       | 94350/400000 [00:12<00:41, 7404.50it/s] 24%|██▍       | 95100/400000 [00:12<00:41, 7431.55it/s] 24%|██▍       | 95852/400000 [00:12<00:40, 7456.43it/s] 24%|██▍       | 96598/400000 [00:13<00:41, 7362.14it/s] 24%|██▍       | 97368/400000 [00:13<00:40, 7459.68it/s] 25%|██▍       | 98120/400000 [00:13<00:40, 7475.06it/s] 25%|██▍       | 98868/400000 [00:13<00:40, 7445.43it/s] 25%|██▍       | 99622/400000 [00:13<00:40, 7471.59it/s] 25%|██▌       | 100370/400000 [00:13<00:40, 7461.73it/s] 25%|██▌       | 101135/400000 [00:13<00:39, 7515.11it/s] 25%|██▌       | 101899/400000 [00:13<00:39, 7549.27it/s] 26%|██▌       | 102662/400000 [00:13<00:39, 7571.97it/s] 26%|██▌       | 103420/400000 [00:13<00:39, 7541.78it/s] 26%|██▌       | 104179/400000 [00:14<00:39, 7555.15it/s] 26%|██▌       | 104949/400000 [00:14<00:38, 7596.93it/s] 26%|██▋       | 105709/400000 [00:14<00:38, 7579.50it/s] 27%|██▋       | 106468/400000 [00:14<00:39, 7493.94it/s] 27%|██▋       | 107218/400000 [00:14<00:39, 7368.73it/s] 27%|██▋       | 107956/400000 [00:14<00:40, 7275.53it/s] 27%|██▋       | 108703/400000 [00:14<00:39, 7332.50it/s] 27%|██▋       | 109450/400000 [00:14<00:39, 7372.08it/s] 28%|██▊       | 110203/400000 [00:14<00:39, 7418.07it/s] 28%|██▊       | 110946/400000 [00:14<00:39, 7406.78it/s] 28%|██▊       | 111704/400000 [00:15<00:38, 7457.02it/s] 28%|██▊       | 112476/400000 [00:15<00:38, 7533.62it/s] 28%|██▊       | 113234/400000 [00:15<00:38, 7545.80it/s] 28%|██▊       | 113989/400000 [00:15<00:38, 7505.54it/s] 29%|██▊       | 114740/400000 [00:15<00:38, 7504.44it/s] 29%|██▉       | 115491/400000 [00:15<00:38, 7472.87it/s] 29%|██▉       | 116255/400000 [00:15<00:37, 7519.99it/s] 29%|██▉       | 117008/400000 [00:15<00:38, 7445.09it/s] 29%|██▉       | 117753/400000 [00:15<00:37, 7438.81it/s] 30%|██▉       | 118518/400000 [00:15<00:37, 7500.74it/s] 30%|██▉       | 119269/400000 [00:16<00:37, 7450.56it/s] 30%|███       | 120015/400000 [00:16<00:37, 7369.98it/s] 30%|███       | 120788/400000 [00:16<00:37, 7473.33it/s] 30%|███       | 121554/400000 [00:16<00:36, 7525.75it/s] 31%|███       | 122308/400000 [00:16<00:37, 7392.34it/s] 31%|███       | 123049/400000 [00:16<00:38, 7229.60it/s] 31%|███       | 123787/400000 [00:16<00:37, 7272.92it/s] 31%|███       | 124544/400000 [00:16<00:37, 7357.01it/s] 31%|███▏      | 125284/400000 [00:16<00:37, 7367.70it/s] 32%|███▏      | 126022/400000 [00:16<00:37, 7283.89it/s] 32%|███▏      | 126752/400000 [00:17<00:38, 7167.77it/s] 32%|███▏      | 127515/400000 [00:17<00:37, 7300.14it/s] 32%|███▏      | 128263/400000 [00:17<00:36, 7352.60it/s] 32%|███▏      | 129025/400000 [00:17<00:36, 7429.59it/s] 32%|███▏      | 129769/400000 [00:17<00:36, 7409.18it/s] 33%|███▎      | 130518/400000 [00:17<00:36, 7431.40it/s] 33%|███▎      | 131276/400000 [00:17<00:35, 7475.28it/s] 33%|███▎      | 132039/400000 [00:17<00:35, 7518.87it/s] 33%|███▎      | 132792/400000 [00:17<00:35, 7471.42it/s] 33%|███▎      | 133540/400000 [00:17<00:36, 7399.70it/s] 34%|███▎      | 134297/400000 [00:18<00:35, 7448.29it/s] 34%|███▍      | 135070/400000 [00:18<00:35, 7528.85it/s] 34%|███▍      | 135824/400000 [00:18<00:35, 7521.48it/s] 34%|███▍      | 136577/400000 [00:18<00:35, 7488.32it/s] 34%|███▍      | 137330/400000 [00:18<00:35, 7499.90it/s] 35%|███▍      | 138088/400000 [00:18<00:34, 7522.16it/s] 35%|███▍      | 138846/400000 [00:18<00:34, 7535.20it/s] 35%|███▍      | 139617/400000 [00:18<00:34, 7586.67it/s] 35%|███▌      | 140376/400000 [00:18<00:34, 7551.53it/s] 35%|███▌      | 141132/400000 [00:19<00:34, 7468.12it/s] 35%|███▌      | 141885/400000 [00:19<00:34, 7484.36it/s] 36%|███▌      | 142634/400000 [00:19<00:34, 7420.96it/s] 36%|███▌      | 143380/400000 [00:19<00:34, 7432.23it/s] 36%|███▌      | 144136/400000 [00:19<00:34, 7467.51it/s] 36%|███▌      | 144900/400000 [00:19<00:33, 7516.79it/s] 36%|███▋      | 145652/400000 [00:19<00:34, 7461.70it/s] 37%|███▋      | 146423/400000 [00:19<00:33, 7531.99it/s] 37%|███▋      | 147177/400000 [00:19<00:33, 7531.84it/s] 37%|███▋      | 147932/400000 [00:19<00:33, 7536.60it/s] 37%|███▋      | 148686/400000 [00:20<00:33, 7469.54it/s] 37%|███▋      | 149438/400000 [00:20<00:33, 7483.98it/s] 38%|███▊      | 150194/400000 [00:20<00:33, 7505.19it/s] 38%|███▊      | 150958/400000 [00:20<00:33, 7542.73it/s] 38%|███▊      | 151713/400000 [00:20<00:32, 7530.26it/s] 38%|███▊      | 152467/400000 [00:20<00:33, 7493.25it/s] 38%|███▊      | 153217/400000 [00:20<00:33, 7466.77it/s] 38%|███▊      | 153964/400000 [00:20<00:33, 7418.41it/s] 39%|███▊      | 154720/400000 [00:20<00:32, 7460.21it/s] 39%|███▉      | 155482/400000 [00:20<00:32, 7506.57it/s] 39%|███▉      | 156238/400000 [00:21<00:32, 7521.43it/s] 39%|███▉      | 156992/400000 [00:21<00:32, 7526.64it/s] 39%|███▉      | 157745/400000 [00:21<00:32, 7507.05it/s] 40%|███▉      | 158496/400000 [00:21<00:32, 7501.11it/s] 40%|███▉      | 159247/400000 [00:21<00:32, 7487.41it/s] 40%|███▉      | 159996/400000 [00:21<00:32, 7344.33it/s] 40%|████      | 160746/400000 [00:21<00:32, 7387.69it/s] 40%|████      | 161486/400000 [00:21<00:32, 7385.67it/s] 41%|████      | 162255/400000 [00:21<00:31, 7471.83it/s] 41%|████      | 163006/400000 [00:21<00:31, 7482.32it/s] 41%|████      | 163763/400000 [00:22<00:31, 7506.32it/s] 41%|████      | 164531/400000 [00:22<00:31, 7556.62it/s] 41%|████▏     | 165287/400000 [00:22<00:31, 7554.48it/s] 42%|████▏     | 166056/400000 [00:22<00:30, 7592.68it/s] 42%|████▏     | 166816/400000 [00:22<00:30, 7563.92it/s] 42%|████▏     | 167573/400000 [00:22<00:30, 7527.86it/s] 42%|████▏     | 168326/400000 [00:22<00:31, 7426.33it/s] 42%|████▏     | 169070/400000 [00:22<00:31, 7400.29it/s] 42%|████▏     | 169827/400000 [00:22<00:30, 7449.38it/s] 43%|████▎     | 170583/400000 [00:22<00:30, 7479.70it/s] 43%|████▎     | 171332/400000 [00:23<00:30, 7470.46it/s] 43%|████▎     | 172088/400000 [00:23<00:30, 7496.99it/s] 43%|████▎     | 172838/400000 [00:23<00:30, 7481.57it/s] 43%|████▎     | 173601/400000 [00:23<00:30, 7524.46it/s] 44%|████▎     | 174354/400000 [00:23<00:30, 7504.31it/s] 44%|████▍     | 175105/400000 [00:23<00:30, 7463.14it/s] 44%|████▍     | 175854/400000 [00:23<00:30, 7468.01it/s] 44%|████▍     | 176606/400000 [00:23<00:29, 7483.08it/s] 44%|████▍     | 177374/400000 [00:23<00:29, 7539.69it/s] 45%|████▍     | 178130/400000 [00:23<00:29, 7543.51it/s] 45%|████▍     | 178885/400000 [00:24<00:29, 7539.64it/s] 45%|████▍     | 179651/400000 [00:24<00:29, 7573.33it/s] 45%|████▌     | 180409/400000 [00:24<00:29, 7558.20it/s] 45%|████▌     | 181165/400000 [00:24<00:29, 7465.63it/s] 45%|████▌     | 181927/400000 [00:24<00:29, 7511.13it/s] 46%|████▌     | 182683/400000 [00:24<00:28, 7525.22it/s] 46%|████▌     | 183447/400000 [00:24<00:28, 7558.90it/s] 46%|████▌     | 184204/400000 [00:24<00:28, 7551.41it/s] 46%|████▌     | 184969/400000 [00:24<00:28, 7580.59it/s] 46%|████▋     | 185728/400000 [00:24<00:28, 7573.01it/s] 47%|████▋     | 186486/400000 [00:25<00:28, 7481.91it/s] 47%|████▋     | 187242/400000 [00:25<00:28, 7504.67it/s] 47%|████▋     | 187993/400000 [00:25<00:28, 7502.69it/s] 47%|████▋     | 188755/400000 [00:25<00:28, 7536.89it/s] 47%|████▋     | 189509/400000 [00:25<00:27, 7521.68it/s] 48%|████▊     | 190262/400000 [00:25<00:27, 7520.01it/s] 48%|████▊     | 191021/400000 [00:25<00:27, 7538.21it/s] 48%|████▊     | 191777/400000 [00:25<00:27, 7542.10it/s] 48%|████▊     | 192535/400000 [00:25<00:27, 7552.69it/s] 48%|████▊     | 193298/400000 [00:25<00:27, 7573.34it/s] 49%|████▊     | 194056/400000 [00:26<00:27, 7439.68it/s] 49%|████▊     | 194801/400000 [00:26<00:27, 7411.89it/s] 49%|████▉     | 195543/400000 [00:26<00:27, 7413.52it/s] 49%|████▉     | 196285/400000 [00:26<00:27, 7394.52it/s] 49%|████▉     | 197045/400000 [00:26<00:27, 7452.79it/s] 49%|████▉     | 197810/400000 [00:26<00:26, 7509.22it/s] 50%|████▉     | 198568/400000 [00:26<00:26, 7529.92it/s] 50%|████▉     | 199326/400000 [00:26<00:26, 7542.95it/s] 50%|█████     | 200099/400000 [00:26<00:26, 7595.83it/s] 50%|█████     | 200859/400000 [00:26<00:26, 7554.17it/s] 50%|█████     | 201615/400000 [00:27<00:26, 7526.69it/s] 51%|█████     | 202377/400000 [00:27<00:26, 7552.10it/s] 51%|█████     | 203133/400000 [00:27<00:26, 7542.27it/s] 51%|█████     | 203888/400000 [00:27<00:26, 7540.44it/s] 51%|█████     | 204643/400000 [00:27<00:26, 7489.49it/s] 51%|█████▏    | 205404/400000 [00:27<00:25, 7523.22it/s] 52%|█████▏    | 206157/400000 [00:27<00:25, 7505.62it/s] 52%|█████▏    | 206908/400000 [00:27<00:25, 7491.03it/s] 52%|█████▏    | 207658/400000 [00:27<00:25, 7465.98it/s] 52%|█████▏    | 208405/400000 [00:27<00:25, 7430.05it/s] 52%|█████▏    | 209149/400000 [00:28<00:25, 7402.70it/s] 52%|█████▏    | 209890/400000 [00:28<00:25, 7367.05it/s] 53%|█████▎    | 210637/400000 [00:28<00:25, 7392.62it/s] 53%|█████▎    | 211386/400000 [00:28<00:25, 7418.77it/s] 53%|█████▎    | 212146/400000 [00:28<00:25, 7472.20it/s] 53%|█████▎    | 212896/400000 [00:28<00:25, 7479.87it/s] 53%|█████▎    | 213668/400000 [00:28<00:24, 7548.83it/s] 54%|█████▎    | 214429/400000 [00:28<00:24, 7566.03it/s] 54%|█████▍    | 215197/400000 [00:28<00:24, 7599.43it/s] 54%|█████▍    | 215965/400000 [00:28<00:24, 7622.73it/s] 54%|█████▍    | 216728/400000 [00:29<00:24, 7613.15it/s] 54%|█████▍    | 217490/400000 [00:29<00:23, 7611.84it/s] 55%|█████▍    | 218252/400000 [00:29<00:24, 7532.58it/s] 55%|█████▍    | 219006/400000 [00:29<00:24, 7374.70it/s] 55%|█████▍    | 219745/400000 [00:29<00:24, 7373.58it/s] 55%|█████▌    | 220494/400000 [00:29<00:24, 7407.29it/s] 55%|█████▌    | 221236/400000 [00:29<00:24, 7389.79it/s] 55%|█████▌    | 221978/400000 [00:29<00:24, 7397.12it/s] 56%|█████▌    | 222718/400000 [00:29<00:24, 7306.75it/s] 56%|█████▌    | 223467/400000 [00:29<00:23, 7358.39it/s] 56%|█████▌    | 224229/400000 [00:30<00:23, 7432.99it/s] 56%|█████▌    | 224989/400000 [00:30<00:23, 7480.04it/s] 56%|█████▋    | 225742/400000 [00:30<00:23, 7493.37it/s] 57%|█████▋    | 226500/400000 [00:30<00:23, 7516.96it/s] 57%|█████▋    | 227260/400000 [00:30<00:22, 7539.06it/s] 57%|█████▋    | 228023/400000 [00:30<00:22, 7564.88it/s] 57%|█████▋    | 228780/400000 [00:30<00:22, 7564.88it/s] 57%|█████▋    | 229542/400000 [00:30<00:22, 7579.79it/s] 58%|█████▊    | 230303/400000 [00:30<00:22, 7587.96it/s] 58%|█████▊    | 231067/400000 [00:30<00:22, 7601.52it/s] 58%|█████▊    | 231828/400000 [00:31<00:22, 7595.77it/s] 58%|█████▊    | 232588/400000 [00:31<00:22, 7583.21it/s] 58%|█████▊    | 233347/400000 [00:31<00:22, 7490.44it/s] 59%|█████▊    | 234097/400000 [00:31<00:22, 7479.66it/s] 59%|█████▊    | 234846/400000 [00:31<00:22, 7475.94it/s] 59%|█████▉    | 235594/400000 [00:31<00:22, 7292.42it/s] 59%|█████▉    | 236357/400000 [00:31<00:22, 7388.37it/s] 59%|█████▉    | 237109/400000 [00:31<00:21, 7426.10it/s] 59%|█████▉    | 237853/400000 [00:31<00:21, 7428.57it/s] 60%|█████▉    | 238597/400000 [00:32<00:22, 7325.78it/s] 60%|█████▉    | 239348/400000 [00:32<00:21, 7378.25it/s] 60%|██████    | 240110/400000 [00:32<00:21, 7448.30it/s] 60%|██████    | 240864/400000 [00:32<00:21, 7473.18it/s] 60%|██████    | 241616/400000 [00:32<00:21, 7486.88it/s] 61%|██████    | 242365/400000 [00:32<00:21, 7486.65it/s] 61%|██████    | 243117/400000 [00:32<00:20, 7495.19it/s] 61%|██████    | 243869/400000 [00:32<00:20, 7501.45it/s] 61%|██████    | 244622/400000 [00:32<00:20, 7509.03it/s] 61%|██████▏   | 245373/400000 [00:32<00:21, 7320.16it/s] 62%|██████▏   | 246107/400000 [00:33<00:21, 7315.33it/s] 62%|██████▏   | 246840/400000 [00:33<00:21, 7267.41it/s] 62%|██████▏   | 247568/400000 [00:33<00:21, 7203.09it/s] 62%|██████▏   | 248289/400000 [00:33<00:21, 7191.21it/s] 62%|██████▏   | 249029/400000 [00:33<00:20, 7250.43it/s] 62%|██████▏   | 249788/400000 [00:33<00:20, 7346.85it/s] 63%|██████▎   | 250544/400000 [00:33<00:20, 7407.37it/s] 63%|██████▎   | 251307/400000 [00:33<00:19, 7471.57it/s] 63%|██████▎   | 252075/400000 [00:33<00:19, 7530.76it/s] 63%|██████▎   | 252829/400000 [00:33<00:19, 7500.38it/s] 63%|██████▎   | 253595/400000 [00:34<00:19, 7547.11it/s] 64%|██████▎   | 254366/400000 [00:34<00:19, 7593.57it/s] 64%|██████▍   | 255139/400000 [00:34<00:18, 7633.55it/s] 64%|██████▍   | 255903/400000 [00:34<00:19, 7537.71it/s] 64%|██████▍   | 256661/400000 [00:34<00:18, 7547.79it/s] 64%|██████▍   | 257417/400000 [00:34<00:19, 7446.73it/s] 65%|██████▍   | 258170/400000 [00:34<00:18, 7469.46it/s] 65%|██████▍   | 258918/400000 [00:34<00:18, 7428.60it/s] 65%|██████▍   | 259665/400000 [00:34<00:18, 7438.46it/s] 65%|██████▌   | 260410/400000 [00:34<00:18, 7381.79it/s] 65%|██████▌   | 261149/400000 [00:35<00:18, 7354.89it/s] 65%|██████▌   | 261902/400000 [00:35<00:18, 7405.91it/s] 66%|██████▌   | 262643/400000 [00:35<00:18, 7354.31it/s] 66%|██████▌   | 263380/400000 [00:35<00:18, 7358.80it/s] 66%|██████▌   | 264118/400000 [00:35<00:18, 7362.97it/s] 66%|██████▌   | 264855/400000 [00:35<00:18, 7362.59it/s] 66%|██████▋   | 265594/400000 [00:35<00:18, 7370.66it/s] 67%|██████▋   | 266361/400000 [00:35<00:17, 7456.95it/s] 67%|██████▋   | 267107/400000 [00:35<00:17, 7444.85it/s] 67%|██████▋   | 267862/400000 [00:35<00:17, 7473.68it/s] 67%|██████▋   | 268610/400000 [00:36<00:17, 7352.32it/s] 67%|██████▋   | 269379/400000 [00:36<00:17, 7447.70it/s] 68%|██████▊   | 270141/400000 [00:36<00:17, 7497.99it/s] 68%|██████▊   | 270903/400000 [00:36<00:17, 7532.12it/s] 68%|██████▊   | 271657/400000 [00:36<00:17, 7514.86it/s] 68%|██████▊   | 272409/400000 [00:36<00:17, 7360.65it/s] 68%|██████▊   | 273146/400000 [00:36<00:17, 7257.47it/s] 68%|██████▊   | 273913/400000 [00:36<00:17, 7374.46it/s] 69%|██████▊   | 274675/400000 [00:36<00:16, 7446.28it/s] 69%|██████▉   | 275436/400000 [00:36<00:16, 7494.40it/s] 69%|██████▉   | 276187/400000 [00:37<00:17, 7223.06it/s] 69%|██████▉   | 276947/400000 [00:37<00:16, 7329.83it/s] 69%|██████▉   | 277696/400000 [00:37<00:16, 7376.39it/s] 70%|██████▉   | 278436/400000 [00:37<00:16, 7356.16it/s] 70%|██████▉   | 279173/400000 [00:37<00:16, 7338.07it/s] 70%|██████▉   | 279920/400000 [00:37<00:16, 7375.18it/s] 70%|███████   | 280674/400000 [00:37<00:16, 7421.07it/s] 70%|███████   | 281424/400000 [00:37<00:15, 7443.18it/s] 71%|███████   | 282188/400000 [00:37<00:15, 7501.10it/s] 71%|███████   | 282946/400000 [00:37<00:15, 7523.01it/s] 71%|███████   | 283699/400000 [00:38<00:15, 7382.22it/s] 71%|███████   | 284460/400000 [00:38<00:15, 7447.17it/s] 71%|███████▏  | 285217/400000 [00:38<00:15, 7482.43it/s] 71%|███████▏  | 285966/400000 [00:38<00:15, 7451.05it/s] 72%|███████▏  | 286724/400000 [00:38<00:15, 7486.75it/s] 72%|███████▏  | 287489/400000 [00:38<00:14, 7534.75it/s] 72%|███████▏  | 288250/400000 [00:38<00:14, 7556.62it/s] 72%|███████▏  | 289006/400000 [00:38<00:14, 7549.73it/s] 72%|███████▏  | 289762/400000 [00:38<00:14, 7493.05it/s] 73%|███████▎  | 290519/400000 [00:38<00:14, 7515.92it/s] 73%|███████▎  | 291281/400000 [00:39<00:14, 7546.05it/s] 73%|███████▎  | 292038/400000 [00:39<00:14, 7552.93it/s] 73%|███████▎  | 292806/400000 [00:39<00:14, 7587.58it/s] 73%|███████▎  | 293565/400000 [00:39<00:14, 7414.40it/s] 74%|███████▎  | 294308/400000 [00:39<00:14, 7394.48it/s] 74%|███████▍  | 295049/400000 [00:39<00:14, 7385.04it/s] 74%|███████▍  | 295788/400000 [00:39<00:14, 7363.65it/s] 74%|███████▍  | 296525/400000 [00:39<00:14, 7058.93it/s] 74%|███████▍  | 297286/400000 [00:39<00:14, 7214.81it/s] 75%|███████▍  | 298022/400000 [00:40<00:14, 7256.19it/s] 75%|███████▍  | 298774/400000 [00:40<00:13, 7332.73it/s] 75%|███████▍  | 299538/400000 [00:40<00:13, 7420.99it/s] 75%|███████▌  | 300307/400000 [00:40<00:13, 7499.10it/s] 75%|███████▌  | 301066/400000 [00:40<00:13, 7525.70it/s] 75%|███████▌  | 301820/400000 [00:40<00:13, 7404.27it/s] 76%|███████▌  | 302588/400000 [00:40<00:13, 7482.21it/s] 76%|███████▌  | 303357/400000 [00:40<00:12, 7542.97it/s] 76%|███████▌  | 304123/400000 [00:40<00:12, 7577.64it/s] 76%|███████▌  | 304882/400000 [00:40<00:12, 7580.84it/s] 76%|███████▋  | 305641/400000 [00:41<00:12, 7544.38it/s] 77%|███████▋  | 306402/400000 [00:41<00:12, 7561.85it/s] 77%|███████▋  | 307159/400000 [00:41<00:12, 7554.81it/s] 77%|███████▋  | 307915/400000 [00:41<00:12, 7536.99it/s] 77%|███████▋  | 308674/400000 [00:41<00:12, 7552.72it/s] 77%|███████▋  | 309430/400000 [00:41<00:12, 7480.01it/s] 78%|███████▊  | 310179/400000 [00:41<00:12, 7166.35it/s] 78%|███████▊  | 310917/400000 [00:41<00:12, 7229.07it/s] 78%|███████▊  | 311666/400000 [00:41<00:12, 7302.84it/s] 78%|███████▊  | 312406/400000 [00:41<00:11, 7329.45it/s] 78%|███████▊  | 313141/400000 [00:42<00:11, 7328.76it/s] 78%|███████▊  | 313901/400000 [00:42<00:11, 7406.65it/s] 79%|███████▊  | 314663/400000 [00:42<00:11, 7467.78it/s] 79%|███████▉  | 315411/400000 [00:42<00:11, 7452.21it/s] 79%|███████▉  | 316162/400000 [00:42<00:11, 7466.93it/s] 79%|███████▉  | 316910/400000 [00:42<00:11, 7400.90it/s] 79%|███████▉  | 317651/400000 [00:42<00:11, 7386.60it/s] 80%|███████▉  | 318413/400000 [00:42<00:10, 7452.98it/s] 80%|███████▉  | 319178/400000 [00:42<00:10, 7509.22it/s] 80%|███████▉  | 319941/400000 [00:42<00:10, 7544.90it/s] 80%|████████  | 320696/400000 [00:43<00:10, 7488.00it/s] 80%|████████  | 321451/400000 [00:43<00:10, 7505.54it/s] 81%|████████  | 322213/400000 [00:43<00:10, 7539.25it/s] 81%|████████  | 322968/400000 [00:43<00:10, 7322.46it/s] 81%|████████  | 323702/400000 [00:43<00:10, 7315.05it/s] 81%|████████  | 324435/400000 [00:43<00:10, 7251.36it/s] 81%|████████▏ | 325161/400000 [00:43<00:10, 6902.64it/s] 81%|████████▏ | 325896/400000 [00:43<00:10, 7028.51it/s] 82%|████████▏ | 326644/400000 [00:43<00:10, 7156.01it/s] 82%|████████▏ | 327407/400000 [00:43<00:09, 7289.24it/s] 82%|████████▏ | 328139/400000 [00:44<00:09, 7279.63it/s] 82%|████████▏ | 328887/400000 [00:44<00:09, 7338.12it/s] 82%|████████▏ | 329623/400000 [00:44<00:09, 7174.40it/s] 83%|████████▎ | 330360/400000 [00:44<00:09, 7231.27it/s] 83%|████████▎ | 331103/400000 [00:44<00:09, 7289.00it/s] 83%|████████▎ | 331860/400000 [00:44<00:09, 7370.63it/s] 83%|████████▎ | 332604/400000 [00:44<00:09, 7389.71it/s] 83%|████████▎ | 333375/400000 [00:44<00:08, 7482.85it/s] 84%|████████▎ | 334129/400000 [00:44<00:08, 7498.29it/s] 84%|████████▎ | 334880/400000 [00:44<00:08, 7477.35it/s] 84%|████████▍ | 335629/400000 [00:45<00:08, 7359.34it/s] 84%|████████▍ | 336375/400000 [00:45<00:08, 7386.61it/s] 84%|████████▍ | 337117/400000 [00:45<00:08, 7395.63it/s] 84%|████████▍ | 337863/400000 [00:45<00:08, 7413.32it/s] 85%|████████▍ | 338605/400000 [00:45<00:08, 7372.85it/s] 85%|████████▍ | 339343/400000 [00:45<00:08, 7295.42it/s] 85%|████████▌ | 340090/400000 [00:45<00:08, 7345.44it/s] 85%|████████▌ | 340845/400000 [00:45<00:07, 7404.46it/s] 85%|████████▌ | 341620/400000 [00:45<00:07, 7504.72it/s] 86%|████████▌ | 342372/400000 [00:46<00:07, 7499.05it/s] 86%|████████▌ | 343123/400000 [00:46<00:07, 7488.04it/s] 86%|████████▌ | 343881/400000 [00:46<00:07, 7514.41it/s] 86%|████████▌ | 344633/400000 [00:46<00:07, 7480.02it/s] 86%|████████▋ | 345382/400000 [00:46<00:07, 7353.32it/s] 87%|████████▋ | 346143/400000 [00:46<00:07, 7427.53it/s] 87%|████████▋ | 346890/400000 [00:46<00:07, 7437.57it/s] 87%|████████▋ | 347638/400000 [00:46<00:07, 7448.55it/s] 87%|████████▋ | 348401/400000 [00:46<00:06, 7500.28it/s] 87%|████████▋ | 349153/400000 [00:46<00:06, 7506.09it/s] 87%|████████▋ | 349904/400000 [00:47<00:06, 7428.65it/s] 88%|████████▊ | 350648/400000 [00:47<00:06, 7249.56it/s] 88%|████████▊ | 351394/400000 [00:47<00:06, 7310.41it/s] 88%|████████▊ | 352132/400000 [00:47<00:06, 7330.05it/s] 88%|████████▊ | 352866/400000 [00:47<00:06, 7325.33it/s] 88%|████████▊ | 353600/400000 [00:47<00:06, 7320.40it/s] 89%|████████▊ | 354334/400000 [00:47<00:06, 7324.24it/s] 89%|████████▉ | 355067/400000 [00:47<00:06, 6961.61it/s] 89%|████████▉ | 355834/400000 [00:47<00:06, 7158.11it/s] 89%|████████▉ | 356558/400000 [00:47<00:06, 7180.03it/s] 89%|████████▉ | 357286/400000 [00:48<00:05, 7208.49it/s] 90%|████████▉ | 358009/400000 [00:48<00:05, 7189.47it/s] 90%|████████▉ | 358764/400000 [00:48<00:05, 7292.88it/s] 90%|████████▉ | 359520/400000 [00:48<00:05, 7368.35it/s] 90%|█████████ | 360262/400000 [00:48<00:05, 7383.55it/s] 90%|█████████ | 361007/400000 [00:48<00:05, 7400.97it/s] 90%|█████████ | 361748/400000 [00:48<00:05, 7207.69it/s] 91%|█████████ | 362472/400000 [00:48<00:05, 7216.25it/s] 91%|█████████ | 363205/400000 [00:48<00:05, 7249.71it/s] 91%|█████████ | 363939/400000 [00:48<00:04, 7275.87it/s] 91%|█████████ | 364668/400000 [00:49<00:04, 7237.60it/s] 91%|█████████▏| 365406/400000 [00:49<00:04, 7277.55it/s] 92%|█████████▏| 366135/400000 [00:49<00:04, 7244.72it/s] 92%|█████████▏| 366864/400000 [00:49<00:04, 7256.84it/s] 92%|█████████▏| 367617/400000 [00:49<00:04, 7336.24it/s] 92%|█████████▏| 368369/400000 [00:49<00:04, 7388.20it/s] 92%|█████████▏| 369109/400000 [00:49<00:04, 7357.42it/s] 92%|█████████▏| 369868/400000 [00:49<00:04, 7425.25it/s] 93%|█████████▎| 370617/400000 [00:49<00:03, 7441.86it/s] 93%|█████████▎| 371362/400000 [00:49<00:03, 7440.45it/s] 93%|█████████▎| 372107/400000 [00:50<00:03, 7437.73it/s] 93%|█████████▎| 372851/400000 [00:50<00:03, 7416.55it/s] 93%|█████████▎| 373609/400000 [00:50<00:03, 7462.39it/s] 94%|█████████▎| 374368/400000 [00:50<00:03, 7499.20it/s] 94%|█████████▍| 375119/400000 [00:50<00:03, 7422.35it/s] 94%|█████████▍| 375862/400000 [00:50<00:03, 7355.32it/s] 94%|█████████▍| 376598/400000 [00:50<00:03, 7342.09it/s] 94%|█████████▍| 377333/400000 [00:50<00:03, 7343.59it/s] 95%|█████████▍| 378068/400000 [00:50<00:03, 7009.14it/s] 95%|█████████▍| 378818/400000 [00:50<00:02, 7149.17it/s] 95%|█████████▍| 379584/400000 [00:51<00:02, 7293.37it/s] 95%|█████████▌| 380334/400000 [00:51<00:02, 7353.82it/s] 95%|█████████▌| 381094/400000 [00:51<00:02, 7425.57it/s] 95%|█████████▌| 381839/400000 [00:51<00:02, 7396.66it/s] 96%|█████████▌| 382586/400000 [00:51<00:02, 7418.36it/s] 96%|█████████▌| 383329/400000 [00:51<00:02, 7411.67it/s] 96%|█████████▌| 384088/400000 [00:51<00:02, 7462.16it/s] 96%|█████████▌| 384846/400000 [00:51<00:02, 7495.51it/s] 96%|█████████▋| 385616/400000 [00:51<00:01, 7553.33it/s] 97%|█████████▋| 386383/400000 [00:51<00:01, 7585.76it/s] 97%|█████████▋| 387145/400000 [00:52<00:01, 7594.18it/s] 97%|█████████▋| 387906/400000 [00:52<00:01, 7598.04it/s] 97%|█████████▋| 388671/400000 [00:52<00:01, 7610.90it/s] 97%|█████████▋| 389436/400000 [00:52<00:01, 7622.36it/s] 98%|█████████▊| 390199/400000 [00:52<00:01, 7570.14it/s] 98%|█████████▊| 390957/400000 [00:52<00:01, 7556.32it/s] 98%|█████████▊| 391713/400000 [00:52<00:01, 7510.12it/s] 98%|█████████▊| 392465/400000 [00:52<00:01, 7441.49it/s] 98%|█████████▊| 393210/400000 [00:52<00:00, 7437.17it/s] 98%|█████████▊| 393954/400000 [00:53<00:00, 7320.10it/s] 99%|█████████▊| 394691/400000 [00:53<00:00, 7332.54it/s] 99%|█████████▉| 395425/400000 [00:53<00:00, 7333.73it/s] 99%|█████████▉| 396159/400000 [00:53<00:00, 7313.66it/s] 99%|█████████▉| 396891/400000 [00:53<00:00, 7271.15it/s] 99%|█████████▉| 397623/400000 [00:53<00:00, 7282.90it/s]100%|█████████▉| 398374/400000 [00:53<00:00, 7349.24it/s]100%|█████████▉| 399110/400000 [00:53<00:00, 7222.87it/s]100%|█████████▉| 399833/400000 [00:53<00:00, 7203.44it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7429.11it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f92bf649940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011388629461733227 	 Accuracy: 50
Train Epoch: 1 	 Loss: 0.011696599199620378 	 Accuracy: 49

  model saves at 49% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15930 out of table with 15903 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15930 out of table with 15903 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-14 17:25:06.637183: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 17:25:06.641522: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-14 17:25:06.641706: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d6eb31f1b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 17:25:06.641742: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f926c028d30> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7510 - accuracy: 0.4945
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6002 - accuracy: 0.5043 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.5976 - accuracy: 0.5045
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6452 - accuracy: 0.5014
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6666 - accuracy: 0.5000
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6863 - accuracy: 0.4987
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.7050 - accuracy: 0.4975
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7126 - accuracy: 0.4970
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6942 - accuracy: 0.4982
11000/25000 [============>.................] - ETA: 4s - loss: 7.6722 - accuracy: 0.4996
12000/25000 [=============>................] - ETA: 4s - loss: 7.6602 - accuracy: 0.5004
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6466 - accuracy: 0.5013
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6480 - accuracy: 0.5012
15000/25000 [=================>............] - ETA: 3s - loss: 7.6482 - accuracy: 0.5012
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6417 - accuracy: 0.5016
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6152 - accuracy: 0.5034
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6138 - accuracy: 0.5034
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6117 - accuracy: 0.5036
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6168 - accuracy: 0.5033
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6272 - accuracy: 0.5026
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6387 - accuracy: 0.5018
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6626 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6577 - accuracy: 0.5006
25000/25000 [==============================] - 10s 404us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f921f411198> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f923c7aa128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.5020 - crf_viterbi_accuracy: 0.1200 - val_loss: 1.4507 - val_crf_viterbi_accuracy: 0.1067

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

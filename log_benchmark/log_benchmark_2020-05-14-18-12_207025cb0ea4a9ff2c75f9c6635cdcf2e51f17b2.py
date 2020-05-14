
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f5bd4075f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 18:12:34.633904
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 18:12:34.637539
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 18:12:34.640696
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 18:12:34.644858
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f5bdfe3f438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 357482.3125
Epoch 2/10

1/1 [==============================] - 0s 91ms/step - loss: 279132.4375
Epoch 3/10

1/1 [==============================] - 0s 95ms/step - loss: 153077.8125
Epoch 4/10

1/1 [==============================] - 0s 94ms/step - loss: 65405.6992
Epoch 5/10

1/1 [==============================] - 0s 99ms/step - loss: 29548.0781
Epoch 6/10

1/1 [==============================] - 0s 94ms/step - loss: 15377.3818
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 9251.8555
Epoch 8/10

1/1 [==============================] - 0s 104ms/step - loss: 6291.7578
Epoch 9/10

1/1 [==============================] - 0s 94ms/step - loss: 4608.5645
Epoch 10/10

1/1 [==============================] - 0s 91ms/step - loss: 3605.7849

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.21314919e+00  1.05765796e+00 -1.28621137e+00  6.71472847e-02
   1.11060083e-01 -2.64076614e+00 -8.68176222e-02  7.83567429e-02
   4.06115949e-01 -1.04566312e+00  9.64360118e-01  2.75497627e+00
  -9.62969422e-01 -7.73531675e-01  1.55046448e-01 -1.11600924e+00
   1.56907487e+00  2.84917712e-01 -5.17537594e-01 -3.49112093e-01
   4.88335073e-01 -1.58784986e+00 -2.49241233e-01 -1.69495118e+00
   1.27107108e+00 -1.46714187e+00  1.87309861e+00  6.23041749e-01
  -1.41751337e+00 -4.76604521e-01  3.19197387e-01  1.09641969e+00
   3.89570892e-01 -4.80867982e-01 -4.78850156e-01 -1.00719586e-01
   4.55125242e-01  3.71593475e-01 -6.11431301e-01 -1.01511943e+00
   2.57103491e+00  2.32241440e+00 -1.71569169e+00  6.51814342e-02
  -2.02734184e+00 -8.77143264e-01 -1.48988962e-01 -3.50663662e-02
  -2.29326510e+00  4.74739552e-01  7.68005848e-03 -4.87202883e-01
   1.67626619e+00  8.25207651e-01 -5.85618377e-01 -8.32360506e-01
   1.05657947e+00 -7.16147006e-01  2.34228897e+00 -7.08148956e-01
   2.73074716e-01  1.16509295e+01  1.07903585e+01  1.14937191e+01
   1.29680738e+01  1.16684418e+01  1.01846428e+01  1.07254095e+01
   1.26077452e+01  1.18716860e+01  1.20211840e+01  1.22501526e+01
   1.28368559e+01  1.07008448e+01  1.22126474e+01  9.69617653e+00
   1.21148758e+01  1.08270016e+01  1.29856911e+01  1.42134542e+01
   1.24998398e+01  9.21500397e+00  1.26888962e+01  1.21593304e+01
   1.25873432e+01  1.15120916e+01  1.15205431e+01  1.06303864e+01
   1.19108725e+01  1.31607561e+01  1.35049477e+01  1.16480083e+01
   1.14232101e+01  1.25963364e+01  1.28127613e+01  1.09081631e+01
   1.14018793e+01  1.02070389e+01  1.09166861e+01  1.17808857e+01
   1.33696976e+01  1.14299469e+01  9.56105995e+00  1.07140646e+01
   1.26260052e+01  1.01359167e+01  1.22307034e+01  1.38157616e+01
   1.23351994e+01  1.16657162e+01  1.15154123e+01  1.10113649e+01
   1.27872992e+01  1.06085491e+01  1.29665537e+01  1.33254538e+01
   1.24254465e+01  1.11910677e+01  1.16345510e+01  1.03678598e+01
  -1.53750813e+00  4.01929617e-01  2.75879234e-01  3.44807029e-01
  -1.36779594e+00  1.47337228e-01  6.15029931e-01  4.12063807e-01
   2.99657285e-01  1.17678165e+00  1.43079901e+00  1.78715956e+00
  -1.37639654e+00 -1.35426962e+00 -6.30223334e-01 -1.99707961e+00
  -1.56978166e+00  5.81730962e-01  7.93725610e-01  1.15865541e+00
   1.37896466e+00 -6.57164097e-01  7.62240738e-02 -3.11509550e-01
  -1.68008161e+00  8.28418970e-01 -1.31541383e+00  9.15528536e-01
  -1.89535451e+00 -1.26503611e+00  1.14414144e+00 -2.26327255e-01
  -1.69424748e+00 -1.14996731e+00 -1.90874875e+00  9.76810455e-01
   1.06211293e+00  1.91416025e+00 -1.15895891e+00 -3.37829083e-01
   8.86589885e-02  1.93579388e+00  8.75151992e-01  2.57709593e-01
   2.22239047e-01  2.82971764e+00 -8.42605233e-01 -1.02114248e+00
  -1.22648275e+00 -7.77464509e-02 -1.48424840e+00  1.72910452e-01
   6.20574474e-01  2.02681255e+00 -6.03323340e-01 -7.38283396e-02
   5.36177278e-01 -1.63643801e+00  6.95493877e-01 -2.41567230e+00
   1.24234378e-01  3.68129849e-01  2.58378530e+00  1.56235850e+00
   3.85822773e-01  3.87650371e-01  1.75716865e+00  3.33117342e+00
   1.67188883e+00  1.25289130e+00  1.16140580e+00  2.01466990e+00
   1.04628801e-01  6.89351082e-01  3.23168755e-01  1.60304368e-01
   1.80424452e-01  2.81475687e+00  5.00422716e-01  2.18070507e-01
   1.88858449e-01  1.89246178e+00  1.58847129e+00  1.60757947e+00
   2.45315313e+00  1.19512272e+00  2.46274567e+00  2.73085773e-01
   2.61420393e+00  1.13752663e-01  2.00220823e+00  4.35701489e-01
   1.40838742e-01  1.92462766e+00  5.07085383e-01  1.30408049e+00
   7.30253041e-01  1.40628338e+00  1.63938594e+00  2.17975569e+00
   3.12221169e-01  1.77920222e-01  2.07956016e-01  8.88154507e-01
   9.58202720e-01  1.49807084e+00  4.25714254e-01  3.23612571e-01
   4.02110159e-01  7.42567778e-01  1.12401450e+00  8.58405650e-01
   7.31171787e-01  1.76535904e+00  2.28667045e+00  2.10602701e-01
   2.20112467e+00  1.09261727e+00  4.65798140e-01  1.64924169e+00
   7.12022305e-01  1.09186049e+01  1.25674868e+01  1.21935024e+01
   1.12054806e+01  1.23357401e+01  1.14097376e+01  1.10769968e+01
   1.25100279e+01  1.15158129e+01  1.24039125e+01  1.29525499e+01
   1.25987206e+01  1.22268200e+01  1.19499178e+01  1.13835325e+01
   1.14126024e+01  1.20750723e+01  1.06970711e+01  1.25861301e+01
   1.29327230e+01  1.05798645e+01  1.07133579e+01  1.15872650e+01
   1.18985415e+01  1.14086723e+01  1.30210829e+01  1.19320784e+01
   1.04611921e+01  1.14140902e+01  1.22733946e+01  1.14113188e+01
   9.68992615e+00  1.16849136e+01  1.06917677e+01  1.26407967e+01
   1.22994070e+01  1.10893879e+01  1.16488695e+01  1.18480921e+01
   1.16914024e+01  1.18969126e+01  1.05654202e+01  1.18027916e+01
   1.25170307e+01  1.27380629e+01  1.03972549e+01  1.20831146e+01
   1.15240431e+01  1.18106089e+01  1.25671120e+01  1.31852579e+01
   9.73776722e+00  1.32834578e+01  1.24493790e+01  1.23722944e+01
   1.27466564e+01  1.14072208e+01  1.30688887e+01  1.04175768e+01
   1.47191703e-01  5.30394971e-01  1.89572334e+00  1.22720087e+00
   1.31465340e+00  1.12330437e+00  1.04152203e+00  3.16881323e+00
   1.02726460e+00  5.23888469e-01  1.64266467e-01  6.72395051e-01
   3.13202977e-01  3.57839763e-01  4.88841832e-01  1.26251066e+00
   6.38702691e-01  1.50019741e+00  2.89072752e-01  1.68796813e+00
   7.54321337e-01  1.66776049e+00  1.64525664e+00  1.39089632e+00
   3.86730385e+00  1.37024593e+00  1.90924764e+00  3.44384730e-01
   2.91755438e+00  1.92071152e+00  2.65250206e-01  1.54283226e+00
   2.71589398e-01  1.49321675e+00  2.78594089e+00  9.80027080e-01
   3.11317265e-01  1.21519423e+00  2.01376581e+00  3.32582831e-01
   5.46275437e-01  1.64551687e+00  6.34168208e-01  1.04820490e-01
   2.52548909e+00  1.83897090e+00  3.22519875e+00  1.34277344e-01
   6.59952044e-01  5.26297688e-02  1.16406858e+00  6.24448299e-01
   4.61529493e-01  9.52689648e-02  5.03026247e-01  2.81410575e-01
   1.56342530e+00  6.63411438e-01  7.94582963e-01  2.95541763e-01
  -1.41913471e+01  9.25084591e+00 -1.26647682e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 18:12:42.621194
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   89.2454
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 18:12:42.624986
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8004.16
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 18:12:42.628362
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    89.085
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 18:12:42.631626
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -715.829
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140032016827728
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140029504401984
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140029504402488
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140029504402992
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140029504403496
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140029504404000

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f5bd3d35b70> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.532255
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.498556
grad_step = 000002, loss = 0.474426
grad_step = 000003, loss = 0.450433
grad_step = 000004, loss = 0.423814
grad_step = 000005, loss = 0.398055
grad_step = 000006, loss = 0.379989
grad_step = 000007, loss = 0.366820
grad_step = 000008, loss = 0.354547
grad_step = 000009, loss = 0.337443
grad_step = 000010, loss = 0.322820
grad_step = 000011, loss = 0.314035
grad_step = 000012, loss = 0.304388
grad_step = 000013, loss = 0.292579
grad_step = 000014, loss = 0.280602
grad_step = 000015, loss = 0.269992
grad_step = 000016, loss = 0.260573
grad_step = 000017, loss = 0.250824
grad_step = 000018, loss = 0.239747
grad_step = 000019, loss = 0.227928
grad_step = 000020, loss = 0.217287
grad_step = 000021, loss = 0.208985
grad_step = 000022, loss = 0.201500
grad_step = 000023, loss = 0.191949
grad_step = 000024, loss = 0.181041
grad_step = 000025, loss = 0.170691
grad_step = 000026, loss = 0.162038
grad_step = 000027, loss = 0.154670
grad_step = 000028, loss = 0.147297
grad_step = 000029, loss = 0.139421
grad_step = 000030, loss = 0.131724
grad_step = 000031, loss = 0.124746
grad_step = 000032, loss = 0.118186
grad_step = 000033, loss = 0.111643
grad_step = 000034, loss = 0.104914
grad_step = 000035, loss = 0.098333
grad_step = 000036, loss = 0.092420
grad_step = 000037, loss = 0.086944
grad_step = 000038, loss = 0.081363
grad_step = 000039, loss = 0.075917
grad_step = 000040, loss = 0.071041
grad_step = 000041, loss = 0.066599
grad_step = 000042, loss = 0.062207
grad_step = 000043, loss = 0.057836
grad_step = 000044, loss = 0.053745
grad_step = 000045, loss = 0.049984
grad_step = 000046, loss = 0.046450
grad_step = 000047, loss = 0.043149
grad_step = 000048, loss = 0.040034
grad_step = 000049, loss = 0.037064
grad_step = 000050, loss = 0.034327
grad_step = 000051, loss = 0.031810
grad_step = 000052, loss = 0.029320
grad_step = 000053, loss = 0.026965
grad_step = 000054, loss = 0.024955
grad_step = 000055, loss = 0.023091
grad_step = 000056, loss = 0.021235
grad_step = 000057, loss = 0.019555
grad_step = 000058, loss = 0.018053
grad_step = 000059, loss = 0.016617
grad_step = 000060, loss = 0.015281
grad_step = 000061, loss = 0.014045
grad_step = 000062, loss = 0.012903
grad_step = 000063, loss = 0.011896
grad_step = 000064, loss = 0.010946
grad_step = 000065, loss = 0.010056
grad_step = 000066, loss = 0.009282
grad_step = 000067, loss = 0.008556
grad_step = 000068, loss = 0.007877
grad_step = 000069, loss = 0.007282
grad_step = 000070, loss = 0.006741
grad_step = 000071, loss = 0.006246
grad_step = 000072, loss = 0.005796
grad_step = 000073, loss = 0.005384
grad_step = 000074, loss = 0.005023
grad_step = 000075, loss = 0.004690
grad_step = 000076, loss = 0.004381
grad_step = 000077, loss = 0.004122
grad_step = 000078, loss = 0.003885
grad_step = 000079, loss = 0.003669
grad_step = 000080, loss = 0.003483
grad_step = 000081, loss = 0.003313
grad_step = 000082, loss = 0.003163
grad_step = 000083, loss = 0.003024
grad_step = 000084, loss = 0.002907
grad_step = 000085, loss = 0.002812
grad_step = 000086, loss = 0.002722
grad_step = 000087, loss = 0.002647
grad_step = 000088, loss = 0.002583
grad_step = 000089, loss = 0.002527
grad_step = 000090, loss = 0.002479
grad_step = 000091, loss = 0.002437
grad_step = 000092, loss = 0.002405
grad_step = 000093, loss = 0.002375
grad_step = 000094, loss = 0.002350
grad_step = 000095, loss = 0.002331
grad_step = 000096, loss = 0.002313
grad_step = 000097, loss = 0.002299
grad_step = 000098, loss = 0.002287
grad_step = 000099, loss = 0.002277
grad_step = 000100, loss = 0.002268
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002261
grad_step = 000102, loss = 0.002253
grad_step = 000103, loss = 0.002247
grad_step = 000104, loss = 0.002243
grad_step = 000105, loss = 0.002238
grad_step = 000106, loss = 0.002232
grad_step = 000107, loss = 0.002228
grad_step = 000108, loss = 0.002224
grad_step = 000109, loss = 0.002220
grad_step = 000110, loss = 0.002214
grad_step = 000111, loss = 0.002210
grad_step = 000112, loss = 0.002206
grad_step = 000113, loss = 0.002202
grad_step = 000114, loss = 0.002197
grad_step = 000115, loss = 0.002192
grad_step = 000116, loss = 0.002187
grad_step = 000117, loss = 0.002183
grad_step = 000118, loss = 0.002179
grad_step = 000119, loss = 0.002175
grad_step = 000120, loss = 0.002171
grad_step = 000121, loss = 0.002165
grad_step = 000122, loss = 0.002160
grad_step = 000123, loss = 0.002155
grad_step = 000124, loss = 0.002151
grad_step = 000125, loss = 0.002148
grad_step = 000126, loss = 0.002144
grad_step = 000127, loss = 0.002141
grad_step = 000128, loss = 0.002136
grad_step = 000129, loss = 0.002131
grad_step = 000130, loss = 0.002126
grad_step = 000131, loss = 0.002123
grad_step = 000132, loss = 0.002120
grad_step = 000133, loss = 0.002117
grad_step = 000134, loss = 0.002115
grad_step = 000135, loss = 0.002113
grad_step = 000136, loss = 0.002108
grad_step = 000137, loss = 0.002103
grad_step = 000138, loss = 0.002098
grad_step = 000139, loss = 0.002096
grad_step = 000140, loss = 0.002095
grad_step = 000141, loss = 0.002092
grad_step = 000142, loss = 0.002088
grad_step = 000143, loss = 0.002083
grad_step = 000144, loss = 0.002080
grad_step = 000145, loss = 0.002078
grad_step = 000146, loss = 0.002077
grad_step = 000147, loss = 0.002073
grad_step = 000148, loss = 0.002069
grad_step = 000149, loss = 0.002065
grad_step = 000150, loss = 0.002062
grad_step = 000151, loss = 0.002060
grad_step = 000152, loss = 0.002058
grad_step = 000153, loss = 0.002055
grad_step = 000154, loss = 0.002051
grad_step = 000155, loss = 0.002047
grad_step = 000156, loss = 0.002043
grad_step = 000157, loss = 0.002040
grad_step = 000158, loss = 0.002037
grad_step = 000159, loss = 0.002034
grad_step = 000160, loss = 0.002032
grad_step = 000161, loss = 0.002032
grad_step = 000162, loss = 0.002028
grad_step = 000163, loss = 0.002023
grad_step = 000164, loss = 0.002021
grad_step = 000165, loss = 0.002015
grad_step = 000166, loss = 0.002011
grad_step = 000167, loss = 0.002011
grad_step = 000168, loss = 0.002008
grad_step = 000169, loss = 0.002004
grad_step = 000170, loss = 0.002002
grad_step = 000171, loss = 0.002002
grad_step = 000172, loss = 0.001999
grad_step = 000173, loss = 0.001994
grad_step = 000174, loss = 0.001988
grad_step = 000175, loss = 0.001984
grad_step = 000176, loss = 0.001980
grad_step = 000177, loss = 0.001976
grad_step = 000178, loss = 0.001973
grad_step = 000179, loss = 0.001971
grad_step = 000180, loss = 0.001971
grad_step = 000181, loss = 0.001971
grad_step = 000182, loss = 0.001975
grad_step = 000183, loss = 0.001989
grad_step = 000184, loss = 0.002004
grad_step = 000185, loss = 0.002009
grad_step = 000186, loss = 0.001976
grad_step = 000187, loss = 0.001952
grad_step = 000188, loss = 0.001957
grad_step = 000189, loss = 0.001963
grad_step = 000190, loss = 0.001963
grad_step = 000191, loss = 0.001960
grad_step = 000192, loss = 0.001947
grad_step = 000193, loss = 0.001935
grad_step = 000194, loss = 0.001933
grad_step = 000195, loss = 0.001943
grad_step = 000196, loss = 0.001945
grad_step = 000197, loss = 0.001931
grad_step = 000198, loss = 0.001924
grad_step = 000199, loss = 0.001926
grad_step = 000200, loss = 0.001920
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001911
grad_step = 000202, loss = 0.001907
grad_step = 000203, loss = 0.001911
grad_step = 000204, loss = 0.001914
grad_step = 000205, loss = 0.001911
grad_step = 000206, loss = 0.001913
grad_step = 000207, loss = 0.001921
grad_step = 000208, loss = 0.001932
grad_step = 000209, loss = 0.001934
grad_step = 000210, loss = 0.001929
grad_step = 000211, loss = 0.001911
grad_step = 000212, loss = 0.001893
grad_step = 000213, loss = 0.001880
grad_step = 000214, loss = 0.001882
grad_step = 000215, loss = 0.001892
grad_step = 000216, loss = 0.001895
grad_step = 000217, loss = 0.001889
grad_step = 000218, loss = 0.001876
grad_step = 000219, loss = 0.001867
grad_step = 000220, loss = 0.001866
grad_step = 000221, loss = 0.001868
grad_step = 000222, loss = 0.001868
grad_step = 000223, loss = 0.001864
grad_step = 000224, loss = 0.001860
grad_step = 000225, loss = 0.001856
grad_step = 000226, loss = 0.001853
grad_step = 000227, loss = 0.001849
grad_step = 000228, loss = 0.001844
grad_step = 000229, loss = 0.001841
grad_step = 000230, loss = 0.001841
grad_step = 000231, loss = 0.001841
grad_step = 000232, loss = 0.001840
grad_step = 000233, loss = 0.001838
grad_step = 000234, loss = 0.001836
grad_step = 000235, loss = 0.001838
grad_step = 000236, loss = 0.001841
grad_step = 000237, loss = 0.001848
grad_step = 000238, loss = 0.001845
grad_step = 000239, loss = 0.001839
grad_step = 000240, loss = 0.001823
grad_step = 000241, loss = 0.001812
grad_step = 000242, loss = 0.001811
grad_step = 000243, loss = 0.001814
grad_step = 000244, loss = 0.001815
grad_step = 000245, loss = 0.001809
grad_step = 000246, loss = 0.001801
grad_step = 000247, loss = 0.001794
grad_step = 000248, loss = 0.001791
grad_step = 000249, loss = 0.001789
grad_step = 000250, loss = 0.001786
grad_step = 000251, loss = 0.001783
grad_step = 000252, loss = 0.001781
grad_step = 000253, loss = 0.001781
grad_step = 000254, loss = 0.001784
grad_step = 000255, loss = 0.001794
grad_step = 000256, loss = 0.001807
grad_step = 000257, loss = 0.001834
grad_step = 000258, loss = 0.001842
grad_step = 000259, loss = 0.001839
grad_step = 000260, loss = 0.001790
grad_step = 000261, loss = 0.001760
grad_step = 000262, loss = 0.001769
grad_step = 000263, loss = 0.001781
grad_step = 000264, loss = 0.001767
grad_step = 000265, loss = 0.001744
grad_step = 000266, loss = 0.001747
grad_step = 000267, loss = 0.001765
grad_step = 000268, loss = 0.001763
grad_step = 000269, loss = 0.001748
grad_step = 000270, loss = 0.001729
grad_step = 000271, loss = 0.001725
grad_step = 000272, loss = 0.001729
grad_step = 000273, loss = 0.001730
grad_step = 000274, loss = 0.001728
grad_step = 000275, loss = 0.001715
grad_step = 000276, loss = 0.001705
grad_step = 000277, loss = 0.001703
grad_step = 000278, loss = 0.001704
grad_step = 000279, loss = 0.001710
grad_step = 000280, loss = 0.001721
grad_step = 000281, loss = 0.001714
grad_step = 000282, loss = 0.001701
grad_step = 000283, loss = 0.001694
grad_step = 000284, loss = 0.001689
grad_step = 000285, loss = 0.001680
grad_step = 000286, loss = 0.001667
grad_step = 000287, loss = 0.001659
grad_step = 000288, loss = 0.001661
grad_step = 000289, loss = 0.001658
grad_step = 000290, loss = 0.001652
grad_step = 000291, loss = 0.001645
grad_step = 000292, loss = 0.001645
grad_step = 000293, loss = 0.001653
grad_step = 000294, loss = 0.001671
grad_step = 000295, loss = 0.001721
grad_step = 000296, loss = 0.001837
grad_step = 000297, loss = 0.001977
grad_step = 000298, loss = 0.001999
grad_step = 000299, loss = 0.001814
grad_step = 000300, loss = 0.001654
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001699
grad_step = 000302, loss = 0.001800
grad_step = 000303, loss = 0.001752
grad_step = 000304, loss = 0.001628
grad_step = 000305, loss = 0.001631
grad_step = 000306, loss = 0.001725
grad_step = 000307, loss = 0.001713
grad_step = 000308, loss = 0.001603
grad_step = 000309, loss = 0.001629
grad_step = 000310, loss = 0.001702
grad_step = 000311, loss = 0.001649
grad_step = 000312, loss = 0.001609
grad_step = 000313, loss = 0.001616
grad_step = 000314, loss = 0.001630
grad_step = 000315, loss = 0.001634
grad_step = 000316, loss = 0.001625
grad_step = 000317, loss = 0.001614
grad_step = 000318, loss = 0.001666
grad_step = 000319, loss = 0.001672
grad_step = 000320, loss = 0.001626
grad_step = 000321, loss = 0.001567
grad_step = 000322, loss = 0.001589
grad_step = 000323, loss = 0.001620
grad_step = 000324, loss = 0.001585
grad_step = 000325, loss = 0.001554
grad_step = 000326, loss = 0.001566
grad_step = 000327, loss = 0.001586
grad_step = 000328, loss = 0.001579
grad_step = 000329, loss = 0.001561
grad_step = 000330, loss = 0.001561
grad_step = 000331, loss = 0.001566
grad_step = 000332, loss = 0.001561
grad_step = 000333, loss = 0.001541
grad_step = 000334, loss = 0.001525
grad_step = 000335, loss = 0.001527
grad_step = 000336, loss = 0.001539
grad_step = 000337, loss = 0.001548
grad_step = 000338, loss = 0.001546
grad_step = 000339, loss = 0.001544
grad_step = 000340, loss = 0.001539
grad_step = 000341, loss = 0.001536
grad_step = 000342, loss = 0.001528
grad_step = 000343, loss = 0.001521
grad_step = 000344, loss = 0.001511
grad_step = 000345, loss = 0.001501
grad_step = 000346, loss = 0.001498
grad_step = 000347, loss = 0.001500
grad_step = 000348, loss = 0.001504
grad_step = 000349, loss = 0.001504
grad_step = 000350, loss = 0.001502
grad_step = 000351, loss = 0.001497
grad_step = 000352, loss = 0.001493
grad_step = 000353, loss = 0.001491
grad_step = 000354, loss = 0.001494
grad_step = 000355, loss = 0.001504
grad_step = 000356, loss = 0.001532
grad_step = 000357, loss = 0.001583
grad_step = 000358, loss = 0.001681
grad_step = 000359, loss = 0.001742
grad_step = 000360, loss = 0.001767
grad_step = 000361, loss = 0.001644
grad_step = 000362, loss = 0.001553
grad_step = 000363, loss = 0.001520
grad_step = 000364, loss = 0.001536
grad_step = 000365, loss = 0.001565
grad_step = 000366, loss = 0.001567
grad_step = 000367, loss = 0.001534
grad_step = 000368, loss = 0.001479
grad_step = 000369, loss = 0.001477
grad_step = 000370, loss = 0.001524
grad_step = 000371, loss = 0.001531
grad_step = 000372, loss = 0.001479
grad_step = 000373, loss = 0.001445
grad_step = 000374, loss = 0.001467
grad_step = 000375, loss = 0.001494
grad_step = 000376, loss = 0.001481
grad_step = 000377, loss = 0.001453
grad_step = 000378, loss = 0.001445
grad_step = 000379, loss = 0.001451
grad_step = 000380, loss = 0.001451
grad_step = 000381, loss = 0.001447
grad_step = 000382, loss = 0.001448
grad_step = 000383, loss = 0.001447
grad_step = 000384, loss = 0.001438
grad_step = 000385, loss = 0.001425
grad_step = 000386, loss = 0.001420
grad_step = 000387, loss = 0.001425
grad_step = 000388, loss = 0.001428
grad_step = 000389, loss = 0.001428
grad_step = 000390, loss = 0.001427
grad_step = 000391, loss = 0.001426
grad_step = 000392, loss = 0.001422
grad_step = 000393, loss = 0.001414
grad_step = 000394, loss = 0.001405
grad_step = 000395, loss = 0.001401
grad_step = 000396, loss = 0.001400
grad_step = 000397, loss = 0.001398
grad_step = 000398, loss = 0.001395
grad_step = 000399, loss = 0.001393
grad_step = 000400, loss = 0.001394
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001396
grad_step = 000402, loss = 0.001400
grad_step = 000403, loss = 0.001407
grad_step = 000404, loss = 0.001425
grad_step = 000405, loss = 0.001449
grad_step = 000406, loss = 0.001498
grad_step = 000407, loss = 0.001525
grad_step = 000408, loss = 0.001561
grad_step = 000409, loss = 0.001540
grad_step = 000410, loss = 0.001515
grad_step = 000411, loss = 0.001452
grad_step = 000412, loss = 0.001398
grad_step = 000413, loss = 0.001380
grad_step = 000414, loss = 0.001404
grad_step = 000415, loss = 0.001436
grad_step = 000416, loss = 0.001424
grad_step = 000417, loss = 0.001387
grad_step = 000418, loss = 0.001363
grad_step = 000419, loss = 0.001369
grad_step = 000420, loss = 0.001386
grad_step = 000421, loss = 0.001387
grad_step = 000422, loss = 0.001373
grad_step = 000423, loss = 0.001354
grad_step = 000424, loss = 0.001347
grad_step = 000425, loss = 0.001356
grad_step = 000426, loss = 0.001368
grad_step = 000427, loss = 0.001369
grad_step = 000428, loss = 0.001363
grad_step = 000429, loss = 0.001362
grad_step = 000430, loss = 0.001388
grad_step = 000431, loss = 0.001430
grad_step = 000432, loss = 0.001487
grad_step = 000433, loss = 0.001504
grad_step = 000434, loss = 0.001492
grad_step = 000435, loss = 0.001408
grad_step = 000436, loss = 0.001338
grad_step = 000437, loss = 0.001323
grad_step = 000438, loss = 0.001353
grad_step = 000439, loss = 0.001385
grad_step = 000440, loss = 0.001382
grad_step = 000441, loss = 0.001359
grad_step = 000442, loss = 0.001332
grad_step = 000443, loss = 0.001319
grad_step = 000444, loss = 0.001318
grad_step = 000445, loss = 0.001321
grad_step = 000446, loss = 0.001329
grad_step = 000447, loss = 0.001332
grad_step = 000448, loss = 0.001323
grad_step = 000449, loss = 0.001303
grad_step = 000450, loss = 0.001287
grad_step = 000451, loss = 0.001286
grad_step = 000452, loss = 0.001296
grad_step = 000453, loss = 0.001304
grad_step = 000454, loss = 0.001301
grad_step = 000455, loss = 0.001292
grad_step = 000456, loss = 0.001285
grad_step = 000457, loss = 0.001292
grad_step = 000458, loss = 0.001313
grad_step = 000459, loss = 0.001357
grad_step = 000460, loss = 0.001388
grad_step = 000461, loss = 0.001434
grad_step = 000462, loss = 0.001386
grad_step = 000463, loss = 0.001333
grad_step = 000464, loss = 0.001312
grad_step = 000465, loss = 0.001333
grad_step = 000466, loss = 0.001334
grad_step = 000467, loss = 0.001285
grad_step = 000468, loss = 0.001251
grad_step = 000469, loss = 0.001268
grad_step = 000470, loss = 0.001302
grad_step = 000471, loss = 0.001306
grad_step = 000472, loss = 0.001270
grad_step = 000473, loss = 0.001241
grad_step = 000474, loss = 0.001243
grad_step = 000475, loss = 0.001259
grad_step = 000476, loss = 0.001264
grad_step = 000477, loss = 0.001248
grad_step = 000478, loss = 0.001227
grad_step = 000479, loss = 0.001221
grad_step = 000480, loss = 0.001231
grad_step = 000481, loss = 0.001242
grad_step = 000482, loss = 0.001240
grad_step = 000483, loss = 0.001229
grad_step = 000484, loss = 0.001216
grad_step = 000485, loss = 0.001212
grad_step = 000486, loss = 0.001216
grad_step = 000487, loss = 0.001219
grad_step = 000488, loss = 0.001216
grad_step = 000489, loss = 0.001209
grad_step = 000490, loss = 0.001200
grad_step = 000491, loss = 0.001196
grad_step = 000492, loss = 0.001197
grad_step = 000493, loss = 0.001205
grad_step = 000494, loss = 0.001216
grad_step = 000495, loss = 0.001232
grad_step = 000496, loss = 0.001248
grad_step = 000497, loss = 0.001277
grad_step = 000498, loss = 0.001299
grad_step = 000499, loss = 0.001341
grad_step = 000500, loss = 0.001370
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001388
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

  date_run                              2020-05-14 18:13:04.839408
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.260957
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 18:13:04.845171
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.18517
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 18:13:04.851951
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.149876
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 18:13:04.857033
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.81372
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
0   2020-05-14 18:12:34.633904  ...    mean_absolute_error
1   2020-05-14 18:12:34.637539  ...     mean_squared_error
2   2020-05-14 18:12:34.640696  ...  median_absolute_error
3   2020-05-14 18:12:34.644858  ...               r2_score
4   2020-05-14 18:12:42.621194  ...    mean_absolute_error
5   2020-05-14 18:12:42.624986  ...     mean_squared_error
6   2020-05-14 18:12:42.628362  ...  median_absolute_error
7   2020-05-14 18:12:42.631626  ...               r2_score
8   2020-05-14 18:13:04.839408  ...    mean_absolute_error
9   2020-05-14 18:13:04.845171  ...     mean_squared_error
10  2020-05-14 18:13:04.851951  ...  median_absolute_error
11  2020-05-14 18:13:04.857033  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a92577a90> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:30, 327886.17it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 416290.58it/s]  9%|▉         | 876544/9912422 [00:00<00:15, 578648.65it/s] 36%|███▌      | 3522560/9912422 [00:00<00:07, 817278.98it/s] 72%|███████▏  | 7127040/9912422 [00:00<00:02, 1153753.67it/s]9920512it [00:00, 10442698.72it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 148615.27it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 314350.24it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 406445.33it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 562704.11it/s]1654784it [00:00, 2801876.01it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 55041.81it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a44f27e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a445590b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a44f27e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a44559048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a41ce84a8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a445590b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a44f27e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a44559048> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a41ce84a8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1a9252feb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fa3910c1208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=075b358822f242dbf163c19e91affc23b106590a9d7df9c86b84fb78047d41ff
  Stored in directory: /tmp/pip-ephem-wheel-cache-mdlbrge2/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fa328ebc780> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 47s
   57344/17464789 [..............................] - ETA: 40s
  106496/17464789 [..............................] - ETA: 32s
  196608/17464789 [..............................] - ETA: 23s
  385024/17464789 [..............................] - ETA: 14s
  753664/17464789 [>.............................] - ETA: 8s 
 1507328/17464789 [=>............................] - ETA: 4s
 2998272/17464789 [====>.........................] - ETA: 2s
 5701632/17464789 [========>.....................] - ETA: 1s
 8224768/17464789 [=============>................] - ETA: 0s
11059200/17464789 [=================>............] - ETA: 0s
13991936/17464789 [=======================>......] - ETA: 0s
16842752/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 18:14:33.448266: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 18:14:33.452727: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-14 18:14:33.453325: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56047bc96ee0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 18:14:33.453342: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.6666 - accuracy: 0.5000
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5900 - accuracy: 0.5050 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6002 - accuracy: 0.5043
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5861 - accuracy: 0.5052
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5470 - accuracy: 0.5078
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5491 - accuracy: 0.5077
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5746 - accuracy: 0.5060
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6321 - accuracy: 0.5023
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6326 - accuracy: 0.5022
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6528 - accuracy: 0.5009
11000/25000 [============>.................] - ETA: 3s - loss: 7.6513 - accuracy: 0.5010
12000/25000 [=============>................] - ETA: 3s - loss: 7.6998 - accuracy: 0.4978
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6938 - accuracy: 0.4982
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6765 - accuracy: 0.4994
15000/25000 [=================>............] - ETA: 2s - loss: 7.6840 - accuracy: 0.4989
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6973 - accuracy: 0.4980
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7036 - accuracy: 0.4976
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7084 - accuracy: 0.4973
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7102 - accuracy: 0.4972
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7034 - accuracy: 0.4976
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6630 - accuracy: 0.5002
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6750 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6680 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6602 - accuracy: 0.5004
25000/25000 [==============================] - 8s 327us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 18:14:48.237053
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 18:14:48.237053  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<10:45:41, 22.3kB/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:00<7:32:53, 31.7kB/s]  .vector_cache/glove.6B.zip:   1%|          | 5.63M/862M [00:00<5:15:12, 45.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.7M/862M [00:00<3:38:05, 64.7kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.0M/862M [00:00<2:30:50, 92.4kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.8M/862M [00:00<1:44:24, 132kB/s] .vector_cache/glove.6B.zip:   5%|▍         | 39.3M/862M [00:01<1:13:06, 188kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.4M/862M [00:02<52:07, 262kB/s]  .vector_cache/glove.6B.zip:   5%|▌         | 46.7M/862M [00:02<36:27, 373kB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.6M/862M [00:02<26:35, 510kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:02<18:46, 719kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<15:00, 896kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:05<11:41, 1.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.3M/862M [00:05<08:17, 1.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<15:00, 891kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:07<11:29, 1.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<10:00, 1.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:09<07:51, 1.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<07:29, 1.76MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<05:46, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<06:16, 2.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<04:55, 2.67MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<05:35, 2.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<04:25, 2.96MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<05:19, 2.45MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<04:41, 2.78MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<05:12, 2.49MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<04:39, 2.78MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<05:10, 2.49MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<05:02, 2.55MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<05:22, 2.38MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<04:32, 2.82MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<05:09, 2.47MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<04:34, 2.78MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:05, 2.49MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:07, 2.47MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<03:43, 3.39MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:33, 1.67MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:26, 1.96MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:20, 1.98MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:53, 2.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<04:13, 2.95MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<12:13, 1.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<09:53, 1.26MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<07:01, 1.77MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<05:44, 2.16MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:33<04:26, 2.78MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<18:36:10, 11.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<13:02:06, 15.8kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<9:06:29, 22.5kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<6:22:55, 32.1kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<4:28:42, 45.5kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<3:08:44, 64.7kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<2:13:21, 91.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<1:34:02, 129kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:44<1:07:54, 178kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:44<49:02, 246kB/s]  .vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<34:17, 351kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:46<35:03, 343kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:46<25:32, 470kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<17:53, 668kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:48<1:03:48, 187kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<45:19, 263kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<31:54, 373kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:48<22:20, 530kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<31:00, 381kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:52<22:37, 522kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<15:50, 742kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<1:27:14, 135kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:54<1:02:27, 188kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<43:41, 268kB/s]  .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:57<38:46, 301kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:57<28:08, 415kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:59<21:19, 545kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:59<16:08, 719kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<11:23, 1.02MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:01<12:29, 924kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:01<09:38, 1.20MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:03<08:27, 1.36MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:03<06:45, 1.70MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:05<06:25, 1.77MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:05<07:39, 1.49MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:05<05:56, 1.92MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<04:20, 2.62MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:07<06:30, 1.74MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:07<09:16, 1.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:07<06:58, 1.62MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<05:05, 2.22MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:07<03:47, 2.97MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:09<17:35, 641kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:09<13:13, 851kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:09<09:18, 1.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:11<23:14, 482kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:11<17:10, 652kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:11<12:03, 923kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:13<29:50, 373kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:13<21:47, 510kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:13<15:26, 719kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<13:26, 824kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:15<10:58, 1.01MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<07:47, 1.41MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:17<09:23, 1.17MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:17<08:10, 1.34MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:17<05:54, 1.86MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:17<04:17, 2.55MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:19<22:01, 496kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:19<16:54, 646kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:19<12:03, 905kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:21<10:40, 1.02MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:21<09:04, 1.20MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<06:29, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<07:29, 1.44MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<06:49, 1.58MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:23<05:17, 2.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:23<04:08, 2.60MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:23<03:10, 3.39MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:25<05:55, 1.81MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:25<05:36, 1.91MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:25<04:36, 2.32MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<03:17, 3.24MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<10:44, 989kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<09:03, 1.17MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:28<06:25, 1.64MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<09:03, 1.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<07:14, 1.46MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:30<05:09, 2.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:32<08:36, 1.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:32<07:23, 1.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:32<05:18, 1.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:34<06:57, 1.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:34<06:18, 1.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<04:46, 2.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<03:24, 3.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<42:37, 242kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:37<31:05, 332kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:37<21:52, 470kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<15:28, 662kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<20:16, 505kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:39<15:10, 675kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<10:42, 953kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:41<11:13, 907kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:41<08:45, 1.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:41<06:15, 1.62MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:43<07:20, 1.38MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:43<06:15, 1.61MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:43<04:28, 2.25MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:45<07:58, 1.26MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:45<06:23, 1.57MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:45<04:32, 2.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:47<12:59, 768kB/s] .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:47<10:07, 984kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<07:10, 1.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:49<08:51, 1.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:49<07:30, 1.32MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:49<05:28, 1.81MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:51<06:00, 1.63MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:51<05:36, 1.75MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<04:00, 2.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<07:37, 1.28MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<06:46, 1.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<04:51, 2.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:53<03:35, 2.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:55<35:15, 275kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:55<25:27, 381kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:55<17:50, 541kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<17:13, 559kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<13:13, 727kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:57<09:22, 1.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:59<09:20, 1.02MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:59<07:17, 1.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:59<05:09, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:01<39:18, 241kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:01<28:13, 336kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<19:46, 477kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:03<18:26, 511kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:03<13:45, 685kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<09:41, 967kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<11:18, 827kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<08:52, 1.05MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<06:16, 1.48MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<10:26, 889kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:07<08:05, 1.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:07<05:43, 1.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<10:56, 842kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:09<08:52, 1.04MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<06:15, 1.46MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<10:44, 852kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:11<08:49, 1.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:11<06:15, 1.45MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:13<08:07, 1.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:13<06:27, 1.41MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:13<04:34, 1.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:15<21:59, 409kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:15<16:23, 549kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<11:31, 777kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:17<11:53, 752kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:17<09:11, 972kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:17<06:43, 1.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:18<05:04, 1.75MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:18<03:35, 2.45MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:21<42:49, 205kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:21<30:42, 286kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:21<21:24, 407kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:23<1:48:03, 80.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:23<1:16:29, 114kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:23<53:20, 163kB/s]  .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:25<42:05, 206kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:25<30:15, 286kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:25<21:08, 407kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:27<20:57, 410kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:27<15:38, 549kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:27<11:07, 770kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:27<07:53, 1.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<05:37, 1.51MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:30<12:55, 656kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<09:45, 868kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<06:51, 1.23MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:32<18:35, 452kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:32<13:42, 613kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:32<09:35, 870kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<28:08, 297kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<20:22, 409kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<14:13, 582kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<9:26:12, 14.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<6:36:40, 20.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:36<4:35:58, 29.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<3:21:03, 40.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<2:21:58, 57.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:38<1:38:55, 82.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:40<1:12:32, 112kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:40<51:30, 158kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<35:51, 225kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:42<1:28:57, 90.7kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:42<1:02:51, 128kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:42<43:43, 183kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:44<3:21:26, 39.7kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:44<2:21:31, 56.5kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:44<1:38:24, 80.6kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:46<1:48:32, 73.0kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<1:16:32, 103kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:48<54:20, 145kB/s]  .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:48<39:14, 200kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:48<27:27, 285kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:51<22:24, 348kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:51<16:13, 480kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:51<11:22, 680kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:52<11:56, 646kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:53<09:31, 811kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:53<06:54, 1.11MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:53<05:06, 1.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:53<03:49, 2.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:54<05:35, 1.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:55<05:00, 1.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:55<03:40, 2.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:55<02:43, 2.79MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:56<05:04, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:56<04:40, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:57<03:26, 2.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:57<02:33, 2.94MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:58<04:47, 1.57MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:58<04:27, 1.68MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:59<03:18, 2.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:59<02:27, 3.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:00<04:42, 1.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:00<04:18, 1.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:01<03:07, 2.37MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:01<02:19, 3.17MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:02<09:29, 777kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:02<07:19, 1.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<05:10, 1.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:04<07:35, 963kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:04<06:03, 1.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:04<04:18, 1.69MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:06<05:35, 1.30MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:06<04:59, 1.45MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:06<03:36, 2.00MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:07<02:39, 2.70MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:08<07:45, 925kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:08<06:21, 1.13MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:08<04:31, 1.57MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:10<05:20, 1.33MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:10<04:18, 1.64MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<03:04, 2.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:12<07:46, 904kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:12<06:31, 1.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:12<04:42, 1.49MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:12<03:23, 2.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:14<09:05, 767kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:14<06:53, 1.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:14<04:52, 1.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:16<06:31, 1.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:16<05:08, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<03:39, 1.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:18<05:21, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:18<04:47, 1.43MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:18<03:23, 1.99MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:20<05:26, 1.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:20<05:04, 1.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:20<03:38, 1.85MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:22<04:11, 1.59MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:22<03:58, 1.68MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:22<02:52, 2.31MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:24<03:39, 1.81MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:24<03:35, 1.85MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:24<02:36, 2.52MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:27<04:08, 1.58MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:27<03:20, 1.96MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:27<02:22, 2.73MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:29<08:20, 777kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:29<06:34, 984kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:29<04:38, 1.39MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:31<06:50, 937kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:31<05:43, 1.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:31<04:03, 1.57MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:33<05:25, 1.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:33<04:42, 1.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:33<03:19, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:35<05:46, 1.09MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:35<04:45, 1.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:35<03:24, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:37<04:27, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:37<03:37, 1.71MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:37<02:36, 2.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:39<03:47, 1.62MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:39<03:36, 1.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:39<02:39, 2.31MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:39<01:58, 3.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:39<01:32, 3.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:41<2:54:13, 34.8kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:41<2:02:40, 49.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:41<1:25:10, 70.6kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:43<1:04:01, 93.7kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:43<45:30, 132kB/s]   .vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:43<31:39, 188kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:45<25:16, 235kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:45<18:27, 321kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:45<12:51, 457kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:47<13:23, 438kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:47<10:03, 582kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:47<07:02, 825kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:49<08:12, 706kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:49<06:12, 932kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:52<05:31, 1.03MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:52<04:18, 1.32MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:52<03:03, 1.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:54<05:14, 1.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:54<04:30, 1.25MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:54<03:11, 1.76MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:56<04:25, 1.26MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:56<03:52, 1.44MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:56<02:44, 2.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:58<04:56, 1.12MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:58<03:56, 1.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:00<03:32, 1.54MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:00<03:24, 1.60MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:00<02:29, 2.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [04:00<01:49, 2.96MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [04:01<05:02, 1.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:02<03:56, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:02<02:46, 1.92MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:03<19:08, 277kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:04<13:50, 383kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:05<10:21, 506kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:06<07:51, 666kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:06<05:31, 940kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:07<05:36, 922kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:07<04:23, 1.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:08<03:06, 1.65MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:09<04:33, 1.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:09<03:36, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:11<03:15, 1.55MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:11<03:00, 1.68MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:12<02:08, 2.33MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:13<03:32, 1.40MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:13<03:26, 1.44MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:13<02:38, 1.87MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:14<01:56, 2.54MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:16<03:14, 1.51MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:16<02:39, 1.84MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:18<02:26, 1.98MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:18<02:01, 2.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:20<02:21, 2.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:20<02:26, 1.94MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:20<01:46, 2.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:22<02:28, 1.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:22<02:18, 2.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:22<01:38, 2.82MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:24<03:22, 1.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:24<03:01, 1.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:24<02:07, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:26<06:01, 755kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:26<04:27, 1.02MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:26<03:07, 1.44MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:28<07:24, 605kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:28<05:44, 782kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:30<04:35, 961kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:30<04:35, 962kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:30<03:29, 1.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:30<02:33, 1.72MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:30<01:53, 2.31MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:33<03:47, 1.14MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:33<03:01, 1.44MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:35<02:41, 1.59MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:35<02:17, 1.86MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:35<01:38, 2.58MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:37<03:28, 1.21MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:37<02:42, 1.55MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:39<02:30, 1.65MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:39<02:10, 1.91MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:39<01:32, 2.66MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:41<04:27, 913kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:41<03:23, 1.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:42<02:56, 1.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:43<02:19, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:44<02:12, 1.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:45<01:54, 2.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:47<02:01, 1.92MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:47<01:44, 2.22MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:49<01:45, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:49<02:10, 1.74MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:49<01:38, 2.31MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:49<01:14, 3.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:49<00:59, 3.78MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:51<02:28, 1.51MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:51<02:02, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:53<01:57, 1.87MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:53<01:40, 2.18MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:53<01:12, 2.99MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:55<02:08, 1.67MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:55<01:48, 1.98MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:57<01:46, 1.99MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:57<01:31, 2.30MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:57<01:04, 3.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:59<28:48, 120kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:59<21:18, 162kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:59<14:58, 230kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [05:01<10:55, 310kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [05:01<08:09, 414kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:01<05:42, 587kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:03<04:55, 673kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:03<03:44, 884kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:03<02:36, 1.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:05<05:27, 595kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:05<04:04, 794kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:07<03:16, 970kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:07<03:08, 1.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:07<02:14, 1.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:07<01:35, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:09<20:10, 154kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:09<14:34, 213kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:09<10:03, 304kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:11<09:06, 334kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:11<06:45, 450kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:11<04:42, 637kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:13<04:41, 634kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:14<03:43, 796kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:14<02:39, 1.11MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:14<01:54, 1.53MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:14<01:24, 2.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:15<1:23:51, 34.6kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:16<59:01, 49.1kB/s]  .vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:16<41:03, 70.1kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:16<28:31, 99.9kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:18<21:46, 130kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:18<15:53, 178kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:18<11:12, 252kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:18<07:55, 354kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:18<05:38, 496kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:18<03:59, 696kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:20<04:47, 578kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:20<03:46, 731kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:20<02:48, 978kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:20<02:02, 1.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:20<01:31, 1.79MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:20<01:08, 2.37MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:22<02:45, 977kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:22<02:09, 1.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:22<01:30, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:24<02:52, 913kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:24<02:17, 1.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:24<01:36, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:26<02:37, 976kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:26<01:58, 1.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:26<01:22, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:28<05:06, 488kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:28<03:50, 646kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:28<02:41, 911kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:30<02:37, 921kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:30<02:09, 1.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:30<01:30, 1.57MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:32<01:59, 1.18MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:32<01:39, 1.42MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:32<01:08, 2.00MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:34<11:58, 191kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:34<08:40, 263kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:34<05:58, 374kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:36<05:09, 429kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:36<03:44, 590kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:36<02:34, 837kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:38<06:08, 350kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:38<04:25, 483kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:38<03:02, 686kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:40<05:34, 373kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:40<04:03, 510kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:40<02:46, 724kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:42<2:49:38, 11.8kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:42<1:58:29, 16.9kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:44<1:20:41, 24.0kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:44<56:33, 34.2kB/s]  .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:44<38:31, 48.8kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:46<28:18, 66.1kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:46<19:53, 93.7kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:46<13:40, 134kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:49<10:13, 176kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:49<07:16, 247kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:51<05:10, 334kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:51<03:51, 448kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:51<02:39, 636kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:53<02:31, 658kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:53<01:53, 873kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:55<01:31, 1.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:55<01:18, 1.22MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:55<00:54, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:57<01:04, 1.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:57<00:55, 1.66MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:59<00:49, 1.76MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:59<00:50, 1.73MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:59<00:35, 2.41MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [06:01<01:00, 1.37MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [06:01<00:51, 1.62MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [06:01<00:36, 2.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [06:03<00:47, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [06:03<00:52, 1.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:03<00:39, 1.98MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:03<00:28, 2.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [06:03<00:21, 3.46MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [06:05<01:24, 888kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [06:05<01:08, 1.09MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [06:05<00:49, 1.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:05<00:35, 2.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:05<00:26, 2.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:08<07:20, 161kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:08<05:17, 222kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:08<03:33, 317kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:11<03:25, 324kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:11<02:28, 447kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:11<01:39, 635kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:12<01:31, 685kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:12<01:03, 963kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:12<00:44, 1.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:15<01:36, 606kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:15<01:11, 810kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:15<00:47, 1.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:17<03:29, 260kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:17<02:31, 356kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:17<01:44, 503kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:17<01:12, 709kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:19<01:15, 661kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:19<00:57, 872kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:22<00:46, 998kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:22<00:35, 1.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:23<00:28, 1.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:23<00:20, 2.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:25<00:19, 1.89MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:25<00:17, 2.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:25<00:11, 3.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:27<00:22, 1.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:27<00:17, 1.87MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:29<00:15, 1.87MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:29<00:13, 2.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:31<00:11, 2.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:31<00:09, 2.60MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:33<00:09, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:33<00:09, 2.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:33<00:07, 2.84MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:34<00:04, 3.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:35<00:11, 1.52MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:35<00:10, 1.65MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:35<00:05, 2.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:37<00:14, 882kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:37<00:12, 1.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:37<00:07, 1.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:39<00:06, 1.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:39<00:04, 1.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:41<00:02, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:41<00:02, 1.80MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:41<00:01, 2.38MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:41<00:00, 3.13MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:43<00:00, 1.92MB/s].vector_cache/glove.6B.zip: 862MB [06:43, 2.14MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 818/400000 [00:00<00:48, 8167.00it/s]  0%|          | 1659/400000 [00:00<00:48, 8237.38it/s]  1%|          | 2523/400000 [00:00<00:47, 8351.84it/s]  1%|          | 3360/400000 [00:00<00:47, 8355.21it/s]  1%|          | 4188/400000 [00:00<00:47, 8330.79it/s]  1%|▏         | 5028/400000 [00:00<00:47, 8350.34it/s]  1%|▏         | 5875/400000 [00:00<00:47, 8385.49it/s]  2%|▏         | 6721/400000 [00:00<00:46, 8407.11it/s]  2%|▏         | 7539/400000 [00:00<00:47, 8334.85it/s]  2%|▏         | 8386/400000 [00:01<00:46, 8371.90it/s]  2%|▏         | 9237/400000 [00:01<00:46, 8410.98it/s]  3%|▎         | 10107/400000 [00:01<00:45, 8480.86it/s]  3%|▎         | 10945/400000 [00:01<00:46, 8284.84it/s]  3%|▎         | 11769/400000 [00:01<00:46, 8269.94it/s]  3%|▎         | 12592/400000 [00:01<00:47, 8165.49it/s]  3%|▎         | 13406/400000 [00:01<00:47, 8156.28it/s]  4%|▎         | 14224/400000 [00:01<00:47, 8162.81it/s]  4%|▍         | 15039/400000 [00:01<00:47, 8124.58it/s]  4%|▍         | 15851/400000 [00:01<00:47, 8028.88it/s]  4%|▍         | 16654/400000 [00:02<00:48, 7925.46it/s]  4%|▍         | 17447/400000 [00:02<00:48, 7868.48it/s]  5%|▍         | 18240/400000 [00:02<00:48, 7885.17it/s]  5%|▍         | 19029/400000 [00:02<00:48, 7823.94it/s]  5%|▍         | 19819/400000 [00:02<00:48, 7845.84it/s]  5%|▌         | 20606/400000 [00:02<00:48, 7852.68it/s]  5%|▌         | 21392/400000 [00:02<00:48, 7815.88it/s]  6%|▌         | 22220/400000 [00:02<00:47, 7949.24it/s]  6%|▌         | 23066/400000 [00:02<00:46, 8094.85it/s]  6%|▌         | 23877/400000 [00:02<00:47, 7971.90it/s]  6%|▌         | 24676/400000 [00:03<00:47, 7936.14it/s]  6%|▋         | 25477/400000 [00:03<00:47, 7957.55it/s]  7%|▋         | 26275/400000 [00:03<00:46, 7962.27it/s]  7%|▋         | 27072/400000 [00:03<00:47, 7808.09it/s]  7%|▋         | 27854/400000 [00:03<00:47, 7779.40it/s]  7%|▋         | 28643/400000 [00:03<00:47, 7810.47it/s]  7%|▋         | 29434/400000 [00:03<00:47, 7838.27it/s]  8%|▊         | 30221/400000 [00:03<00:47, 7847.56it/s]  8%|▊         | 31012/400000 [00:03<00:46, 7863.89it/s]  8%|▊         | 31799/400000 [00:03<00:47, 7811.17it/s]  8%|▊         | 32586/400000 [00:04<00:46, 7827.31it/s]  8%|▊         | 33387/400000 [00:04<00:46, 7878.41it/s]  9%|▊         | 34185/400000 [00:04<00:46, 7907.89it/s]  9%|▊         | 34987/400000 [00:04<00:45, 7938.59it/s]  9%|▉         | 35796/400000 [00:04<00:45, 7982.06it/s]  9%|▉         | 36595/400000 [00:04<00:45, 7954.91it/s]  9%|▉         | 37412/400000 [00:04<00:45, 8015.91it/s] 10%|▉         | 38268/400000 [00:04<00:44, 8169.67it/s] 10%|▉         | 39139/400000 [00:04<00:43, 8322.29it/s] 10%|▉         | 39994/400000 [00:04<00:42, 8386.41it/s] 10%|█         | 40834/400000 [00:05<00:43, 8274.74it/s] 10%|█         | 41675/400000 [00:05<00:43, 8314.17it/s] 11%|█         | 42524/400000 [00:05<00:42, 8363.65it/s] 11%|█         | 43390/400000 [00:05<00:42, 8449.98it/s] 11%|█         | 44253/400000 [00:05<00:41, 8500.46it/s] 11%|█▏        | 45104/400000 [00:05<00:42, 8271.99it/s] 11%|█▏        | 45933/400000 [00:05<00:43, 8150.61it/s] 12%|█▏        | 46750/400000 [00:05<00:43, 8107.41it/s] 12%|█▏        | 47562/400000 [00:05<00:43, 8097.62it/s] 12%|█▏        | 48376/400000 [00:05<00:43, 8108.53it/s] 12%|█▏        | 49188/400000 [00:06<00:44, 7878.86it/s] 13%|█▎        | 50057/400000 [00:06<00:43, 8103.94it/s] 13%|█▎        | 50926/400000 [00:06<00:42, 8268.24it/s] 13%|█▎        | 51782/400000 [00:06<00:41, 8351.66it/s] 13%|█▎        | 52638/400000 [00:06<00:41, 8410.20it/s] 13%|█▎        | 53481/400000 [00:06<00:42, 8183.39it/s] 14%|█▎        | 54305/400000 [00:06<00:42, 8199.67it/s] 14%|█▍        | 55141/400000 [00:06<00:41, 8246.39it/s] 14%|█▍        | 56003/400000 [00:06<00:41, 8352.35it/s] 14%|█▍        | 56876/400000 [00:06<00:40, 8461.92it/s] 14%|█▍        | 57724/400000 [00:07<00:40, 8380.49it/s] 15%|█▍        | 58592/400000 [00:07<00:40, 8467.13it/s] 15%|█▍        | 59461/400000 [00:07<00:39, 8530.35it/s] 15%|█▌        | 60331/400000 [00:07<00:39, 8578.32it/s] 15%|█▌        | 61190/400000 [00:07<00:39, 8546.86it/s] 16%|█▌        | 62046/400000 [00:07<00:39, 8487.53it/s] 16%|█▌        | 62896/400000 [00:07<00:41, 8122.54it/s] 16%|█▌        | 63753/400000 [00:07<00:40, 8251.49it/s] 16%|█▌        | 64604/400000 [00:07<00:40, 8326.35it/s] 16%|█▋        | 65462/400000 [00:08<00:39, 8398.34it/s] 17%|█▋        | 66304/400000 [00:08<00:40, 8310.12it/s] 17%|█▋        | 67138/400000 [00:08<00:40, 8318.67it/s] 17%|█▋        | 67971/400000 [00:08<00:40, 8201.47it/s] 17%|█▋        | 68793/400000 [00:08<00:40, 8126.36it/s] 17%|█▋        | 69660/400000 [00:08<00:39, 8281.71it/s] 18%|█▊        | 70490/400000 [00:08<00:40, 8089.19it/s] 18%|█▊        | 71351/400000 [00:08<00:39, 8238.44it/s] 18%|█▊        | 72229/400000 [00:08<00:39, 8392.38it/s] 18%|█▊        | 73100/400000 [00:08<00:38, 8484.53it/s] 18%|█▊        | 73965/400000 [00:09<00:38, 8532.29it/s] 19%|█▊        | 74820/400000 [00:09<00:39, 8211.20it/s] 19%|█▉        | 75669/400000 [00:09<00:39, 8291.96it/s] 19%|█▉        | 76526/400000 [00:09<00:38, 8372.79it/s] 19%|█▉        | 77380/400000 [00:09<00:38, 8421.20it/s] 20%|█▉        | 78224/400000 [00:09<00:38, 8363.59it/s] 20%|█▉        | 79062/400000 [00:09<00:38, 8350.94it/s] 20%|█▉        | 79914/400000 [00:09<00:38, 8399.10it/s] 20%|██        | 80786/400000 [00:09<00:37, 8491.18it/s] 20%|██        | 81636/400000 [00:09<00:38, 8277.03it/s] 21%|██        | 82466/400000 [00:10<00:38, 8215.80it/s] 21%|██        | 83289/400000 [00:10<00:38, 8151.49it/s] 21%|██        | 84106/400000 [00:10<00:38, 8126.50it/s] 21%|██        | 84956/400000 [00:10<00:38, 8233.95it/s] 21%|██▏       | 85822/400000 [00:10<00:37, 8354.82it/s] 22%|██▏       | 86659/400000 [00:10<00:37, 8315.27it/s] 22%|██▏       | 87524/400000 [00:10<00:37, 8410.94it/s] 22%|██▏       | 88366/400000 [00:10<00:37, 8400.64it/s] 22%|██▏       | 89207/400000 [00:10<00:37, 8349.00it/s] 23%|██▎       | 90062/400000 [00:10<00:36, 8407.64it/s] 23%|██▎       | 90904/400000 [00:11<00:36, 8372.43it/s] 23%|██▎       | 91742/400000 [00:11<00:36, 8360.75it/s] 23%|██▎       | 92604/400000 [00:11<00:36, 8435.25it/s] 23%|██▎       | 93471/400000 [00:11<00:36, 8502.91it/s] 24%|██▎       | 94328/400000 [00:11<00:35, 8522.07it/s] 24%|██▍       | 95181/400000 [00:11<00:36, 8451.78it/s] 24%|██▍       | 96027/400000 [00:11<00:36, 8377.95it/s] 24%|██▍       | 96866/400000 [00:11<00:36, 8276.39it/s] 24%|██▍       | 97720/400000 [00:11<00:36, 8353.09it/s] 25%|██▍       | 98574/400000 [00:11<00:35, 8405.76it/s] 25%|██▍       | 99416/400000 [00:12<00:36, 8334.99it/s] 25%|██▌       | 100250/400000 [00:12<00:36, 8276.20it/s] 25%|██▌       | 101097/400000 [00:12<00:35, 8331.25it/s] 25%|██▌       | 101931/400000 [00:12<00:35, 8325.77it/s] 26%|██▌       | 102789/400000 [00:12<00:35, 8397.60it/s] 26%|██▌       | 103630/400000 [00:12<00:35, 8367.48it/s] 26%|██▌       | 104468/400000 [00:12<00:35, 8338.62it/s] 26%|██▋       | 105303/400000 [00:12<00:35, 8302.06it/s] 27%|██▋       | 106134/400000 [00:12<00:35, 8288.45it/s] 27%|██▋       | 106996/400000 [00:12<00:34, 8384.04it/s] 27%|██▋       | 107835/400000 [00:13<00:35, 8330.99it/s] 27%|██▋       | 108669/400000 [00:13<00:34, 8324.75it/s] 27%|██▋       | 109537/400000 [00:13<00:34, 8425.70it/s] 28%|██▊       | 110381/400000 [00:13<00:34, 8312.33it/s] 28%|██▊       | 111213/400000 [00:13<00:35, 8220.73it/s] 28%|██▊       | 112036/400000 [00:13<00:35, 8200.31it/s] 28%|██▊       | 112864/400000 [00:13<00:34, 8223.60it/s] 28%|██▊       | 113695/400000 [00:13<00:34, 8248.27it/s] 29%|██▊       | 114521/400000 [00:13<00:34, 8244.72it/s] 29%|██▉       | 115346/400000 [00:14<00:34, 8204.27it/s] 29%|██▉       | 116167/400000 [00:14<00:34, 8189.94it/s] 29%|██▉       | 117013/400000 [00:14<00:34, 8266.83it/s] 29%|██▉       | 117840/400000 [00:14<00:34, 8192.31it/s] 30%|██▉       | 118660/400000 [00:14<00:34, 8149.42it/s] 30%|██▉       | 119499/400000 [00:14<00:34, 8219.30it/s] 30%|███       | 120322/400000 [00:14<00:34, 8221.42it/s] 30%|███       | 121175/400000 [00:14<00:33, 8310.67it/s] 31%|███       | 122035/400000 [00:14<00:33, 8393.78it/s] 31%|███       | 122875/400000 [00:14<00:33, 8381.02it/s] 31%|███       | 123714/400000 [00:15<00:33, 8239.44it/s] 31%|███       | 124539/400000 [00:15<00:33, 8103.60it/s] 31%|███▏      | 125404/400000 [00:15<00:33, 8259.44it/s] 32%|███▏      | 126249/400000 [00:15<00:32, 8314.78it/s] 32%|███▏      | 127101/400000 [00:15<00:32, 8374.29it/s] 32%|███▏      | 127964/400000 [00:15<00:32, 8449.31it/s] 32%|███▏      | 128810/400000 [00:15<00:32, 8415.50it/s] 32%|███▏      | 129653/400000 [00:15<00:32, 8335.29it/s] 33%|███▎      | 130493/400000 [00:15<00:32, 8351.66it/s] 33%|███▎      | 131346/400000 [00:15<00:31, 8403.13it/s] 33%|███▎      | 132191/400000 [00:16<00:31, 8415.51it/s] 33%|███▎      | 133033/400000 [00:16<00:31, 8361.70it/s] 33%|███▎      | 133870/400000 [00:16<00:31, 8354.03it/s] 34%|███▎      | 134737/400000 [00:16<00:31, 8444.51it/s] 34%|███▍      | 135591/400000 [00:16<00:31, 8471.52it/s] 34%|███▍      | 136456/400000 [00:16<00:30, 8523.72it/s] 34%|███▍      | 137309/400000 [00:16<00:31, 8425.58it/s] 35%|███▍      | 138153/400000 [00:16<00:31, 8366.13it/s] 35%|███▍      | 138991/400000 [00:16<00:31, 8346.98it/s] 35%|███▍      | 139834/400000 [00:16<00:31, 8371.46it/s] 35%|███▌      | 140703/400000 [00:17<00:30, 8463.47it/s] 35%|███▌      | 141550/400000 [00:17<00:30, 8461.61it/s] 36%|███▌      | 142411/400000 [00:17<00:30, 8505.09it/s] 36%|███▌      | 143280/400000 [00:17<00:29, 8558.38it/s] 36%|███▌      | 144150/400000 [00:17<00:29, 8598.35it/s] 36%|███▋      | 145011/400000 [00:17<00:29, 8575.71it/s] 36%|███▋      | 145872/400000 [00:17<00:29, 8585.82it/s] 37%|███▋      | 146731/400000 [00:17<00:29, 8454.65it/s] 37%|███▋      | 147578/400000 [00:17<00:29, 8455.50it/s] 37%|███▋      | 148428/400000 [00:17<00:29, 8468.31it/s] 37%|███▋      | 149300/400000 [00:18<00:29, 8539.95it/s] 38%|███▊      | 150159/400000 [00:18<00:29, 8546.87it/s] 38%|███▊      | 151014/400000 [00:18<00:29, 8417.13it/s] 38%|███▊      | 151857/400000 [00:18<00:29, 8362.75it/s] 38%|███▊      | 152694/400000 [00:18<00:29, 8334.61it/s] 38%|███▊      | 153531/400000 [00:18<00:29, 8342.52it/s] 39%|███▊      | 154381/400000 [00:18<00:29, 8388.11it/s] 39%|███▉      | 155221/400000 [00:18<00:29, 8389.81it/s] 39%|███▉      | 156097/400000 [00:18<00:28, 8497.20it/s] 39%|███▉      | 156948/400000 [00:18<00:28, 8423.35it/s] 39%|███▉      | 157791/400000 [00:19<00:29, 8269.02it/s] 40%|███▉      | 158619/400000 [00:19<00:29, 8214.71it/s] 40%|███▉      | 159442/400000 [00:19<00:29, 8186.09it/s] 40%|████      | 160303/400000 [00:19<00:28, 8306.48it/s] 40%|████      | 161153/400000 [00:19<00:28, 8362.75it/s] 41%|████      | 162009/400000 [00:19<00:28, 8420.86it/s] 41%|████      | 162876/400000 [00:19<00:27, 8491.47it/s] 41%|████      | 163726/400000 [00:19<00:27, 8463.20it/s] 41%|████      | 164573/400000 [00:19<00:28, 8349.00it/s] 41%|████▏     | 165409/400000 [00:19<00:28, 8238.08it/s] 42%|████▏     | 166234/400000 [00:20<00:29, 7959.38it/s] 42%|████▏     | 167033/400000 [00:20<00:29, 7926.31it/s] 42%|████▏     | 167832/400000 [00:20<00:29, 7943.66it/s] 42%|████▏     | 168647/400000 [00:20<00:28, 8001.71it/s] 42%|████▏     | 169449/400000 [00:20<00:28, 7978.86it/s] 43%|████▎     | 170248/400000 [00:20<00:28, 7969.12it/s] 43%|████▎     | 171087/400000 [00:20<00:28, 8087.64it/s] 43%|████▎     | 171897/400000 [00:20<00:28, 7882.63it/s] 43%|████▎     | 172687/400000 [00:20<00:28, 7861.34it/s] 43%|████▎     | 173549/400000 [00:21<00:28, 8072.49it/s] 44%|████▎     | 174359/400000 [00:21<00:28, 8037.13it/s] 44%|████▍     | 175165/400000 [00:21<00:28, 7942.29it/s] 44%|████▍     | 175961/400000 [00:21<00:28, 7935.86it/s] 44%|████▍     | 176756/400000 [00:21<00:28, 7923.92it/s] 44%|████▍     | 177550/400000 [00:21<00:28, 7860.83it/s] 45%|████▍     | 178337/400000 [00:21<00:28, 7819.82it/s] 45%|████▍     | 179120/400000 [00:21<00:28, 7796.82it/s] 45%|████▍     | 179901/400000 [00:21<00:28, 7713.69it/s] 45%|████▌     | 180714/400000 [00:21<00:27, 7832.86it/s] 45%|████▌     | 181570/400000 [00:22<00:27, 8037.37it/s] 46%|████▌     | 182431/400000 [00:22<00:26, 8199.77it/s] 46%|████▌     | 183254/400000 [00:22<00:26, 8194.26it/s] 46%|████▌     | 184115/400000 [00:22<00:25, 8313.56it/s] 46%|████▌     | 184965/400000 [00:22<00:25, 8368.57it/s] 46%|████▋     | 185808/400000 [00:22<00:25, 8384.86it/s] 47%|████▋     | 186648/400000 [00:22<00:25, 8341.11it/s] 47%|████▋     | 187483/400000 [00:22<00:26, 8060.20it/s] 47%|████▋     | 188292/400000 [00:22<00:26, 8008.72it/s] 47%|████▋     | 189100/400000 [00:22<00:26, 8027.79it/s] 47%|████▋     | 189905/400000 [00:23<00:26, 8018.93it/s] 48%|████▊     | 190708/400000 [00:23<00:26, 7970.99it/s] 48%|████▊     | 191506/400000 [00:23<00:26, 7879.87it/s] 48%|████▊     | 192295/400000 [00:23<00:26, 7833.29it/s] 48%|████▊     | 193125/400000 [00:23<00:25, 7966.97it/s] 48%|████▊     | 193993/400000 [00:23<00:25, 8166.14it/s] 49%|████▊     | 194853/400000 [00:23<00:24, 8291.30it/s] 49%|████▉     | 195684/400000 [00:23<00:24, 8283.64it/s] 49%|████▉     | 196519/400000 [00:23<00:24, 8301.49it/s] 49%|████▉     | 197351/400000 [00:23<00:25, 8083.89it/s] 50%|████▉     | 198162/400000 [00:24<00:25, 8028.57it/s] 50%|████▉     | 198967/400000 [00:24<00:25, 7959.99it/s] 50%|████▉     | 199786/400000 [00:24<00:24, 8027.56it/s] 50%|█████     | 200590/400000 [00:24<00:24, 8006.45it/s] 50%|█████     | 201392/400000 [00:24<00:24, 7982.98it/s] 51%|█████     | 202191/400000 [00:24<00:25, 7833.59it/s] 51%|█████     | 203005/400000 [00:24<00:24, 7922.11it/s] 51%|█████     | 203849/400000 [00:24<00:24, 8068.41it/s] 51%|█████     | 204663/400000 [00:24<00:24, 8088.22it/s] 51%|█████▏    | 205487/400000 [00:24<00:23, 8131.03it/s] 52%|█████▏    | 206332/400000 [00:25<00:23, 8223.38it/s] 52%|█████▏    | 207197/400000 [00:25<00:23, 8344.45it/s] 52%|█████▏    | 208033/400000 [00:25<00:23, 8304.51it/s] 52%|█████▏    | 208899/400000 [00:25<00:22, 8406.46it/s] 52%|█████▏    | 209742/400000 [00:25<00:22, 8411.68it/s] 53%|█████▎    | 210607/400000 [00:25<00:22, 8481.52it/s] 53%|█████▎    | 211461/400000 [00:25<00:22, 8498.49it/s] 53%|█████▎    | 212312/400000 [00:25<00:22, 8299.56it/s] 53%|█████▎    | 213144/400000 [00:25<00:22, 8222.06it/s] 53%|█████▎    | 213968/400000 [00:25<00:22, 8163.90it/s] 54%|█████▎    | 214786/400000 [00:26<00:22, 8165.88it/s] 54%|█████▍    | 215604/400000 [00:26<00:22, 8063.88it/s] 54%|█████▍    | 216412/400000 [00:26<00:23, 7968.32it/s] 54%|█████▍    | 217210/400000 [00:26<00:22, 7966.51it/s] 55%|█████▍    | 218016/400000 [00:26<00:22, 7992.53it/s] 55%|█████▍    | 218819/400000 [00:26<00:22, 8003.09it/s] 55%|█████▍    | 219620/400000 [00:26<00:22, 7991.26it/s] 55%|█████▌    | 220420/400000 [00:26<00:22, 7909.35it/s] 55%|█████▌    | 221224/400000 [00:26<00:22, 7947.91it/s] 56%|█████▌    | 222032/400000 [00:27<00:22, 7984.18it/s] 56%|█████▌    | 222836/400000 [00:27<00:22, 7998.75it/s] 56%|█████▌    | 223637/400000 [00:27<00:22, 7996.61it/s] 56%|█████▌    | 224437/400000 [00:27<00:22, 7967.01it/s] 56%|█████▋    | 225238/400000 [00:27<00:21, 7979.11it/s] 57%|█████▋    | 226036/400000 [00:27<00:21, 7946.43it/s] 57%|█████▋    | 226831/400000 [00:27<00:21, 7907.66it/s] 57%|█████▋    | 227630/400000 [00:27<00:21, 7931.16it/s] 57%|█████▋    | 228424/400000 [00:27<00:21, 7896.92it/s] 57%|█████▋    | 229222/400000 [00:27<00:21, 7920.52it/s] 58%|█████▊    | 230078/400000 [00:28<00:20, 8101.81it/s] 58%|█████▊    | 230928/400000 [00:28<00:20, 8214.64it/s] 58%|█████▊    | 231800/400000 [00:28<00:20, 8359.30it/s] 58%|█████▊    | 232638/400000 [00:28<00:20, 8290.12it/s] 58%|█████▊    | 233469/400000 [00:28<00:20, 8178.84it/s] 59%|█████▊    | 234289/400000 [00:28<00:20, 8082.27it/s] 59%|█████▉    | 235099/400000 [00:28<00:20, 8032.24it/s] 59%|█████▉    | 235904/400000 [00:28<00:20, 8015.86it/s] 59%|█████▉    | 236707/400000 [00:28<00:20, 7973.37it/s] 59%|█████▉    | 237505/400000 [00:28<00:20, 7963.53it/s] 60%|█████▉    | 238304/400000 [00:29<00:20, 7969.18it/s] 60%|█████▉    | 239102/400000 [00:29<00:20, 7813.46it/s] 60%|█████▉    | 239961/400000 [00:29<00:19, 8028.98it/s] 60%|██████    | 240775/400000 [00:29<00:19, 8059.52it/s] 60%|██████    | 241614/400000 [00:29<00:19, 8153.49it/s] 61%|██████    | 242487/400000 [00:29<00:18, 8317.40it/s] 61%|██████    | 243341/400000 [00:29<00:18, 8382.56it/s] 61%|██████    | 244215/400000 [00:29<00:18, 8485.21it/s] 61%|██████▏   | 245065/400000 [00:29<00:18, 8362.73it/s] 61%|██████▏   | 245903/400000 [00:29<00:18, 8273.90it/s] 62%|██████▏   | 246732/400000 [00:30<00:19, 7986.29it/s] 62%|██████▏   | 247534/400000 [00:30<00:19, 7881.87it/s] 62%|██████▏   | 248325/400000 [00:30<00:19, 7728.10it/s] 62%|██████▏   | 249101/400000 [00:30<00:19, 7637.47it/s] 62%|██████▏   | 249946/400000 [00:30<00:19, 7863.89it/s] 63%|██████▎   | 250807/400000 [00:30<00:18, 8073.36it/s] 63%|██████▎   | 251646/400000 [00:30<00:18, 8164.81it/s] 63%|██████▎   | 252470/400000 [00:30<00:18, 8185.31it/s] 63%|██████▎   | 253291/400000 [00:30<00:18, 8010.98it/s] 64%|██████▎   | 254095/400000 [00:30<00:18, 7980.93it/s] 64%|██████▎   | 254895/400000 [00:31<00:18, 7973.79it/s] 64%|██████▍   | 255694/400000 [00:31<00:18, 7926.99it/s] 64%|██████▍   | 256488/400000 [00:31<00:18, 7927.27it/s] 64%|██████▍   | 257282/400000 [00:31<00:18, 7842.91it/s] 65%|██████▍   | 258067/400000 [00:31<00:18, 7844.14it/s] 65%|██████▍   | 258866/400000 [00:31<00:17, 7885.04it/s] 65%|██████▍   | 259655/400000 [00:31<00:17, 7812.70it/s] 65%|██████▌   | 260437/400000 [00:31<00:17, 7814.14it/s] 65%|██████▌   | 261219/400000 [00:31<00:17, 7795.95it/s] 66%|██████▌   | 262062/400000 [00:31<00:17, 7973.61it/s] 66%|██████▌   | 262920/400000 [00:32<00:16, 8145.01it/s] 66%|██████▌   | 263799/400000 [00:32<00:16, 8327.46it/s] 66%|██████▌   | 264661/400000 [00:32<00:16, 8411.17it/s] 66%|██████▋   | 265524/400000 [00:32<00:15, 8473.24it/s] 67%|██████▋   | 266373/400000 [00:32<00:15, 8354.43it/s] 67%|██████▋   | 267210/400000 [00:32<00:16, 8236.99it/s] 67%|██████▋   | 268035/400000 [00:32<00:16, 8132.39it/s] 67%|██████▋   | 268850/400000 [00:32<00:16, 8077.06it/s] 67%|██████▋   | 269659/400000 [00:32<00:16, 8059.88it/s] 68%|██████▊   | 270466/400000 [00:33<00:16, 8032.25it/s] 68%|██████▊   | 271270/400000 [00:33<00:16, 7982.93it/s] 68%|██████▊   | 272135/400000 [00:33<00:15, 8170.97it/s] 68%|██████▊   | 272962/400000 [00:33<00:15, 8198.77it/s] 68%|██████▊   | 273783/400000 [00:33<00:15, 8058.55it/s] 69%|██████▊   | 274634/400000 [00:33<00:15, 8187.70it/s] 69%|██████▉   | 275455/400000 [00:33<00:15, 8135.22it/s] 69%|██████▉   | 276324/400000 [00:33<00:14, 8293.33it/s] 69%|██████▉   | 277187/400000 [00:33<00:14, 8389.73it/s] 70%|██████▉   | 278028/400000 [00:33<00:14, 8198.72it/s] 70%|██████▉   | 278850/400000 [00:34<00:14, 8129.40it/s] 70%|██████▉   | 279665/400000 [00:34<00:14, 8108.27it/s] 70%|███████   | 280477/400000 [00:34<00:14, 8079.89it/s] 70%|███████   | 281286/400000 [00:34<00:14, 8074.92it/s] 71%|███████   | 282095/400000 [00:34<00:15, 7851.07it/s] 71%|███████   | 282882/400000 [00:34<00:15, 7742.10it/s] 71%|███████   | 283688/400000 [00:34<00:14, 7834.60it/s] 71%|███████   | 284533/400000 [00:34<00:14, 8007.50it/s] 71%|███████▏  | 285408/400000 [00:34<00:13, 8214.31it/s] 72%|███████▏  | 286233/400000 [00:34<00:13, 8213.37it/s] 72%|███████▏  | 287071/400000 [00:35<00:13, 8260.85it/s] 72%|███████▏  | 287917/400000 [00:35<00:13, 8318.98it/s] 72%|███████▏  | 288783/400000 [00:35<00:13, 8416.85it/s] 72%|███████▏  | 289645/400000 [00:35<00:13, 8475.53it/s] 73%|███████▎  | 290505/400000 [00:35<00:12, 8509.26it/s] 73%|███████▎  | 291357/400000 [00:35<00:13, 8310.88it/s] 73%|███████▎  | 292190/400000 [00:35<00:13, 8223.29it/s] 73%|███████▎  | 293014/400000 [00:35<00:13, 8113.22it/s] 73%|███████▎  | 293827/400000 [00:35<00:13, 8114.26it/s] 74%|███████▎  | 294640/400000 [00:35<00:13, 8081.71it/s] 74%|███████▍  | 295480/400000 [00:36<00:12, 8173.97it/s] 74%|███████▍  | 296349/400000 [00:36<00:12, 8320.44it/s] 74%|███████▍  | 297219/400000 [00:36<00:12, 8429.97it/s] 75%|███████▍  | 298096/400000 [00:36<00:11, 8526.90it/s] 75%|███████▍  | 298958/400000 [00:36<00:11, 8552.69it/s] 75%|███████▍  | 299815/400000 [00:36<00:11, 8530.86it/s] 75%|███████▌  | 300670/400000 [00:36<00:11, 8533.52it/s] 75%|███████▌  | 301524/400000 [00:36<00:11, 8458.01it/s] 76%|███████▌  | 302402/400000 [00:36<00:11, 8549.88it/s] 76%|███████▌  | 303258/400000 [00:36<00:11, 8541.37it/s] 76%|███████▌  | 304113/400000 [00:37<00:11, 8529.03it/s] 76%|███████▌  | 304980/400000 [00:37<00:11, 8568.45it/s] 76%|███████▋  | 305856/400000 [00:37<00:10, 8623.00it/s] 77%|███████▋  | 306720/400000 [00:37<00:10, 8626.40it/s] 77%|███████▋  | 307583/400000 [00:37<00:11, 8342.59it/s] 77%|███████▋  | 308420/400000 [00:37<00:11, 8294.64it/s] 77%|███████▋  | 309252/400000 [00:37<00:11, 8213.28it/s] 78%|███████▊  | 310075/400000 [00:37<00:11, 8159.28it/s] 78%|███████▊  | 310892/400000 [00:37<00:10, 8106.60it/s] 78%|███████▊  | 311704/400000 [00:38<00:11, 8018.86it/s] 78%|███████▊  | 312535/400000 [00:38<00:10, 8102.50it/s] 78%|███████▊  | 313402/400000 [00:38<00:10, 8263.13it/s] 79%|███████▊  | 314254/400000 [00:38<00:10, 8336.87it/s] 79%|███████▉  | 315089/400000 [00:38<00:10, 8157.21it/s] 79%|███████▉  | 315907/400000 [00:38<00:10, 8064.17it/s] 79%|███████▉  | 316760/400000 [00:38<00:10, 8198.09it/s] 79%|███████▉  | 317628/400000 [00:38<00:09, 8335.76it/s] 80%|███████▉  | 318503/400000 [00:38<00:09, 8455.03it/s] 80%|███████▉  | 319374/400000 [00:38<00:09, 8529.55it/s] 80%|████████  | 320229/400000 [00:39<00:09, 8453.54it/s] 80%|████████  | 321096/400000 [00:39<00:09, 8516.37it/s] 80%|████████  | 321949/400000 [00:39<00:09, 8409.70it/s] 81%|████████  | 322791/400000 [00:39<00:09, 8313.80it/s] 81%|████████  | 323624/400000 [00:39<00:09, 8245.06it/s] 81%|████████  | 324450/400000 [00:39<00:09, 8069.49it/s] 81%|████████▏ | 325259/400000 [00:39<00:09, 7932.22it/s] 82%|████████▏ | 326054/400000 [00:39<00:09, 7916.99it/s] 82%|████████▏ | 326855/400000 [00:39<00:09, 7942.64it/s] 82%|████████▏ | 327651/400000 [00:39<00:09, 7853.25it/s] 82%|████████▏ | 328438/400000 [00:40<00:09, 7846.62it/s] 82%|████████▏ | 329296/400000 [00:40<00:08, 8050.62it/s] 83%|████████▎ | 330161/400000 [00:40<00:08, 8221.42it/s] 83%|████████▎ | 331034/400000 [00:40<00:08, 8366.17it/s] 83%|████████▎ | 331908/400000 [00:40<00:08, 8474.84it/s] 83%|████████▎ | 332758/400000 [00:40<00:07, 8429.52it/s] 83%|████████▎ | 333603/400000 [00:40<00:08, 8250.66it/s] 84%|████████▎ | 334430/400000 [00:40<00:08, 8137.88it/s] 84%|████████▍ | 335246/400000 [00:40<00:08, 8086.96it/s] 84%|████████▍ | 336056/400000 [00:40<00:07, 8073.65it/s] 84%|████████▍ | 336865/400000 [00:41<00:07, 7980.27it/s] 84%|████████▍ | 337664/400000 [00:41<00:07, 7968.27it/s] 85%|████████▍ | 338467/400000 [00:41<00:07, 7984.78it/s] 85%|████████▍ | 339272/400000 [00:41<00:07, 8001.28it/s] 85%|████████▌ | 340089/400000 [00:41<00:07, 8048.38it/s] 85%|████████▌ | 340895/400000 [00:41<00:07, 7869.07it/s] 85%|████████▌ | 341683/400000 [00:41<00:07, 7813.90it/s] 86%|████████▌ | 342466/400000 [00:41<00:07, 7720.79it/s] 86%|████████▌ | 343239/400000 [00:41<00:07, 7703.39it/s] 86%|████████▌ | 344039/400000 [00:41<00:07, 7789.08it/s] 86%|████████▌ | 344830/400000 [00:42<00:07, 7822.52it/s] 86%|████████▋ | 345620/400000 [00:42<00:06, 7844.28it/s] 87%|████████▋ | 346435/400000 [00:42<00:06, 7932.45it/s] 87%|████████▋ | 347243/400000 [00:42<00:06, 7974.13it/s] 87%|████████▋ | 348055/400000 [00:42<00:06, 8015.98it/s] 87%|████████▋ | 348857/400000 [00:42<00:06, 7945.19it/s] 87%|████████▋ | 349652/400000 [00:42<00:06, 7913.99it/s] 88%|████████▊ | 350464/400000 [00:42<00:06, 7973.10it/s] 88%|████████▊ | 351274/400000 [00:42<00:06, 8010.03it/s] 88%|████████▊ | 352113/400000 [00:42<00:05, 8117.71it/s] 88%|████████▊ | 352952/400000 [00:43<00:05, 8195.11it/s] 88%|████████▊ | 353819/400000 [00:43<00:05, 8330.63it/s] 89%|████████▊ | 354679/400000 [00:43<00:05, 8409.02it/s] 89%|████████▉ | 355521/400000 [00:43<00:05, 8351.80it/s] 89%|████████▉ | 356392/400000 [00:43<00:05, 8454.98it/s] 89%|████████▉ | 357239/400000 [00:43<00:05, 8408.55it/s] 90%|████████▉ | 358081/400000 [00:43<00:05, 8173.66it/s] 90%|████████▉ | 358901/400000 [00:43<00:05, 8128.29it/s] 90%|████████▉ | 359716/400000 [00:43<00:04, 8109.08it/s] 90%|█████████ | 360528/400000 [00:44<00:04, 8030.51it/s] 90%|█████████ | 361332/400000 [00:44<00:04, 7950.81it/s] 91%|█████████ | 362129/400000 [00:44<00:04, 7951.29it/s] 91%|█████████ | 362925/400000 [00:44<00:04, 7762.59it/s] 91%|█████████ | 363738/400000 [00:44<00:04, 7867.98it/s] 91%|█████████ | 364557/400000 [00:44<00:04, 7961.29it/s] 91%|█████████▏| 365355/400000 [00:44<00:04, 7895.72it/s] 92%|█████████▏| 366162/400000 [00:44<00:04, 7944.77it/s] 92%|█████████▏| 366958/400000 [00:44<00:04, 7912.73it/s] 92%|█████████▏| 367750/400000 [00:44<00:04, 7868.03it/s] 92%|█████████▏| 368546/400000 [00:45<00:03, 7892.58it/s] 92%|█████████▏| 369371/400000 [00:45<00:03, 7993.96it/s] 93%|█████████▎| 370241/400000 [00:45<00:03, 8191.52it/s] 93%|█████████▎| 371100/400000 [00:45<00:03, 8304.67it/s] 93%|█████████▎| 371968/400000 [00:45<00:03, 8411.22it/s] 93%|█████████▎| 372831/400000 [00:45<00:03, 8473.12it/s] 93%|█████████▎| 373693/400000 [00:45<00:03, 8515.29it/s] 94%|█████████▎| 374546/400000 [00:45<00:03, 8415.72it/s] 94%|█████████▍| 375389/400000 [00:45<00:02, 8267.90it/s] 94%|█████████▍| 376217/400000 [00:45<00:02, 8214.11it/s] 94%|█████████▍| 377040/400000 [00:46<00:02, 8129.37it/s] 94%|█████████▍| 377854/400000 [00:46<00:02, 8033.66it/s] 95%|█████████▍| 378659/400000 [00:46<00:02, 8019.67it/s] 95%|█████████▍| 379517/400000 [00:46<00:02, 8179.05it/s] 95%|█████████▌| 380339/400000 [00:46<00:02, 8189.39it/s] 95%|█████████▌| 381159/400000 [00:46<00:02, 8144.99it/s] 95%|█████████▌| 381975/400000 [00:46<00:02, 8117.58it/s] 96%|█████████▌| 382788/400000 [00:46<00:02, 8052.07it/s] 96%|█████████▌| 383657/400000 [00:46<00:01, 8231.94it/s] 96%|█████████▌| 384522/400000 [00:46<00:01, 8352.29it/s] 96%|█████████▋| 385359/400000 [00:47<00:01, 8225.68it/s] 97%|█████████▋| 386183/400000 [00:47<00:01, 8195.80it/s] 97%|█████████▋| 387004/400000 [00:47<00:01, 8060.75it/s] 97%|█████████▋| 387812/400000 [00:47<00:01, 8016.37it/s] 97%|█████████▋| 388629/400000 [00:47<00:01, 8060.87it/s] 97%|█████████▋| 389436/400000 [00:47<00:01, 8016.85it/s] 98%|█████████▊| 390239/400000 [00:47<00:01, 7503.90it/s] 98%|█████████▊| 390997/400000 [00:47<00:01, 7462.18it/s] 98%|█████████▊| 391867/400000 [00:47<00:01, 7793.37it/s] 98%|█████████▊| 392742/400000 [00:48<00:00, 8056.64it/s] 98%|█████████▊| 393601/400000 [00:48<00:00, 8207.56it/s] 99%|█████████▊| 394470/400000 [00:48<00:00, 8345.75it/s] 99%|█████████▉| 395310/400000 [00:48<00:00, 8307.35it/s] 99%|█████████▉| 396145/400000 [00:48<00:00, 8242.90it/s] 99%|█████████▉| 397019/400000 [00:48<00:00, 8385.60it/s] 99%|█████████▉| 397860/400000 [00:48<00:00, 8375.72it/s]100%|█████████▉| 398719/400000 [00:48<00:00, 8435.74it/s]100%|█████████▉| 399571/400000 [00:48<00:00, 8460.61it/s]100%|█████████▉| 399999/400000 [00:48<00:00, 8186.50it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fb3c47fdcf8> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010863811161385073 	 Accuracy: 54
Train Epoch: 1 	 Loss: 0.010940673558608345 	 Accuracy: 61

  model saves at 61% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15739 out of table with 15729 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15739 out of table with 15729 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-14 18:24:02.214887: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 18:24:02.218919: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-14 18:24:02.219054: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558e7cd950b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 18:24:02.219070: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fb3711ded30> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8660 - accuracy: 0.4870
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8123 - accuracy: 0.4905 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7382 - accuracy: 0.4953
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6130 - accuracy: 0.5035
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5685 - accuracy: 0.5064
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6206 - accuracy: 0.5030
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5878 - accuracy: 0.5051
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6225 - accuracy: 0.5029
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6377 - accuracy: 0.5019
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6114 - accuracy: 0.5036
11000/25000 [============>.................] - ETA: 3s - loss: 7.6346 - accuracy: 0.5021
12000/25000 [=============>................] - ETA: 3s - loss: 7.6206 - accuracy: 0.5030
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6218 - accuracy: 0.5029
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6151 - accuracy: 0.5034
15000/25000 [=================>............] - ETA: 2s - loss: 7.6625 - accuracy: 0.5003
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6436 - accuracy: 0.5015
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6693 - accuracy: 0.4998
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6649 - accuracy: 0.5001
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6650 - accuracy: 0.5001
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6605 - accuracy: 0.5004
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6732 - accuracy: 0.4996
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6562 - accuracy: 0.5007
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6466 - accuracy: 0.5013
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6596 - accuracy: 0.5005
25000/25000 [==============================] - 8s 330us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fb34070a160> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fb3cbf5a128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 985ms/step - loss: 1.6660 - crf_viterbi_accuracy: 0.1067 - val_loss: 1.6724 - val_crf_viterbi_accuracy: 0.0933

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


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f264e328fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 19:11:56.338639
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 19:11:56.342680
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 19:11:56.345877
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 19:11:56.349233
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f265a340438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 355330.6875
Epoch 2/10

1/1 [==============================] - 0s 118ms/step - loss: 249304.0625
Epoch 3/10

1/1 [==============================] - 0s 104ms/step - loss: 142446.2031
Epoch 4/10

1/1 [==============================] - 0s 98ms/step - loss: 73482.3047
Epoch 5/10

1/1 [==============================] - 0s 99ms/step - loss: 40101.5820
Epoch 6/10

1/1 [==============================] - 0s 120ms/step - loss: 23825.0977
Epoch 7/10

1/1 [==============================] - 0s 93ms/step - loss: 15292.2734
Epoch 8/10

1/1 [==============================] - 0s 105ms/step - loss: 10578.0361
Epoch 9/10

1/1 [==============================] - 0s 105ms/step - loss: 7623.5015
Epoch 10/10

1/1 [==============================] - 0s 105ms/step - loss: 5761.6675

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.2885348   1.1786218   1.4723839   0.89848584 -1.1820965   0.40827346
   1.3461473  -0.40944183  0.7325788   0.91874087 -1.2633812   0.2956872
  -2.0523193  -0.24580884  1.3468146   2.058608    0.21034557 -0.48121208
  -0.834674    0.6155367   2.3100202  -2.2619588   0.14519489 -1.2920363
   0.43585855 -0.9321553   1.8051288   0.76482946  0.13117245  1.3107969
   1.1138966   0.47198525  0.5737471   0.751307    0.84675443 -0.18078747
   2.0396066  -0.26368982  0.46263304  1.1075667  -0.50920355  1.7034229
  -0.9714388  -0.16686064 -0.7622151  -0.70569485  1.4310433  -1.9096856
  -1.13363     0.02681422  0.3490959   1.2502842   1.2362547  -0.26083225
  -0.7515257  -1.026654    0.33750325 -0.55399615  1.9968604  -1.7111157
  -1.9084553   0.41478795  0.02290437  0.54102105  0.6235901   1.1312746
   0.81302    -0.77834857 -0.09644997 -0.28205234 -1.480988   -1.250247
  -0.09394783  0.31213802  0.67078954 -0.8144661   0.0832662   0.99500257
   0.7428375   0.8051483  -0.56488895  0.01494202  0.17545876  0.5423123
  -1.7196035  -1.2851725  -1.4499042   1.148253    0.7174128   1.6310422
   0.46376482 -0.40569595  1.0146044   0.47138178 -0.11468726  0.8892177
  -0.5575883  -1.0353518  -1.7829736   1.7047305   1.7650273   1.0289565
  -1.7752421  -0.42242217  1.2700083  -0.735536   -1.7642832  -0.75608724
   1.2833891  -1.1509306   1.0574151   0.12995075  0.47434425  1.5549272
   1.3304793  -1.1071799  -1.452147   -0.73907524 -1.9273002   1.8208003
  -0.4191375   8.871503   10.26046     9.456517    7.6088295   8.177051
   9.177983    8.968239   11.129322    7.5536857   8.960257    7.523703
   9.436752    8.46791    11.383095    8.788692    7.8517275   8.882963
   9.348981    7.0145764  11.329811    8.234157   11.081109    8.958997
   7.428466    9.285488    7.6764      9.07986     8.056287    9.012236
   8.590079    7.669971    8.929167   11.06278     9.592791    8.58266
   9.827926   11.421141   10.08003    10.374325    7.5785007  10.387041
   8.249949    7.7720666   9.136476    6.771932    9.753449    7.131703
   8.555392    7.3839417   7.5958943   9.931469    9.429764    8.639515
   9.126931    8.749559   10.752215    6.3474936   8.386664    9.251666
   0.45884156  0.5158629   1.6329422   1.4215331   3.1251225   1.3843602
   2.1555963   0.42888153  1.6456294   2.4492867   0.4545319   0.22153997
   0.7360147   1.4626622   1.6821892   1.9324037   0.17216331  0.13060838
   0.40226293  2.3120885   0.65577936  0.49150884  2.100665    1.726722
   0.52212477  0.49637777  1.1071845   0.16612077  2.8743405   1.3130634
   2.6984844   1.8726017   0.21576846  0.56931615  0.68332636  2.87646
   0.11464971  0.61945516  0.8368461   0.58301914  0.11939812  1.4095403
   1.1484752   1.1155857   0.8843799   1.416898    1.7488412   2.0108109
   1.5826843   0.41138673  0.4548943   0.19724035  1.5960312   2.1429129
   1.0793942   0.979417    0.40433735  0.458678    1.8230009   0.19636357
   2.7934902   0.7913837   2.1050606   2.1115837   0.81973135  0.26629627
   2.099433    0.6005944   0.60594785  0.7462828   0.27508092  0.3688923
   0.23731649  2.103383    0.9145339   0.3226614   3.2476182   0.740224
   1.3089907   1.5497103   2.201085    0.8764376   1.7459393   1.2584395
   1.3129307   1.1863045   0.65921426  1.1368054   0.11504334  0.2428245
   0.08739835  2.1900578   0.3077402   0.6455786   0.28048503  2.2808108
   0.21236545  0.08025122  2.1482737   2.3685808   1.0447792   2.1003976
   0.3930832   0.5115182   2.5278635   0.18186438  2.1734014   1.6593778
   0.4784441   0.07984096  2.0386107   0.44257557  2.272019    0.47720987
   0.23212826  0.18459749  1.4318385   1.683001    2.3434029   0.5593462
   0.08434767 10.853026    9.911623    8.937531    8.6642885   9.763813
   7.771795    8.963983   10.445019    9.738272   10.164821    9.882461
   7.4945064   8.870595    8.757052    9.792665   10.207574   11.043408
   9.813154    7.4756165  10.563507    9.4779005   7.782694   10.815433
   6.739614    9.067027   10.162591    8.308829    7.770996    9.374684
   8.697863    9.531885    8.324343    8.891814    8.426552    8.082297
  10.4804     10.065669    7.493842    9.282874    8.890532    7.3736696
   9.572888    8.727505    7.945077    8.972275    9.263464    8.0563345
   8.295695    7.803263    9.516637    9.327619    8.414168    9.489445
   8.92631     9.231482    8.7832985   7.4735637   9.845323    9.496994
  -7.815896   -7.970498    7.053428  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 19:12:05.397445
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.9462
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 19:12:05.401744
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8848.23
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 19:12:05.404880
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.2178
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 19:12:05.407931
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -791.421
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139802160704368
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139801199153784
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139801199154288
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139801199154792
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139801199155296
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139801199155800

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f26561c1ef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.518389
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.491801
grad_step = 000002, loss = 0.470906
grad_step = 000003, loss = 0.449532
grad_step = 000004, loss = 0.426595
grad_step = 000005, loss = 0.404678
grad_step = 000006, loss = 0.386202
grad_step = 000007, loss = 0.380036
grad_step = 000008, loss = 0.378146
grad_step = 000009, loss = 0.364779
grad_step = 000010, loss = 0.349269
grad_step = 000011, loss = 0.338663
grad_step = 000012, loss = 0.331868
grad_step = 000013, loss = 0.325633
grad_step = 000014, loss = 0.318056
grad_step = 000015, loss = 0.308785
grad_step = 000016, loss = 0.298383
grad_step = 000017, loss = 0.287828
grad_step = 000018, loss = 0.278314
grad_step = 000019, loss = 0.270662
grad_step = 000020, loss = 0.263908
grad_step = 000021, loss = 0.256396
grad_step = 000022, loss = 0.247727
grad_step = 000023, loss = 0.239028
grad_step = 000024, loss = 0.231288
grad_step = 000025, loss = 0.224383
grad_step = 000026, loss = 0.217580
grad_step = 000027, loss = 0.210382
grad_step = 000028, loss = 0.202894
grad_step = 000029, loss = 0.195684
grad_step = 000030, loss = 0.189172
grad_step = 000031, loss = 0.183008
grad_step = 000032, loss = 0.176620
grad_step = 000033, loss = 0.170110
grad_step = 000034, loss = 0.163944
grad_step = 000035, loss = 0.158237
grad_step = 000036, loss = 0.152684
grad_step = 000037, loss = 0.147038
grad_step = 000038, loss = 0.141391
grad_step = 000039, loss = 0.136035
grad_step = 000040, loss = 0.131072
grad_step = 000041, loss = 0.126241
grad_step = 000042, loss = 0.121267
grad_step = 000043, loss = 0.116364
grad_step = 000044, loss = 0.111852
grad_step = 000045, loss = 0.107603
grad_step = 000046, loss = 0.103335
grad_step = 000047, loss = 0.099029
grad_step = 000048, loss = 0.094908
grad_step = 000049, loss = 0.091114
grad_step = 000050, loss = 0.087470
grad_step = 000051, loss = 0.083760
grad_step = 000052, loss = 0.080129
grad_step = 000053, loss = 0.076762
grad_step = 000054, loss = 0.073554
grad_step = 000055, loss = 0.070371
grad_step = 000056, loss = 0.067260
grad_step = 000057, loss = 0.064334
grad_step = 000058, loss = 0.061535
grad_step = 000059, loss = 0.058755
grad_step = 000060, loss = 0.056088
grad_step = 000061, loss = 0.053580
grad_step = 000062, loss = 0.051142
grad_step = 000063, loss = 0.048746
grad_step = 000064, loss = 0.046482
grad_step = 000065, loss = 0.044332
grad_step = 000066, loss = 0.041917
grad_step = 000067, loss = 0.039410
grad_step = 000068, loss = 0.037123
grad_step = 000069, loss = 0.035142
grad_step = 000070, loss = 0.033334
grad_step = 000071, loss = 0.031476
grad_step = 000072, loss = 0.029362
grad_step = 000073, loss = 0.027273
grad_step = 000074, loss = 0.025580
grad_step = 000075, loss = 0.024051
grad_step = 000076, loss = 0.022486
grad_step = 000077, loss = 0.020926
grad_step = 000078, loss = 0.019445
grad_step = 000079, loss = 0.018095
grad_step = 000080, loss = 0.016786
grad_step = 000081, loss = 0.015535
grad_step = 000082, loss = 0.014445
grad_step = 000083, loss = 0.013377
grad_step = 000084, loss = 0.012333
grad_step = 000085, loss = 0.011407
grad_step = 000086, loss = 0.010516
grad_step = 000087, loss = 0.009687
grad_step = 000088, loss = 0.008950
grad_step = 000089, loss = 0.008259
grad_step = 000090, loss = 0.007619
grad_step = 000091, loss = 0.007016
grad_step = 000092, loss = 0.006497
grad_step = 000093, loss = 0.006065
grad_step = 000094, loss = 0.005590
grad_step = 000095, loss = 0.005133
grad_step = 000096, loss = 0.004721
grad_step = 000097, loss = 0.004388
grad_step = 000098, loss = 0.004137
grad_step = 000099, loss = 0.003889
grad_step = 000100, loss = 0.003631
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003373
grad_step = 000102, loss = 0.003177
grad_step = 000103, loss = 0.003037
grad_step = 000104, loss = 0.002907
grad_step = 000105, loss = 0.002783
grad_step = 000106, loss = 0.002653
grad_step = 000107, loss = 0.002535
grad_step = 000108, loss = 0.002444
grad_step = 000109, loss = 0.002377
grad_step = 000110, loss = 0.002325
grad_step = 000111, loss = 0.002281
grad_step = 000112, loss = 0.002245
grad_step = 000113, loss = 0.002199
grad_step = 000114, loss = 0.002153
grad_step = 000115, loss = 0.002118
grad_step = 000116, loss = 0.002094
grad_step = 000117, loss = 0.002083
grad_step = 000118, loss = 0.002078
grad_step = 000119, loss = 0.002076
grad_step = 000120, loss = 0.002070
grad_step = 000121, loss = 0.002063
grad_step = 000122, loss = 0.002051
grad_step = 000123, loss = 0.002037
grad_step = 000124, loss = 0.002026
grad_step = 000125, loss = 0.002017
grad_step = 000126, loss = 0.002012
grad_step = 000127, loss = 0.002010
grad_step = 000128, loss = 0.002011
grad_step = 000129, loss = 0.002012
grad_step = 000130, loss = 0.002015
grad_step = 000131, loss = 0.002017
grad_step = 000132, loss = 0.002020
grad_step = 000133, loss = 0.002021
grad_step = 000134, loss = 0.002020
grad_step = 000135, loss = 0.002011
grad_step = 000136, loss = 0.001998
grad_step = 000137, loss = 0.001982
grad_step = 000138, loss = 0.001966
grad_step = 000139, loss = 0.001954
grad_step = 000140, loss = 0.001948
grad_step = 000141, loss = 0.001947
grad_step = 000142, loss = 0.001948
grad_step = 000143, loss = 0.001950
grad_step = 000144, loss = 0.001950
grad_step = 000145, loss = 0.001949
grad_step = 000146, loss = 0.001943
grad_step = 000147, loss = 0.001937
grad_step = 000148, loss = 0.001927
grad_step = 000149, loss = 0.001916
grad_step = 000150, loss = 0.001906
grad_step = 000151, loss = 0.001898
grad_step = 000152, loss = 0.001892
grad_step = 000153, loss = 0.001887
grad_step = 000154, loss = 0.001885
grad_step = 000155, loss = 0.001883
grad_step = 000156, loss = 0.001883
grad_step = 000157, loss = 0.001884
grad_step = 000158, loss = 0.001887
grad_step = 000159, loss = 0.001894
grad_step = 000160, loss = 0.001912
grad_step = 000161, loss = 0.001934
grad_step = 000162, loss = 0.001969
grad_step = 000163, loss = 0.001982
grad_step = 000164, loss = 0.001971
grad_step = 000165, loss = 0.001908
grad_step = 000166, loss = 0.001851
grad_step = 000167, loss = 0.001840
grad_step = 000168, loss = 0.001868
grad_step = 000169, loss = 0.001900
grad_step = 000170, loss = 0.001891
grad_step = 000171, loss = 0.001856
grad_step = 000172, loss = 0.001822
grad_step = 000173, loss = 0.001822
grad_step = 000174, loss = 0.001842
grad_step = 000175, loss = 0.001852
grad_step = 000176, loss = 0.001840
grad_step = 000177, loss = 0.001813
grad_step = 000178, loss = 0.001798
grad_step = 000179, loss = 0.001801
grad_step = 000180, loss = 0.001811
grad_step = 000181, loss = 0.001816
grad_step = 000182, loss = 0.001806
grad_step = 000183, loss = 0.001791
grad_step = 000184, loss = 0.001778
grad_step = 000185, loss = 0.001773
grad_step = 000186, loss = 0.001775
grad_step = 000187, loss = 0.001779
grad_step = 000188, loss = 0.001782
grad_step = 000189, loss = 0.001781
grad_step = 000190, loss = 0.001778
grad_step = 000191, loss = 0.001771
grad_step = 000192, loss = 0.001765
grad_step = 000193, loss = 0.001757
grad_step = 000194, loss = 0.001752
grad_step = 000195, loss = 0.001746
grad_step = 000196, loss = 0.001742
grad_step = 000197, loss = 0.001739
grad_step = 000198, loss = 0.001737
grad_step = 000199, loss = 0.001739
grad_step = 000200, loss = 0.001747
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001766
grad_step = 000202, loss = 0.001814
grad_step = 000203, loss = 0.001883
grad_step = 000204, loss = 0.002000
grad_step = 000205, loss = 0.002018
grad_step = 000206, loss = 0.001951
grad_step = 000207, loss = 0.001763
grad_step = 000208, loss = 0.001699
grad_step = 000209, loss = 0.001789
grad_step = 000210, loss = 0.001859
grad_step = 000211, loss = 0.001812
grad_step = 000212, loss = 0.001701
grad_step = 000213, loss = 0.001705
grad_step = 000214, loss = 0.001781
grad_step = 000215, loss = 0.001779
grad_step = 000216, loss = 0.001709
grad_step = 000217, loss = 0.001676
grad_step = 000218, loss = 0.001716
grad_step = 000219, loss = 0.001749
grad_step = 000220, loss = 0.001710
grad_step = 000221, loss = 0.001668
grad_step = 000222, loss = 0.001676
grad_step = 000223, loss = 0.001704
grad_step = 000224, loss = 0.001707
grad_step = 000225, loss = 0.001672
grad_step = 000226, loss = 0.001653
grad_step = 000227, loss = 0.001665
grad_step = 000228, loss = 0.001681
grad_step = 000229, loss = 0.001677
grad_step = 000230, loss = 0.001655
grad_step = 000231, loss = 0.001641
grad_step = 000232, loss = 0.001646
grad_step = 000233, loss = 0.001655
grad_step = 000234, loss = 0.001658
grad_step = 000235, loss = 0.001647
grad_step = 000236, loss = 0.001634
grad_step = 000237, loss = 0.001627
grad_step = 000238, loss = 0.001629
grad_step = 000239, loss = 0.001635
grad_step = 000240, loss = 0.001638
grad_step = 000241, loss = 0.001638
grad_step = 000242, loss = 0.001630
grad_step = 000243, loss = 0.001622
grad_step = 000244, loss = 0.001615
grad_step = 000245, loss = 0.001609
grad_step = 000246, loss = 0.001606
grad_step = 000247, loss = 0.001605
grad_step = 000248, loss = 0.001605
grad_step = 000249, loss = 0.001606
grad_step = 000250, loss = 0.001612
grad_step = 000251, loss = 0.001621
grad_step = 000252, loss = 0.001644
grad_step = 000253, loss = 0.001674
grad_step = 000254, loss = 0.001738
grad_step = 000255, loss = 0.001769
grad_step = 000256, loss = 0.001810
grad_step = 000257, loss = 0.001730
grad_step = 000258, loss = 0.001633
grad_step = 000259, loss = 0.001583
grad_step = 000260, loss = 0.001616
grad_step = 000261, loss = 0.001678
grad_step = 000262, loss = 0.001680
grad_step = 000263, loss = 0.001631
grad_step = 000264, loss = 0.001577
grad_step = 000265, loss = 0.001572
grad_step = 000266, loss = 0.001607
grad_step = 000267, loss = 0.001639
grad_step = 000268, loss = 0.001655
grad_step = 000269, loss = 0.001614
grad_step = 000270, loss = 0.001574
grad_step = 000271, loss = 0.001552
grad_step = 000272, loss = 0.001558
grad_step = 000273, loss = 0.001582
grad_step = 000274, loss = 0.001597
grad_step = 000275, loss = 0.001596
grad_step = 000276, loss = 0.001568
grad_step = 000277, loss = 0.001544
grad_step = 000278, loss = 0.001533
grad_step = 000279, loss = 0.001536
grad_step = 000280, loss = 0.001548
grad_step = 000281, loss = 0.001555
grad_step = 000282, loss = 0.001556
grad_step = 000283, loss = 0.001543
grad_step = 000284, loss = 0.001532
grad_step = 000285, loss = 0.001520
grad_step = 000286, loss = 0.001510
grad_step = 000287, loss = 0.001505
grad_step = 000288, loss = 0.001503
grad_step = 000289, loss = 0.001504
grad_step = 000290, loss = 0.001506
grad_step = 000291, loss = 0.001513
grad_step = 000292, loss = 0.001524
grad_step = 000293, loss = 0.001535
grad_step = 000294, loss = 0.001546
grad_step = 000295, loss = 0.001578
grad_step = 000296, loss = 0.001564
grad_step = 000297, loss = 0.001563
grad_step = 000298, loss = 0.001530
grad_step = 000299, loss = 0.001493
grad_step = 000300, loss = 0.001472
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001466
grad_step = 000302, loss = 0.001472
grad_step = 000303, loss = 0.001482
grad_step = 000304, loss = 0.001491
grad_step = 000305, loss = 0.001497
grad_step = 000306, loss = 0.001513
grad_step = 000307, loss = 0.001492
grad_step = 000308, loss = 0.001477
grad_step = 000309, loss = 0.001463
grad_step = 000310, loss = 0.001449
grad_step = 000311, loss = 0.001437
grad_step = 000312, loss = 0.001435
grad_step = 000313, loss = 0.001439
grad_step = 000314, loss = 0.001441
grad_step = 000315, loss = 0.001456
grad_step = 000316, loss = 0.001474
grad_step = 000317, loss = 0.001503
grad_step = 000318, loss = 0.001504
grad_step = 000319, loss = 0.001506
grad_step = 000320, loss = 0.001472
grad_step = 000321, loss = 0.001435
grad_step = 000322, loss = 0.001414
grad_step = 000323, loss = 0.001417
grad_step = 000324, loss = 0.001444
grad_step = 000325, loss = 0.001478
grad_step = 000326, loss = 0.001517
grad_step = 000327, loss = 0.001452
grad_step = 000328, loss = 0.001407
grad_step = 000329, loss = 0.001425
grad_step = 000330, loss = 0.001447
grad_step = 000331, loss = 0.001431
grad_step = 000332, loss = 0.001404
grad_step = 000333, loss = 0.001416
grad_step = 000334, loss = 0.001442
grad_step = 000335, loss = 0.001425
grad_step = 000336, loss = 0.001393
grad_step = 000337, loss = 0.001395
grad_step = 000338, loss = 0.001411
grad_step = 000339, loss = 0.001402
grad_step = 000340, loss = 0.001385
grad_step = 000341, loss = 0.001380
grad_step = 000342, loss = 0.001387
grad_step = 000343, loss = 0.001392
grad_step = 000344, loss = 0.001389
grad_step = 000345, loss = 0.001379
grad_step = 000346, loss = 0.001375
grad_step = 000347, loss = 0.001380
grad_step = 000348, loss = 0.001382
grad_step = 000349, loss = 0.001378
grad_step = 000350, loss = 0.001373
grad_step = 000351, loss = 0.001372
grad_step = 000352, loss = 0.001376
grad_step = 000353, loss = 0.001384
grad_step = 000354, loss = 0.001399
grad_step = 000355, loss = 0.001404
grad_step = 000356, loss = 0.001423
grad_step = 000357, loss = 0.001445
grad_step = 000358, loss = 0.001485
grad_step = 000359, loss = 0.001510
grad_step = 000360, loss = 0.001522
grad_step = 000361, loss = 0.001483
grad_step = 000362, loss = 0.001426
grad_step = 000363, loss = 0.001367
grad_step = 000364, loss = 0.001342
grad_step = 000365, loss = 0.001358
grad_step = 000366, loss = 0.001389
grad_step = 000367, loss = 0.001410
grad_step = 000368, loss = 0.001399
grad_step = 000369, loss = 0.001369
grad_step = 000370, loss = 0.001341
grad_step = 000371, loss = 0.001331
grad_step = 000372, loss = 0.001338
grad_step = 000373, loss = 0.001351
grad_step = 000374, loss = 0.001360
grad_step = 000375, loss = 0.001359
grad_step = 000376, loss = 0.001355
grad_step = 000377, loss = 0.001350
grad_step = 000378, loss = 0.001343
grad_step = 000379, loss = 0.001332
grad_step = 000380, loss = 0.001323
grad_step = 000381, loss = 0.001316
grad_step = 000382, loss = 0.001316
grad_step = 000383, loss = 0.001320
grad_step = 000384, loss = 0.001324
grad_step = 000385, loss = 0.001326
grad_step = 000386, loss = 0.001321
grad_step = 000387, loss = 0.001313
grad_step = 000388, loss = 0.001306
grad_step = 000389, loss = 0.001302
grad_step = 000390, loss = 0.001300
grad_step = 000391, loss = 0.001300
grad_step = 000392, loss = 0.001298
grad_step = 000393, loss = 0.001295
grad_step = 000394, loss = 0.001292
grad_step = 000395, loss = 0.001291
grad_step = 000396, loss = 0.001291
grad_step = 000397, loss = 0.001291
grad_step = 000398, loss = 0.001289
grad_step = 000399, loss = 0.001286
grad_step = 000400, loss = 0.001282
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001281
grad_step = 000402, loss = 0.001284
grad_step = 000403, loss = 0.001293
grad_step = 000404, loss = 0.001323
grad_step = 000405, loss = 0.001367
grad_step = 000406, loss = 0.001462
grad_step = 000407, loss = 0.001477
grad_step = 000408, loss = 0.001502
grad_step = 000409, loss = 0.001488
grad_step = 000410, loss = 0.001481
grad_step = 000411, loss = 0.001422
grad_step = 000412, loss = 0.001316
grad_step = 000413, loss = 0.001266
grad_step = 000414, loss = 0.001308
grad_step = 000415, loss = 0.001366
grad_step = 000416, loss = 0.001359
grad_step = 000417, loss = 0.001303
grad_step = 000418, loss = 0.001274
grad_step = 000419, loss = 0.001289
grad_step = 000420, loss = 0.001293
grad_step = 000421, loss = 0.001274
grad_step = 000422, loss = 0.001257
grad_step = 000423, loss = 0.001266
grad_step = 000424, loss = 0.001317
grad_step = 000425, loss = 0.001347
grad_step = 000426, loss = 0.001330
grad_step = 000427, loss = 0.001263
grad_step = 000428, loss = 0.001261
grad_step = 000429, loss = 0.001293
grad_step = 000430, loss = 0.001286
grad_step = 000431, loss = 0.001267
grad_step = 000432, loss = 0.001258
grad_step = 000433, loss = 0.001275
grad_step = 000434, loss = 0.001279
grad_step = 000435, loss = 0.001250
grad_step = 000436, loss = 0.001227
grad_step = 000437, loss = 0.001237
grad_step = 000438, loss = 0.001256
grad_step = 000439, loss = 0.001249
grad_step = 000440, loss = 0.001233
grad_step = 000441, loss = 0.001234
grad_step = 000442, loss = 0.001241
grad_step = 000443, loss = 0.001230
grad_step = 000444, loss = 0.001215
grad_step = 000445, loss = 0.001213
grad_step = 000446, loss = 0.001222
grad_step = 000447, loss = 0.001226
grad_step = 000448, loss = 0.001216
grad_step = 000449, loss = 0.001210
grad_step = 000450, loss = 0.001212
grad_step = 000451, loss = 0.001213
grad_step = 000452, loss = 0.001206
grad_step = 000453, loss = 0.001199
grad_step = 000454, loss = 0.001198
grad_step = 000455, loss = 0.001200
grad_step = 000456, loss = 0.001200
grad_step = 000457, loss = 0.001196
grad_step = 000458, loss = 0.001193
grad_step = 000459, loss = 0.001194
grad_step = 000460, loss = 0.001195
grad_step = 000461, loss = 0.001192
grad_step = 000462, loss = 0.001189
grad_step = 000463, loss = 0.001187
grad_step = 000464, loss = 0.001186
grad_step = 000465, loss = 0.001186
grad_step = 000466, loss = 0.001184
grad_step = 000467, loss = 0.001181
grad_step = 000468, loss = 0.001178
grad_step = 000469, loss = 0.001176
grad_step = 000470, loss = 0.001175
grad_step = 000471, loss = 0.001174
grad_step = 000472, loss = 0.001172
grad_step = 000473, loss = 0.001170
grad_step = 000474, loss = 0.001169
grad_step = 000475, loss = 0.001168
grad_step = 000476, loss = 0.001166
grad_step = 000477, loss = 0.001165
grad_step = 000478, loss = 0.001164
grad_step = 000479, loss = 0.001163
grad_step = 000480, loss = 0.001162
grad_step = 000481, loss = 0.001162
grad_step = 000482, loss = 0.001164
grad_step = 000483, loss = 0.001167
grad_step = 000484, loss = 0.001174
grad_step = 000485, loss = 0.001187
grad_step = 000486, loss = 0.001216
grad_step = 000487, loss = 0.001260
grad_step = 000488, loss = 0.001352
grad_step = 000489, loss = 0.001414
grad_step = 000490, loss = 0.001495
grad_step = 000491, loss = 0.001404
grad_step = 000492, loss = 0.001269
grad_step = 000493, loss = 0.001190
grad_step = 000494, loss = 0.001202
grad_step = 000495, loss = 0.001240
grad_step = 000496, loss = 0.001247
grad_step = 000497, loss = 0.001239
grad_step = 000498, loss = 0.001211
grad_step = 000499, loss = 0.001159
grad_step = 000500, loss = 0.001160
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

  date_run                              2020-05-14 19:12:27.772642
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.269537
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 19:12:27.778349
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.190433
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 19:12:27.785436
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.147731
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 19:12:27.791177
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.89369
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
0   2020-05-14 19:11:56.338639  ...    mean_absolute_error
1   2020-05-14 19:11:56.342680  ...     mean_squared_error
2   2020-05-14 19:11:56.345877  ...  median_absolute_error
3   2020-05-14 19:11:56.349233  ...               r2_score
4   2020-05-14 19:12:05.397445  ...    mean_absolute_error
5   2020-05-14 19:12:05.401744  ...     mean_squared_error
6   2020-05-14 19:12:05.404880  ...  median_absolute_error
7   2020-05-14 19:12:05.407931  ...               r2_score
8   2020-05-14 19:12:27.772642  ...    mean_absolute_error
9   2020-05-14 19:12:27.778349  ...     mean_squared_error
10  2020-05-14 19:12:27.785436  ...  median_absolute_error
11  2020-05-14 19:12:27.791177  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a86bcafd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 464521.66it/s] 79%|███████▊  | 7790592/9912422 [00:00<00:03, 661897.45it/s]9920512it [00:00, 44630024.72it/s]                           
0it [00:00, ?it/s]32768it [00:00, 694929.33it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 478687.89it/s]1654784it [00:00, 12031888.63it/s]                         
0it [00:00, ?it/s]8192it [00:00, 208861.09it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a395cbe80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a38bfc0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a395cbe80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a38b520f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a3638d4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a36378c50> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a395cbe80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a38b10710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a3638d4e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2a38bfc128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f50fa5191d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=7b0bafebd3efc87504386238a39b70ff71329ae3731826167fa697d5bd457fc5
  Stored in directory: /tmp/pip-ephem-wheel-cache-qbtg7g5k/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f5092314748> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1384448/17464789 [=>............................] - ETA: 0s
 6397952/17464789 [=========>....................] - ETA: 0s
16506880/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 19:13:52.312850: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 19:13:52.316932: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-14 19:13:52.317070: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5602b843ac80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 19:13:52.317085: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5746 - accuracy: 0.5060
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7663 - accuracy: 0.4935 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7893 - accuracy: 0.4920
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7203 - accuracy: 0.4965
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7280 - accuracy: 0.4960
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7407 - accuracy: 0.4952
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.8068 - accuracy: 0.4909
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7701 - accuracy: 0.4933
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7603 - accuracy: 0.4939
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7510 - accuracy: 0.4945
11000/25000 [============>.................] - ETA: 4s - loss: 7.7614 - accuracy: 0.4938
12000/25000 [=============>................] - ETA: 4s - loss: 7.7241 - accuracy: 0.4963
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7114 - accuracy: 0.4971
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6940 - accuracy: 0.4982
15000/25000 [=================>............] - ETA: 3s - loss: 7.6850 - accuracy: 0.4988
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6331 - accuracy: 0.5022
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6486 - accuracy: 0.5012
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6530 - accuracy: 0.5009
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6790 - accuracy: 0.4992
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6715 - accuracy: 0.4997
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6653 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6634 - accuracy: 0.5002
25000/25000 [==============================] - 9s 367us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 19:14:08.420890
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 19:14:08.420890  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<51:09:48, 4.68kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<36:03:01, 6.64kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:02<25:17:18, 9.47kB/s] .vector_cache/glove.6B.zip:   0%|          | 664k/862M [00:02<17:42:34, 13.5kB/s].vector_cache/glove.6B.zip:   0%|          | 1.79M/862M [00:02<12:23:13, 19.3kB/s].vector_cache/glove.6B.zip:   1%|          | 4.66M/862M [00:02<8:38:40, 27.6kB/s] .vector_cache/glove.6B.zip:   1%|          | 9.49M/862M [00:02<6:01:07, 39.4kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.0M/862M [00:02<4:12:12, 56.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.1M/862M [00:02<2:55:48, 80.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.1M/862M [00:02<2:02:58, 114kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 21.8M/862M [00:02<1:25:48, 163kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.3M/862M [00:02<1:00:03, 232kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.2M/862M [00:03<41:57, 331kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.9M/862M [00:03<29:21, 471kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.7M/862M [00:03<20:34, 670kB/s].vector_cache/glove.6B.zip:   5%|▍         | 38.8M/862M [00:03<14:28, 948kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.8M/862M [00:03<10:19, 1.33MB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.4M/862M [00:03<07:27, 1.83MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.3M/862M [00:03<05:25, 2.51MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.7M/862M [00:03<04:04, 3.33MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.5M/862M [00:03<02:57, 4.58MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.4M/862M [00:03<02:16, 5.95MB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.2M/862M [00:04<03:36, 3.73MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:04<02:44, 4.89MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.3M/862M [00:08<09:13, 1.45MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.3M/862M [00:08<20:07, 667kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:08<17:18, 775kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.2M/862M [00:08<12:51, 1.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:08<09:07, 1.46MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:09<15:26, 864kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:10<14:04, 948kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:10<13:07, 1.02MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:10<11:18, 1.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:10<08:59, 1.48MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.0M/862M [00:10<06:47, 1.96MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:10<04:59, 2.66MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:10<04:17, 3.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:13<05:56, 2.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:13<17:51, 740kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:14<17:02, 775kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:14<13:16, 994kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:14<09:54, 1.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:14<07:12, 1.83MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:14<05:17, 2.49MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:15<3:34:03, 61.4kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:15<2:31:04, 87.0kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:16<1:46:13, 124kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:16<1:14:30, 176kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:17<55:59, 234kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:17<41:10, 317kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:17<29:09, 448kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:18<20:39, 631kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:19<19:08, 680kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:19<15:21, 847kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:20<11:13, 1.16MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:20<08:02, 1.61MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:21<12:58, 997kB/s] .vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:21<11:02, 1.17MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:21<08:10, 1.58MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:22<05:53, 2.19MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:23<13:43, 937kB/s] .vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:23<11:34, 1.11MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:23<08:34, 1.50MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:25<08:19, 1.54MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:25<07:45, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:25<05:54, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:27<06:27, 1.97MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:27<06:25, 1.98MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:27<04:59, 2.55MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:29<06:16, 2.02MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:30<12:18, 1.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:30<10:18, 1.23MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:30<07:58, 1.59MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:30<05:47, 2.18MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:31<07:49, 1.61MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:32<07:24, 1.70MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:32<05:47, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<04:15, 2.95MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:34<07:58, 1.57MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:34<09:14, 1.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:34<07:17, 1.72MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:34<05:26, 2.30MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:36<06:19, 1.97MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:36<06:21, 1.96MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:36<05:03, 2.46MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:36<03:53, 3.19MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<02:59, 4.13MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:38<09:08, 1.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:38<08:52, 1.40MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:38<06:51, 1.80MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<05:00, 2.46MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<07:39, 1.61MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:40<07:15, 1.69MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:40<05:51, 2.10MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:40<04:34, 2.68MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:40<03:54, 3.15MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<02:57, 4.14MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:43<40:30, 302kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:43<40:48, 300kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:43<31:46, 385kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:43<23:48, 514kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:43<17:19, 706kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:43<12:19, 990kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<09:03, 1.35MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:44<06:52, 1.77MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:45<1:13:07, 167kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:45<52:53, 230kB/s]  .vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:45<38:19, 317kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:45<27:40, 439kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:45<20:13, 601kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:45<14:27, 839kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:47<13:17, 910kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:47<13:05, 924kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:47<11:22, 1.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:47<09:04, 1.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:47<06:59, 1.73MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<05:05, 2.36MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:49<08:47, 1.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:49<14:44, 817kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:49<12:21, 973kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:49<09:10, 1.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:49<06:40, 1.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:51<07:55, 1.51MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:51<09:04, 1.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:51<07:40, 1.56MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:51<06:27, 1.85MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:51<05:38, 2.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:51<05:11, 2.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:52<04:44, 2.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:52<04:25, 2.70MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<04:04, 2.93MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<03:37, 3.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:52<02:49, 4.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:53<7:23:50, 26.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:53<5:11:32, 38.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:53<3:38:11, 54.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:53<2:32:55, 77.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<1:47:17, 110kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:55<1:21:16, 146kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:55<1:01:05, 194kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:55<43:37, 271kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:55<31:05, 380kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:55<22:10, 532kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:56<16:00, 736kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:56<11:35, 1.02MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:57<16:13, 725kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:57<13:26, 874kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:57<09:55, 1.18MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:57<07:04, 1.65MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:59<13:13, 884kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:59<20:58, 557kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [01:00<17:36, 664kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [01:00<13:39, 855kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [01:00<10:05, 1.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:00<07:32, 1.55MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:00<05:59, 1.94MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<04:41, 2.48MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:01<07:16, 1.60MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:01<07:11, 1.62MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:02<05:42, 2.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:02<04:36, 2.51MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:02<03:45, 3.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:02<03:05, 3.75MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:02<02:34, 4.48MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:03<09:30, 1.21MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:03<08:43, 1.32MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:04<06:38, 1.74MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:04<04:57, 2.32MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:04<03:48, 3.02MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:04<03:08, 3.66MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:05<14:47, 776kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:06<14:26, 795kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:06<11:07, 1.03MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:06<08:00, 1.43MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:06<05:47, 1.97MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:07<1:21:18, 140kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:08<59:56, 190kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:08<42:44, 267kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:08<30:14, 376kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:08<21:35, 527kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:08<15:25, 736kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:09<17:03, 665kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:10<15:05, 751kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:10<11:22, 996kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:10<08:15, 1.37MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:10<06:08, 1.84MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:10<04:39, 2.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:11<13:38, 827kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:11<11:29, 981kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:12<08:32, 1.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:12<06:10, 1.82MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:12<04:44, 2.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:13<11:59, 935kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:13<12:23, 904kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:14<09:38, 1.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:14<07:00, 1.59MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:14<05:03, 2.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:15<18:54, 589kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:15<16:11, 688kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:16<12:00, 926kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:16<08:41, 1.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:16<06:26, 1.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:16<04:54, 2.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:17<13:29, 820kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:17<12:21, 895kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:18<09:57, 1.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:18<07:18, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:18<05:23, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:18<04:02, 2.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:20<15:56, 690kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:20<22:16, 493kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:20<18:13, 603kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:20<13:25, 818kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:20<09:34, 1.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:20<07:00, 1.56MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:22<14:26, 756kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:22<12:35, 867kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:22<09:19, 1.17MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:22<06:50, 1.59MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:22<05:18, 2.05MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:22<03:56, 2.76MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<02:56, 3.68MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:25<22:05, 489kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:25<27:07, 399kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:25<22:23, 483kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:25<17:29, 618kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:25<12:47, 844kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:26<09:11, 1.17MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:26<06:38, 1.62MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:28<22:11, 484kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:28<26:40, 403kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:28<20:51, 515kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:29<16:38, 645kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:29<12:27, 861kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:29<09:20, 1.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:29<06:57, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:29<05:17, 2.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:29<04:01, 2.65MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:31<17:50, 598kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:31<17:18, 616kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:31<13:19, 800kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:31<09:32, 1.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:31<07:00, 1.52MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:33<08:31, 1.24MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:33<10:26, 1.02MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:33<08:17, 1.28MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:33<06:22, 1.66MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:33<04:38, 2.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:33<03:29, 3.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:35<19:50, 531kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:35<18:23, 573kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:35<14:59, 702kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:35<11:12, 939kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:35<08:10, 1.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:35<05:59, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:35<04:36, 2.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:37<09:44, 1.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:37<09:00, 1.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:37<06:51, 1.52MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:37<05:00, 2.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:37<03:42, 2.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:39<21:02, 494kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:39<18:45, 554kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:39<14:12, 731kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:39<10:13, 1.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:39<07:17, 1.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:40<19:41, 524kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:41<17:43, 582kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:41<15:04, 685kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:41<12:11, 846kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:41<09:39, 1.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:41<08:20, 1.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:41<06:57, 1.48MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:41<05:48, 1.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:41<04:54, 2.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:41<03:48, 2.70MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:43<05:56, 1.73MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:43<08:24, 1.22MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:43<06:58, 1.47MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:43<05:09, 1.98MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:43<03:50, 2.65MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:45<06:26, 1.58MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:45<08:40, 1.17MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:45<09:35, 1.06MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:45<09:13, 1.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:45<07:51, 1.29MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:45<06:12, 1.64MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:46<04:46, 2.13MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:46<03:36, 2.81MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:47<05:22, 1.88MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:47<05:53, 1.72MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:47<04:47, 2.11MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:47<03:36, 2.79MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:47<02:52, 3.50MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:47<02:18, 4.34MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:49<10:46, 933kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:49<12:06, 830kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:49<09:42, 1.03MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:49<07:20, 1.37MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:49<05:28, 1.83MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:49<04:05, 2.45MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:51<06:33, 1.52MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:51<06:42, 1.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:51<05:07, 1.94MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:51<03:53, 2.55MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:51<02:55, 3.39MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:53<07:42, 1.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:53<09:37, 1.03MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:53<08:46, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:53<07:30, 1.32MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:53<06:09, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:53<05:01, 1.97MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:53<03:47, 2.61MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:54<02:51, 3.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:55<09:45, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:55<08:48, 1.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:55<06:52, 1.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:55<05:19, 1.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:55<03:58, 2.46MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:55<03:02, 3.22MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:57<08:26, 1.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:57<10:00, 976kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:57<08:15, 1.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:57<06:32, 1.49MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:57<04:54, 1.99MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:57<03:37, 2.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:59<06:23, 1.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:59<08:13, 1.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:59<06:52, 1.41MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:59<05:16, 1.83MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:59<03:54, 2.47MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:59<02:53, 3.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:01<58:11, 166kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:01<44:32, 216kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:01<32:44, 294kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:01<23:39, 407kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [02:01<16:45, 573kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:01<11:53, 805kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:03<13:53, 688kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:03<13:27, 711kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:03<10:18, 927kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:03<07:23, 1.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:03<05:26, 1.75MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:05<07:56, 1.20MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:05<09:33, 993kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:05<07:30, 1.26MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:05<05:27, 1.73MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:05<04:09, 2.28MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:07<06:01, 1.56MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:07<07:54, 1.19MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:07<06:23, 1.47MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:07<04:55, 1.91MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:07<03:36, 2.60MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:07<02:45, 3.40MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:09<31:36, 296kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:09<25:15, 370kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:09<18:21, 509kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:09<13:11, 707kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:09<09:20, 996kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:11<13:06, 709kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:11<13:31, 686kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:11<12:09, 764kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:11<10:20, 898kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:12<07:56, 1.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:12<06:04, 1.53MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:12<04:28, 2.07MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:13<05:30, 1.67MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:13<05:14, 1.76MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:13<04:04, 2.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:13<03:02, 3.02MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:15<04:53, 1.87MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:15<06:31, 1.40MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:15<06:04, 1.51MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:15<05:24, 1.69MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:16<04:19, 2.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:16<03:23, 2.69MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:16<02:33, 3.56MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:17<05:43, 1.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:17<05:16, 1.72MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:17<04:00, 2.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:17<02:53, 3.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:20<49:54, 181kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:20<42:33, 212kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:20<32:03, 281kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:20<22:53, 393kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:20<16:08, 556kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:21<11:47, 760kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:21<08:38, 1.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:21<09:16, 965kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:21<06:58, 1.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:21<05:18, 1.68MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:21<04:02, 2.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:22<03:04, 2.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:25<09:59, 888kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:25<16:04, 552kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:25<13:21, 664kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:25<09:52, 897kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:25<07:02, 1.25MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:27<08:17, 1.06MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:27<09:53, 890kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:27<10:35, 831kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:27<11:00, 799kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:27<09:41, 909kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:27<09:17, 946kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:27<08:00, 1.10MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:27<07:14, 1.21MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:28<06:32, 1.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:28<05:16, 1.67MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:28<03:59, 2.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:29<04:03, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:29<03:37, 2.41MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:29<03:03, 2.85MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:29<02:26, 3.56MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:29<02:05, 4.15MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:29<01:44, 4.97MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:31<07:15, 1.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:31<12:16, 706kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:31<10:16, 843kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:31<07:36, 1.14MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:31<05:31, 1.56MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:32<04:10, 2.06MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:32<03:26, 2.50MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:33<07:50, 1.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:33<07:04, 1.21MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:33<05:28, 1.57MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:33<04:00, 2.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:33<03:00, 2.85MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:35<08:07, 1.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:35<09:14, 922kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:35<08:50, 964kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:35<07:16, 1.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:35<05:41, 1.50MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:35<04:28, 1.90MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:36<03:32, 2.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:36<02:40, 3.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:37<05:39, 1.50MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:37<05:12, 1.62MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:37<03:57, 2.13MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:37<03:00, 2.81MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:37<02:23, 3.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:37<01:59, 4.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:39<12:29, 672kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:39<11:35, 724kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:39<08:40, 966kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:39<06:27, 1.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:39<04:37, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:42<11:17, 737kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:42<16:31, 503kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:42<13:34, 612kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:43<09:58, 832kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:43<07:13, 1.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:44<06:35, 1.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:44<05:59, 1.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:44<05:05, 1.62MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:44<04:00, 2.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:45<03:01, 2.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:46<04:03, 2.02MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:46<04:20, 1.88MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:46<04:26, 1.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:46<04:30, 1.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:47<04:11, 1.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:47<03:38, 2.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:47<02:46, 2.94MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:48<03:40, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:48<03:21, 2.41MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:48<02:32, 3.18MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:50<03:28, 2.32MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:50<04:02, 1.99MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:50<03:09, 2.55MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:50<02:20, 3.41MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:52<04:19, 1.84MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:52<04:34, 1.74MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:52<03:29, 2.28MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:52<02:40, 2.97MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:52<02:00, 3.93MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:54<24:20, 325kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:54<18:35, 425kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:54<13:23, 589kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:54<09:29, 829kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:56<09:06, 861kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:56<07:55, 989kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:56<05:55, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:58<05:23, 1.44MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:58<05:18, 1.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:58<04:04, 1.90MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [03:00<04:06, 1.87MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [03:00<04:20, 1.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [03:00<03:24, 2.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [03:03<12:26, 617kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [03:03<16:16, 471kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [03:04<13:11, 581kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [03:04<09:40, 791kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [03:04<06:50, 1.11MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [03:05<10:43, 709kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [03:05<08:06, 937kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [03:05<06:00, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:05<04:16, 1.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:37<1:56:23, 64.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:38<1:29:01, 84.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:38<1:04:03, 118kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:38<45:09, 167kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:38<31:36, 237kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:40<25:05, 298kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:40<23:47, 314kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:40<18:28, 404kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:40<14:05, 529kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:40<10:30, 709kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:40<07:44, 961kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:41<05:32, 1.34MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:42<06:06, 1.21MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:42<04:58, 1.49MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [03:42<03:37, 2.03MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:44<04:21, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:44<05:50, 1.25MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:44<04:43, 1.55MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:44<03:32, 2.06MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:44<02:35, 2.80MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:47<07:04, 1.03MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:47<12:07, 598kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:47<10:12, 710kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:47<07:41, 942kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:47<05:35, 1.29MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:47<04:04, 1.77MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:47<03:02, 2.36MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:49<33:04, 217kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:49<23:50, 301kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:49<16:56, 423kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:49<12:08, 590kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:49<08:53, 804kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:49<06:32, 1.09MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:49<04:54, 1.45MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:51<11:41, 609kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:51<09:19, 763kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:51<06:53, 1.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:51<04:57, 1.43MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:51<03:45, 1.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:51<02:51, 2.46MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:53<12:41, 555kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:53<10:17, 684kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:53<07:44, 909kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:53<05:36, 1.25MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:53<04:14, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:53<03:14, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:53<02:31, 2.76MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:55<21:29, 325kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:55<15:53, 439kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:55<11:17, 616kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:55<07:59, 867kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:57<09:42, 712kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:57<07:50, 881kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:57<05:47, 1.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:57<04:15, 1.62MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:57<03:12, 2.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:57<02:37, 2.61MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:57<02:09, 3.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:59<5:28:16, 20.9kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:59<3:50:16, 29.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:59<2:41:24, 42.3kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:59<1:52:49, 60.4kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:59<1:18:54, 86.1kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:59<55:27, 122kB/s]   .vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [04:01<1:01:07, 111kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [04:01<43:28, 156kB/s]  .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [04:01<30:29, 221kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [04:01<21:28, 313kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [04:02<18:00, 373kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [04:03<13:58, 480kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [04:03<10:07, 661kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [04:03<07:10, 928kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [04:05<06:48, 975kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [04:05<07:27, 890kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [04:05<06:07, 1.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [04:05<04:36, 1.43MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [04:05<03:23, 1.94MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [04:05<02:36, 2.52MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [04:05<02:02, 3.21MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [04:07<11:20, 579kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [04:07<08:40, 755kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [04:07<06:14, 1.05MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [04:07<04:26, 1.46MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [04:09<09:36, 676kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [04:09<08:00, 811kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [04:09<06:03, 1.07MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [04:09<04:25, 1.46MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [04:09<03:15, 1.97MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [04:09<02:31, 2.55MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [04:11<06:44, 954kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [04:11<05:24, 1.19MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [04:11<04:01, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [04:11<02:55, 2.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [04:13<04:12, 1.51MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [04:13<04:13, 1.50MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [04:13<03:11, 1.98MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [04:13<02:24, 2.62MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [04:14<03:14, 1.94MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [04:15<03:25, 1.84MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [04:15<02:41, 2.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [04:15<01:56, 3.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [04:16<20:26, 304kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [04:17<15:29, 401kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [04:17<11:04, 560kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [04:17<07:52, 785kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [04:18<07:06, 866kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [04:19<06:08, 1.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [04:19<04:35, 1.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [04:19<03:15, 1.86MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [04:20<07:38, 795kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [04:20<06:33, 928kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [04:21<04:52, 1.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [04:21<03:29, 1.73MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [04:22<05:00, 1.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [04:22<04:38, 1.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [04:23<03:38, 1.65MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [04:23<02:39, 2.24MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [04:23<01:56, 3.05MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [04:24<1:52:28, 52.9kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [04:24<1:20:17, 74.0kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [04:25<57:13, 104kB/s]   .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [04:25<40:52, 145kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [04:25<28:44, 206kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [04:25<20:08, 293kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [04:26<15:59, 367kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [04:27<12:22, 474kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [04:27<09:14, 635kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [04:27<06:39, 879kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [04:27<04:42, 1.23MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [04:28<08:17, 700kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [04:29<06:45, 857kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [04:29<05:28, 1.06MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [04:29<04:23, 1.32MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [04:29<03:22, 1.72MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [04:29<02:33, 2.25MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [04:29<01:53, 3.04MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [04:30<09:16, 618kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [04:31<07:13, 794kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [04:31<05:27, 1.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [04:31<03:56, 1.44MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [04:31<02:51, 1.98MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [04:32<05:54, 959kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [04:32<05:42, 993kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [04:33<05:06, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [04:33<04:19, 1.31MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [04:33<03:32, 1.60MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [04:33<02:47, 2.02MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [04:33<02:04, 2.71MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [04:34<03:00, 1.86MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [04:34<02:51, 1.96MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [04:35<02:34, 2.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [04:35<02:06, 2.65MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [04:35<01:38, 3.40MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [04:35<01:17, 4.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [04:36<03:37, 1.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [04:36<03:27, 1.60MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [04:37<03:07, 1.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [04:37<02:44, 2.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:37<02:14, 2.45MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:37<01:46, 3.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:37<01:19, 4.12MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:39<15:01, 363kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:40<15:54, 343kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:40<12:26, 438kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:40<08:56, 609kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:40<06:25, 845kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [04:41<05:30, 980kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [04:42<04:50, 1.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:42<03:36, 1.49MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:42<02:38, 2.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:44<03:24, 1.56MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:44<04:17, 1.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:44<03:30, 1.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:44<02:32, 2.08MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:46<02:58, 1.76MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:46<03:52, 1.35MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:46<03:34, 1.46MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:46<02:53, 1.81MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:46<02:15, 2.31MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:46<01:41, 3.08MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:46<01:19, 3.93MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:48<10:11, 508kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:48<09:27, 547kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:49<07:05, 728kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:49<05:11, 992kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:49<03:39, 1.39MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:50<08:09, 626kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:50<06:57, 732kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:50<06:09, 829kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:51<05:15, 970kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:51<04:36, 1.10MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:51<03:41, 1.38MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:51<02:49, 1.80MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:51<02:03, 2.45MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:53<04:08, 1.21MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:53<04:33, 1.10MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:53<03:32, 1.42MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:53<02:36, 1.91MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:53<01:53, 2.64MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:55<09:28, 524kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:55<07:27, 666kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:55<05:22, 921kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:55<03:50, 1.28MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:55<02:47, 1.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:58<44:06, 111kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:58<35:45, 137kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:58<26:11, 187kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:58<18:31, 263kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:58<13:02, 373kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [05:00<10:07, 477kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [05:00<08:04, 597kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [05:00<06:01, 799kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [05:00<04:22, 1.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [05:00<03:08, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [05:02<04:07, 1.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [05:02<03:40, 1.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [05:02<02:45, 1.72MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [05:04<02:46, 1.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [05:04<03:29, 1.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [05:04<02:44, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [05:04<02:08, 2.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [05:04<01:35, 2.90MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [05:04<01:14, 3.73MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [05:06<05:03, 913kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [05:06<04:00, 1.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [05:06<02:53, 1.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [05:08<03:18, 1.38MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [05:08<05:34, 816kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [05:08<04:54, 926kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [05:08<03:47, 1.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [05:08<02:50, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [05:09<02:10, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [05:09<01:42, 2.63MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [05:09<01:21, 3.29MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [05:09<01:14, 3.59MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [05:10<21:34, 208kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [05:10<15:48, 283kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [05:10<11:25, 391kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [05:10<08:03, 552kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [05:10<05:43, 773kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [05:12<06:03, 729kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [05:12<05:31, 799kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [05:12<04:46, 923kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [05:12<03:49, 1.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [05:12<02:58, 1.48MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [05:12<02:13, 1.97MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [05:13<01:42, 2.55MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [05:15<03:00, 1.44MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [05:15<03:46, 1.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [05:15<03:04, 1.41MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [05:15<02:13, 1.94MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [05:19<03:55, 1.09MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [05:19<06:17, 679kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [05:19<05:16, 810kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [05:19<03:55, 1.09MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [05:19<02:52, 1.48MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [05:20<02:03, 2.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [05:21<04:03, 1.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [05:21<03:26, 1.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [05:21<03:05, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [05:21<02:59, 1.41MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [05:21<02:44, 1.53MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [05:21<02:13, 1.89MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [05:21<01:38, 2.53MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [05:25<04:04, 1.02MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [05:25<06:57, 594kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [05:25<05:51, 707kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [05:26<04:19, 952kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [05:26<03:03, 1.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [05:27<03:37, 1.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [05:27<03:13, 1.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [05:27<02:25, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [05:27<01:48, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [05:28<01:18, 3.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [05:29<21:15, 188kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [05:29<15:27, 259kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [05:29<11:10, 357kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [05:29<07:56, 501kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [05:30<05:42, 695kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [05:30<04:02, 976kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [05:31<08:59, 437kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [05:31<06:55, 567kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [05:31<04:59, 783kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [05:31<03:31, 1.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [05:33<04:47, 807kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [05:33<04:37, 835kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [05:33<03:29, 1.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [05:34<02:36, 1.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [05:34<01:53, 2.02MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [05:34<01:25, 2.66MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [05:35<06:40, 569kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [05:35<04:56, 766kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [05:35<03:35, 1.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [05:36<02:32, 1.47MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [05:37<05:10, 720kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [05:37<04:13, 879kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [05:37<03:05, 1.19MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [05:41<03:25, 1.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [05:41<06:01, 607kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [05:41<05:05, 717kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [05:41<03:44, 973kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [05:41<02:43, 1.33MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [05:41<01:57, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [05:42<03:28, 1.03MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [05:43<02:41, 1.33MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [05:43<02:10, 1.64MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [05:43<01:54, 1.87MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [05:43<01:34, 2.25MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [05:43<01:20, 2.64MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [05:43<01:03, 3.35MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [05:43<00:50, 4.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [05:45<10:12, 344kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [05:45<08:56, 394kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [05:45<06:41, 525kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [05:45<04:45, 734kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [05:46<03:21, 1.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [05:49<07:29, 460kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [05:49<08:40, 397kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [05:49<06:53, 500kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [05:50<05:00, 685kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [05:50<03:30, 968kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [05:51<04:45, 710kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [05:51<04:16, 791kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:51<04:11, 805kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:51<04:31, 746kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:52<04:21, 775kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:52<03:34, 941kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:52<02:41, 1.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:52<01:56, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:55<03:26, 960kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:55<05:42, 580kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:55<04:46, 692kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:55<03:30, 939kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:55<02:33, 1.29MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:56<01:49, 1.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:58<07:00, 462kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:58<08:13, 394kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:58<06:28, 500kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [05:58<04:47, 675kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:59<03:25, 940kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:59<02:24, 1.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:59<02:17, 1.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:59<01:39, 1.90MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:59<01:12, 2.59MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [06:01<02:38, 1.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [06:01<02:53, 1.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [06:01<02:16, 1.36MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [06:01<01:38, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [06:03<02:02, 1.49MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [06:03<02:07, 1.42MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [06:03<01:43, 1.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [06:03<01:16, 2.36MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [06:03<00:55, 3.19MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [06:05<03:15, 911kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [06:05<03:05, 957kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [06:05<03:07, 951kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [06:05<02:46, 1.07MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [06:05<02:07, 1.39MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [06:05<01:31, 1.92MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [06:07<01:40, 1.73MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [06:07<01:48, 1.59MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [06:08<01:25, 2.01MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [06:08<01:01, 2.77MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [06:09<03:42, 757kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [06:09<02:56, 956kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [06:10<02:07, 1.31MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [06:12<02:11, 1.25MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [06:12<03:14, 845kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [06:12<02:42, 1.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [06:12<01:59, 1.37MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [06:12<01:25, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [06:14<02:15, 1.18MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [06:14<01:47, 1.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [06:14<01:16, 2.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [06:16<01:37, 1.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [06:16<01:57, 1.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [06:16<01:33, 1.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [06:16<01:11, 2.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [06:16<00:51, 2.97MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [06:18<01:53, 1.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [06:18<01:42, 1.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [06:18<01:15, 2.00MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [06:18<00:55, 2.69MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [06:20<01:32, 1.61MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [06:20<01:24, 1.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [06:20<01:10, 2.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [06:20<00:52, 2.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [06:20<00:38, 3.72MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [06:22<03:11, 752kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [06:22<02:32, 940kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [06:22<01:54, 1.25MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [06:22<01:21, 1.73MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [06:24<01:39, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [06:24<01:52, 1.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [06:24<01:27, 1.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [06:24<01:03, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [06:24<00:47, 2.85MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [06:26<01:36, 1.40MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [06:26<01:33, 1.45MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [06:26<01:14, 1.81MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [06:26<00:59, 2.26MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [06:26<00:45, 2.91MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [06:26<00:38, 3.49MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [06:26<00:30, 4.33MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [06:28<02:33, 855kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [06:28<02:04, 1.06MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [06:28<01:30, 1.44MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [06:28<01:03, 2.00MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [06:30<10:19, 206kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [06:30<09:19, 227kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [06:30<06:59, 303kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [06:31<05:08, 411kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [06:31<03:40, 573kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [06:31<02:33, 809kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [06:32<02:19, 886kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [06:32<01:40, 1.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [06:32<01:13, 1.65MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [06:32<00:56, 2.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [06:32<00:44, 2.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [06:34<01:53, 1.05MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [06:34<01:54, 1.04MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [06:34<01:27, 1.35MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [06:34<01:04, 1.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [06:34<00:47, 2.44MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [06:34<00:35, 3.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [06:36<20:25, 93.8kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [06:36<14:52, 129kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [06:36<10:42, 178kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [06:36<07:40, 248kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [06:36<05:32, 343kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [06:36<04:01, 470kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [06:36<02:56, 641kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [06:36<02:06, 887kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [06:36<01:31, 1.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [06:38<02:26, 756kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [06:38<01:50, 997kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [06:38<01:20, 1.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [06:38<00:57, 1.87MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [06:38<00:43, 2.46MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [06:40<03:05, 576kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [06:40<02:48, 634kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [06:40<02:09, 822kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [06:40<01:34, 1.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [06:40<01:06, 1.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [06:41<01:28, 1.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [06:42<01:19, 1.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [06:42<01:04, 1.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [06:42<00:50, 2.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [06:42<00:39, 2.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [06:42<00:30, 3.26MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [06:42<00:23, 4.12MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [06:43<02:31, 649kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [06:44<01:59, 819kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [06:44<01:31, 1.07MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [06:44<01:08, 1.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [06:44<00:50, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [06:44<00:37, 2.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [06:47<02:22, 661kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [06:47<03:14, 485kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [06:47<02:37, 595kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [06:48<01:54, 816kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [06:48<01:21, 1.13MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [06:49<01:14, 1.20MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [06:49<01:12, 1.24MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [06:49<01:12, 1.24MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [06:49<01:13, 1.23MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [06:50<01:21, 1.10MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [06:50<01:23, 1.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [06:50<01:12, 1.23MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [06:50<00:54, 1.63MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [06:50<00:38, 2.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [06:51<01:04, 1.34MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [06:51<00:53, 1.61MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [06:51<00:40, 2.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [06:51<00:29, 2.84MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [06:53<00:48, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [06:53<00:46, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [06:53<00:35, 2.29MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:54<00:25, 3.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:55<01:08, 1.14MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:55<00:59, 1.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:55<00:43, 1.75MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [06:55<00:31, 2.38MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [06:57<00:49, 1.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:57<00:45, 1.59MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:57<00:34, 2.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:57<00:24, 2.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [07:38<14:12, 81.3kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [07:38<11:06, 104kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [07:38<08:02, 143kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [07:38<05:38, 202kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [07:38<03:50, 288kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [07:39<02:53, 376kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 798M/862M [07:39<02:04, 520kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [07:39<01:30, 709kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [07:40<01:07, 949kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [07:40<00:49, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [07:40<00:35, 1.73MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [07:40<00:26, 2.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [07:45<18:53, 53.8kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [07:45<14:15, 71.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [07:45<10:09, 99.8kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [07:45<07:13, 140kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [07:45<05:02, 198kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [07:45<03:30, 280kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [07:45<02:27, 394kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [07:46<01:41, 557kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [07:46<01:16, 725kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [07:46<00:50, 1.03MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [07:49<01:49, 467kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [07:49<02:08, 400kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [07:49<01:42, 501kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [07:49<01:13, 688kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [07:49<00:51, 954kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [07:49<00:35, 1.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [07:51<01:01, 772kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [07:51<00:50, 928kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [07:51<00:37, 1.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [07:51<00:27, 1.65MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [07:51<00:19, 2.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [07:51<00:14, 2.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [07:53<12:39, 56.7kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [07:53<08:57, 79.9kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [07:53<06:10, 114kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [07:53<04:07, 162kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [07:56<03:22, 192kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [07:56<02:54, 223kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [07:56<02:10, 297kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [07:57<01:33, 410kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [07:57<01:04, 576kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [07:57<00:43, 811kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [07:58<00:51, 671kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [07:58<00:41, 828kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [07:58<00:30, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [07:58<00:21, 1.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [07:59<00:15, 2.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [07:59<00:11, 2.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [08:00<00:42, 727kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [08:00<00:36, 829kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [08:00<00:26, 1.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [08:00<00:19, 1.47MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [08:01<00:14, 1.94MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [08:01<00:10, 2.57MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [08:02<00:19, 1.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [08:02<00:15, 1.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [08:02<00:11, 2.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [08:02<00:08, 2.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [08:03<00:06, 3.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [08:04<00:17, 1.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [08:04<00:16, 1.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [08:04<00:12, 1.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [08:04<00:09, 2.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [08:05<00:06, 2.95MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [08:05<00:04, 3.83MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [08:06<00:41, 442kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [08:06<00:32, 559kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [08:06<00:22, 774kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [08:06<00:16, 1.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [08:06<00:10, 1.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [08:07<00:07, 1.94MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [08:08<00:37, 381kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [08:08<00:28, 497kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [08:08<00:20, 667kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [08:08<00:13, 921kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [08:08<00:09, 1.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [08:09<00:05, 1.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [08:10<00:32, 313kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [08:10<00:23, 411kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [08:10<00:16, 571kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [08:10<00:09, 800kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [08:10<00:05, 1.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [08:12<00:10, 575kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [08:12<00:07, 726kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [08:12<00:04, 997kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [08:12<00:02, 1.38MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [08:12<00:00, 1.90MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [08:14<00:34, 52.2kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [08:14<00:20, 73.8kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [08:14<00:07, 105kB/s] .vector_cache/glove.6B.zip: 862MB [08:14, 1.74MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 789/400000 [00:00<00:50, 7886.00it/s]  0%|          | 1645/400000 [00:00<00:49, 8076.22it/s]  1%|          | 2526/400000 [00:00<00:47, 8282.36it/s]  1%|          | 3440/400000 [00:00<00:46, 8518.79it/s]  1%|          | 4248/400000 [00:00<00:47, 8382.11it/s]  1%|▏         | 5079/400000 [00:00<00:47, 8359.06it/s]  1%|▏         | 5972/400000 [00:00<00:46, 8519.14it/s]  2%|▏         | 6883/400000 [00:00<00:45, 8688.12it/s]  2%|▏         | 7731/400000 [00:00<00:45, 8622.56it/s]  2%|▏         | 8630/400000 [00:01<00:44, 8729.59it/s]  2%|▏         | 9483/400000 [00:01<00:45, 8585.04it/s]  3%|▎         | 10334/400000 [00:01<00:45, 8562.07it/s]  3%|▎         | 11232/400000 [00:01<00:44, 8682.60it/s]  3%|▎         | 12208/400000 [00:01<00:43, 8979.52it/s]  3%|▎         | 13144/400000 [00:01<00:42, 9090.16it/s]  4%|▎         | 14052/400000 [00:01<00:43, 8874.91it/s]  4%|▍         | 15000/400000 [00:01<00:42, 9046.92it/s]  4%|▍         | 15912/400000 [00:01<00:42, 9068.11it/s]  4%|▍         | 16821/400000 [00:01<00:42, 9073.09it/s]  4%|▍         | 17729/400000 [00:02<00:42, 9071.85it/s]  5%|▍         | 18637/400000 [00:02<00:43, 8840.20it/s]  5%|▍         | 19542/400000 [00:02<00:42, 8901.30it/s]  5%|▌         | 20448/400000 [00:02<00:42, 8944.24it/s]  5%|▌         | 21344/400000 [00:02<00:42, 8882.22it/s]  6%|▌         | 22288/400000 [00:02<00:41, 9039.72it/s]  6%|▌         | 23194/400000 [00:02<00:41, 8999.38it/s]  6%|▌         | 24140/400000 [00:02<00:41, 9131.06it/s]  6%|▋         | 25055/400000 [00:02<00:41, 8977.06it/s]  6%|▋         | 25955/400000 [00:02<00:42, 8848.90it/s]  7%|▋         | 26842/400000 [00:03<00:42, 8854.33it/s]  7%|▋         | 27729/400000 [00:03<00:44, 8427.03it/s]  7%|▋         | 28577/400000 [00:03<00:43, 8442.40it/s]  7%|▋         | 29489/400000 [00:03<00:42, 8634.54it/s]  8%|▊         | 30454/400000 [00:03<00:41, 8914.35it/s]  8%|▊         | 31351/400000 [00:03<00:42, 8726.23it/s]  8%|▊         | 32246/400000 [00:03<00:41, 8791.78it/s]  8%|▊         | 33217/400000 [00:03<00:40, 9046.79it/s]  9%|▊         | 34163/400000 [00:03<00:39, 9166.05it/s]  9%|▉         | 35083/400000 [00:03<00:40, 9007.66it/s]  9%|▉         | 35987/400000 [00:04<00:40, 8996.73it/s]  9%|▉         | 36896/400000 [00:04<00:40, 9023.97it/s]  9%|▉         | 37845/400000 [00:04<00:39, 9157.40it/s] 10%|▉         | 38808/400000 [00:04<00:38, 9292.03it/s] 10%|▉         | 39739/400000 [00:04<00:39, 9206.01it/s] 10%|█         | 40661/400000 [00:04<00:40, 8860.33it/s] 10%|█         | 41554/400000 [00:04<00:40, 8879.98it/s] 11%|█         | 42480/400000 [00:04<00:39, 8989.08it/s] 11%|█         | 43382/400000 [00:04<00:39, 8957.96it/s] 11%|█         | 44344/400000 [00:04<00:38, 9145.65it/s] 11%|█▏        | 45267/400000 [00:05<00:38, 9168.59it/s] 12%|█▏        | 46186/400000 [00:05<00:39, 8960.63it/s] 12%|█▏        | 47145/400000 [00:05<00:38, 9139.84it/s] 12%|█▏        | 48062/400000 [00:05<00:39, 9017.16it/s] 12%|█▏        | 48966/400000 [00:05<00:39, 8897.48it/s] 12%|█▏        | 49899/400000 [00:05<00:38, 9019.18it/s] 13%|█▎        | 50803/400000 [00:05<00:39, 8881.70it/s] 13%|█▎        | 51694/400000 [00:05<00:39, 8889.79it/s] 13%|█▎        | 52585/400000 [00:05<00:39, 8783.13it/s] 13%|█▎        | 53465/400000 [00:06<00:39, 8767.32it/s] 14%|█▎        | 54343/400000 [00:06<00:39, 8696.36it/s] 14%|█▍        | 55214/400000 [00:06<00:40, 8409.52it/s] 14%|█▍        | 56149/400000 [00:06<00:39, 8670.04it/s] 14%|█▍        | 57051/400000 [00:06<00:39, 8770.39it/s] 14%|█▍        | 57967/400000 [00:06<00:38, 8882.09it/s] 15%|█▍        | 58858/400000 [00:06<00:39, 8641.61it/s] 15%|█▍        | 59726/400000 [00:06<00:39, 8564.43it/s] 15%|█▌        | 60606/400000 [00:06<00:39, 8633.04it/s] 15%|█▌        | 61520/400000 [00:06<00:38, 8776.12it/s] 16%|█▌        | 62400/400000 [00:07<00:39, 8599.94it/s] 16%|█▌        | 63263/400000 [00:07<00:39, 8500.35it/s] 16%|█▌        | 64167/400000 [00:07<00:38, 8654.70it/s] 16%|█▋        | 65064/400000 [00:07<00:38, 8744.78it/s] 17%|█▋        | 66007/400000 [00:07<00:37, 8937.74it/s] 17%|█▋        | 66924/400000 [00:07<00:36, 9005.04it/s] 17%|█▋        | 67827/400000 [00:07<00:36, 9002.27it/s] 17%|█▋        | 68729/400000 [00:07<00:38, 8717.45it/s] 17%|█▋        | 69613/400000 [00:07<00:37, 8752.71it/s] 18%|█▊        | 70544/400000 [00:07<00:36, 8911.09it/s] 18%|█▊        | 71468/400000 [00:08<00:36, 9005.84it/s] 18%|█▊        | 72399/400000 [00:08<00:36, 9092.69it/s] 18%|█▊        | 73310/400000 [00:08<00:37, 8716.14it/s] 19%|█▊        | 74186/400000 [00:08<00:38, 8431.67it/s] 19%|█▉        | 75035/400000 [00:08<00:39, 8292.21it/s] 19%|█▉        | 75883/400000 [00:08<00:38, 8346.65it/s] 19%|█▉        | 76727/400000 [00:08<00:38, 8373.35it/s] 19%|█▉        | 77593/400000 [00:08<00:38, 8457.20it/s] 20%|█▉        | 78538/400000 [00:08<00:36, 8732.26it/s] 20%|█▉        | 79449/400000 [00:08<00:36, 8838.92it/s] 20%|██        | 80336/400000 [00:09<00:37, 8638.83it/s] 20%|██        | 81203/400000 [00:09<00:36, 8645.05it/s] 21%|██        | 82127/400000 [00:09<00:36, 8814.72it/s] 21%|██        | 83037/400000 [00:09<00:35, 8897.40it/s] 21%|██        | 83929/400000 [00:09<00:36, 8583.51it/s] 21%|██        | 84865/400000 [00:09<00:35, 8801.57it/s] 21%|██▏       | 85750/400000 [00:09<00:35, 8804.65it/s] 22%|██▏       | 86667/400000 [00:09<00:35, 8909.37it/s] 22%|██▏       | 87601/400000 [00:09<00:34, 9032.94it/s] 22%|██▏       | 88507/400000 [00:10<00:34, 9002.84it/s] 22%|██▏       | 89409/400000 [00:10<00:35, 8870.10it/s] 23%|██▎       | 90298/400000 [00:10<00:37, 8302.82it/s] 23%|██▎       | 91137/400000 [00:10<00:37, 8232.69it/s] 23%|██▎       | 92056/400000 [00:10<00:36, 8496.96it/s] 23%|██▎       | 92993/400000 [00:10<00:35, 8739.55it/s] 23%|██▎       | 93926/400000 [00:10<00:34, 8906.17it/s] 24%|██▎       | 94834/400000 [00:10<00:34, 8957.14it/s] 24%|██▍       | 95771/400000 [00:10<00:33, 9076.12it/s] 24%|██▍       | 96708/400000 [00:10<00:33, 9160.63it/s] 24%|██▍       | 97627/400000 [00:11<00:33, 9052.37it/s] 25%|██▍       | 98535/400000 [00:11<00:34, 8852.73it/s] 25%|██▍       | 99423/400000 [00:11<00:34, 8632.44it/s] 25%|██▌       | 100295/400000 [00:11<00:34, 8657.74it/s] 25%|██▌       | 101165/400000 [00:11<00:34, 8668.99it/s] 26%|██▌       | 102034/400000 [00:11<00:35, 8318.78it/s] 26%|██▌       | 102918/400000 [00:11<00:35, 8467.14it/s] 26%|██▌       | 103769/400000 [00:11<00:35, 8446.70it/s] 26%|██▌       | 104669/400000 [00:11<00:34, 8602.95it/s] 26%|██▋       | 105577/400000 [00:11<00:33, 8739.46it/s] 27%|██▋       | 106454/400000 [00:12<00:33, 8746.88it/s] 27%|██▋       | 107331/400000 [00:12<00:34, 8502.74it/s] 27%|██▋       | 108193/400000 [00:12<00:34, 8534.12it/s] 27%|██▋       | 109139/400000 [00:12<00:33, 8791.65it/s] 28%|██▊       | 110089/400000 [00:12<00:32, 8989.43it/s] 28%|██▊       | 110992/400000 [00:12<00:32, 8976.87it/s] 28%|██▊       | 111893/400000 [00:12<00:32, 8983.83it/s] 28%|██▊       | 112794/400000 [00:12<00:33, 8623.59it/s] 28%|██▊       | 113661/400000 [00:12<00:33, 8556.52it/s] 29%|██▊       | 114559/400000 [00:13<00:32, 8678.65it/s] 29%|██▉       | 115470/400000 [00:13<00:32, 8803.32it/s] 29%|██▉       | 116353/400000 [00:13<00:32, 8735.42it/s] 29%|██▉       | 117229/400000 [00:13<00:32, 8580.56it/s] 30%|██▉       | 118089/400000 [00:13<00:32, 8568.62it/s] 30%|██▉       | 118995/400000 [00:13<00:32, 8708.56it/s] 30%|██▉       | 119935/400000 [00:13<00:31, 8904.18it/s] 30%|███       | 120882/400000 [00:13<00:30, 9066.67it/s] 30%|███       | 121791/400000 [00:13<00:30, 9017.61it/s] 31%|███       | 122715/400000 [00:13<00:30, 9082.10it/s] 31%|███       | 123628/400000 [00:14<00:30, 9095.74it/s] 31%|███       | 124550/400000 [00:14<00:30, 9132.42it/s] 31%|███▏      | 125479/400000 [00:14<00:29, 9176.92it/s] 32%|███▏      | 126398/400000 [00:14<00:30, 9109.42it/s] 32%|███▏      | 127310/400000 [00:14<00:29, 9103.20it/s] 32%|███▏      | 128221/400000 [00:14<00:30, 8928.68it/s] 32%|███▏      | 129148/400000 [00:14<00:30, 9026.88it/s] 33%|███▎      | 130052/400000 [00:14<00:30, 8862.42it/s] 33%|███▎      | 130940/400000 [00:14<00:30, 8847.80it/s] 33%|███▎      | 131886/400000 [00:14<00:29, 9021.31it/s] 33%|███▎      | 132790/400000 [00:15<00:29, 9005.10it/s] 33%|███▎      | 133692/400000 [00:15<00:29, 8899.28it/s] 34%|███▎      | 134599/400000 [00:15<00:29, 8948.36it/s] 34%|███▍      | 135495/400000 [00:15<00:29, 8941.33it/s] 34%|███▍      | 136438/400000 [00:15<00:29, 9082.13it/s] 34%|███▍      | 137348/400000 [00:15<00:28, 9071.70it/s] 35%|███▍      | 138308/400000 [00:15<00:28, 9221.64it/s] 35%|███▍      | 139232/400000 [00:15<00:28, 9191.29it/s] 35%|███▌      | 140152/400000 [00:15<00:30, 8583.38it/s] 35%|███▌      | 141037/400000 [00:15<00:29, 8659.90it/s] 35%|███▌      | 141919/400000 [00:16<00:29, 8706.92it/s] 36%|███▌      | 142795/400000 [00:16<00:30, 8501.15it/s] 36%|███▌      | 143666/400000 [00:16<00:29, 8560.95it/s] 36%|███▌      | 144545/400000 [00:16<00:29, 8627.41it/s] 36%|███▋      | 145458/400000 [00:16<00:29, 8771.73it/s] 37%|███▋      | 146342/400000 [00:16<00:28, 8790.40it/s] 37%|███▋      | 147262/400000 [00:16<00:28, 8908.49it/s] 37%|███▋      | 148155/400000 [00:16<00:28, 8824.91it/s] 37%|███▋      | 149039/400000 [00:16<00:28, 8751.86it/s] 37%|███▋      | 149981/400000 [00:16<00:27, 8940.17it/s] 38%|███▊      | 150929/400000 [00:17<00:27, 9093.73it/s] 38%|███▊      | 151841/400000 [00:17<00:27, 9098.08it/s] 38%|███▊      | 152761/400000 [00:17<00:27, 9125.85it/s] 38%|███▊      | 153678/400000 [00:17<00:26, 9134.79it/s] 39%|███▊      | 154593/400000 [00:17<00:26, 9129.50it/s] 39%|███▉      | 155507/400000 [00:17<00:27, 8889.86it/s] 39%|███▉      | 156398/400000 [00:17<00:27, 8852.86it/s] 39%|███▉      | 157285/400000 [00:17<00:28, 8519.14it/s] 40%|███▉      | 158141/400000 [00:17<00:28, 8523.16it/s] 40%|███▉      | 159038/400000 [00:18<00:27, 8650.23it/s] 40%|███▉      | 159926/400000 [00:18<00:27, 8716.13it/s] 40%|████      | 160823/400000 [00:18<00:27, 8788.72it/s] 40%|████      | 161757/400000 [00:18<00:26, 8946.29it/s] 41%|████      | 162665/400000 [00:18<00:26, 8984.89it/s] 41%|████      | 163592/400000 [00:18<00:26, 9067.53it/s] 41%|████      | 164500/400000 [00:18<00:26, 9022.66it/s] 41%|████▏     | 165404/400000 [00:18<00:26, 8950.44it/s] 42%|████▏     | 166338/400000 [00:18<00:25, 9063.00it/s] 42%|████▏     | 167246/400000 [00:18<00:26, 8704.19it/s] 42%|████▏     | 168121/400000 [00:19<00:26, 8649.60it/s] 42%|████▏     | 168989/400000 [00:19<00:26, 8604.10it/s] 42%|████▏     | 169915/400000 [00:19<00:26, 8788.79it/s] 43%|████▎     | 170821/400000 [00:19<00:25, 8866.91it/s] 43%|████▎     | 171710/400000 [00:19<00:26, 8708.28it/s] 43%|████▎     | 172583/400000 [00:19<00:26, 8612.73it/s] 43%|████▎     | 173446/400000 [00:19<00:26, 8529.92it/s] 44%|████▎     | 174337/400000 [00:19<00:26, 8638.76it/s] 44%|████▍     | 175236/400000 [00:19<00:25, 8739.73it/s] 44%|████▍     | 176112/400000 [00:19<00:25, 8649.44it/s] 44%|████▍     | 176978/400000 [00:20<00:25, 8608.36it/s] 44%|████▍     | 177884/400000 [00:20<00:25, 8737.21it/s] 45%|████▍     | 178812/400000 [00:20<00:24, 8890.30it/s] 45%|████▍     | 179729/400000 [00:20<00:24, 8971.02it/s] 45%|████▌     | 180633/400000 [00:20<00:24, 8989.17it/s] 45%|████▌     | 181567/400000 [00:20<00:24, 9091.55it/s] 46%|████▌     | 182477/400000 [00:20<00:24, 9037.41it/s] 46%|████▌     | 183383/400000 [00:20<00:23, 9042.55it/s] 46%|████▌     | 184293/400000 [00:20<00:23, 9059.50it/s] 46%|████▋     | 185200/400000 [00:20<00:23, 8953.11it/s] 47%|████▋     | 186108/400000 [00:21<00:23, 8988.59it/s] 47%|████▋     | 187008/400000 [00:21<00:23, 8969.64it/s] 47%|████▋     | 187906/400000 [00:21<00:24, 8702.26it/s] 47%|████▋     | 188808/400000 [00:21<00:24, 8793.62it/s] 47%|████▋     | 189689/400000 [00:21<00:23, 8762.99it/s] 48%|████▊     | 190602/400000 [00:21<00:23, 8869.34it/s] 48%|████▊     | 191501/400000 [00:21<00:23, 8902.48it/s] 48%|████▊     | 192416/400000 [00:21<00:23, 8974.16it/s] 48%|████▊     | 193315/400000 [00:21<00:23, 8902.61it/s] 49%|████▊     | 194206/400000 [00:22<00:25, 8202.65it/s] 49%|████▉     | 195094/400000 [00:22<00:24, 8394.38it/s] 49%|████▉     | 195958/400000 [00:22<00:24, 8466.15it/s] 49%|████▉     | 196820/400000 [00:22<00:23, 8511.38it/s] 49%|████▉     | 197693/400000 [00:22<00:23, 8574.63it/s] 50%|████▉     | 198570/400000 [00:22<00:23, 8632.16it/s] 50%|████▉     | 199459/400000 [00:22<00:23, 8705.19it/s] 50%|█████     | 200363/400000 [00:22<00:22, 8801.31it/s] 50%|█████     | 201270/400000 [00:22<00:22, 8876.82it/s] 51%|█████     | 202159/400000 [00:22<00:23, 8583.27it/s] 51%|█████     | 203021/400000 [00:23<00:23, 8386.67it/s] 51%|█████     | 203913/400000 [00:23<00:22, 8539.57it/s] 51%|█████     | 204827/400000 [00:23<00:22, 8711.03it/s] 51%|█████▏    | 205725/400000 [00:23<00:22, 8789.99it/s] 52%|█████▏    | 206607/400000 [00:23<00:22, 8788.45it/s] 52%|█████▏    | 207500/400000 [00:23<00:21, 8827.99it/s] 52%|█████▏    | 208448/400000 [00:23<00:21, 9012.68it/s] 52%|█████▏    | 209351/400000 [00:23<00:21, 8850.45it/s] 53%|█████▎    | 210253/400000 [00:23<00:21, 8900.44it/s] 53%|█████▎    | 211145/400000 [00:23<00:21, 8826.00it/s] 53%|█████▎    | 212029/400000 [00:24<00:21, 8763.59it/s] 53%|█████▎    | 212927/400000 [00:24<00:21, 8826.55it/s] 53%|█████▎    | 213830/400000 [00:24<00:20, 8885.98it/s] 54%|█████▎    | 214792/400000 [00:24<00:20, 9093.92it/s] 54%|█████▍    | 215727/400000 [00:24<00:20, 9167.39it/s] 54%|█████▍    | 216646/400000 [00:24<00:20, 9031.00it/s] 54%|█████▍    | 217551/400000 [00:24<00:20, 8950.46it/s] 55%|█████▍    | 218501/400000 [00:24<00:19, 9107.24it/s] 55%|█████▍    | 219438/400000 [00:24<00:19, 9182.59it/s] 55%|█████▌    | 220367/400000 [00:24<00:19, 9212.50it/s] 55%|█████▌    | 221290/400000 [00:25<00:20, 8670.28it/s] 56%|█████▌    | 222187/400000 [00:25<00:20, 8756.74it/s] 56%|█████▌    | 223068/400000 [00:25<00:20, 8745.28it/s] 56%|█████▌    | 223947/400000 [00:25<00:20, 8733.46it/s] 56%|█████▌    | 224841/400000 [00:25<00:19, 8793.49it/s] 56%|█████▋    | 225723/400000 [00:25<00:20, 8675.40it/s] 57%|█████▋    | 226618/400000 [00:25<00:19, 8753.92it/s] 57%|█████▋    | 227508/400000 [00:25<00:19, 8794.60it/s] 57%|█████▋    | 228441/400000 [00:25<00:19, 8947.44it/s] 57%|█████▋    | 229338/400000 [00:25<00:19, 8818.20it/s] 58%|█████▊    | 230222/400000 [00:26<00:20, 8349.60it/s] 58%|█████▊    | 231119/400000 [00:26<00:19, 8526.22it/s] 58%|█████▊    | 231977/400000 [00:26<00:19, 8427.21it/s] 58%|█████▊    | 232827/400000 [00:26<00:19, 8447.27it/s] 58%|█████▊    | 233734/400000 [00:26<00:19, 8623.38it/s] 59%|█████▊    | 234671/400000 [00:26<00:18, 8833.81it/s] 59%|█████▉    | 235574/400000 [00:26<00:18, 8889.90it/s] 59%|█████▉    | 236466/400000 [00:26<00:18, 8710.52it/s] 59%|█████▉    | 237396/400000 [00:26<00:18, 8878.94it/s] 60%|█████▉    | 238287/400000 [00:27<00:18, 8528.54it/s] 60%|█████▉    | 239145/400000 [00:27<00:19, 8418.25it/s] 60%|██████    | 240035/400000 [00:27<00:18, 8556.38it/s] 60%|██████    | 240894/400000 [00:27<00:19, 8339.45it/s] 60%|██████    | 241802/400000 [00:27<00:18, 8547.22it/s] 61%|██████    | 242661/400000 [00:27<00:18, 8504.00it/s] 61%|██████    | 243515/400000 [00:27<00:18, 8499.73it/s] 61%|██████    | 244408/400000 [00:27<00:18, 8621.24it/s] 61%|██████▏   | 245272/400000 [00:27<00:17, 8623.92it/s] 62%|██████▏   | 246136/400000 [00:27<00:17, 8568.59it/s] 62%|██████▏   | 246994/400000 [00:28<00:18, 8277.68it/s] 62%|██████▏   | 247845/400000 [00:28<00:18, 8344.97it/s] 62%|██████▏   | 248788/400000 [00:28<00:17, 8642.40it/s] 62%|██████▏   | 249729/400000 [00:28<00:16, 8858.48it/s] 63%|██████▎   | 250670/400000 [00:28<00:16, 9016.36it/s] 63%|██████▎   | 251576/400000 [00:28<00:16, 8957.41it/s] 63%|██████▎   | 252475/400000 [00:28<00:16, 8960.40it/s] 63%|██████▎   | 253403/400000 [00:28<00:16, 9052.40it/s] 64%|██████▎   | 254329/400000 [00:28<00:15, 9111.84it/s] 64%|██████▍   | 255242/400000 [00:28<00:16, 8937.01it/s] 64%|██████▍   | 256138/400000 [00:29<00:16, 8812.68it/s] 64%|██████▍   | 257050/400000 [00:29<00:16, 8899.87it/s] 64%|██████▍   | 257942/400000 [00:29<00:16, 8811.57it/s] 65%|██████▍   | 258859/400000 [00:29<00:15, 8915.40it/s] 65%|██████▍   | 259752/400000 [00:29<00:15, 8814.24it/s] 65%|██████▌   | 260635/400000 [00:29<00:16, 8498.25it/s] 65%|██████▌   | 261494/400000 [00:29<00:16, 8523.89it/s] 66%|██████▌   | 262425/400000 [00:29<00:15, 8743.23it/s] 66%|██████▌   | 263382/400000 [00:29<00:15, 8975.48it/s] 66%|██████▌   | 264304/400000 [00:30<00:15, 9044.81it/s] 66%|██████▋   | 265212/400000 [00:30<00:15, 8646.74it/s] 67%|██████▋   | 266085/400000 [00:30<00:15, 8671.19it/s] 67%|██████▋   | 266972/400000 [00:30<00:15, 8727.60it/s] 67%|██████▋   | 267866/400000 [00:30<00:15, 8788.28it/s] 67%|██████▋   | 268819/400000 [00:30<00:14, 8996.52it/s] 67%|██████▋   | 269722/400000 [00:30<00:14, 8901.15it/s] 68%|██████▊   | 270667/400000 [00:30<00:14, 9058.71it/s] 68%|██████▊   | 271595/400000 [00:30<00:14, 9122.38it/s] 68%|██████▊   | 272525/400000 [00:30<00:13, 9173.52it/s] 68%|██████▊   | 273444/400000 [00:31<00:13, 9069.08it/s] 69%|██████▊   | 274353/400000 [00:31<00:14, 8827.33it/s] 69%|██████▉   | 275238/400000 [00:31<00:14, 8774.59it/s] 69%|██████▉   | 276118/400000 [00:31<00:14, 8734.47it/s] 69%|██████▉   | 277006/400000 [00:31<00:14, 8774.33it/s] 69%|██████▉   | 277953/400000 [00:31<00:13, 8969.50it/s] 70%|██████▉   | 278852/400000 [00:31<00:13, 8822.47it/s] 70%|██████▉   | 279736/400000 [00:31<00:13, 8708.92it/s] 70%|███████   | 280613/400000 [00:31<00:13, 8726.14it/s] 70%|███████   | 281549/400000 [00:31<00:13, 8905.90it/s] 71%|███████   | 282442/400000 [00:32<00:13, 8686.85it/s] 71%|███████   | 283319/400000 [00:32<00:13, 8711.10it/s] 71%|███████   | 284255/400000 [00:32<00:13, 8892.58it/s] 71%|███████▏  | 285149/400000 [00:32<00:12, 8904.10it/s] 72%|███████▏  | 286062/400000 [00:32<00:12, 8970.52it/s] 72%|███████▏  | 286979/400000 [00:32<00:12, 9028.01it/s] 72%|███████▏  | 287883/400000 [00:32<00:12, 9023.48it/s] 72%|███████▏  | 288799/400000 [00:32<00:12, 9061.86it/s] 72%|███████▏  | 289706/400000 [00:32<00:12, 8977.92it/s] 73%|███████▎  | 290605/400000 [00:32<00:12, 8699.68it/s] 73%|███████▎  | 291478/400000 [00:33<00:12, 8488.89it/s] 73%|███████▎  | 292330/400000 [00:33<00:12, 8462.85it/s] 73%|███████▎  | 293197/400000 [00:33<00:12, 8522.83it/s] 74%|███████▎  | 294113/400000 [00:33<00:12, 8702.51it/s] 74%|███████▍  | 295078/400000 [00:33<00:11, 8965.83it/s] 74%|███████▍  | 295978/400000 [00:33<00:11, 8768.89it/s] 74%|███████▍  | 296900/400000 [00:33<00:11, 8897.39it/s] 74%|███████▍  | 297885/400000 [00:33<00:11, 9161.46it/s] 75%|███████▍  | 298866/400000 [00:33<00:10, 9345.74it/s] 75%|███████▍  | 299821/400000 [00:33<00:10, 9405.36it/s] 75%|███████▌  | 300765/400000 [00:34<00:11, 8936.25it/s] 75%|███████▌  | 301666/400000 [00:34<00:11, 8841.09it/s] 76%|███████▌  | 302562/400000 [00:34<00:10, 8875.32it/s] 76%|███████▌  | 303523/400000 [00:34<00:10, 9082.52it/s] 76%|███████▌  | 304442/400000 [00:34<00:10, 9113.37it/s] 76%|███████▋  | 305356/400000 [00:34<00:10, 8867.40it/s] 77%|███████▋  | 306246/400000 [00:34<00:10, 8855.84it/s] 77%|███████▋  | 307134/400000 [00:34<00:10, 8778.02it/s] 77%|███████▋  | 308014/400000 [00:34<00:10, 8697.90it/s] 77%|███████▋  | 308914/400000 [00:35<00:10, 8784.23it/s] 77%|███████▋  | 309794/400000 [00:35<00:10, 8446.75it/s] 78%|███████▊  | 310728/400000 [00:35<00:10, 8695.18it/s] 78%|███████▊  | 311605/400000 [00:35<00:10, 8715.38it/s] 78%|███████▊  | 312542/400000 [00:35<00:09, 8901.18it/s] 78%|███████▊  | 313468/400000 [00:35<00:09, 9005.28it/s] 79%|███████▊  | 314371/400000 [00:35<00:09, 8810.34it/s] 79%|███████▉  | 315315/400000 [00:35<00:09, 8988.85it/s] 79%|███████▉  | 316272/400000 [00:35<00:09, 9154.08it/s] 79%|███████▉  | 317191/400000 [00:35<00:10, 8247.69it/s] 80%|███████▉  | 318035/400000 [00:36<00:10, 7941.44it/s] 80%|███████▉  | 318845/400000 [00:36<00:10, 7831.45it/s] 80%|███████▉  | 319640/400000 [00:36<00:10, 7695.75it/s] 80%|████████  | 320419/400000 [00:36<00:10, 7577.90it/s] 80%|████████  | 321219/400000 [00:36<00:10, 7698.02it/s] 80%|████████  | 321994/400000 [00:36<00:10, 7627.85it/s] 81%|████████  | 322884/400000 [00:36<00:09, 7968.69it/s] 81%|████████  | 323788/400000 [00:36<00:09, 8261.28it/s] 81%|████████  | 324686/400000 [00:36<00:08, 8462.49it/s] 81%|████████▏ | 325542/400000 [00:37<00:08, 8490.37it/s] 82%|████████▏ | 326418/400000 [00:37<00:08, 8569.05it/s] 82%|████████▏ | 327346/400000 [00:37<00:08, 8769.11it/s] 82%|████████▏ | 328227/400000 [00:37<00:08, 8574.48it/s] 82%|████████▏ | 329156/400000 [00:37<00:08, 8775.70it/s] 83%|████████▎ | 330045/400000 [00:37<00:07, 8807.76it/s] 83%|████████▎ | 330933/400000 [00:37<00:07, 8827.32it/s] 83%|████████▎ | 331869/400000 [00:37<00:07, 8977.31it/s] 83%|████████▎ | 332769/400000 [00:37<00:07, 8848.96it/s] 83%|████████▎ | 333656/400000 [00:37<00:07, 8743.60it/s] 84%|████████▎ | 334532/400000 [00:38<00:07, 8441.13it/s] 84%|████████▍ | 335407/400000 [00:38<00:07, 8528.87it/s] 84%|████████▍ | 336305/400000 [00:38<00:07, 8658.72it/s] 84%|████████▍ | 337174/400000 [00:38<00:07, 8614.18it/s] 85%|████████▍ | 338080/400000 [00:38<00:07, 8739.63it/s] 85%|████████▍ | 339019/400000 [00:38<00:06, 8922.98it/s] 85%|████████▍ | 339914/400000 [00:38<00:06, 8868.64it/s] 85%|████████▌ | 340803/400000 [00:38<00:06, 8676.76it/s] 85%|████████▌ | 341754/400000 [00:38<00:06, 8909.81it/s] 86%|████████▌ | 342693/400000 [00:38<00:06, 9047.29it/s] 86%|████████▌ | 343601/400000 [00:39<00:06, 8926.95it/s] 86%|████████▌ | 344496/400000 [00:39<00:06, 8774.93it/s] 86%|████████▋ | 345376/400000 [00:39<00:06, 8652.16it/s] 87%|████████▋ | 346244/400000 [00:39<00:06, 8489.90it/s] 87%|████████▋ | 347195/400000 [00:39<00:06, 8759.61it/s] 87%|████████▋ | 348075/400000 [00:39<00:06, 8624.33it/s] 87%|████████▋ | 348941/400000 [00:39<00:05, 8550.70it/s] 87%|████████▋ | 349855/400000 [00:39<00:05, 8718.93it/s] 88%|████████▊ | 350809/400000 [00:39<00:05, 8948.33it/s] 88%|████████▊ | 351707/400000 [00:40<00:05, 8943.95it/s] 88%|████████▊ | 352604/400000 [00:40<00:05, 8838.03it/s] 88%|████████▊ | 353490/400000 [00:40<00:05, 8834.72it/s] 89%|████████▊ | 354458/400000 [00:40<00:05, 9070.06it/s] 89%|████████▉ | 355368/400000 [00:40<00:05, 8912.71it/s] 89%|████████▉ | 356340/400000 [00:40<00:04, 9139.39it/s] 89%|████████▉ | 357308/400000 [00:40<00:04, 9293.08it/s] 90%|████████▉ | 358241/400000 [00:40<00:04, 9205.40it/s] 90%|████████▉ | 359217/400000 [00:40<00:04, 9364.94it/s] 90%|█████████ | 360156/400000 [00:40<00:04, 9360.16it/s] 90%|█████████ | 361094/400000 [00:41<00:04, 9282.25it/s] 91%|█████████ | 362024/400000 [00:41<00:04, 9127.24it/s] 91%|█████████ | 362939/400000 [00:41<00:04, 8874.33it/s] 91%|█████████ | 363829/400000 [00:41<00:04, 8830.37it/s] 91%|█████████ | 364783/400000 [00:41<00:03, 9031.40it/s] 91%|█████████▏| 365753/400000 [00:41<00:03, 9222.05it/s] 92%|█████████▏| 366678/400000 [00:41<00:03, 9157.77it/s] 92%|█████████▏| 367596/400000 [00:41<00:03, 8902.30it/s] 92%|█████████▏| 368521/400000 [00:41<00:03, 9001.20it/s] 92%|█████████▏| 369483/400000 [00:41<00:03, 9176.88it/s] 93%|█████████▎| 370404/400000 [00:42<00:03, 9004.39it/s] 93%|█████████▎| 371307/400000 [00:42<00:03, 8859.03it/s] 93%|█████████▎| 372196/400000 [00:42<00:03, 8808.54it/s] 93%|█████████▎| 373079/400000 [00:42<00:03, 8744.66it/s] 93%|█████████▎| 373995/400000 [00:42<00:02, 8863.16it/s] 94%|█████████▎| 374917/400000 [00:42<00:02, 8966.59it/s] 94%|█████████▍| 375863/400000 [00:42<00:02, 9106.97it/s] 94%|█████████▍| 376776/400000 [00:42<00:02, 9076.07it/s] 94%|█████████▍| 377704/400000 [00:42<00:02, 9135.93it/s] 95%|█████████▍| 378619/400000 [00:42<00:02, 8971.35it/s] 95%|█████████▍| 379518/400000 [00:43<00:02, 8735.54it/s] 95%|█████████▌| 380446/400000 [00:43<00:02, 8891.34it/s] 95%|█████████▌| 381351/400000 [00:43<00:02, 8936.99it/s] 96%|█████████▌| 382309/400000 [00:43<00:01, 9118.58it/s] 96%|█████████▌| 383223/400000 [00:43<00:01, 8977.59it/s] 96%|█████████▌| 384123/400000 [00:43<00:01, 8979.63it/s] 96%|█████████▋| 385023/400000 [00:43<00:01, 8807.86it/s] 96%|█████████▋| 385915/400000 [00:43<00:01, 8839.13it/s] 97%|█████████▋| 386849/400000 [00:43<00:01, 8981.94it/s] 97%|█████████▋| 387749/400000 [00:44<00:01, 8959.25it/s] 97%|█████████▋| 388677/400000 [00:44<00:01, 9052.24it/s] 97%|█████████▋| 389586/400000 [00:44<00:01, 9061.20it/s] 98%|█████████▊| 390493/400000 [00:44<00:01, 9026.33it/s] 98%|█████████▊| 391408/400000 [00:44<00:00, 9061.24it/s] 98%|█████████▊| 392315/400000 [00:44<00:00, 8971.56it/s] 98%|█████████▊| 393213/400000 [00:44<00:00, 8943.29it/s] 99%|█████████▊| 394148/400000 [00:44<00:00, 9059.07it/s] 99%|█████████▉| 395055/400000 [00:44<00:00, 9029.18it/s] 99%|█████████▉| 396019/400000 [00:44<00:00, 9201.33it/s] 99%|█████████▉| 396973/400000 [00:45<00:00, 9297.47it/s] 99%|█████████▉| 397904/400000 [00:45<00:00, 9077.44it/s]100%|█████████▉| 398814/400000 [00:45<00:00, 8765.88it/s]100%|█████████▉| 399769/400000 [00:45<00:00, 8985.32it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8818.76it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f77a1180be0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011276575031880828 	 Accuracy: 50
Train Epoch: 1 	 Loss: 0.011391955075854042 	 Accuracy: 51

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
2020-05-14 19:24:47.474501: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 19:24:47.479146: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-14 19:24:47.479412: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557ca919dd70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 19:24:47.479429: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f774db69128> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.3446 - accuracy: 0.5210
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6820 - accuracy: 0.4990 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6360 - accuracy: 0.5020
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6245 - accuracy: 0.5027
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5746 - accuracy: 0.5060
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6130 - accuracy: 0.5035
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6184 - accuracy: 0.5031
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5823 - accuracy: 0.5055
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5951 - accuracy: 0.5047
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6114 - accuracy: 0.5036
11000/25000 [============>.................] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
12000/25000 [=============>................] - ETA: 3s - loss: 7.6206 - accuracy: 0.5030
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6065 - accuracy: 0.5039
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6086 - accuracy: 0.5038
15000/25000 [=================>............] - ETA: 3s - loss: 7.6165 - accuracy: 0.5033
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6292 - accuracy: 0.5024
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6360 - accuracy: 0.5020
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6573 - accuracy: 0.5006
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6473 - accuracy: 0.5013
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6329 - accuracy: 0.5022
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6411 - accuracy: 0.5017
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6367 - accuracy: 0.5020
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6520 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6577 - accuracy: 0.5006
25000/25000 [==============================] - 9s 362us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f77050476d8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f774a724da0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 920ms/step - loss: 1.3713 - crf_viterbi_accuracy: 0.6800 - val_loss: 1.3934 - val_crf_viterbi_accuracy: 0.6533

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

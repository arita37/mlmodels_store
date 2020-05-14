
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fd81b1e8fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 20:15:18.479989
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 20:15:18.483621
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 20:15:18.486839
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 20:15:18.489640
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fd827200438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353562.5938
Epoch 2/10

1/1 [==============================] - 0s 97ms/step - loss: 263909.8125
Epoch 3/10

1/1 [==============================] - 0s 85ms/step - loss: 148936.6406
Epoch 4/10

1/1 [==============================] - 0s 89ms/step - loss: 74591.6484
Epoch 5/10

1/1 [==============================] - 0s 93ms/step - loss: 38116.4922
Epoch 6/10

1/1 [==============================] - 0s 96ms/step - loss: 21041.7871
Epoch 7/10

1/1 [==============================] - 0s 93ms/step - loss: 12640.8643
Epoch 8/10

1/1 [==============================] - 0s 96ms/step - loss: 8219.9453
Epoch 9/10

1/1 [==============================] - 0s 94ms/step - loss: 5796.5869
Epoch 10/10

1/1 [==============================] - 0s 111ms/step - loss: 4408.1553

  #### Inference Need return ypred, ytrue ######################### 
[[-5.31973600e-01  5.82882643e-01  1.11351037e+00  4.47049677e-01
   5.02678216e-01 -1.40600586e+00 -3.19315761e-01 -1.28658974e+00
  -5.88649631e-01 -1.67421639e+00  5.35126150e-01  8.28823447e-01
   1.62562144e+00 -8.69141102e-01 -7.88460195e-01  1.32835960e+00
   1.87794602e+00  2.33504581e+00 -1.21299386e+00 -1.18931139e+00
   7.64733851e-02  3.84407341e-02  1.37009561e-01  7.57193327e-01
   5.65658510e-01 -1.47169399e+00 -6.71151340e-01  1.70923007e+00
  -5.12216091e-02 -2.34224582e+00 -2.71456748e-01  1.54619920e+00
  -4.90825623e-01 -2.58282471e+00  3.71358097e-01 -8.57544184e-01
  -5.73309183e-01  6.79347634e-01  1.20952463e+00  8.58660638e-01
  -2.42230916e+00  5.07722497e-01 -4.06285614e-01 -1.52044511e+00
  -6.19560480e-03 -2.75347620e-01 -2.83803076e-01 -1.19124854e+00
   5.99656105e-01 -9.83595192e-01 -1.33448660e+00  1.37362421e+00
  -5.49836874e-01  2.52020288e+00  8.44459891e-01 -2.81159252e-01
   6.05452299e-01 -2.34432310e-01 -4.61621761e-01 -1.23611502e-01
  -3.27019006e-01  9.82553196e+00  9.30573559e+00  8.19749069e+00
   1.24329891e+01  1.20830364e+01  1.12669439e+01  1.09810629e+01
   9.05760574e+00  1.12787094e+01  1.06306391e+01  1.28294277e+01
   1.22740145e+01  8.91986752e+00  1.13484707e+01  9.37545109e+00
   1.06378307e+01  1.16117744e+01  1.07791271e+01  1.02230930e+01
   9.41923141e+00  8.44287777e+00  1.17972288e+01  1.20436401e+01
   1.03648634e+01  1.07179527e+01  1.16874561e+01  1.11424465e+01
   9.71578789e+00  1.15858288e+01  1.04168091e+01  1.18084154e+01
   9.82551765e+00  1.15649996e+01  9.08833122e+00  1.00654039e+01
   9.98359585e+00  1.13301592e+01  1.14345264e+01  1.00171108e+01
   1.18454933e+01  8.30003738e+00  1.03700733e+01  1.02944794e+01
   1.02425756e+01  1.18524628e+01  9.23069286e+00  1.07606325e+01
   1.07294254e+01  9.78898430e+00  1.14039268e+01  9.82782173e+00
   8.54000473e+00  1.08811159e+01  1.33883247e+01  1.05429010e+01
   1.05592995e+01  1.18221922e+01  9.34345818e+00  9.65602684e+00
   2.34476298e-01  1.25174451e+00  1.08502865e-01 -1.16019499e+00
  -5.08776844e-01 -1.23846436e+00  5.03696561e-01  3.17777693e-03
   8.51588011e-01 -8.34656060e-01 -1.25601625e+00  9.41438556e-01
  -1.41357481e+00 -3.64571184e-01 -5.68012953e-01 -1.51438320e+00
  -2.71282703e-01 -9.83495593e-01 -7.73450494e-01 -3.54160309e-01
   8.58423710e-01 -1.02095032e+00  1.41490579e+00  8.74668896e-01
  -8.09596777e-02 -1.03424621e+00 -1.29773831e+00  2.33022594e+00
   8.19003224e-01  3.24360847e-01 -9.06552196e-01 -9.42943394e-02
   1.56723309e+00  4.53542113e-01  2.09248924e+00 -6.52769327e-01
   6.42390013e-01  1.68009663e+00  7.69704580e-04 -3.12182724e-01
  -1.37478441e-01  9.87123430e-01 -6.42914772e-01  6.76879644e-01
  -1.49195755e+00  8.77254605e-01  1.74488759e+00 -6.15141809e-01
   8.15998256e-01 -5.26224494e-01  1.73289448e-01  7.90959239e-01
   2.40329337e+00  8.51252854e-01  1.98565513e-01  1.36279070e+00
   9.93302822e-01 -6.88454509e-01  1.48710907e+00  1.84038687e+00
   2.05721200e-01  7.26128578e-01  3.60717177e-01  2.27079630e-01
   1.79391348e+00  1.10663128e+00  2.48591363e-01  1.06289577e+00
   1.84872484e+00  1.79656696e+00  1.10645795e+00  2.47923970e-01
   1.43558741e+00  2.85356069e+00  8.76862407e-01  1.41521549e+00
   4.73012447e-01  6.56178832e-01  1.80578589e+00  1.97620320e+00
   4.41275716e-01  3.58723164e-01  1.02940845e+00  2.30558991e-01
   2.67032385e+00  1.61591029e+00  9.61525142e-01  1.51255441e+00
   2.05825758e+00  8.95307362e-01  9.77141976e-01  7.92323589e-01
   2.06879473e+00  4.89812613e-01  3.75072718e-01  4.39327097e+00
   2.87099028e+00  1.27907991e-01  3.08560705e+00  1.58293629e+00
   3.53086805e+00  2.08861411e-01  1.88188291e+00  8.15137684e-01
   1.44854152e+00  9.82367814e-01  2.42170990e-01  6.44141197e-01
   4.01852429e-01  3.71913373e-01  1.92950702e+00  5.03554881e-01
   1.19095254e+00  2.92213082e-01  1.23550069e+00  2.65843630e-01
   1.31361890e+00  2.58164167e+00  2.36655951e-01  4.77099597e-01
   1.68772578e-01  1.16022339e+01  1.06974211e+01  1.08079863e+01
   8.76101112e+00  1.16787252e+01  1.17619524e+01  1.06179209e+01
   1.28155689e+01  1.11581326e+01  1.10464001e+01  1.06145725e+01
   1.12951679e+01  9.37654591e+00  1.03451920e+01  1.13368797e+01
   1.05567455e+01  1.19824162e+01  1.16600370e+01  1.12886410e+01
   1.06977272e+01  1.01914749e+01  1.22234125e+01  9.09815216e+00
   8.75585938e+00  1.19993706e+01  1.00439739e+01  1.15411234e+01
   1.06425180e+01  9.95557499e+00  1.05859003e+01  1.19918871e+01
   1.07567663e+01  1.04888887e+01  1.18322020e+01  1.00633793e+01
   1.08464165e+01  1.06799574e+01  1.19479275e+01  9.77113724e+00
   1.08891411e+01  9.57013035e+00  1.13063326e+01  1.10750332e+01
   9.31877041e+00  1.19271631e+01  1.04621363e+01  9.38006592e+00
   1.06999378e+01  1.02991943e+01  1.17450085e+01  1.07361040e+01
   9.61746025e+00  1.06398354e+01  9.87083054e+00  1.00749922e+01
   7.93655062e+00  9.38900471e+00  8.97390366e+00  1.08085785e+01
   5.04388690e-01  3.63820732e-01  1.02896047e+00  1.47788954e+00
   1.52741337e+00  3.31278801e-01  2.10616517e+00  2.28629827e-01
   9.58573222e-01  3.30093479e+00  1.99968135e+00  1.47762239e-01
   1.83955765e+00  7.17178524e-01  2.43588734e+00  1.54483294e+00
   1.58684170e+00  1.60156369e+00  2.33019972e+00  3.37840676e-01
   5.20888865e-01  1.31935012e+00  3.04058886e+00  9.22170579e-01
   1.94866002e-01  2.69736624e+00  2.57368207e-01  7.44701803e-01
   2.23546672e+00  1.44868076e+00  1.08939230e+00  1.80570674e+00
   1.17815697e+00  3.18125427e-01  6.66909933e-01  3.39240193e-01
   1.78847885e+00  2.14943695e+00  4.20060039e-01  1.13125813e+00
   8.72043312e-01  1.77765632e+00  2.93238044e-01  3.73129511e+00
   5.14063478e-01  1.35545373e+00  2.31657720e+00  7.91304529e-01
   1.97313941e+00  7.36236572e-01  1.34766269e+00  4.84316111e-01
   1.92998600e+00  1.78463924e+00  2.38410044e+00  2.60823250e+00
   1.68984950e+00  6.37910306e-01  1.74133301e-01  7.89453030e-01
  -1.50609083e+01  1.14167833e+01 -1.33033724e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 20:15:26.973873
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.1798
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 20:15:26.977552
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8521.27
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 20:15:26.980937
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.3538
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 20:15:26.984099
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -762.14
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140565787891024
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140564846355008
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140564846355512
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140564846356016
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140564846356520
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140564846357024

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fd81b101438> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.476512
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.445510
grad_step = 000002, loss = 0.422075
grad_step = 000003, loss = 0.400911
grad_step = 000004, loss = 0.378617
grad_step = 000005, loss = 0.356668
grad_step = 000006, loss = 0.335622
grad_step = 000007, loss = 0.316670
grad_step = 000008, loss = 0.305071
grad_step = 000009, loss = 0.296387
grad_step = 000010, loss = 0.282436
grad_step = 000011, loss = 0.269538
grad_step = 000012, loss = 0.260689
grad_step = 000013, loss = 0.253938
grad_step = 000014, loss = 0.246169
grad_step = 000015, loss = 0.236361
grad_step = 000016, loss = 0.225219
grad_step = 000017, loss = 0.214166
grad_step = 000018, loss = 0.204086
grad_step = 000019, loss = 0.195511
grad_step = 000020, loss = 0.186351
grad_step = 000021, loss = 0.175918
grad_step = 000022, loss = 0.165419
grad_step = 000023, loss = 0.156056
grad_step = 000024, loss = 0.147853
grad_step = 000025, loss = 0.139932
grad_step = 000026, loss = 0.131723
grad_step = 000027, loss = 0.123413
grad_step = 000028, loss = 0.115774
grad_step = 000029, loss = 0.109098
grad_step = 000030, loss = 0.102714
grad_step = 000031, loss = 0.096038
grad_step = 000032, loss = 0.089462
grad_step = 000033, loss = 0.083431
grad_step = 000034, loss = 0.077637
grad_step = 000035, loss = 0.071846
grad_step = 000036, loss = 0.066375
grad_step = 000037, loss = 0.061227
grad_step = 000038, loss = 0.056385
grad_step = 000039, loss = 0.051776
grad_step = 000040, loss = 0.047214
grad_step = 000041, loss = 0.042771
grad_step = 000042, loss = 0.038714
grad_step = 000043, loss = 0.035015
grad_step = 000044, loss = 0.031508
grad_step = 000045, loss = 0.028244
grad_step = 000046, loss = 0.025377
grad_step = 000047, loss = 0.022836
grad_step = 000048, loss = 0.020365
grad_step = 000049, loss = 0.018164
grad_step = 000050, loss = 0.016310
grad_step = 000051, loss = 0.014655
grad_step = 000052, loss = 0.013085
grad_step = 000053, loss = 0.011659
grad_step = 000054, loss = 0.010441
grad_step = 000055, loss = 0.009283
grad_step = 000056, loss = 0.008209
grad_step = 000057, loss = 0.007324
grad_step = 000058, loss = 0.006580
grad_step = 000059, loss = 0.005960
grad_step = 000060, loss = 0.005458
grad_step = 000061, loss = 0.005032
grad_step = 000062, loss = 0.004664
grad_step = 000063, loss = 0.004369
grad_step = 000064, loss = 0.004133
grad_step = 000065, loss = 0.003935
grad_step = 000066, loss = 0.003790
grad_step = 000067, loss = 0.003673
grad_step = 000068, loss = 0.003545
grad_step = 000069, loss = 0.003438
grad_step = 000070, loss = 0.003353
grad_step = 000071, loss = 0.003268
grad_step = 000072, loss = 0.003186
grad_step = 000073, loss = 0.003121
grad_step = 000074, loss = 0.003051
grad_step = 000075, loss = 0.002976
grad_step = 000076, loss = 0.002909
grad_step = 000077, loss = 0.002841
grad_step = 000078, loss = 0.002773
grad_step = 000079, loss = 0.002708
grad_step = 000080, loss = 0.002647
grad_step = 000081, loss = 0.002590
grad_step = 000082, loss = 0.002542
grad_step = 000083, loss = 0.002497
grad_step = 000084, loss = 0.002456
grad_step = 000085, loss = 0.002422
grad_step = 000086, loss = 0.002389
grad_step = 000087, loss = 0.002360
grad_step = 000088, loss = 0.002337
grad_step = 000089, loss = 0.002317
grad_step = 000090, loss = 0.002299
grad_step = 000091, loss = 0.002287
grad_step = 000092, loss = 0.002275
grad_step = 000093, loss = 0.002264
grad_step = 000094, loss = 0.002257
grad_step = 000095, loss = 0.002250
grad_step = 000096, loss = 0.002244
grad_step = 000097, loss = 0.002239
grad_step = 000098, loss = 0.002235
grad_step = 000099, loss = 0.002231
grad_step = 000100, loss = 0.002228
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002224
grad_step = 000102, loss = 0.002220
grad_step = 000103, loss = 0.002217
grad_step = 000104, loss = 0.002213
grad_step = 000105, loss = 0.002210
grad_step = 000106, loss = 0.002207
grad_step = 000107, loss = 0.002203
grad_step = 000108, loss = 0.002199
grad_step = 000109, loss = 0.002195
grad_step = 000110, loss = 0.002190
grad_step = 000111, loss = 0.002186
grad_step = 000112, loss = 0.002182
grad_step = 000113, loss = 0.002177
grad_step = 000114, loss = 0.002173
grad_step = 000115, loss = 0.002168
grad_step = 000116, loss = 0.002164
grad_step = 000117, loss = 0.002160
grad_step = 000118, loss = 0.002156
grad_step = 000119, loss = 0.002151
grad_step = 000120, loss = 0.002147
grad_step = 000121, loss = 0.002143
grad_step = 000122, loss = 0.002139
grad_step = 000123, loss = 0.002134
grad_step = 000124, loss = 0.002130
grad_step = 000125, loss = 0.002126
grad_step = 000126, loss = 0.002121
grad_step = 000127, loss = 0.002117
grad_step = 000128, loss = 0.002113
grad_step = 000129, loss = 0.002110
grad_step = 000130, loss = 0.002105
grad_step = 000131, loss = 0.002100
grad_step = 000132, loss = 0.002097
grad_step = 000133, loss = 0.002092
grad_step = 000134, loss = 0.002088
grad_step = 000135, loss = 0.002083
grad_step = 000136, loss = 0.002079
grad_step = 000137, loss = 0.002073
grad_step = 000138, loss = 0.002069
grad_step = 000139, loss = 0.002064
grad_step = 000140, loss = 0.002060
grad_step = 000141, loss = 0.002054
grad_step = 000142, loss = 0.002050
grad_step = 000143, loss = 0.002044
grad_step = 000144, loss = 0.002039
grad_step = 000145, loss = 0.002033
grad_step = 000146, loss = 0.002028
grad_step = 000147, loss = 0.002023
grad_step = 000148, loss = 0.002017
grad_step = 000149, loss = 0.002012
grad_step = 000150, loss = 0.002006
grad_step = 000151, loss = 0.002000
grad_step = 000152, loss = 0.001994
grad_step = 000153, loss = 0.001989
grad_step = 000154, loss = 0.001983
grad_step = 000155, loss = 0.001978
grad_step = 000156, loss = 0.001974
grad_step = 000157, loss = 0.001972
grad_step = 000158, loss = 0.001969
grad_step = 000159, loss = 0.001963
grad_step = 000160, loss = 0.001954
grad_step = 000161, loss = 0.001944
grad_step = 000162, loss = 0.001936
grad_step = 000163, loss = 0.001930
grad_step = 000164, loss = 0.001927
grad_step = 000165, loss = 0.001924
grad_step = 000166, loss = 0.001925
grad_step = 000167, loss = 0.001930
grad_step = 000168, loss = 0.001936
grad_step = 000169, loss = 0.001933
grad_step = 000170, loss = 0.001915
grad_step = 000171, loss = 0.001896
grad_step = 000172, loss = 0.001885
grad_step = 000173, loss = 0.001882
grad_step = 000174, loss = 0.001884
grad_step = 000175, loss = 0.001887
grad_step = 000176, loss = 0.001886
grad_step = 000177, loss = 0.001876
grad_step = 000178, loss = 0.001862
grad_step = 000179, loss = 0.001848
grad_step = 000180, loss = 0.001842
grad_step = 000181, loss = 0.001844
grad_step = 000182, loss = 0.001846
grad_step = 000183, loss = 0.001848
grad_step = 000184, loss = 0.001848
grad_step = 000185, loss = 0.001858
grad_step = 000186, loss = 0.001890
grad_step = 000187, loss = 0.001905
grad_step = 000188, loss = 0.001919
grad_step = 000189, loss = 0.001873
grad_step = 000190, loss = 0.001842
grad_step = 000191, loss = 0.001819
grad_step = 000192, loss = 0.001830
grad_step = 000193, loss = 0.001878
grad_step = 000194, loss = 0.001861
grad_step = 000195, loss = 0.001827
grad_step = 000196, loss = 0.001787
grad_step = 000197, loss = 0.001789
grad_step = 000198, loss = 0.001817
grad_step = 000199, loss = 0.001825
grad_step = 000200, loss = 0.001823
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001792
grad_step = 000202, loss = 0.001778
grad_step = 000203, loss = 0.001773
grad_step = 000204, loss = 0.001773
grad_step = 000205, loss = 0.001777
grad_step = 000206, loss = 0.001778
grad_step = 000207, loss = 0.001784
grad_step = 000208, loss = 0.001778
grad_step = 000209, loss = 0.001769
grad_step = 000210, loss = 0.001759
grad_step = 000211, loss = 0.001752
grad_step = 000212, loss = 0.001749
grad_step = 000213, loss = 0.001750
grad_step = 000214, loss = 0.001760
grad_step = 000215, loss = 0.001765
grad_step = 000216, loss = 0.001775
grad_step = 000217, loss = 0.001779
grad_step = 000218, loss = 0.001772
grad_step = 000219, loss = 0.001754
grad_step = 000220, loss = 0.001736
grad_step = 000221, loss = 0.001731
grad_step = 000222, loss = 0.001738
grad_step = 000223, loss = 0.001760
grad_step = 000224, loss = 0.001779
grad_step = 000225, loss = 0.001817
grad_step = 000226, loss = 0.001791
grad_step = 000227, loss = 0.001777
grad_step = 000228, loss = 0.001766
grad_step = 000229, loss = 0.001807
grad_step = 000230, loss = 0.001851
grad_step = 000231, loss = 0.001789
grad_step = 000232, loss = 0.001722
grad_step = 000233, loss = 0.001722
grad_step = 000234, loss = 0.001759
grad_step = 000235, loss = 0.001770
grad_step = 000236, loss = 0.001721
grad_step = 000237, loss = 0.001711
grad_step = 000238, loss = 0.001743
grad_step = 000239, loss = 0.001736
grad_step = 000240, loss = 0.001713
grad_step = 000241, loss = 0.001695
grad_step = 000242, loss = 0.001706
grad_step = 000243, loss = 0.001724
grad_step = 000244, loss = 0.001713
grad_step = 000245, loss = 0.001696
grad_step = 000246, loss = 0.001688
grad_step = 000247, loss = 0.001696
grad_step = 000248, loss = 0.001706
grad_step = 000249, loss = 0.001703
grad_step = 000250, loss = 0.001693
grad_step = 000251, loss = 0.001691
grad_step = 000252, loss = 0.001704
grad_step = 000253, loss = 0.001739
grad_step = 000254, loss = 0.001775
grad_step = 000255, loss = 0.001848
grad_step = 000256, loss = 0.001865
grad_step = 000257, loss = 0.001875
grad_step = 000258, loss = 0.001797
grad_step = 000259, loss = 0.001727
grad_step = 000260, loss = 0.001727
grad_step = 000261, loss = 0.001723
grad_step = 000262, loss = 0.001732
grad_step = 000263, loss = 0.001748
grad_step = 000264, loss = 0.001709
grad_step = 000265, loss = 0.001688
grad_step = 000266, loss = 0.001695
grad_step = 000267, loss = 0.001715
grad_step = 000268, loss = 0.001694
grad_step = 000269, loss = 0.001660
grad_step = 000270, loss = 0.001691
grad_step = 000271, loss = 0.001710
grad_step = 000272, loss = 0.001664
grad_step = 000273, loss = 0.001658
grad_step = 000274, loss = 0.001688
grad_step = 000275, loss = 0.001673
grad_step = 000276, loss = 0.001650
grad_step = 000277, loss = 0.001652
grad_step = 000278, loss = 0.001664
grad_step = 000279, loss = 0.001664
grad_step = 000280, loss = 0.001643
grad_step = 000281, loss = 0.001641
grad_step = 000282, loss = 0.001653
grad_step = 000283, loss = 0.001651
grad_step = 000284, loss = 0.001638
grad_step = 000285, loss = 0.001630
grad_step = 000286, loss = 0.001635
grad_step = 000287, loss = 0.001640
grad_step = 000288, loss = 0.001631
grad_step = 000289, loss = 0.001624
grad_step = 000290, loss = 0.001626
grad_step = 000291, loss = 0.001627
grad_step = 000292, loss = 0.001623
grad_step = 000293, loss = 0.001616
grad_step = 000294, loss = 0.001615
grad_step = 000295, loss = 0.001616
grad_step = 000296, loss = 0.001616
grad_step = 000297, loss = 0.001612
grad_step = 000298, loss = 0.001607
grad_step = 000299, loss = 0.001605
grad_step = 000300, loss = 0.001605
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001606
grad_step = 000302, loss = 0.001607
grad_step = 000303, loss = 0.001607
grad_step = 000304, loss = 0.001613
grad_step = 000305, loss = 0.001623
grad_step = 000306, loss = 0.001652
grad_step = 000307, loss = 0.001686
grad_step = 000308, loss = 0.001752
grad_step = 000309, loss = 0.001728
grad_step = 000310, loss = 0.001686
grad_step = 000311, loss = 0.001597
grad_step = 000312, loss = 0.001596
grad_step = 000313, loss = 0.001653
grad_step = 000314, loss = 0.001643
grad_step = 000315, loss = 0.001599
grad_step = 000316, loss = 0.001575
grad_step = 000317, loss = 0.001601
grad_step = 000318, loss = 0.001627
grad_step = 000319, loss = 0.001600
grad_step = 000320, loss = 0.001570
grad_step = 000321, loss = 0.001563
grad_step = 000322, loss = 0.001580
grad_step = 000323, loss = 0.001595
grad_step = 000324, loss = 0.001583
grad_step = 000325, loss = 0.001563
grad_step = 000326, loss = 0.001551
grad_step = 000327, loss = 0.001556
grad_step = 000328, loss = 0.001566
grad_step = 000329, loss = 0.001563
grad_step = 000330, loss = 0.001552
grad_step = 000331, loss = 0.001540
grad_step = 000332, loss = 0.001537
grad_step = 000333, loss = 0.001541
grad_step = 000334, loss = 0.001544
grad_step = 000335, loss = 0.001544
grad_step = 000336, loss = 0.001538
grad_step = 000337, loss = 0.001531
grad_step = 000338, loss = 0.001525
grad_step = 000339, loss = 0.001524
grad_step = 000340, loss = 0.001527
grad_step = 000341, loss = 0.001537
grad_step = 000342, loss = 0.001551
grad_step = 000343, loss = 0.001583
grad_step = 000344, loss = 0.001609
grad_step = 000345, loss = 0.001674
grad_step = 000346, loss = 0.001642
grad_step = 000347, loss = 0.001646
grad_step = 000348, loss = 0.001583
grad_step = 000349, loss = 0.001580
grad_step = 000350, loss = 0.001573
grad_step = 000351, loss = 0.001531
grad_step = 000352, loss = 0.001503
grad_step = 000353, loss = 0.001524
grad_step = 000354, loss = 0.001564
grad_step = 000355, loss = 0.001552
grad_step = 000356, loss = 0.001507
grad_step = 000357, loss = 0.001481
grad_step = 000358, loss = 0.001492
grad_step = 000359, loss = 0.001516
grad_step = 000360, loss = 0.001524
grad_step = 000361, loss = 0.001505
grad_step = 000362, loss = 0.001480
grad_step = 000363, loss = 0.001468
grad_step = 000364, loss = 0.001476
grad_step = 000365, loss = 0.001490
grad_step = 000366, loss = 0.001492
grad_step = 000367, loss = 0.001483
grad_step = 000368, loss = 0.001473
grad_step = 000369, loss = 0.001483
grad_step = 000370, loss = 0.001510
grad_step = 000371, loss = 0.001574
grad_step = 000372, loss = 0.001588
grad_step = 000373, loss = 0.001648
grad_step = 000374, loss = 0.001585
grad_step = 000375, loss = 0.001574
grad_step = 000376, loss = 0.001567
grad_step = 000377, loss = 0.001510
grad_step = 000378, loss = 0.001458
grad_step = 000379, loss = 0.001487
grad_step = 000380, loss = 0.001540
grad_step = 000381, loss = 0.001515
grad_step = 000382, loss = 0.001454
grad_step = 000383, loss = 0.001444
grad_step = 000384, loss = 0.001482
grad_step = 000385, loss = 0.001497
grad_step = 000386, loss = 0.001466
grad_step = 000387, loss = 0.001437
grad_step = 000388, loss = 0.001452
grad_step = 000389, loss = 0.001470
grad_step = 000390, loss = 0.001466
grad_step = 000391, loss = 0.001439
grad_step = 000392, loss = 0.001443
grad_step = 000393, loss = 0.001466
grad_step = 000394, loss = 0.001480
grad_step = 000395, loss = 0.001480
grad_step = 000396, loss = 0.001506
grad_step = 000397, loss = 0.001525
grad_step = 000398, loss = 0.001548
grad_step = 000399, loss = 0.001483
grad_step = 000400, loss = 0.001433
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001413
grad_step = 000402, loss = 0.001433
grad_step = 000403, loss = 0.001465
grad_step = 000404, loss = 0.001458
grad_step = 000405, loss = 0.001439
grad_step = 000406, loss = 0.001412
grad_step = 000407, loss = 0.001408
grad_step = 000408, loss = 0.001424
grad_step = 000409, loss = 0.001431
grad_step = 000410, loss = 0.001425
grad_step = 000411, loss = 0.001406
grad_step = 000412, loss = 0.001395
grad_step = 000413, loss = 0.001398
grad_step = 000414, loss = 0.001407
grad_step = 000415, loss = 0.001417
grad_step = 000416, loss = 0.001415
grad_step = 000417, loss = 0.001410
grad_step = 000418, loss = 0.001401
grad_step = 000419, loss = 0.001394
grad_step = 000420, loss = 0.001388
grad_step = 000421, loss = 0.001382
grad_step = 000422, loss = 0.001379
grad_step = 000423, loss = 0.001379
grad_step = 000424, loss = 0.001381
grad_step = 000425, loss = 0.001384
grad_step = 000426, loss = 0.001387
grad_step = 000427, loss = 0.001387
grad_step = 000428, loss = 0.001391
grad_step = 000429, loss = 0.001392
grad_step = 000430, loss = 0.001396
grad_step = 000431, loss = 0.001395
grad_step = 000432, loss = 0.001396
grad_step = 000433, loss = 0.001390
grad_step = 000434, loss = 0.001386
grad_step = 000435, loss = 0.001377
grad_step = 000436, loss = 0.001371
grad_step = 000437, loss = 0.001364
grad_step = 000438, loss = 0.001359
grad_step = 000439, loss = 0.001355
grad_step = 000440, loss = 0.001352
grad_step = 000441, loss = 0.001350
grad_step = 000442, loss = 0.001349
grad_step = 000443, loss = 0.001348
grad_step = 000444, loss = 0.001347
grad_step = 000445, loss = 0.001347
grad_step = 000446, loss = 0.001349
grad_step = 000447, loss = 0.001355
grad_step = 000448, loss = 0.001368
grad_step = 000449, loss = 0.001405
grad_step = 000450, loss = 0.001464
grad_step = 000451, loss = 0.001605
grad_step = 000452, loss = 0.001633
grad_step = 000453, loss = 0.001673
grad_step = 000454, loss = 0.001427
grad_step = 000455, loss = 0.001338
grad_step = 000456, loss = 0.001433
grad_step = 000457, loss = 0.001454
grad_step = 000458, loss = 0.001387
grad_step = 000459, loss = 0.001338
grad_step = 000460, loss = 0.001403
grad_step = 000461, loss = 0.001433
grad_step = 000462, loss = 0.001341
grad_step = 000463, loss = 0.001348
grad_step = 000464, loss = 0.001414
grad_step = 000465, loss = 0.001370
grad_step = 000466, loss = 0.001328
grad_step = 000467, loss = 0.001339
grad_step = 000468, loss = 0.001358
grad_step = 000469, loss = 0.001350
grad_step = 000470, loss = 0.001333
grad_step = 000471, loss = 0.001340
grad_step = 000472, loss = 0.001344
grad_step = 000473, loss = 0.001323
grad_step = 000474, loss = 0.001309
grad_step = 000475, loss = 0.001319
grad_step = 000476, loss = 0.001332
grad_step = 000477, loss = 0.001328
grad_step = 000478, loss = 0.001311
grad_step = 000479, loss = 0.001302
grad_step = 000480, loss = 0.001307
grad_step = 000481, loss = 0.001310
grad_step = 000482, loss = 0.001305
grad_step = 000483, loss = 0.001296
grad_step = 000484, loss = 0.001293
grad_step = 000485, loss = 0.001298
grad_step = 000486, loss = 0.001300
grad_step = 000487, loss = 0.001297
grad_step = 000488, loss = 0.001290
grad_step = 000489, loss = 0.001287
grad_step = 000490, loss = 0.001289
grad_step = 000491, loss = 0.001289
grad_step = 000492, loss = 0.001285
grad_step = 000493, loss = 0.001280
grad_step = 000494, loss = 0.001278
grad_step = 000495, loss = 0.001278
grad_step = 000496, loss = 0.001278
grad_step = 000497, loss = 0.001276
grad_step = 000498, loss = 0.001272
grad_step = 000499, loss = 0.001271
grad_step = 000500, loss = 0.001271
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001271
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

  date_run                              2020-05-14 20:15:44.396133
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.228506
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 20:15:44.569038
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.126517
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 20:15:44.579736
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.13979
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 20:15:44.585215
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.922466
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
0   2020-05-14 20:15:18.479989  ...    mean_absolute_error
1   2020-05-14 20:15:18.483621  ...     mean_squared_error
2   2020-05-14 20:15:18.486839  ...  median_absolute_error
3   2020-05-14 20:15:18.489640  ...               r2_score
4   2020-05-14 20:15:26.973873  ...    mean_absolute_error
5   2020-05-14 20:15:26.977552  ...     mean_squared_error
6   2020-05-14 20:15:26.980937  ...  median_absolute_error
7   2020-05-14 20:15:26.984099  ...               r2_score
8   2020-05-14 20:15:44.396133  ...    mean_absolute_error
9   2020-05-14 20:15:44.569038  ...     mean_squared_error
10  2020-05-14 20:15:44.579736  ...  median_absolute_error
11  2020-05-14 20:15:44.585215  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f79765a4fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 18%|█▊        | 1785856/9912422 [00:00<00:00, 12881142.43it/s]9920512it [00:00, 30288234.34it/s]                             
0it [00:00, ?it/s]32768it [00:00, 703873.04it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  1%|          | 16384/1648877 [00:00<00:31, 52246.40it/s]1654784it [00:00, 3830872.29it/s]                         
0it [00:00, ?it/s]8192it [00:00, 211561.72it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7928fa6e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f79285d90b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7928fa6e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f792852b0b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7925d684a8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7925d51c18> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7928fa6e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f79284ea6d8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7925d684a8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f79765afef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f8cf1a39208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=22992e8b979b87740c4321807d54eb6ab1be9cf47aa82eff91488c6383a6971c
  Stored in directory: /tmp/pip-ephem-wheel-cache-zswx3msd/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f8ce7ba4048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 2s
 2686976/17464789 [===>..........................] - ETA: 0s
11223040/17464789 [==================>...........] - ETA: 0s
16064512/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 20:17:10.435356: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 20:17:10.439313: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-14 20:17:10.439461: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5614113ef6c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 20:17:10.439473: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 8.3873 - accuracy: 0.4530
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8660 - accuracy: 0.4870 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.8966 - accuracy: 0.4850
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8851 - accuracy: 0.4857
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7770 - accuracy: 0.4928
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7356 - accuracy: 0.4955
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7148 - accuracy: 0.4969
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7433 - accuracy: 0.4950
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7160 - accuracy: 0.4968
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6988 - accuracy: 0.4979
11000/25000 [============>.................] - ETA: 3s - loss: 7.6792 - accuracy: 0.4992
12000/25000 [=============>................] - ETA: 3s - loss: 7.6896 - accuracy: 0.4985
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7138 - accuracy: 0.4969
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7225 - accuracy: 0.4964
15000/25000 [=================>............] - ETA: 2s - loss: 7.7014 - accuracy: 0.4977
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6973 - accuracy: 0.4980
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6910 - accuracy: 0.4984
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6871 - accuracy: 0.4987
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6844 - accuracy: 0.4988
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6751 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6695 - accuracy: 0.4998
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6792 - accuracy: 0.4992
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6713 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6909 - accuracy: 0.4984
25000/25000 [==============================] - 7s 276us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 20:17:23.770087
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 20:17:23.770087  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<29:35:40, 8.09kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<20:57:13, 11.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<14:43:16, 16.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<10:18:40, 23.2kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<7:11:54, 33.1kB/s].vector_cache/glove.6B.zip:   1%|          | 9.29M/862M [00:01<5:00:26, 47.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:01<3:29:00, 67.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.0M/862M [00:01<2:25:43, 96.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.6M/862M [00:02<1:41:33, 138kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 27.4M/862M [00:02<1:10:52, 196kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.5M/862M [00:02<49:27, 280kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 34.1M/862M [00:02<34:40, 398kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.7M/862M [00:02<24:22, 565kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.3M/862M [00:02<17:09, 799kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.8M/862M [00:02<12:08, 1.13MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.1M/862M [00:02<08:39, 1.58MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.3M/862M [00:02<06:09, 2.20MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.0M/862M [00:02<04:27, 3.04MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:03<03:52, 3.48MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:03<02:48, 4.78MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.4M/862M [00:06<06:18, 2.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.4M/862M [00:06<16:01, 836kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 58.6M/862M [00:06<14:20, 934kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.3M/862M [00:06<10:44, 1.25MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<07:52, 1.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:06<05:42, 2.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:08<25:37, 520kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 62.7M/862M [00:08<20:24, 653kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.9M/862M [00:08<16:19, 816kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.6M/862M [00:08<11:57, 1.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:08<08:36, 1.54MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:10<10:20, 1.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.8M/862M [00:10<09:48, 1.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.2M/862M [00:10<07:57, 1.66MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<05:57, 2.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:10<04:19, 3.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:12<33:21, 395kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:12<24:47, 532kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<17:36, 747kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:12<12:29, 1.05MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:14<33:01, 397kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 75.3M/862M [00:14<24:18, 540kB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.8M/862M [00:14<17:46, 737kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:14<12:33, 1.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:16<19:00, 686kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:16<14:39, 890kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<10:34, 1.23MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:18<10:27, 1.24MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:18<09:34, 1.36MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.6M/862M [00:18<08:04, 1.61MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<06:04, 2.13MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:20<06:34, 1.96MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:20<05:48, 2.23MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.1M/862M [00:20<04:50, 2.66MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<03:47, 3.39MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<03:21, 3.84MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<02:41, 4.77MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.4M/862M [00:20<02:29, 5.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.4M/862M [00:22<1:49:23, 117kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:22<1:25:49, 150kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:22<1:02:15, 206kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:22<44:07, 291kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<31:20, 409kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:22<22:21, 572kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:24<21:37, 591kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:24<19:24, 658kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.2M/862M [00:24<14:28, 882kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<10:41, 1.19MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<07:51, 1.62MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:24<05:50, 2.18MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.7M/862M [00:26<24:23, 521kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:26<21:17, 597kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:26<15:55, 797kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<11:37, 1.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<08:31, 1.48MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<06:21, 1.99MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:28<1:20:40, 157kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:28<1:00:42, 208kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:28<43:20, 291kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<30:42, 411kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<22:02, 572kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<15:50, 794kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<11:32, 1.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:30<5:05:28, 41.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:30<3:37:47, 57.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<2:33:13, 82.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<1:47:31, 117kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<1:15:46, 165kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<53:16, 235kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<45:03, 277kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:32<39:20, 318kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:32<29:11, 428kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<20:56, 596kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<15:12, 820kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<11:00, 1.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<08:10, 1.52MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<14:07, 880kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 116M/862M [00:34<13:44, 905kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<10:36, 1.17MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:48, 1.59MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:46, 2.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:36, 2.69MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<11:49, 1.05MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<12:21, 1.00MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<09:37, 1.28MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:10, 1.72MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:23, 2.29MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<08:12, 1.50MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:38<13:26, 914kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:38<11:00, 1.12MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<08:16, 1.48MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:15, 1.96MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<04:45, 2.58MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<03:42, 3.29MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<14:11, 862kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:40<13:31, 904kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:40<10:23, 1.18MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:36, 1.60MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<05:42, 2.14MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<04:19, 2.81MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<29:49, 408kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:42<24:14, 501kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:42<17:50, 681kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<12:49, 945kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<09:17, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<13:50, 873kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<17:14, 701kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:44<13:44, 879kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<10:01, 1.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:21, 1.64MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<05:26, 2.21MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<15:25, 779kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<13:56, 862kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:46<10:32, 1.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:40, 1.56MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<08:27, 1.41MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<11:21, 1.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<09:02, 1.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:42, 1.78MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<04:59, 2.38MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<03:50, 3.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<12:59, 915kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<14:25, 823kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:50<11:26, 1.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<08:23, 1.41MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<06:04, 1.95MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<13:49, 854kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<14:32, 812kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:52<11:11, 1.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<08:20, 1.41MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:03, 1.94MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:32, 2.59MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<25:52, 454kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<22:58, 511kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<17:06, 686kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<12:33, 933kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<09:02, 1.29MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<09:35, 1.22MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<10:50, 1.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<08:30, 1.37MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<06:26, 1.81MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:41, 2.48MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<08:01, 1.45MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<09:26, 1.23MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<07:34, 1.53MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:31, 2.09MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<04:03, 2.84MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<1:05:33, 176kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<49:26, 233kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<35:22, 326kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<25:00, 460kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<17:38, 650kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<25:57, 442kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<21:20, 537kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<15:44, 727kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<11:12, 1.02MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<11:23, 1.00MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<10:58, 1.04MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<08:19, 1.37MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:06, 1.86MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<04:25, 2.56MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<30:48, 368kB/s] .vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<25:03, 452kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<18:28, 613kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<13:08, 859kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<12:23, 908kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<12:10, 924kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<09:27, 1.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<06:49, 1.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<04:56, 2.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<5:11:04, 36.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<3:40:46, 50.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<2:35:08, 72.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<1:48:29, 103kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<1:19:44, 139kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<58:26, 190kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<41:25, 268kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<29:12, 379kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<23:02, 479kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<18:19, 603kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<13:21, 826kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<09:29, 1.16MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<11:02, 994kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<10:11, 1.08MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<07:45, 1.41MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<05:34, 1.96MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<11:41, 933kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<10:55, 999kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<08:17, 1.31MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<05:56, 1.83MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<10:32, 1.03MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<09:13, 1.18MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:53, 1.57MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<06:41, 1.61MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<07:08, 1.51MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:33, 1.93MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<04:00, 2.67MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<09:22, 1.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<08:12, 1.30MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<06:08, 1.74MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<06:12, 1.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<06:29, 1.64MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:05, 2.09MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<05:15, 2.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:52, 1.80MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:38, 2.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<03:22, 3.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<08:28, 1.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<08:02, 1.31MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<06:06, 1.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<04:24, 2.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<08:35, 1.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<08:11, 1.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<06:15, 1.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<06:02, 1.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<06:21, 1.63MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:51, 2.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:37, 2.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<05:18, 1.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<05:46, 1.78MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:33, 2.25MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<04:50, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<05:30, 1.85MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:22, 2.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<04:41, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<04:41, 2.16MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<03:39, 2.77MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<04:28, 2.25MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<04:55, 2.05MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<03:54, 2.58MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:43<04:24, 2.27MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:43<04:54, 2.04MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<03:53, 2.57MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<02:49, 3.53MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:45<36:05, 276kB/s] .vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<27:06, 367kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<19:19, 514kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<13:37, 726kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<15:11, 650kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<12:26, 794kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<09:08, 1.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<06:29, 1.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<16:28, 596kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<13:18, 737kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<09:45, 1.00MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<06:54, 1.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<23:43, 410kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<18:23, 529kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<13:17, 731kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<09:23, 1.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<13:12, 732kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<11:01, 877kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<08:09, 1.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<05:47, 1.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<13:41, 701kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:55<11:21, 845kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<08:19, 1.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<05:58, 1.60MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<07:36, 1.25MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:57<07:00, 1.36MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<05:19, 1.79MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<03:50, 2.47MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<09:57, 951kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<08:42, 1.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<06:31, 1.45MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:41, 2.01MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<07:20, 1.28MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:54, 1.36MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<05:12, 1.80MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<03:47, 2.47MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<06:45, 1.38MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<06:26, 1.45MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<04:56, 1.89MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<03:32, 2.62MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<15:01, 617kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<12:09, 761kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<08:55, 1.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:20, 1.45MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<11:34, 794kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<09:47, 939kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<07:16, 1.26MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<05:09, 1.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<24:05, 379kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<18:32, 492kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<13:22, 680kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<09:27, 958kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<11:17, 801kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<09:35, 944kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<07:07, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:07, 1.76MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<06:54, 1.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<06:27, 1.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:55, 1.82MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:34, 2.50MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<06:40, 1.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<06:18, 1.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:45, 1.87MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:31, 2.51MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<04:55, 1.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<05:05, 1.74MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<03:57, 2.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<02:52, 3.06MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<09:44, 901kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<08:26, 1.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<06:14, 1.40MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<04:30, 1.94MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<06:27, 1.35MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<06:05, 1.43MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:36, 1.88MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:25, 2.53MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:40, 1.85MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:25, 1.95MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:23, 2.54MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:02, 2.12MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:18, 1.99MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:23, 2.52MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<03:48, 2.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<04:11, 2.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:18, 2.57MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<03:44, 2.26MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<04:07, 2.04MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:15, 2.59MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<03:41, 2.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<04:04, 2.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:13, 2.59MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<03:39, 2.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:03, 2.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:11, 2.59MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<03:38, 2.26MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<03:57, 2.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:07, 2.63MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<03:34, 2.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<03:57, 2.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:04, 2.64MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<02:17, 3.53MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<04:33, 1.78MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<04:37, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:31, 2.29MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<02:37, 3.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<04:22, 1.83MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<04:28, 1.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:25, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<02:30, 3.17MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<05:15, 1.51MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<05:05, 1.56MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:51, 2.06MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<02:48, 2.81MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<05:39, 1.39MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<05:19, 1.48MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<04:01, 1.95MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<02:54, 2.69MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<08:37, 907kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<07:33, 1.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<05:56, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:20, 1.79MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<04:37, 1.67MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<04:37, 1.68MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<03:34, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:47, 2.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<04:01, 1.90MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:09, 2.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:29, 2.18MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<03:27, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<02:39, 2.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:20, 2.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<03:38, 2.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<02:49, 2.67MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<02:05, 3.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:30, 1.66MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<04:29, 1.67MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:25, 2.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<02:28, 3.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<12:23, 597kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:58<09:59, 741kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<07:18, 1.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<06:20, 1.16MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:41, 1.29MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<04:18, 1.70MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:14, 1.72MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<04:15, 1.70MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<03:18, 2.20MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:35, 2.00MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<04:55, 1.46MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<03:59, 1.80MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<02:59, 2.39MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:26, 2.07MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:41, 1.93MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<02:50, 2.50MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<02:07, 3.33MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:41, 1.91MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:50, 1.84MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<02:56, 2.39MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<02:09, 3.25MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<05:08, 1.36MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<05:27, 1.28MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<05:02, 1.38MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<04:15, 1.64MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<03:19, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<02:28, 2.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:38, 1.90MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:32, 1.96MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:21, 2.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<02:35, 2.66MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<01:58, 3.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<01:31, 4.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<19:24, 353kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<15:00, 456kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<10:50, 630kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<07:37, 891kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<10:53, 623kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<08:50, 766kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<06:30, 1.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:40, 1.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<05:05, 1.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:43, 1.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:31, 1.90MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<02:37, 2.54MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:19<03:41, 1.80MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:47, 1.75MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:03, 2.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:16, 2.91MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:21, 1.96MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:31, 1.87MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:44, 2.39MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<01:59, 3.28MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<11:38, 559kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<09:18, 699kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<06:44, 962kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<04:47, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<06:32, 983kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<05:43, 1.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:16, 1.50MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<03:02, 2.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<2:06:24, 50.4kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<1:29:31, 71.1kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<1:02:47, 101kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<44:41, 141kB/s]  .vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<32:23, 194kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<22:53, 274kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<16:00, 390kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<16:11, 385kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<12:23, 503kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<08:55, 696kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<07:15, 850kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<06:10, 999kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:34, 1.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:13, 1.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:02, 1.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:05, 1.96MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:10, 1.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:22, 1.38MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:35, 1.67MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:38, 2.27MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:27, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:28, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:40, 2.21MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<02:52, 2.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<03:00, 1.95MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:21, 2.48MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<02:38, 2.20MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<02:52, 2.02MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:13, 2.60MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<01:39, 3.49MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:45<03:28, 1.65MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:27, 1.66MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:40, 2.15MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<01:54, 2.97MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<38:08, 149kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<27:42, 205kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<19:35, 289kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<14:33, 385kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<11:10, 501kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<08:03, 694kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<05:39, 981kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<14:22, 385kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<11:01, 502kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<07:56, 696kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<06:26, 850kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<05:29, 996kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<04:04, 1.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<03:44, 1.44MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<03:35, 1.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:43, 1.98MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:01, 2.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<02:52, 1.85MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<02:57, 1.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:18, 2.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<02:29, 2.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<02:41, 1.96MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:06, 2.49MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<01:31, 3.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<05:40, 917kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<04:50, 1.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<03:35, 1.44MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:34, 1.99MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<03:56, 1.30MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<03:39, 1.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:44, 1.86MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:00, 2.53MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<03:12, 1.58MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<03:07, 1.61MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:23, 2.11MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<01:43, 2.91MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<04:53, 1.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<04:16, 1.16MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:12, 1.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:04, 1.60MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<03:00, 1.63MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:19, 2.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:26, 1.99MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:11<02:31, 1.92MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<01:57, 2.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<01:25, 3.38MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<04:12, 1.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<03:48, 1.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<02:51, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:47, 1.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:47, 1.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<02:06, 2.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<01:32, 3.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:24, 1.36MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<03:10, 1.46MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:25, 1.91MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:27, 1.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:32, 1.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<01:58, 2.31MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:08, 2.11MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:18, 1.96MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<01:46, 2.52MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<01:17, 3.44MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<04:29, 989kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:54, 1.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:53, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:05, 2.11MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:15, 1.34MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:03, 1.43MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:19, 1.88MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:39, 2.60MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<07:00, 614kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<05:38, 763kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<04:07, 1.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:53, 1.46MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<32:57, 128kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<23:48, 178kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<16:47, 251kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<12:19, 338kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<09:21, 445kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<06:42, 618kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<04:41, 874kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<10:07, 405kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<07:46, 526kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<05:36, 727kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<04:34, 882kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<03:54, 1.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:54, 1.38MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:41, 1.47MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:33, 1.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:57, 2.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:01, 1.93MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:06, 1.85MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:38, 2.37MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<01:47, 2.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<01:55, 1.99MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:29, 2.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<01:04, 3.51MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<09:59, 376kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<07:37, 492kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<05:27, 684kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<03:48, 967kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<19:24, 190kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<14:13, 259kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<10:03, 364kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<07:01, 517kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<06:40, 541kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<05:17, 682kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<03:50, 935kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<03:15, 1.09MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:54, 1.22MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<02:11, 1.61MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<02:06, 1.65MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<02:05, 1.67MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:36, 2.16MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<01:41, 2.01MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<01:46, 1.92MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:23, 2.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<01:30, 2.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<02:14, 1.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:51, 1.79MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:21, 2.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<01:50, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:50, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:24, 2.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:01, 3.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<02:04, 1.54MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<02:01, 1.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:32, 2.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:06, 2.83MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<02:22, 1.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<02:08, 1.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:40, 1.86MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:13, 2.53MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:02<01:40, 1.82MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<01:42, 1.78MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:18, 2.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<00:58, 3.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:36, 1.86MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:38, 1.81MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:16, 2.32MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<00:55, 3.17MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<02:33, 1.14MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<02:13, 1.31MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:43, 1.68MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:15, 2.29MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<01:38, 1.74MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<01:38, 1.73MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:15, 2.26MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<00:54, 3.09MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<02:19, 1.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:56, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:25, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:01, 2.64MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<02:07, 1.27MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<02:23, 1.13MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:54, 1.42MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:22, 1.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:41, 1.57MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:38, 1.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:15, 2.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<01:18, 1.97MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<01:21, 1.89MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:02, 2.46MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<00:45, 3.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<01:59, 1.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<01:50, 1.36MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:23, 1.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<01:22, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<01:22, 1.77MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:02, 2.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<00:45, 3.16MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<02:09, 1.10MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<01:54, 1.23MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:25, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:00, 2.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<02:08, 1.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<01:53, 1.21MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<01:25, 1.61MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:21, 1.65MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:19, 1.68MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:01, 2.17MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:04, 2.02MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:28<01:07, 1.93MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<00:52, 2.46MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:57, 2.19MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:30<01:01, 2.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<00:48, 2.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:53, 2.26MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<00:59, 2.05MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<00:46, 2.59MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:51, 2.26MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:56, 2.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<00:44, 2.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:49, 2.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:54, 2.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<00:42, 2.62MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:47, 2.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:52, 2.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<00:41, 2.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:46, 2.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:50, 2.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:38, 2.68MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:27, 3.65MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:49, 919kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:33, 1.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:08, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:48, 2.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<01:29, 1.08MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:19, 1.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:58, 1.63MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:41, 2.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:25, 1.08MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:15, 1.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:55, 1.64MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:38, 2.28MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<02:28, 595kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:59, 741kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:26, 1.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:12, 1.16MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:06, 1.27MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:49, 1.68MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:47, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:47, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:35, 2.20MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:37, 2.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:39, 1.94MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:30, 2.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:32, 2.20MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:35, 2.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:27, 2.59MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:29, 2.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:32, 2.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:25, 2.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:17, 3.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<21:07, 50.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<14:54, 70.9kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<10:19, 101kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<06:59, 144kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<05:26, 183kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<03:57, 250kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<02:46, 351kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:59, 462kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:33, 591kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:06, 814kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:52, 969kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:45, 1.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:33, 1.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:30, 1.56MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:29, 1.60MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:21, 2.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:15, 2.81MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:09<00:23, 1.85MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:09<00:23, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:18, 2.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:18, 2.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:11<00:19, 1.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:14, 2.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:15, 2.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:16, 2.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:12, 2.61MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:13, 2.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:14, 2.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:11, 2.67MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:07, 3.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<01:27, 303kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<01:04, 407kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:47, 550kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:31, 776kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:26, 852kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:22, 1.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:15, 1.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:09, 1.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:25, 721kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:20, 870kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:14, 1.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:10, 1.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:09, 1.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:07, 1.84MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:05, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:03, 2.29MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:02, 2.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:02, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:01, 2.53MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 3.48MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:05, 302kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:03, 401kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:01, 561kB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 791/400000 [00:00<00:50, 7909.50it/s]  0%|          | 1628/400000 [00:00<00:49, 8041.44it/s]  1%|          | 2458/400000 [00:00<00:48, 8116.10it/s]  1%|          | 3229/400000 [00:00<00:49, 7988.36it/s]  1%|          | 4012/400000 [00:00<00:49, 7938.09it/s]  1%|          | 4823/400000 [00:00<00:49, 7988.78it/s]  1%|▏         | 5659/400000 [00:00<00:48, 8094.04it/s]  2%|▏         | 6516/400000 [00:00<00:47, 8230.35it/s]  2%|▏         | 7343/400000 [00:00<00:47, 8239.39it/s]  2%|▏         | 8168/400000 [00:01<00:47, 8241.67it/s]  2%|▏         | 8990/400000 [00:01<00:47, 8234.52it/s]  2%|▏         | 9824/400000 [00:01<00:47, 8263.19it/s]  3%|▎         | 10662/400000 [00:01<00:46, 8296.04it/s]  3%|▎         | 11491/400000 [00:01<00:46, 8291.82it/s]  3%|▎         | 12315/400000 [00:01<00:48, 8009.01it/s]  3%|▎         | 13134/400000 [00:01<00:47, 8061.60it/s]  3%|▎         | 13940/400000 [00:01<00:48, 8025.78it/s]  4%|▎         | 14770/400000 [00:01<00:47, 8104.45it/s]  4%|▍         | 15581/400000 [00:01<00:47, 8052.04it/s]  4%|▍         | 16406/400000 [00:02<00:47, 8108.08it/s]  4%|▍         | 17242/400000 [00:02<00:46, 8180.90it/s]  5%|▍         | 18071/400000 [00:02<00:46, 8211.14it/s]  5%|▍         | 18903/400000 [00:02<00:46, 8241.16it/s]  5%|▍         | 19738/400000 [00:02<00:45, 8272.89it/s]  5%|▌         | 20567/400000 [00:02<00:45, 8276.09it/s]  5%|▌         | 21405/400000 [00:02<00:45, 8306.14it/s]  6%|▌         | 22245/400000 [00:02<00:45, 8332.52it/s]  6%|▌         | 23095/400000 [00:02<00:44, 8380.24it/s]  6%|▌         | 23937/400000 [00:02<00:44, 8391.49it/s]  6%|▌         | 24777/400000 [00:03<00:44, 8350.20it/s]  6%|▋         | 25618/400000 [00:03<00:44, 8367.90it/s]  7%|▋         | 26455/400000 [00:03<00:45, 8264.12it/s]  7%|▋         | 27295/400000 [00:03<00:44, 8303.30it/s]  7%|▋         | 28146/400000 [00:03<00:44, 8362.29it/s]  7%|▋         | 28983/400000 [00:03<00:44, 8355.77it/s]  7%|▋         | 29819/400000 [00:03<00:44, 8352.67it/s]  8%|▊         | 30667/400000 [00:03<00:44, 8390.29it/s]  8%|▊         | 31507/400000 [00:03<00:44, 8325.59it/s]  8%|▊         | 32353/400000 [00:03<00:43, 8363.63it/s]  8%|▊         | 33215/400000 [00:04<00:43, 8437.63it/s]  9%|▊         | 34082/400000 [00:04<00:43, 8505.55it/s]  9%|▊         | 34933/400000 [00:04<00:42, 8498.64it/s]  9%|▉         | 35784/400000 [00:04<00:43, 8428.04it/s]  9%|▉         | 36648/400000 [00:04<00:42, 8490.50it/s]  9%|▉         | 37498/400000 [00:04<00:42, 8463.73it/s] 10%|▉         | 38376/400000 [00:04<00:42, 8556.15it/s] 10%|▉         | 39242/400000 [00:04<00:42, 8585.01it/s] 10%|█         | 40108/400000 [00:04<00:41, 8604.87it/s] 10%|█         | 40969/400000 [00:04<00:41, 8557.79it/s] 10%|█         | 41825/400000 [00:05<00:42, 8454.99it/s] 11%|█         | 42674/400000 [00:05<00:42, 8464.30it/s] 11%|█         | 43525/400000 [00:05<00:42, 8475.30it/s] 11%|█         | 44373/400000 [00:05<00:42, 8452.66it/s] 11%|█▏        | 45219/400000 [00:05<00:42, 8384.56it/s] 12%|█▏        | 46058/400000 [00:05<00:42, 8337.48it/s] 12%|█▏        | 46892/400000 [00:05<00:42, 8280.93it/s] 12%|█▏        | 47752/400000 [00:05<00:42, 8372.97it/s] 12%|█▏        | 48597/400000 [00:05<00:41, 8394.70it/s] 12%|█▏        | 49448/400000 [00:05<00:41, 8428.76it/s] 13%|█▎        | 50292/400000 [00:06<00:41, 8408.46it/s] 13%|█▎        | 51155/400000 [00:06<00:41, 8472.90it/s] 13%|█▎        | 52003/400000 [00:06<00:41, 8470.55it/s] 13%|█▎        | 52851/400000 [00:06<00:41, 8435.27it/s] 13%|█▎        | 53724/400000 [00:06<00:40, 8520.95it/s] 14%|█▎        | 54577/400000 [00:06<00:40, 8506.55it/s] 14%|█▍        | 55442/400000 [00:06<00:40, 8546.15it/s] 14%|█▍        | 56297/400000 [00:06<00:40, 8451.01it/s] 14%|█▍        | 57143/400000 [00:06<00:41, 8335.05it/s] 15%|█▍        | 58014/400000 [00:06<00:40, 8443.82it/s] 15%|█▍        | 58874/400000 [00:07<00:40, 8489.84it/s] 15%|█▍        | 59733/400000 [00:07<00:39, 8519.12it/s] 15%|█▌        | 60599/400000 [00:07<00:39, 8560.58it/s] 15%|█▌        | 61477/400000 [00:07<00:39, 8624.49it/s] 16%|█▌        | 62349/400000 [00:07<00:39, 8651.41it/s] 16%|█▌        | 63217/400000 [00:07<00:38, 8658.80it/s] 16%|█▌        | 64103/400000 [00:07<00:38, 8717.15it/s] 16%|█▌        | 64975/400000 [00:07<00:38, 8631.42it/s] 16%|█▋        | 65864/400000 [00:07<00:38, 8705.40it/s] 17%|█▋        | 66749/400000 [00:07<00:38, 8746.86it/s] 17%|█▋        | 67625/400000 [00:08<00:38, 8691.21it/s] 17%|█▋        | 68495/400000 [00:08<00:38, 8692.68it/s] 17%|█▋        | 69365/400000 [00:08<00:38, 8687.64it/s] 18%|█▊        | 70255/400000 [00:08<00:37, 8747.43it/s] 18%|█▊        | 71130/400000 [00:08<00:37, 8740.25it/s] 18%|█▊        | 72005/400000 [00:08<00:37, 8671.17it/s] 18%|█▊        | 72875/400000 [00:08<00:37, 8679.55it/s] 18%|█▊        | 73744/400000 [00:08<00:37, 8676.65it/s] 19%|█▊        | 74614/400000 [00:08<00:37, 8681.96it/s] 19%|█▉        | 75483/400000 [00:08<00:37, 8678.12it/s] 19%|█▉        | 76351/400000 [00:09<00:37, 8608.83it/s] 19%|█▉        | 77214/400000 [00:09<00:37, 8614.78it/s] 20%|█▉        | 78076/400000 [00:09<00:37, 8496.76it/s] 20%|█▉        | 78948/400000 [00:09<00:37, 8561.49it/s] 20%|█▉        | 79818/400000 [00:09<00:37, 8601.13it/s] 20%|██        | 80691/400000 [00:09<00:36, 8638.83it/s] 20%|██        | 81575/400000 [00:09<00:36, 8696.94it/s] 21%|██        | 82445/400000 [00:09<00:36, 8686.95it/s] 21%|██        | 83317/400000 [00:09<00:36, 8694.90it/s] 21%|██        | 84187/400000 [00:09<00:36, 8673.14it/s] 21%|██▏       | 85055/400000 [00:10<00:36, 8646.53it/s] 21%|██▏       | 85924/400000 [00:10<00:36, 8657.44it/s] 22%|██▏       | 86796/400000 [00:10<00:36, 8676.04it/s] 22%|██▏       | 87675/400000 [00:10<00:35, 8709.48it/s] 22%|██▏       | 88547/400000 [00:10<00:35, 8703.89it/s] 22%|██▏       | 89422/400000 [00:10<00:35, 8716.33it/s] 23%|██▎       | 90294/400000 [00:10<00:35, 8711.35it/s] 23%|██▎       | 91166/400000 [00:10<00:35, 8709.43it/s] 23%|██▎       | 92041/400000 [00:10<00:35, 8720.22it/s] 23%|██▎       | 92914/400000 [00:10<00:35, 8689.83it/s] 23%|██▎       | 93786/400000 [00:11<00:35, 8697.85it/s] 24%|██▎       | 94675/400000 [00:11<00:34, 8752.01it/s] 24%|██▍       | 95551/400000 [00:11<00:34, 8732.45it/s] 24%|██▍       | 96425/400000 [00:11<00:34, 8726.33it/s] 24%|██▍       | 97298/400000 [00:11<00:34, 8716.70it/s] 25%|██▍       | 98170/400000 [00:11<00:34, 8697.97it/s] 25%|██▍       | 99040/400000 [00:11<00:34, 8663.40it/s] 25%|██▍       | 99907/400000 [00:11<00:34, 8587.21it/s] 25%|██▌       | 100769/400000 [00:11<00:34, 8595.89it/s] 25%|██▌       | 101629/400000 [00:11<00:34, 8571.67it/s] 26%|██▌       | 102487/400000 [00:12<00:35, 8477.35it/s] 26%|██▌       | 103351/400000 [00:12<00:34, 8523.41it/s] 26%|██▌       | 104252/400000 [00:12<00:34, 8661.66it/s] 26%|██▋       | 105126/400000 [00:12<00:33, 8681.61it/s] 27%|██▋       | 106009/400000 [00:12<00:33, 8724.76it/s] 27%|██▋       | 106882/400000 [00:12<00:34, 8493.51it/s] 27%|██▋       | 107733/400000 [00:12<00:34, 8486.95it/s] 27%|██▋       | 108599/400000 [00:12<00:34, 8537.33it/s] 27%|██▋       | 109473/400000 [00:12<00:33, 8595.54it/s] 28%|██▊       | 110354/400000 [00:13<00:33, 8656.45it/s] 28%|██▊       | 111229/400000 [00:13<00:33, 8682.54it/s] 28%|██▊       | 112098/400000 [00:13<00:33, 8631.50it/s] 28%|██▊       | 112972/400000 [00:13<00:33, 8663.30it/s] 28%|██▊       | 113842/400000 [00:13<00:32, 8672.02it/s] 29%|██▊       | 114730/400000 [00:13<00:32, 8730.61it/s] 29%|██▉       | 115604/400000 [00:13<00:32, 8691.17it/s] 29%|██▉       | 116474/400000 [00:13<00:33, 8483.45it/s] 29%|██▉       | 117355/400000 [00:13<00:32, 8576.64it/s] 30%|██▉       | 118233/400000 [00:13<00:32, 8634.33it/s] 30%|██▉       | 119104/400000 [00:14<00:32, 8654.26it/s] 30%|██▉       | 119971/400000 [00:14<00:32, 8637.30it/s] 30%|███       | 120836/400000 [00:14<00:32, 8622.59it/s] 30%|███       | 121699/400000 [00:14<00:32, 8503.36it/s] 31%|███       | 122550/400000 [00:14<00:32, 8472.94it/s] 31%|███       | 123422/400000 [00:14<00:32, 8544.52it/s] 31%|███       | 124287/400000 [00:14<00:32, 8575.19it/s] 31%|███▏      | 125145/400000 [00:14<00:32, 8570.09it/s] 32%|███▏      | 126014/400000 [00:14<00:31, 8603.56it/s] 32%|███▏      | 126887/400000 [00:14<00:31, 8639.22it/s] 32%|███▏      | 127758/400000 [00:15<00:31, 8657.56it/s] 32%|███▏      | 128627/400000 [00:15<00:31, 8666.67it/s] 32%|███▏      | 129494/400000 [00:15<00:31, 8571.35it/s] 33%|███▎      | 130364/400000 [00:15<00:31, 8608.78it/s] 33%|███▎      | 131230/400000 [00:15<00:31, 8622.52it/s] 33%|███▎      | 132110/400000 [00:15<00:30, 8672.93it/s] 33%|███▎      | 132988/400000 [00:15<00:30, 8703.45it/s] 33%|███▎      | 133859/400000 [00:15<00:30, 8701.04it/s] 34%|███▎      | 134730/400000 [00:15<00:30, 8695.78it/s] 34%|███▍      | 135604/400000 [00:15<00:30, 8706.48it/s] 34%|███▍      | 136475/400000 [00:16<00:30, 8638.47it/s] 34%|███▍      | 137340/400000 [00:16<00:30, 8606.59it/s] 35%|███▍      | 138201/400000 [00:16<00:31, 8351.87it/s] 35%|███▍      | 139089/400000 [00:16<00:30, 8503.54it/s] 35%|███▍      | 139972/400000 [00:16<00:30, 8597.52it/s] 35%|███▌      | 140861/400000 [00:16<00:29, 8682.97it/s] 35%|███▌      | 141731/400000 [00:16<00:30, 8566.28it/s] 36%|███▌      | 142602/400000 [00:16<00:29, 8606.81it/s] 36%|███▌      | 143483/400000 [00:16<00:29, 8664.09it/s] 36%|███▌      | 144351/400000 [00:16<00:29, 8658.84it/s] 36%|███▋      | 145218/400000 [00:17<00:29, 8612.30it/s] 37%|███▋      | 146080/400000 [00:17<00:29, 8519.06it/s] 37%|███▋      | 146933/400000 [00:17<00:30, 8308.49it/s] 37%|███▋      | 147798/400000 [00:17<00:29, 8406.75it/s] 37%|███▋      | 148672/400000 [00:17<00:29, 8502.51it/s] 37%|███▋      | 149524/400000 [00:17<00:29, 8480.29it/s] 38%|███▊      | 150386/400000 [00:17<00:29, 8520.74it/s] 38%|███▊      | 151239/400000 [00:17<00:29, 8508.25it/s] 38%|███▊      | 152091/400000 [00:17<00:29, 8466.27it/s] 38%|███▊      | 152951/400000 [00:17<00:29, 8505.67it/s] 38%|███▊      | 153802/400000 [00:18<00:31, 7862.00it/s] 39%|███▊      | 154604/400000 [00:18<00:31, 7906.09it/s] 39%|███▉      | 155467/400000 [00:18<00:30, 8109.63it/s] 39%|███▉      | 156331/400000 [00:18<00:29, 8261.76it/s] 39%|███▉      | 157201/400000 [00:18<00:28, 8386.11it/s] 40%|███▉      | 158070/400000 [00:18<00:28, 8473.69it/s] 40%|███▉      | 158922/400000 [00:18<00:28, 8486.57it/s] 40%|███▉      | 159773/400000 [00:18<00:29, 8209.86it/s] 40%|████      | 160638/400000 [00:18<00:28, 8336.81it/s] 40%|████      | 161475/400000 [00:19<00:28, 8330.10it/s] 41%|████      | 162345/400000 [00:19<00:28, 8436.03it/s] 41%|████      | 163192/400000 [00:19<00:28, 8444.49it/s] 41%|████      | 164047/400000 [00:19<00:27, 8475.18it/s] 41%|████      | 164910/400000 [00:19<00:27, 8520.90it/s] 41%|████▏     | 165763/400000 [00:19<00:27, 8495.00it/s] 42%|████▏     | 166613/400000 [00:19<00:27, 8494.88it/s] 42%|████▏     | 167463/400000 [00:19<00:27, 8336.75it/s] 42%|████▏     | 168298/400000 [00:19<00:27, 8287.12it/s] 42%|████▏     | 169144/400000 [00:19<00:27, 8337.08it/s] 43%|████▎     | 170008/400000 [00:20<00:27, 8425.02it/s] 43%|████▎     | 170873/400000 [00:20<00:26, 8490.48it/s] 43%|████▎     | 171723/400000 [00:20<00:26, 8484.10it/s] 43%|████▎     | 172590/400000 [00:20<00:26, 8536.31it/s] 43%|████▎     | 173461/400000 [00:20<00:26, 8587.21it/s] 44%|████▎     | 174321/400000 [00:20<00:26, 8434.64it/s] 44%|████▍     | 175185/400000 [00:20<00:26, 8493.35it/s] 44%|████▍     | 176036/400000 [00:20<00:26, 8492.99it/s] 44%|████▍     | 176899/400000 [00:20<00:26, 8533.18it/s] 44%|████▍     | 177757/400000 [00:20<00:26, 8544.24it/s] 45%|████▍     | 178630/400000 [00:21<00:25, 8598.55it/s] 45%|████▍     | 179491/400000 [00:21<00:25, 8601.24it/s] 45%|████▌     | 180352/400000 [00:21<00:25, 8537.66it/s] 45%|████▌     | 181252/400000 [00:21<00:25, 8670.37it/s] 46%|████▌     | 182128/400000 [00:21<00:25, 8696.57it/s] 46%|████▌     | 183007/400000 [00:21<00:24, 8723.18it/s] 46%|████▌     | 183880/400000 [00:21<00:24, 8709.01it/s] 46%|████▌     | 184752/400000 [00:21<00:24, 8639.22it/s] 46%|████▋     | 185624/400000 [00:21<00:24, 8662.19it/s] 47%|████▋     | 186513/400000 [00:21<00:24, 8729.05it/s] 47%|████▋     | 187387/400000 [00:22<00:24, 8696.81it/s] 47%|████▋     | 188258/400000 [00:22<00:24, 8700.24it/s] 47%|████▋     | 189129/400000 [00:22<00:24, 8587.22it/s] 47%|████▋     | 189989/400000 [00:22<00:24, 8566.40it/s] 48%|████▊     | 190856/400000 [00:22<00:24, 8595.44it/s] 48%|████▊     | 191720/400000 [00:22<00:24, 8607.12it/s] 48%|████▊     | 192585/400000 [00:22<00:24, 8617.01it/s] 48%|████▊     | 193447/400000 [00:22<00:24, 8545.05it/s] 49%|████▊     | 194302/400000 [00:22<00:24, 8531.31it/s] 49%|████▉     | 195170/400000 [00:22<00:23, 8574.70it/s] 49%|████▉     | 196028/400000 [00:23<00:23, 8509.91it/s] 49%|████▉     | 196897/400000 [00:23<00:23, 8562.44it/s] 49%|████▉     | 197764/400000 [00:23<00:23, 8592.95it/s] 50%|████▉     | 198637/400000 [00:23<00:23, 8632.19it/s] 50%|████▉     | 199511/400000 [00:23<00:23, 8661.86it/s] 50%|█████     | 200383/400000 [00:23<00:23, 8676.68it/s] 50%|█████     | 201251/400000 [00:23<00:22, 8676.95it/s] 51%|█████     | 202119/400000 [00:23<00:23, 8415.27it/s] 51%|█████     | 202980/400000 [00:23<00:23, 8471.16it/s] 51%|█████     | 203841/400000 [00:23<00:23, 8510.97it/s] 51%|█████     | 204702/400000 [00:24<00:22, 8539.26it/s] 51%|█████▏    | 205568/400000 [00:24<00:22, 8574.48it/s] 52%|█████▏    | 206426/400000 [00:24<00:22, 8542.46it/s] 52%|█████▏    | 207293/400000 [00:24<00:22, 8579.06it/s] 52%|█████▏    | 208163/400000 [00:24<00:22, 8614.91it/s] 52%|█████▏    | 209037/400000 [00:24<00:22, 8649.37it/s] 52%|█████▏    | 209919/400000 [00:24<00:21, 8697.28it/s] 53%|█████▎    | 210789/400000 [00:24<00:22, 8588.25it/s] 53%|█████▎    | 211666/400000 [00:24<00:21, 8639.67it/s] 53%|█████▎    | 212541/400000 [00:24<00:21, 8670.70it/s] 53%|█████▎    | 213409/400000 [00:25<00:21, 8652.81it/s] 54%|█████▎    | 214279/400000 [00:25<00:21, 8665.39it/s] 54%|█████▍    | 215159/400000 [00:25<00:21, 8704.24it/s] 54%|█████▍    | 216031/400000 [00:25<00:21, 8706.72it/s] 54%|█████▍    | 216902/400000 [00:25<00:21, 8495.12it/s] 54%|█████▍    | 217786/400000 [00:25<00:21, 8593.72it/s] 55%|█████▍    | 218676/400000 [00:25<00:20, 8682.58it/s] 55%|█████▍    | 219546/400000 [00:25<00:20, 8625.87it/s] 55%|█████▌    | 220425/400000 [00:25<00:20, 8672.60it/s] 55%|█████▌    | 221293/400000 [00:25<00:20, 8667.22it/s] 56%|█████▌    | 222161/400000 [00:26<00:20, 8637.25it/s] 56%|█████▌    | 223031/400000 [00:26<00:20, 8655.42it/s] 56%|█████▌    | 223897/400000 [00:26<00:20, 8450.57it/s] 56%|█████▌    | 224766/400000 [00:26<00:20, 8519.93it/s] 56%|█████▋    | 225628/400000 [00:26<00:20, 8549.58it/s] 57%|█████▋    | 226510/400000 [00:26<00:20, 8628.94it/s] 57%|█████▋    | 227396/400000 [00:26<00:19, 8695.11it/s] 57%|█████▋    | 228280/400000 [00:26<00:19, 8737.92it/s] 57%|█████▋    | 229155/400000 [00:26<00:19, 8625.87it/s] 58%|█████▊    | 230019/400000 [00:26<00:19, 8607.87it/s] 58%|█████▊    | 230881/400000 [00:27<00:19, 8463.56it/s] 58%|█████▊    | 231738/400000 [00:27<00:19, 8493.13it/s] 58%|█████▊    | 232596/400000 [00:27<00:19, 8516.07it/s] 58%|█████▊    | 233461/400000 [00:27<00:19, 8553.61it/s] 59%|█████▊    | 234324/400000 [00:27<00:19, 8574.72it/s] 59%|█████▉    | 235182/400000 [00:27<00:19, 8405.29it/s] 59%|█████▉    | 236024/400000 [00:27<00:19, 8326.32it/s] 59%|█████▉    | 236870/400000 [00:27<00:19, 8365.82it/s] 59%|█████▉    | 237708/400000 [00:27<00:19, 8251.66it/s] 60%|█████▉    | 238585/400000 [00:28<00:19, 8398.66it/s] 60%|█████▉    | 239427/400000 [00:28<00:19, 8318.78it/s] 60%|██████    | 240302/400000 [00:28<00:18, 8441.99it/s] 60%|██████    | 241158/400000 [00:28<00:18, 8477.02it/s] 61%|██████    | 242025/400000 [00:28<00:18, 8532.23it/s] 61%|██████    | 242888/400000 [00:28<00:18, 8560.42it/s] 61%|██████    | 243757/400000 [00:28<00:18, 8597.70it/s] 61%|██████    | 244628/400000 [00:28<00:18, 8628.26it/s] 61%|██████▏   | 245492/400000 [00:28<00:18, 8529.75it/s] 62%|██████▏   | 246349/400000 [00:28<00:17, 8540.37it/s] 62%|██████▏   | 247205/400000 [00:29<00:17, 8545.41it/s] 62%|██████▏   | 248076/400000 [00:29<00:17, 8591.79it/s] 62%|██████▏   | 248949/400000 [00:29<00:17, 8631.72it/s] 62%|██████▏   | 249813/400000 [00:29<00:17, 8620.60it/s] 63%|██████▎   | 250677/400000 [00:29<00:17, 8624.45it/s] 63%|██████▎   | 251540/400000 [00:29<00:17, 8340.61it/s] 63%|██████▎   | 252377/400000 [00:29<00:17, 8233.40it/s] 63%|██████▎   | 253234/400000 [00:29<00:17, 8329.37it/s] 64%|██████▎   | 254101/400000 [00:29<00:17, 8426.80it/s] 64%|██████▎   | 254946/400000 [00:29<00:17, 8367.02it/s] 64%|██████▍   | 255809/400000 [00:30<00:17, 8442.19it/s] 64%|██████▍   | 256657/400000 [00:30<00:16, 8453.01it/s] 64%|██████▍   | 257505/400000 [00:30<00:16, 8450.52it/s] 65%|██████▍   | 258351/400000 [00:30<00:17, 8281.32it/s] 65%|██████▍   | 259183/400000 [00:30<00:16, 8291.49it/s] 65%|██████▌   | 260042/400000 [00:30<00:16, 8376.85it/s] 65%|██████▌   | 260917/400000 [00:30<00:16, 8483.97it/s] 65%|██████▌   | 261767/400000 [00:30<00:16, 8433.83it/s] 66%|██████▌   | 262613/400000 [00:30<00:16, 8441.46it/s] 66%|██████▌   | 263494/400000 [00:30<00:15, 8547.46it/s] 66%|██████▌   | 264371/400000 [00:31<00:15, 8612.78it/s] 66%|██████▋   | 265233/400000 [00:31<00:15, 8595.42it/s] 67%|██████▋   | 266113/400000 [00:31<00:15, 8653.86it/s] 67%|██████▋   | 266983/400000 [00:31<00:15, 8665.25it/s] 67%|██████▋   | 267871/400000 [00:31<00:15, 8726.04it/s] 67%|██████▋   | 268744/400000 [00:31<00:15, 8667.66it/s] 67%|██████▋   | 269612/400000 [00:31<00:15, 8610.12it/s] 68%|██████▊   | 270474/400000 [00:31<00:15, 8586.77it/s] 68%|██████▊   | 271343/400000 [00:31<00:14, 8615.46it/s] 68%|██████▊   | 272205/400000 [00:31<00:14, 8587.10it/s] 68%|██████▊   | 273064/400000 [00:32<00:14, 8575.88it/s] 68%|██████▊   | 273922/400000 [00:32<00:14, 8505.46it/s] 69%|██████▊   | 274783/400000 [00:32<00:14, 8534.18it/s] 69%|██████▉   | 275640/400000 [00:32<00:14, 8542.10it/s] 69%|██████▉   | 276514/400000 [00:32<00:14, 8598.35it/s] 69%|██████▉   | 277375/400000 [00:32<00:14, 8460.68it/s] 70%|██████▉   | 278240/400000 [00:32<00:14, 8516.41it/s] 70%|██████▉   | 279093/400000 [00:32<00:14, 8172.97it/s] 70%|██████▉   | 279914/400000 [00:32<00:14, 8087.63it/s] 70%|███████   | 280776/400000 [00:32<00:14, 8240.20it/s] 70%|███████   | 281635/400000 [00:33<00:14, 8339.99it/s] 71%|███████   | 282500/400000 [00:33<00:13, 8428.18it/s] 71%|███████   | 283373/400000 [00:33<00:13, 8514.38it/s] 71%|███████   | 284226/400000 [00:33<00:13, 8452.71it/s] 71%|███████▏  | 285080/400000 [00:33<00:13, 8476.72it/s] 71%|███████▏  | 285929/400000 [00:33<00:13, 8434.32it/s] 72%|███████▏  | 286774/400000 [00:33<00:13, 8437.19it/s] 72%|███████▏  | 287641/400000 [00:33<00:13, 8501.43it/s] 72%|███████▏  | 288496/400000 [00:33<00:13, 8515.38it/s] 72%|███████▏  | 289359/400000 [00:33<00:12, 8546.87it/s] 73%|███████▎  | 290227/400000 [00:34<00:12, 8583.80it/s] 73%|███████▎  | 291100/400000 [00:34<00:12, 8626.58it/s] 73%|███████▎  | 291963/400000 [00:34<00:12, 8623.99it/s] 73%|███████▎  | 292834/400000 [00:34<00:12, 8648.98it/s] 73%|███████▎  | 293700/400000 [00:34<00:12, 8630.03it/s] 74%|███████▎  | 294568/400000 [00:34<00:12, 8643.67it/s] 74%|███████▍  | 295433/400000 [00:34<00:12, 8645.40it/s] 74%|███████▍  | 296314/400000 [00:34<00:11, 8691.41it/s] 74%|███████▍  | 297193/400000 [00:34<00:11, 8719.46it/s] 75%|███████▍  | 298066/400000 [00:34<00:11, 8684.33it/s] 75%|███████▍  | 298946/400000 [00:35<00:11, 8717.94it/s] 75%|███████▍  | 299818/400000 [00:35<00:11, 8695.71it/s] 75%|███████▌  | 300688/400000 [00:35<00:11, 8684.44it/s] 75%|███████▌  | 301557/400000 [00:35<00:11, 8579.15it/s] 76%|███████▌  | 302416/400000 [00:35<00:11, 8564.66it/s] 76%|███████▌  | 303273/400000 [00:35<00:11, 8449.66it/s] 76%|███████▌  | 304159/400000 [00:35<00:11, 8567.29it/s] 76%|███████▋  | 305036/400000 [00:35<00:11, 8624.28it/s] 76%|███████▋  | 305900/400000 [00:35<00:10, 8606.39it/s] 77%|███████▋  | 306762/400000 [00:35<00:10, 8570.36it/s] 77%|███████▋  | 307629/400000 [00:36<00:10, 8597.34it/s] 77%|███████▋  | 308489/400000 [00:36<00:10, 8559.17it/s] 77%|███████▋  | 309346/400000 [00:36<00:10, 8552.76it/s] 78%|███████▊  | 310202/400000 [00:36<00:10, 8454.09it/s] 78%|███████▊  | 311072/400000 [00:36<00:10, 8525.49it/s] 78%|███████▊  | 311928/400000 [00:36<00:10, 8533.03it/s] 78%|███████▊  | 312802/400000 [00:36<00:10, 8592.19it/s] 78%|███████▊  | 313677/400000 [00:36<00:09, 8638.82it/s] 79%|███████▊  | 314542/400000 [00:36<00:10, 8526.67it/s] 79%|███████▉  | 315396/400000 [00:37<00:09, 8472.55it/s] 79%|███████▉  | 316244/400000 [00:37<00:09, 8440.97it/s] 79%|███████▉  | 317107/400000 [00:37<00:09, 8494.69it/s] 79%|███████▉  | 317993/400000 [00:37<00:09, 8600.54it/s] 80%|███████▉  | 318869/400000 [00:37<00:09, 8646.56it/s] 80%|███████▉  | 319758/400000 [00:37<00:09, 8716.26it/s] 80%|████████  | 320631/400000 [00:37<00:09, 8693.48it/s] 80%|████████  | 321501/400000 [00:37<00:09, 8668.37it/s] 81%|████████  | 322369/400000 [00:37<00:08, 8647.00it/s] 81%|████████  | 323234/400000 [00:37<00:08, 8643.22it/s] 81%|████████  | 324099/400000 [00:38<00:08, 8610.00it/s] 81%|████████  | 324968/400000 [00:38<00:08, 8632.70it/s] 81%|████████▏ | 325841/400000 [00:38<00:08, 8660.61it/s] 82%|████████▏ | 326708/400000 [00:38<00:08, 8656.56it/s] 82%|████████▏ | 327574/400000 [00:38<00:08, 8527.92it/s] 82%|████████▏ | 328428/400000 [00:38<00:08, 8437.10it/s] 82%|████████▏ | 329273/400000 [00:38<00:08, 8366.12it/s] 83%|████████▎ | 330111/400000 [00:38<00:08, 8328.25it/s] 83%|████████▎ | 330951/400000 [00:38<00:08, 8347.89it/s] 83%|████████▎ | 331787/400000 [00:38<00:08, 8253.10it/s] 83%|████████▎ | 332613/400000 [00:39<00:08, 8228.60it/s] 83%|████████▎ | 333455/400000 [00:39<00:08, 8283.38it/s] 84%|████████▎ | 334339/400000 [00:39<00:07, 8440.52it/s] 84%|████████▍ | 335185/400000 [00:39<00:07, 8391.57it/s] 84%|████████▍ | 336055/400000 [00:39<00:07, 8479.39it/s] 84%|████████▍ | 336926/400000 [00:39<00:07, 8545.04it/s] 84%|████████▍ | 337796/400000 [00:39<00:07, 8588.86it/s] 85%|████████▍ | 338666/400000 [00:39<00:07, 8619.31it/s] 85%|████████▍ | 339576/400000 [00:39<00:06, 8757.17it/s] 85%|████████▌ | 340453/400000 [00:39<00:06, 8701.07it/s] 85%|████████▌ | 341324/400000 [00:40<00:06, 8693.53it/s] 86%|████████▌ | 342194/400000 [00:40<00:06, 8487.26it/s] 86%|████████▌ | 343045/400000 [00:40<00:06, 8405.00it/s] 86%|████████▌ | 343901/400000 [00:40<00:06, 8450.01it/s] 86%|████████▌ | 344756/400000 [00:40<00:06, 8477.51it/s] 86%|████████▋ | 345605/400000 [00:40<00:06, 8464.64it/s] 87%|████████▋ | 346452/400000 [00:40<00:06, 8377.57it/s] 87%|████████▋ | 347300/400000 [00:40<00:06, 8405.32it/s] 87%|████████▋ | 348163/400000 [00:40<00:06, 8470.57it/s] 87%|████████▋ | 349011/400000 [00:40<00:06, 8401.58it/s] 87%|████████▋ | 349868/400000 [00:41<00:05, 8450.71it/s] 88%|████████▊ | 350734/400000 [00:41<00:05, 8509.89it/s] 88%|████████▊ | 351605/400000 [00:41<00:05, 8568.50it/s] 88%|████████▊ | 352469/400000 [00:41<00:05, 8589.09it/s] 88%|████████▊ | 353329/400000 [00:41<00:05, 8575.67it/s] 89%|████████▊ | 354187/400000 [00:41<00:05, 8561.93it/s] 89%|████████▉ | 355044/400000 [00:41<00:05, 8494.25it/s] 89%|████████▉ | 355907/400000 [00:41<00:05, 8532.57it/s] 89%|████████▉ | 356777/400000 [00:41<00:05, 8579.44it/s] 89%|████████▉ | 357636/400000 [00:41<00:04, 8577.95it/s] 90%|████████▉ | 358494/400000 [00:42<00:04, 8559.16it/s] 90%|████████▉ | 359364/400000 [00:42<00:04, 8599.42it/s] 90%|█████████ | 360234/400000 [00:42<00:04, 8628.31it/s] 90%|█████████ | 361097/400000 [00:42<00:04, 8527.44it/s] 90%|█████████ | 361981/400000 [00:42<00:04, 8617.34it/s] 91%|█████████ | 362844/400000 [00:42<00:04, 8615.73it/s] 91%|█████████ | 363706/400000 [00:42<00:04, 8588.62it/s] 91%|█████████ | 364566/400000 [00:42<00:04, 8560.47it/s] 91%|█████████▏| 365423/400000 [00:42<00:04, 8485.10it/s] 92%|█████████▏| 366276/400000 [00:42<00:03, 8497.11it/s] 92%|█████████▏| 367126/400000 [00:43<00:03, 8487.44it/s] 92%|█████████▏| 367991/400000 [00:43<00:03, 8533.98it/s] 92%|█████████▏| 368845/400000 [00:43<00:03, 8496.45it/s] 92%|█████████▏| 369695/400000 [00:43<00:03, 8343.50it/s] 93%|█████████▎| 370563/400000 [00:43<00:03, 8440.85it/s] 93%|█████████▎| 371408/400000 [00:43<00:03, 8442.07it/s] 93%|█████████▎| 372270/400000 [00:43<00:03, 8493.55it/s] 93%|█████████▎| 373120/400000 [00:43<00:03, 8421.91it/s] 93%|█████████▎| 373995/400000 [00:43<00:03, 8516.35it/s] 94%|█████████▎| 374870/400000 [00:43<00:02, 8583.81it/s] 94%|█████████▍| 375729/400000 [00:44<00:02, 8469.92it/s] 94%|█████████▍| 376594/400000 [00:44<00:02, 8522.59it/s] 94%|█████████▍| 377468/400000 [00:44<00:02, 8584.49it/s] 95%|█████████▍| 378358/400000 [00:44<00:02, 8676.74it/s] 95%|█████████▍| 379227/400000 [00:44<00:02, 8664.08it/s] 95%|█████████▌| 380094/400000 [00:44<00:02, 8530.57it/s] 95%|█████████▌| 380971/400000 [00:44<00:02, 8600.79it/s] 95%|█████████▌| 381842/400000 [00:44<00:02, 8631.24it/s] 96%|█████████▌| 382733/400000 [00:44<00:01, 8712.79it/s] 96%|█████████▌| 383605/400000 [00:45<00:01, 8662.66it/s] 96%|█████████▌| 384472/400000 [00:45<00:01, 8648.03it/s] 96%|█████████▋| 385338/400000 [00:45<00:01, 8632.36it/s] 97%|█████████▋| 386214/400000 [00:45<00:01, 8667.37it/s] 97%|█████████▋| 387085/400000 [00:45<00:01, 8679.22it/s] 97%|█████████▋| 387965/400000 [00:45<00:01, 8713.81it/s] 97%|█████████▋| 388837/400000 [00:45<00:01, 8714.58it/s] 97%|█████████▋| 389712/400000 [00:45<00:01, 8722.58it/s] 98%|█████████▊| 390589/400000 [00:45<00:01, 8733.90it/s] 98%|█████████▊| 391463/400000 [00:45<00:00, 8647.15it/s] 98%|█████████▊| 392342/400000 [00:46<00:00, 8687.52it/s] 98%|█████████▊| 393211/400000 [00:46<00:00, 8625.88it/s] 99%|█████████▊| 394081/400000 [00:46<00:00, 8645.81it/s] 99%|█████████▊| 394946/400000 [00:46<00:00, 8647.03it/s] 99%|█████████▉| 395811/400000 [00:46<00:00, 8587.04it/s] 99%|█████████▉| 396670/400000 [00:46<00:00, 8573.54it/s] 99%|█████████▉| 397530/400000 [00:46<00:00, 8578.55it/s]100%|█████████▉| 398388/400000 [00:46<00:00, 8317.00it/s]100%|█████████▉| 399263/400000 [00:46<00:00, 8442.17it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8526.80it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f79f8db4940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011451698253424212 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.011660040820322707 	 Accuracy: 48

  model saves at 48% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15872 out of table with 15861 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15872 out of table with 15861 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-14 20:26:34.714417: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 20:26:34.718812: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-14 20:26:34.718944: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563e04deb400 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 20:26:34.718957: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f79a5789d30> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.7126 - accuracy: 0.4970
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7050 - accuracy: 0.4975 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7535 - accuracy: 0.4943
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6015 - accuracy: 0.5042
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5685 - accuracy: 0.5064
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6206 - accuracy: 0.5030
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6338 - accuracy: 0.5021
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6417 - accuracy: 0.5016
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6172 - accuracy: 0.5032
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5808 - accuracy: 0.5056
11000/25000 [============>.................] - ETA: 3s - loss: 7.5983 - accuracy: 0.5045
12000/25000 [=============>................] - ETA: 3s - loss: 7.6053 - accuracy: 0.5040
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6301 - accuracy: 0.5024
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6436 - accuracy: 0.5015
15000/25000 [=================>............] - ETA: 2s - loss: 7.6615 - accuracy: 0.5003
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6551 - accuracy: 0.5008
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6756 - accuracy: 0.4994
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6624 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6620 - accuracy: 0.5003
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6555 - accuracy: 0.5007
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6546 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6577 - accuracy: 0.5006
25000/25000 [==============================] - 7s 277us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f7974d42748> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f7a00506128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.2106 - crf_viterbi_accuracy: 0.6800 - val_loss: 1.1293 - val_crf_viterbi_accuracy: 0.6533

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

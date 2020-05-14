
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fc3abfc6fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 21:14:51.851438
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 21:14:51.855637
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 21:14:51.859175
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 21:14:51.863746
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fc3b7fde470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354234.3438
Epoch 2/10

1/1 [==============================] - 0s 108ms/step - loss: 267713.1250
Epoch 3/10

1/1 [==============================] - 0s 102ms/step - loss: 170030.0469
Epoch 4/10

1/1 [==============================] - 0s 105ms/step - loss: 85744.0938
Epoch 5/10

1/1 [==============================] - 0s 122ms/step - loss: 41141.3984
Epoch 6/10

1/1 [==============================] - 0s 102ms/step - loss: 21046.2422
Epoch 7/10

1/1 [==============================] - 0s 101ms/step - loss: 12109.8740
Epoch 8/10

1/1 [==============================] - 0s 101ms/step - loss: 7770.0723
Epoch 9/10

1/1 [==============================] - 0s 103ms/step - loss: 5477.4580
Epoch 10/10

1/1 [==============================] - 0s 106ms/step - loss: 4160.3696

  #### Inference Need return ypred, ytrue ######################### 
[[ -0.74463296  -0.36996037   1.2864242    1.3246906   -0.22091565
   -0.7691022    0.7851974   -0.12123257  -2.6391332    0.5239316
   -0.9572528    3.287966    -1.1969843   -0.47273284  -0.7994481
    1.2693291    1.5728052   -1.040105     2.0735548    0.2279762
    1.5097992    1.0495355   -1.9826057    1.1098123    0.62559813
    0.197599     1.5885926    0.6705525   -0.5672369    0.38828614
    2.4798923    0.22973514   0.40525174  -0.18871616   2.6543493
    1.6965215    1.0169666   -0.3059595   -0.53526556  -0.47254795
    1.9089587    1.0153435    0.34464207   1.8461709   -0.07344827
    0.83597755   2.8527749   -2.2841      -2.1187234    0.7996313
   -1.6422366   -0.1991694   -0.7083466    0.35977376   0.54847455
   -1.2004983   -0.34642485   0.2670223    0.70059836   1.9101026
   -0.40708366   8.896353    11.120386    13.012511    10.901053
   10.01036     12.3728      11.126959    10.363072     9.996568
   11.490896     9.499538    11.059988    11.503307    10.664775
    8.600403    10.410255    11.23721     10.723714     9.923863
   10.700801    10.603506    11.072092     9.667028    11.960201
   11.908113    10.189626    10.438575    10.9743595   10.259275
   10.482073    10.606159    10.073727    11.441144     9.908425
   11.657984    10.332284     9.753757    10.115975     9.713015
   12.265639    11.0043545    9.490089    11.276831    10.094661
   12.471595    11.91736     12.5306635   12.422423    11.21066
   11.624256    13.243634    10.384371    11.248649     9.349999
   11.217275     9.292944    11.098885    12.241639    11.844019
    0.5400879    0.49881414  -0.23337346   1.9079874    1.2229285
   -0.31818897  -0.64371294  -2.9539146   -1.5972354    2.5219922
   -0.33999276  -2.1028807   -0.2544407    1.0750444   -0.23639065
   -0.5584494    0.31930447  -1.3823781   -1.1355785   -0.5553098
   -1.9160627   -1.5910454    1.2521162    0.5970245   -1.277724
   -0.36851773  -0.8939456   -0.70193714   0.5945645   -1.0810308
    0.4688716   -0.70424986   0.14667633  -0.85164547  -1.3102012
   -1.034524     0.6324156    0.12004549   1.6806642   -0.7337004
    0.7433024    2.2636542    0.6392181   -0.5832265   -0.6201937
    1.9757122    2.734524     0.91813225   1.057709     0.61865306
   -0.43841064  -0.13623083  -0.42376897   1.2121665   -0.3451584
    0.17423153   1.2843541    0.15665817  -0.05153108   0.7615715
    1.2806947    1.1074781    0.45560813   2.0322013    0.81054395
    1.5089493    2.565576     1.8494389    0.09378552   0.51243174
    0.6260058    0.2539208    0.2755822    0.82882404   1.8158963
    1.6700993    0.36308825   0.39484262   1.2556081    1.0440938
    1.9531641    0.7684191    1.958608     0.51440537   0.7520915
    2.8704433    0.90225124   1.4660838    0.5988972    1.1722794
    0.32447052   0.63642186   0.21579617   1.5831206    1.4834535
    1.5054507    0.43901783   0.10913742   0.55106825   0.9004793
    1.5625203    1.2726853    1.1702633    1.9590932    0.47260833
    0.15255392   0.32830364   0.49090207   1.3232785    0.21847647
    1.5362473    0.89356184   0.56046474   0.4681993    2.8081713
    1.727464     0.31677127   1.58044      0.8256057    1.0049461
    0.45821834  10.098215    11.98398      9.763343    11.205222
   10.203788    11.491836    11.770602    10.572282    10.255693
   10.837343    11.682239    11.3116455   10.891724    10.186457
   11.836512    10.353095     9.637913    10.309558    11.89165
   11.551044    11.422196    10.050418    10.700228    11.720761
   10.037443    11.028028    12.168177    10.6590185   11.197447
    9.699731    10.225651     9.11887      9.8251505   11.492247
   11.797867    10.813171    10.706673     9.744894    11.47251
    9.860503    10.773117    12.002922    11.259583    10.6377325
    9.927038    10.983601     9.679893    10.845768    11.447429
   10.75277     12.57135     10.3149605   10.574926    10.050032
   11.162148    11.211979    11.447945    10.347704    11.026398
    0.17227519   1.6414078    0.536749     0.2736861    2.1665502
    0.16782415   1.6948023    0.73641664   2.0565405    1.9683168
    0.733548     1.4566114    0.9973743    0.5179783    0.509761
    1.3205396    0.4852643    2.1744351    1.7531438    3.5902934
    0.35533208   0.14993095   1.5828       1.3833079    3.04222
    4.2816157    2.0865917    0.4178381    0.9492238    0.21248847
    2.3361359    0.28993452   2.7100039    1.6679573    0.29887235
    0.23386157   0.7275005    2.001946     1.6039371    0.08315539
    0.3970151    0.15393865   0.66086835   0.2341522    3.3686662
    0.6380186    0.9474025    1.3377712    2.2388716    0.8460999
    2.8374043    1.3309746    0.49196148   1.2430224    1.9684801
    1.255758     0.629079     3.4477167    0.23892301   1.8671728
  -12.8321       6.2279067   -8.739193  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 21:15:00.814812
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    90.554
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 21:15:00.819469
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    8227.2
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 21:15:00.823872
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.3229
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 21:15:00.827567
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -735.804
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140478041293320
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140475511484992
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140475511485496
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140475511486000
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140475511486504
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140475511487008

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fc3abed4be0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.524219
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.501691
grad_step = 000002, loss = 0.481933
grad_step = 000003, loss = 0.458567
grad_step = 000004, loss = 0.432503
grad_step = 000005, loss = 0.404856
grad_step = 000006, loss = 0.381486
grad_step = 000007, loss = 0.373292
grad_step = 000008, loss = 0.371925
grad_step = 000009, loss = 0.356727
grad_step = 000010, loss = 0.341199
grad_step = 000011, loss = 0.329954
grad_step = 000012, loss = 0.322015
grad_step = 000013, loss = 0.314431
grad_step = 000014, loss = 0.305659
grad_step = 000015, loss = 0.295543
grad_step = 000016, loss = 0.284661
grad_step = 000017, loss = 0.274298
grad_step = 000018, loss = 0.265758
grad_step = 000019, loss = 0.258581
grad_step = 000020, loss = 0.250568
grad_step = 000021, loss = 0.241079
grad_step = 000022, loss = 0.231597
grad_step = 000023, loss = 0.223047
grad_step = 000024, loss = 0.215448
grad_step = 000025, loss = 0.208252
grad_step = 000026, loss = 0.201099
grad_step = 000027, loss = 0.194105
grad_step = 000028, loss = 0.187372
grad_step = 000029, loss = 0.180503
grad_step = 000030, loss = 0.173359
grad_step = 000031, loss = 0.166445
grad_step = 000032, loss = 0.160092
grad_step = 000033, loss = 0.154129
grad_step = 000034, loss = 0.148295
grad_step = 000035, loss = 0.142526
grad_step = 000036, loss = 0.136921
grad_step = 000037, loss = 0.131544
grad_step = 000038, loss = 0.126324
grad_step = 000039, loss = 0.121147
grad_step = 000040, loss = 0.116035
grad_step = 000041, loss = 0.111173
grad_step = 000042, loss = 0.106631
grad_step = 000043, loss = 0.102233
grad_step = 000044, loss = 0.097855
grad_step = 000045, loss = 0.093558
grad_step = 000046, loss = 0.089485
grad_step = 000047, loss = 0.085640
grad_step = 000048, loss = 0.081901
grad_step = 000049, loss = 0.078251
grad_step = 000050, loss = 0.074749
grad_step = 000051, loss = 0.071409
grad_step = 000052, loss = 0.068184
grad_step = 000053, loss = 0.065061
grad_step = 000054, loss = 0.062040
grad_step = 000055, loss = 0.059141
grad_step = 000056, loss = 0.056385
grad_step = 000057, loss = 0.053774
grad_step = 000058, loss = 0.051256
grad_step = 000059, loss = 0.048804
grad_step = 000060, loss = 0.046449
grad_step = 000061, loss = 0.044212
grad_step = 000062, loss = 0.042068
grad_step = 000063, loss = 0.040006
grad_step = 000064, loss = 0.038041
grad_step = 000065, loss = 0.036169
grad_step = 000066, loss = 0.034370
grad_step = 000067, loss = 0.032643
grad_step = 000068, loss = 0.030990
grad_step = 000069, loss = 0.029413
grad_step = 000070, loss = 0.027916
grad_step = 000071, loss = 0.026490
grad_step = 000072, loss = 0.025124
grad_step = 000073, loss = 0.023818
grad_step = 000074, loss = 0.022577
grad_step = 000075, loss = 0.021388
grad_step = 000076, loss = 0.020249
grad_step = 000077, loss = 0.019181
grad_step = 000078, loss = 0.018173
grad_step = 000079, loss = 0.017196
grad_step = 000080, loss = 0.016272
grad_step = 000081, loss = 0.015402
grad_step = 000082, loss = 0.014426
grad_step = 000083, loss = 0.013457
grad_step = 000084, loss = 0.012683
grad_step = 000085, loss = 0.012015
grad_step = 000086, loss = 0.011285
grad_step = 000087, loss = 0.010579
grad_step = 000088, loss = 0.009934
grad_step = 000089, loss = 0.009329
grad_step = 000090, loss = 0.008753
grad_step = 000091, loss = 0.008191
grad_step = 000092, loss = 0.007668
grad_step = 000093, loss = 0.007199
grad_step = 000094, loss = 0.006748
grad_step = 000095, loss = 0.006310
grad_step = 000096, loss = 0.005906
grad_step = 000097, loss = 0.005540
grad_step = 000098, loss = 0.005206
grad_step = 000099, loss = 0.004892
grad_step = 000100, loss = 0.004597
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.004327
grad_step = 000102, loss = 0.004092
grad_step = 000103, loss = 0.003886
grad_step = 000104, loss = 0.003708
grad_step = 000105, loss = 0.003539
grad_step = 000106, loss = 0.003402
grad_step = 000107, loss = 0.003238
grad_step = 000108, loss = 0.003052
grad_step = 000109, loss = 0.002908
grad_step = 000110, loss = 0.002862
grad_step = 000111, loss = 0.002774
grad_step = 000112, loss = 0.002646
grad_step = 000113, loss = 0.002601
grad_step = 000114, loss = 0.002517
grad_step = 000115, loss = 0.002450
grad_step = 000116, loss = 0.002459
grad_step = 000117, loss = 0.002382
grad_step = 000118, loss = 0.002316
grad_step = 000119, loss = 0.002318
grad_step = 000120, loss = 0.002275
grad_step = 000121, loss = 0.002244
grad_step = 000122, loss = 0.002251
grad_step = 000123, loss = 0.002229
grad_step = 000124, loss = 0.002173
grad_step = 000125, loss = 0.002175
grad_step = 000126, loss = 0.002156
grad_step = 000127, loss = 0.002130
grad_step = 000128, loss = 0.002136
grad_step = 000129, loss = 0.002123
grad_step = 000130, loss = 0.002098
grad_step = 000131, loss = 0.002099
grad_step = 000132, loss = 0.002094
grad_step = 000133, loss = 0.002069
grad_step = 000134, loss = 0.002064
grad_step = 000135, loss = 0.002065
grad_step = 000136, loss = 0.002046
grad_step = 000137, loss = 0.002032
grad_step = 000138, loss = 0.002031
grad_step = 000139, loss = 0.002025
grad_step = 000140, loss = 0.002012
grad_step = 000141, loss = 0.001999
grad_step = 000142, loss = 0.001995
grad_step = 000143, loss = 0.001992
grad_step = 000144, loss = 0.001983
grad_step = 000145, loss = 0.001974
grad_step = 000146, loss = 0.001970
grad_step = 000147, loss = 0.001991
grad_step = 000148, loss = 0.002069
grad_step = 000149, loss = 0.002290
grad_step = 000150, loss = 0.002371
grad_step = 000151, loss = 0.002332
grad_step = 000152, loss = 0.001940
grad_step = 000153, loss = 0.002187
grad_step = 000154, loss = 0.002265
grad_step = 000155, loss = 0.001918
grad_step = 000156, loss = 0.002285
grad_step = 000157, loss = 0.002042
grad_step = 000158, loss = 0.002039
grad_step = 000159, loss = 0.002058
grad_step = 000160, loss = 0.001947
grad_step = 000161, loss = 0.002002
grad_step = 000162, loss = 0.001917
grad_step = 000163, loss = 0.002012
grad_step = 000164, loss = 0.001829
grad_step = 000165, loss = 0.001962
grad_step = 000166, loss = 0.001811
grad_step = 000167, loss = 0.001894
grad_step = 000168, loss = 0.001842
grad_step = 000169, loss = 0.001824
grad_step = 000170, loss = 0.001822
grad_step = 000171, loss = 0.001793
grad_step = 000172, loss = 0.001792
grad_step = 000173, loss = 0.001746
grad_step = 000174, loss = 0.001786
grad_step = 000175, loss = 0.001709
grad_step = 000176, loss = 0.001726
grad_step = 000177, loss = 0.001710
grad_step = 000178, loss = 0.001661
grad_step = 000179, loss = 0.001683
grad_step = 000180, loss = 0.001644
grad_step = 000181, loss = 0.001623
grad_step = 000182, loss = 0.001631
grad_step = 000183, loss = 0.001590
grad_step = 000184, loss = 0.001581
grad_step = 000185, loss = 0.001582
grad_step = 000186, loss = 0.001551
grad_step = 000187, loss = 0.001559
grad_step = 000188, loss = 0.001622
grad_step = 000189, loss = 0.001730
grad_step = 000190, loss = 0.002008
grad_step = 000191, loss = 0.001709
grad_step = 000192, loss = 0.001504
grad_step = 000193, loss = 0.001581
grad_step = 000194, loss = 0.001639
grad_step = 000195, loss = 0.001500
grad_step = 000196, loss = 0.001473
grad_step = 000197, loss = 0.001555
grad_step = 000198, loss = 0.001476
grad_step = 000199, loss = 0.001419
grad_step = 000200, loss = 0.001490
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001451
grad_step = 000202, loss = 0.001384
grad_step = 000203, loss = 0.001424
grad_step = 000204, loss = 0.001430
grad_step = 000205, loss = 0.001390
grad_step = 000206, loss = 0.001370
grad_step = 000207, loss = 0.001342
grad_step = 000208, loss = 0.001362
grad_step = 000209, loss = 0.001378
grad_step = 000210, loss = 0.001322
grad_step = 000211, loss = 0.001307
grad_step = 000212, loss = 0.001305
grad_step = 000213, loss = 0.001301
grad_step = 000214, loss = 0.001323
grad_step = 000215, loss = 0.001305
grad_step = 000216, loss = 0.001291
grad_step = 000217, loss = 0.001271
grad_step = 000218, loss = 0.001241
grad_step = 000219, loss = 0.001223
grad_step = 000220, loss = 0.001217
grad_step = 000221, loss = 0.001211
grad_step = 000222, loss = 0.001220
grad_step = 000223, loss = 0.001244
grad_step = 000224, loss = 0.001251
grad_step = 000225, loss = 0.001277
grad_step = 000226, loss = 0.001231
grad_step = 000227, loss = 0.001184
grad_step = 000228, loss = 0.001145
grad_step = 000229, loss = 0.001144
grad_step = 000230, loss = 0.001166
grad_step = 000231, loss = 0.001158
grad_step = 000232, loss = 0.001133
grad_step = 000233, loss = 0.001103
grad_step = 000234, loss = 0.001091
grad_step = 000235, loss = 0.001092
grad_step = 000236, loss = 0.001104
grad_step = 000237, loss = 0.001123
grad_step = 000238, loss = 0.001104
grad_step = 000239, loss = 0.001083
grad_step = 000240, loss = 0.001048
grad_step = 000241, loss = 0.001030
grad_step = 000242, loss = 0.001038
grad_step = 000243, loss = 0.001052
grad_step = 000244, loss = 0.001075
grad_step = 000245, loss = 0.001068
grad_step = 000246, loss = 0.001068
grad_step = 000247, loss = 0.001023
grad_step = 000248, loss = 0.000989
grad_step = 000249, loss = 0.000969
grad_step = 000250, loss = 0.000976
grad_step = 000251, loss = 0.000996
grad_step = 000252, loss = 0.000985
grad_step = 000253, loss = 0.000971
grad_step = 000254, loss = 0.000946
grad_step = 000255, loss = 0.000928
grad_step = 000256, loss = 0.000912
grad_step = 000257, loss = 0.000906
grad_step = 000258, loss = 0.000906
grad_step = 000259, loss = 0.000911
grad_step = 000260, loss = 0.000931
grad_step = 000261, loss = 0.000954
grad_step = 000262, loss = 0.001024
grad_step = 000263, loss = 0.001040
grad_step = 000264, loss = 0.001082
grad_step = 000265, loss = 0.000939
grad_step = 000266, loss = 0.000855
grad_step = 000267, loss = 0.000860
grad_step = 000268, loss = 0.000889
grad_step = 000269, loss = 0.000900
grad_step = 000270, loss = 0.000843
grad_step = 000271, loss = 0.000813
grad_step = 000272, loss = 0.000839
grad_step = 000273, loss = 0.000842
grad_step = 000274, loss = 0.000815
grad_step = 000275, loss = 0.000786
grad_step = 000276, loss = 0.000792
grad_step = 000277, loss = 0.000814
grad_step = 000278, loss = 0.000817
grad_step = 000279, loss = 0.000817
grad_step = 000280, loss = 0.000794
grad_step = 000281, loss = 0.000764
grad_step = 000282, loss = 0.000742
grad_step = 000283, loss = 0.000742
grad_step = 000284, loss = 0.000746
grad_step = 000285, loss = 0.000746
grad_step = 000286, loss = 0.000747
grad_step = 000287, loss = 0.000733
grad_step = 000288, loss = 0.000720
grad_step = 000289, loss = 0.000711
grad_step = 000290, loss = 0.000704
grad_step = 000291, loss = 0.000694
grad_step = 000292, loss = 0.000686
grad_step = 000293, loss = 0.000682
grad_step = 000294, loss = 0.000675
grad_step = 000295, loss = 0.000673
grad_step = 000296, loss = 0.000681
grad_step = 000297, loss = 0.000715
grad_step = 000298, loss = 0.000824
grad_step = 000299, loss = 0.000949
grad_step = 000300, loss = 0.001299
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000986
grad_step = 000302, loss = 0.000735
grad_step = 000303, loss = 0.000670
grad_step = 000304, loss = 0.000819
grad_step = 000305, loss = 0.000878
grad_step = 000306, loss = 0.000674
grad_step = 000307, loss = 0.000757
grad_step = 000308, loss = 0.000878
grad_step = 000309, loss = 0.000686
grad_step = 000310, loss = 0.000849
grad_step = 000311, loss = 0.000872
grad_step = 000312, loss = 0.000708
grad_step = 000313, loss = 0.000919
grad_step = 000314, loss = 0.000791
grad_step = 000315, loss = 0.000749
grad_step = 000316, loss = 0.000839
grad_step = 000317, loss = 0.000641
grad_step = 000318, loss = 0.000762
grad_step = 000319, loss = 0.000634
grad_step = 000320, loss = 0.000646
grad_step = 000321, loss = 0.000705
grad_step = 000322, loss = 0.000582
grad_step = 000323, loss = 0.000696
grad_step = 000324, loss = 0.000609
grad_step = 000325, loss = 0.000613
grad_step = 000326, loss = 0.000632
grad_step = 000327, loss = 0.000557
grad_step = 000328, loss = 0.000612
grad_step = 000329, loss = 0.000561
grad_step = 000330, loss = 0.000556
grad_step = 000331, loss = 0.000584
grad_step = 000332, loss = 0.000533
grad_step = 000333, loss = 0.000566
grad_step = 000334, loss = 0.000554
grad_step = 000335, loss = 0.000533
grad_step = 000336, loss = 0.000562
grad_step = 000337, loss = 0.000530
grad_step = 000338, loss = 0.000532
grad_step = 000339, loss = 0.000535
grad_step = 000340, loss = 0.000514
grad_step = 000341, loss = 0.000526
grad_step = 000342, loss = 0.000512
grad_step = 000343, loss = 0.000500
grad_step = 000344, loss = 0.000515
grad_step = 000345, loss = 0.000498
grad_step = 000346, loss = 0.000491
grad_step = 000347, loss = 0.000499
grad_step = 000348, loss = 0.000488
grad_step = 000349, loss = 0.000487
grad_step = 000350, loss = 0.000485
grad_step = 000351, loss = 0.000476
grad_step = 000352, loss = 0.000479
grad_step = 000353, loss = 0.000477
grad_step = 000354, loss = 0.000469
grad_step = 000355, loss = 0.000469
grad_step = 000356, loss = 0.000465
grad_step = 000357, loss = 0.000461
grad_step = 000358, loss = 0.000462
grad_step = 000359, loss = 0.000458
grad_step = 000360, loss = 0.000453
grad_step = 000361, loss = 0.000453
grad_step = 000362, loss = 0.000450
grad_step = 000363, loss = 0.000447
grad_step = 000364, loss = 0.000446
grad_step = 000365, loss = 0.000442
grad_step = 000366, loss = 0.000440
grad_step = 000367, loss = 0.000439
grad_step = 000368, loss = 0.000436
grad_step = 000369, loss = 0.000434
grad_step = 000370, loss = 0.000432
grad_step = 000371, loss = 0.000430
grad_step = 000372, loss = 0.000428
grad_step = 000373, loss = 0.000427
grad_step = 000374, loss = 0.000425
grad_step = 000375, loss = 0.000423
grad_step = 000376, loss = 0.000422
grad_step = 000377, loss = 0.000420
grad_step = 000378, loss = 0.000418
grad_step = 000379, loss = 0.000417
grad_step = 000380, loss = 0.000416
grad_step = 000381, loss = 0.000416
grad_step = 000382, loss = 0.000417
grad_step = 000383, loss = 0.000419
grad_step = 000384, loss = 0.000418
grad_step = 000385, loss = 0.000418
grad_step = 000386, loss = 0.000414
grad_step = 000387, loss = 0.000412
grad_step = 000388, loss = 0.000407
grad_step = 000389, loss = 0.000403
grad_step = 000390, loss = 0.000399
grad_step = 000391, loss = 0.000397
grad_step = 000392, loss = 0.000394
grad_step = 000393, loss = 0.000392
grad_step = 000394, loss = 0.000391
grad_step = 000395, loss = 0.000390
grad_step = 000396, loss = 0.000389
grad_step = 000397, loss = 0.000388
grad_step = 000398, loss = 0.000388
grad_step = 000399, loss = 0.000388
grad_step = 000400, loss = 0.000391
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000394
grad_step = 000402, loss = 0.000403
grad_step = 000403, loss = 0.000408
grad_step = 000404, loss = 0.000418
grad_step = 000405, loss = 0.000415
grad_step = 000406, loss = 0.000415
grad_step = 000407, loss = 0.000402
grad_step = 000408, loss = 0.000391
grad_step = 000409, loss = 0.000377
grad_step = 000410, loss = 0.000372
grad_step = 000411, loss = 0.000371
grad_step = 000412, loss = 0.000375
grad_step = 000413, loss = 0.000382
grad_step = 000414, loss = 0.000389
grad_step = 000415, loss = 0.000395
grad_step = 000416, loss = 0.000388
grad_step = 000417, loss = 0.000383
grad_step = 000418, loss = 0.000374
grad_step = 000419, loss = 0.000368
grad_step = 000420, loss = 0.000363
grad_step = 000421, loss = 0.000360
grad_step = 000422, loss = 0.000360
grad_step = 000423, loss = 0.000360
grad_step = 000424, loss = 0.000362
grad_step = 000425, loss = 0.000365
grad_step = 000426, loss = 0.000374
grad_step = 000427, loss = 0.000382
grad_step = 000428, loss = 0.000400
grad_step = 000429, loss = 0.000406
grad_step = 000430, loss = 0.000423
grad_step = 000431, loss = 0.000416
grad_step = 000432, loss = 0.000414
grad_step = 000433, loss = 0.000388
grad_step = 000434, loss = 0.000369
grad_step = 000435, loss = 0.000353
grad_step = 000436, loss = 0.000348
grad_step = 000437, loss = 0.000353
grad_step = 000438, loss = 0.000364
grad_step = 000439, loss = 0.000376
grad_step = 000440, loss = 0.000375
grad_step = 000441, loss = 0.000372
grad_step = 000442, loss = 0.000358
grad_step = 000443, loss = 0.000349
grad_step = 000444, loss = 0.000343
grad_step = 000445, loss = 0.000341
grad_step = 000446, loss = 0.000341
grad_step = 000447, loss = 0.000342
grad_step = 000448, loss = 0.000346
grad_step = 000449, loss = 0.000352
grad_step = 000450, loss = 0.000360
grad_step = 000451, loss = 0.000365
grad_step = 000452, loss = 0.000377
grad_step = 000453, loss = 0.000378
grad_step = 000454, loss = 0.000386
grad_step = 000455, loss = 0.000375
grad_step = 000456, loss = 0.000368
grad_step = 000457, loss = 0.000353
grad_step = 000458, loss = 0.000343
grad_step = 000459, loss = 0.000335
grad_step = 000460, loss = 0.000332
grad_step = 000461, loss = 0.000332
grad_step = 000462, loss = 0.000335
grad_step = 000463, loss = 0.000341
grad_step = 000464, loss = 0.000349
grad_step = 000465, loss = 0.000362
grad_step = 000466, loss = 0.000371
grad_step = 000467, loss = 0.000389
grad_step = 000468, loss = 0.000389
grad_step = 000469, loss = 0.000399
grad_step = 000470, loss = 0.000380
grad_step = 000471, loss = 0.000368
grad_step = 000472, loss = 0.000344
grad_step = 000473, loss = 0.000331
grad_step = 000474, loss = 0.000325
grad_step = 000475, loss = 0.000326
grad_step = 000476, loss = 0.000331
grad_step = 000477, loss = 0.000339
grad_step = 000478, loss = 0.000348
grad_step = 000479, loss = 0.000352
grad_step = 000480, loss = 0.000358
grad_step = 000481, loss = 0.000352
grad_step = 000482, loss = 0.000347
grad_step = 000483, loss = 0.000339
grad_step = 000484, loss = 0.000332
grad_step = 000485, loss = 0.000326
grad_step = 000486, loss = 0.000323
grad_step = 000487, loss = 0.000321
grad_step = 000488, loss = 0.000319
grad_step = 000489, loss = 0.000318
grad_step = 000490, loss = 0.000318
grad_step = 000491, loss = 0.000318
grad_step = 000492, loss = 0.000318
grad_step = 000493, loss = 0.000319
grad_step = 000494, loss = 0.000321
grad_step = 000495, loss = 0.000328
grad_step = 000496, loss = 0.000341
grad_step = 000497, loss = 0.000371
grad_step = 000498, loss = 0.000404
grad_step = 000499, loss = 0.000484
grad_step = 000500, loss = 0.000523
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000605
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

  date_run                              2020-05-14 21:15:24.332325
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.269475
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 21:15:24.338214
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.186468
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 21:15:24.345181
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.146307
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 21:15:24.351281
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.83345
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
0   2020-05-14 21:14:51.851438  ...    mean_absolute_error
1   2020-05-14 21:14:51.855637  ...     mean_squared_error
2   2020-05-14 21:14:51.859175  ...  median_absolute_error
3   2020-05-14 21:14:51.863746  ...               r2_score
4   2020-05-14 21:15:00.814812  ...    mean_absolute_error
5   2020-05-14 21:15:00.819469  ...     mean_squared_error
6   2020-05-14 21:15:00.823872  ...  median_absolute_error
7   2020-05-14 21:15:00.827567  ...               r2_score
8   2020-05-14 21:15:24.332325  ...    mean_absolute_error
9   2020-05-14 21:15:24.338214  ...     mean_squared_error
10  2020-05-14 21:15:24.345181  ...  median_absolute_error
11  2020-05-14 21:15:24.351281  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f09f3234898> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 489510.75it/s] 45%|████▍     | 4415488/9912422 [00:00<00:07, 695891.96it/s] 95%|█████████▍| 9388032/9912422 [00:00<00:00, 988101.28it/s]9920512it [00:00, 31807787.25it/s]                           
0it [00:00, ?it/s]32768it [00:00, 579351.40it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 997313.12it/s]1654784it [00:00, 11641918.20it/s]                          
0it [00:00, ?it/s]8192it [00:00, 180388.81it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f09a5be2dd8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f09a2a2f048> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f09a5be2dd8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f09a2a2f048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f09a29a4438> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f09a2a2f048> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f09a5be2dd8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f09a2a2f048> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f09a29a4438> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f09f31ece80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fdfe20811d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=b69ea472845342ef80569da8dba5eb611b9134919b44c496f2d917c15ca7eff5
  Stored in directory: /tmp/pip-ephem-wheel-cache-os0mzi8d/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fdf79e7d6d8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2129920/17464789 [==>...........................] - ETA: 0s
 6963200/17464789 [==========>...................] - ETA: 0s
10027008/17464789 [================>.............] - ETA: 0s
13131776/17464789 [=====================>........] - ETA: 0s
16228352/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 21:16:51.181298: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 21:16:51.185267: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-14 21:16:51.185490: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ca62e9f230 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 21:16:51.185506: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.9426 - accuracy: 0.4820
 2000/25000 [=>............................] - ETA: 10s - loss: 7.8276 - accuracy: 0.4895
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8353 - accuracy: 0.4890 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8315 - accuracy: 0.4893
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.8108 - accuracy: 0.4906
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7893 - accuracy: 0.4920
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7915 - accuracy: 0.4919
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.8008 - accuracy: 0.4913
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7518 - accuracy: 0.4944
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7295 - accuracy: 0.4959
11000/25000 [============>.................] - ETA: 4s - loss: 7.7098 - accuracy: 0.4972
12000/25000 [=============>................] - ETA: 4s - loss: 7.6730 - accuracy: 0.4996
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6607 - accuracy: 0.5004
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6655 - accuracy: 0.5001
15000/25000 [=================>............] - ETA: 3s - loss: 7.6717 - accuracy: 0.4997
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6567 - accuracy: 0.5006
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6428 - accuracy: 0.5016
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6440 - accuracy: 0.5015
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6406 - accuracy: 0.5017
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6681 - accuracy: 0.4999
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6694 - accuracy: 0.4998
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6506 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6583 - accuracy: 0.5005
25000/25000 [==============================] - 10s 389us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 21:17:08.165088
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 21:17:08.165088  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<25:40:04, 9.33kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<18:12:52, 13.1kB/s].vector_cache/glove.6B.zip:   0%|          | 172k/862M [00:01<12:49:40, 18.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 721k/862M [00:01<8:59:29, 26.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.92M/862M [00:01<6:16:56, 38.0kB/s].vector_cache/glove.6B.zip:   1%|          | 8.81M/862M [00:01<4:22:07, 54.3kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.0M/862M [00:01<3:02:55, 77.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.9M/862M [00:01<2:07:23, 111kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.1M/862M [00:01<1:28:51, 158kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.7M/862M [00:01<1:01:53, 225kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.0M/862M [00:02<43:07, 321kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.3M/862M [00:02<30:12, 456kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.3M/862M [00:02<21:05, 649kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.9M/862M [00:02<14:49, 920kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.4M/862M [00:02<10:22, 1.31MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:03<08:04, 1.67MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<07:32, 1.78MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<07:07, 1.89MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<05:22, 2.49MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:07<06:19, 2.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<06:08, 2.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:07<04:43, 2.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:09<05:56, 2.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<05:42, 2.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:09<04:18, 3.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:11<05:49, 2.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:11<06:47, 1.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:11<05:25, 2.44MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<05:54, 2.22MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:13<05:30, 2.39MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:13<04:11, 3.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<05:59, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:15<05:31, 2.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:15<04:11, 3.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<06:00, 2.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:17<06:52, 1.89MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:17<05:28, 2.37MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<05:55, 2.19MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<05:29, 2.35MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:19<04:10, 3.09MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<05:54, 2.18MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<05:25, 2.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:21<04:04, 3.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<05:54, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<06:45, 1.90MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:23<05:17, 2.42MB/s].vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:23<03:50, 3.31MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<10:47, 1.18MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<08:51, 1.44MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.7M/862M [00:25<06:30, 1.95MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:31, 1.68MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:33, 1.93MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:54, 2.57MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:24, 1.97MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:46, 2.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:21, 2.88MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:00, 2.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:46, 1.85MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:22, 2.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:01, 2.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:28, 2.78MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:52, 2.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:24, 2.29MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:02, 3.06MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:43, 2.16MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:31, 1.89MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:12, 2.37MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:36, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:13, 2.35MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<03:55, 3.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:35, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:09, 2.36MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<03:55, 3.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:36, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:09, 2.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<03:54, 3.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:35, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:08, 2.34MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<03:54, 3.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:29, 2.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:18, 1.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:00, 2.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:26, 2.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:03, 2.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<03:48, 3.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:23, 2.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<04:58, 2.38MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:46, 3.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:25, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:00, 2.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<03:45, 3.13MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:22, 2.18MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<04:57, 2.36MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:54<03:45, 3.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:22, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<04:57, 2.35MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:45, 3.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:22, 2.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<04:55, 2.35MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:44, 3.09MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:19, 2.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<04:54, 2.34MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<03:43, 3.08MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:18, 2.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<04:52, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<03:41, 3.08MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:16, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<04:51, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<03:41, 3.07MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:14, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<04:50, 2.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<03:40, 3.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:13, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<04:48, 2.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<03:36, 3.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:10, 2.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:54, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<04:38, 2.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:22, 3.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<10:56, 1.01MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<08:47, 1.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<06:26, 1.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:04, 1.56MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:14, 1.76MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<04:41, 2.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:35, 1.96MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:29, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<05:11, 2.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<03:45, 2.89MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<11:41, 931kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<09:25, 1.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<06:54, 1.57MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:06, 1.52MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:30, 1.44MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<05:53, 1.83MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<04:15, 2.53MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<11:48, 911kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<09:31, 1.13MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:55, 1.55MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<07:06, 1.50MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<07:28, 1.43MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<05:51, 1.82MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<04:14, 2.50MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<11:45, 903kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<09:28, 1.12MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:55, 1.53MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:03, 1.49MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:24, 1.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:43, 1.84MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:07, 2.54MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<07:43, 1.36MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:35, 1.59MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:54, 2.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:37, 1.85MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:08, 2.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<03:52, 2.68MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:53, 2.11MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:38, 2.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<03:32, 2.91MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:42, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:41, 1.80MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<04:34, 2.24MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:20, 3.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<10:54, 934kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<08:48, 1.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<06:23, 1.59MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:38, 1.52MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<07:05, 1.43MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:28, 1.84MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:01, 2.50MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:31, 1.82MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:03, 1.99MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<03:49, 2.62MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<04:47, 2.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<05:41, 1.75MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:34, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<03:19, 2.98MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<10:32, 941kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<08:32, 1.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<06:14, 1.59MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:47<06:26, 1.53MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<06:48, 1.45MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:14, 1.87MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<03:48, 2.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:18, 1.55MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:32, 1.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:08, 2.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<04:58, 1.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<05:48, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:31, 2.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:20, 2.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:57, 1.94MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:35, 2.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<03:27, 2.78MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:26, 2.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:21, 1.79MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:13, 2.26MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<03:06, 3.07MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:22, 1.77MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:51, 1.95MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<03:36, 2.62MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<02:38, 3.56MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<26:25, 357kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<20:42, 455kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<15:00, 627kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<10:35, 884kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<16:36, 563kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<12:43, 735kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<09:06, 1.02MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<08:19, 1.12MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<08:00, 1.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<06:03, 1.53MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<04:22, 2.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:04, 1.52MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:19, 1.73MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:59, 2.31MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:43, 1.94MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:26, 1.68MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:15, 2.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<03:08, 2.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<04:55, 1.84MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:28, 2.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<03:20, 2.71MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:17, 2.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:00, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:00, 2.24MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<02:54, 3.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<11:06, 805kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<08:46, 1.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<06:20, 1.40MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<06:21, 1.40MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<06:31, 1.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:01, 1.76MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<03:35, 2.45MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<09:32, 923kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<07:39, 1.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<05:35, 1.57MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:48, 1.51MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<06:06, 1.43MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:42, 1.85MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<03:24, 2.55MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:50, 1.48MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:05, 1.70MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:48, 2.27MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:28, 1.92MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:02, 1.70MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:00, 2.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<02:53, 2.94MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<11:49, 721kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<09:16, 918kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<06:43, 1.26MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<06:28, 1.31MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<06:20, 1.33MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:53, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<03:30, 2.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<13:54, 603kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<10:42, 782kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<07:43, 1.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<07:08, 1.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<06:56, 1.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:15, 1.58MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:48, 2.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:19, 1.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:41, 1.76MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:30, 2.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:10, 1.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:40, 1.75MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:42, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:35<02:40, 3.03MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<12:58, 625kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<10:01, 808kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<07:12, 1.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<06:43, 1.19MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<06:25, 1.25MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:53, 1.64MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<03:30, 2.27MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<06:18, 1.26MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<05:21, 1.48MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:58, 2.00MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:26, 1.78MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:57, 1.59MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:51, 2.04MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:43<02:46, 2.82MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<07:31, 1.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<06:06, 1.28MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:26, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:50, 1.60MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:12, 1.49MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:05, 1.89MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<02:57, 2.60MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<08:32, 900kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<06:46, 1.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:14, 1.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:49, 2.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:33, 1.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:11, 1.82MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:07, 2.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:41, 2.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:27, 2.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<02:37, 2.86MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<03:25, 2.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<04:00, 1.86MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:09, 2.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:17, 3.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<05:08, 1.44MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:24, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:16, 2.25MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:57, 1.86MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:29, 1.64MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:33, 2.06MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<02:34, 2.83MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<07:26, 977kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<06:03, 1.20MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:24, 1.64MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:36, 1.56MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:45, 1.51MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:40, 1.96MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<02:38, 2.70MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<06:02, 1.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<05:01, 1.42MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:39, 1.94MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<04:08, 1.71MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<04:32, 1.55MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:31, 2.00MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:33, 2.74MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<04:18, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:48, 1.83MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<02:51, 2.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:27, 2.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:58, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:05, 2.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:17, 3.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:31, 1.95MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:11, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:13<02:23, 2.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:10, 2.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:48, 1.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<02:59, 2.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:10, 3.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<04:04, 1.65MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<03:37, 1.85MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:41, 2.48MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<01:58, 3.37MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<10:50, 614kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<08:21, 795kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<06:01, 1.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<05:35, 1.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:40, 1.41MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:27, 1.90MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:47, 1.72MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:10, 1.56MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:17, 1.97MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<02:22, 2.72MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<05:28, 1.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:34, 1.41MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:20, 1.92MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:41, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<04:05, 1.56MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:10, 2.01MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:19, 2.72MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<03:23, 1.86MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<03:06, 2.03MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:20, 2.67MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<02:57, 2.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<03:31, 1.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:49, 2.21MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<02:02, 3.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<06:35, 934kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<05:16, 1.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:50, 1.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:02, 1.51MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:15, 1.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:20, 1.82MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<02:23, 2.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<06:34, 916kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<05:17, 1.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:50, 1.56MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:55, 1.51MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<04:08, 1.43MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:15, 1.83MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<02:20, 2.51MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<06:30, 903kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<05:14, 1.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:48, 1.54MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:52, 1.50MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<04:04, 1.43MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:11, 1.82MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<02:17, 2.51MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<06:22, 902kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<05:07, 1.12MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:44, 1.53MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:48, 1.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:55, 1.44MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:01, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:11, 2.56MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:27, 1.62MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<03:05, 1.82MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:17, 2.44MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:46, 1.99MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<03:10, 1.74MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:30, 2.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<01:48, 3.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:19, 1.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:38, 1.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:40, 2.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:01, 1.78MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<02:41, 2.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:01, 2.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:35, 2.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:04, 1.73MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:27, 2.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<01:47, 2.95MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<05:39, 929kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<04:34, 1.15MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:20, 1.57MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:24, 1.52MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:59, 1.74MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:13, 2.31MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:38, 1.93MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:24, 2.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:47, 2.84MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:22, 2.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:51, 1.77MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:17, 2.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<01:38, 3.03MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<04:54, 1.01MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<04:01, 1.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:56, 1.68MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:05, 1.59MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:12, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:28, 1.97MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<01:46, 2.73MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<04:32, 1.07MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:44, 1.29MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:43, 1.76MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:54, 1.64MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<03:09, 1.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:29, 1.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<01:47, 2.63MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<05:08, 914kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<04:09, 1.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<03:01, 1.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<03:05, 1.50MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<03:08, 1.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:27, 1.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:45, 2.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<08:22, 545kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<06:21, 717kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<04:33, 997kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<04:09, 1.08MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<03:57, 1.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:58, 1.50MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<02:08, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:59, 1.48MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:19, 1.90MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<01:39, 2.62MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<04:48, 908kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<04:22, 994kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<03:17, 1.32MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:24<02:20, 1.83MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<03:26, 1.25MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:54, 1.47MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<02:09, 1.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:22, 1.77MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:39, 1.59MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:06, 2.00MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:28<01:30, 2.74MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<04:27, 930kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<03:34, 1.16MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<02:35, 1.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:42, 1.50MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:20, 1.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:44, 2.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:05, 1.91MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:24, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:51, 2.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:34<01:20, 2.94MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:50, 1.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:24, 1.63MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:46, 2.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:05, 1.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:20, 1.65MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:50, 2.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:38<01:19, 2.89MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<03:24, 1.11MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:49, 1.34MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<02:04, 1.82MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:14, 1.67MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:57, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:27, 2.53MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<01:50, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:08, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:42, 2.14MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<01:13, 2.92MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<04:23, 819kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<03:29, 1.03MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:31, 1.41MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<02:29, 1.41MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:34, 1.37MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:00, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<01:25, 2.42MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<03:52, 892kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<03:04, 1.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<02:13, 1.53MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:17, 1.47MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:00, 1.69MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:29, 2.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:43, 1.91MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:59, 1.66MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:35, 2.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:08, 2.86MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<03:28, 936kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:48, 1.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:02, 1.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:05, 1.52MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:12, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:43, 1.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<01:13, 2.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<03:26, 903kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:45, 1.12MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:59, 1.54MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<02:01, 1.50MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:44, 1.74MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:17, 2.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:33, 1.91MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:46, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:24, 2.09MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<01:00, 2.87MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<03:04, 944kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:29, 1.16MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:48, 1.59MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:50, 1.53MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:57, 1.44MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:30, 1.87MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:05, 2.57MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:44, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:31, 1.80MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:07, 2.42MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:21, 1.99MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:34, 1.70MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:15, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<00:54, 2.91MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:50, 925kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:16, 1.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:38, 1.57MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:40, 1.52MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:46, 1.44MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:22, 1.84MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<00:59, 2.52MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<02:44, 904kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:12, 1.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:36, 1.53MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:36, 1.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:38, 1.46MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:16, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<00:54, 2.59MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<04:12, 556kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<03:12, 727kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<02:17, 1.01MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:03, 1.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:58, 1.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:29, 1.52MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<01:03, 2.11MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:47, 1.23MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:30, 1.45MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:06, 1.96MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:12, 1.75MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:05, 1.95MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:49, 2.57MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<00:59, 2.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:10, 1.75MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:56, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:40, 2.97MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<02:07, 940kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:41, 1.17MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:14, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<00:52, 2.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<03:35, 535kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:57, 649kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:09, 885kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<01:30, 1.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:46, 1.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:27, 1.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:03, 1.72MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<00:45, 2.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<04:02, 442kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<03:01, 589kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<02:08, 821kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:49, 938kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:27, 1.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:03, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:05, 1.51MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:08, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:52, 1.85MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<00:37, 2.57MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:28, 1.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:12, 1.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:52, 1.76MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:55, 1.64MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:49, 1.84MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:35, 2.47MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:42, 2.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:50, 1.72MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:39, 2.14MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:28, 2.93MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:27, 937kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<01:10, 1.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:50, 1.60MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:51, 1.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:44, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:33, 2.31MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:38, 1.94MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:44, 1.64MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:35, 2.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:24, 2.82MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:14, 930kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:59, 1.16MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:42, 1.59MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:43, 1.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:44, 1.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:34, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:23, 2.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<02:42, 376kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<02:00, 506kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<01:24, 708kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<01:08, 833kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:54, 1.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:38, 1.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:37, 1.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:38, 1.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:29, 1.79MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:20, 2.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:37, 1.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:31, 1.53MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:22, 2.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:24, 1.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:27, 1.61MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:21, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:14, 2.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:43, 919kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:35, 1.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:25, 1.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:24, 1.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:25, 1.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:19, 1.82MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:12<00:13, 2.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:34, 918kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:27, 1.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:19, 1.57MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:18, 1.49MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:19, 1.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:14, 1.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:10, 2.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:13, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:12, 1.92MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:08, 2.56MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:05, 3.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:49, 399kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:39, 497kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:27, 681kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:16, 961kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:23, 645kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:17, 837kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:11, 1.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:09, 1.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:08, 1.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:06, 1.64MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:03, 2.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:05, 1.35MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:04, 1.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.13MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.85MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 1.63MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:00, 2.05MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 805/400000 [00:00<00:49, 8046.15it/s]  0%|          | 1556/400000 [00:00<00:50, 7874.76it/s]  1%|          | 2299/400000 [00:00<00:51, 7734.12it/s]  1%|          | 3056/400000 [00:00<00:51, 7681.27it/s]  1%|          | 3858/400000 [00:00<00:50, 7779.03it/s]  1%|          | 4678/400000 [00:00<00:50, 7899.17it/s]  1%|▏         | 5447/400000 [00:00<00:50, 7834.19it/s]  2%|▏         | 6202/400000 [00:00<00:50, 7745.87it/s]  2%|▏         | 6971/400000 [00:00<00:50, 7728.03it/s]  2%|▏         | 7742/400000 [00:01<00:50, 7721.28it/s]  2%|▏         | 8541/400000 [00:01<00:50, 7798.32it/s]  2%|▏         | 9313/400000 [00:01<00:50, 7774.28it/s]  3%|▎         | 10080/400000 [00:01<00:51, 7611.36it/s]  3%|▎         | 10843/400000 [00:01<00:51, 7615.41it/s]  3%|▎         | 11617/400000 [00:01<00:50, 7650.55it/s]  3%|▎         | 12396/400000 [00:01<00:50, 7690.67it/s]  3%|▎         | 13190/400000 [00:01<00:49, 7761.77it/s]  3%|▎         | 13965/400000 [00:01<00:50, 7573.36it/s]  4%|▎         | 14759/400000 [00:01<00:50, 7677.55it/s]  4%|▍         | 15528/400000 [00:02<00:50, 7656.19it/s]  4%|▍         | 16335/400000 [00:02<00:49, 7773.61it/s]  4%|▍         | 17139/400000 [00:02<00:48, 7851.19it/s]  4%|▍         | 17925/400000 [00:02<00:48, 7849.53it/s]  5%|▍         | 18745/400000 [00:02<00:47, 7951.41it/s]  5%|▍         | 19550/400000 [00:02<00:47, 7980.42it/s]  5%|▌         | 20350/400000 [00:02<00:47, 7984.65it/s]  5%|▌         | 21149/400000 [00:02<00:47, 7977.27it/s]  5%|▌         | 21947/400000 [00:02<00:48, 7836.31it/s]  6%|▌         | 22736/400000 [00:02<00:48, 7850.45it/s]  6%|▌         | 23525/400000 [00:03<00:47, 7860.10it/s]  6%|▌         | 24351/400000 [00:03<00:47, 7975.67it/s]  6%|▋         | 25150/400000 [00:03<00:47, 7826.60it/s]  6%|▋         | 25934/400000 [00:03<00:48, 7712.74it/s]  7%|▋         | 26758/400000 [00:03<00:47, 7860.99it/s]  7%|▋         | 27546/400000 [00:03<00:47, 7778.76it/s]  7%|▋         | 28326/400000 [00:03<00:48, 7653.97it/s]  7%|▋         | 29101/400000 [00:03<00:48, 7680.29it/s]  7%|▋         | 29870/400000 [00:03<00:48, 7614.46it/s]  8%|▊         | 30665/400000 [00:03<00:47, 7710.07it/s]  8%|▊         | 31445/400000 [00:04<00:47, 7734.70it/s]  8%|▊         | 32235/400000 [00:04<00:47, 7780.87it/s]  8%|▊         | 33014/400000 [00:04<00:47, 7744.85it/s]  8%|▊         | 33789/400000 [00:04<00:48, 7558.50it/s]  9%|▊         | 34581/400000 [00:04<00:47, 7662.24it/s]  9%|▉         | 35355/400000 [00:04<00:47, 7684.78it/s]  9%|▉         | 36154/400000 [00:04<00:46, 7771.96it/s]  9%|▉         | 36964/400000 [00:04<00:46, 7867.17it/s]  9%|▉         | 37752/400000 [00:04<00:46, 7779.61it/s] 10%|▉         | 38532/400000 [00:04<00:46, 7780.12it/s] 10%|▉         | 39311/400000 [00:05<00:46, 7700.43it/s] 10%|█         | 40082/400000 [00:05<00:46, 7669.01it/s] 10%|█         | 40850/400000 [00:05<00:46, 7657.92it/s] 10%|█         | 41617/400000 [00:05<00:47, 7509.83it/s] 11%|█         | 42416/400000 [00:05<00:46, 7646.07it/s] 11%|█         | 43182/400000 [00:05<00:47, 7563.74it/s] 11%|█         | 43940/400000 [00:05<00:47, 7422.19it/s] 11%|█         | 44684/400000 [00:05<00:48, 7365.02it/s] 11%|█▏        | 45463/400000 [00:05<00:47, 7487.38it/s] 12%|█▏        | 46228/400000 [00:05<00:46, 7534.22it/s] 12%|█▏        | 47013/400000 [00:06<00:46, 7624.31it/s] 12%|█▏        | 47777/400000 [00:06<00:46, 7600.84it/s] 12%|█▏        | 48550/400000 [00:06<00:46, 7638.84it/s] 12%|█▏        | 49337/400000 [00:06<00:45, 7704.72it/s] 13%|█▎        | 50159/400000 [00:06<00:44, 7850.74it/s] 13%|█▎        | 50972/400000 [00:06<00:44, 7932.45it/s] 13%|█▎        | 51767/400000 [00:06<00:44, 7857.94it/s] 13%|█▎        | 52572/400000 [00:06<00:43, 7913.54it/s] 13%|█▎        | 53365/400000 [00:06<00:44, 7853.93it/s] 14%|█▎        | 54151/400000 [00:06<00:44, 7770.84it/s] 14%|█▎        | 54929/400000 [00:07<00:46, 7407.18it/s] 14%|█▍        | 55674/400000 [00:07<00:48, 7130.68it/s] 14%|█▍        | 56393/400000 [00:07<00:48, 7039.39it/s] 14%|█▍        | 57101/400000 [00:07<00:49, 6966.98it/s] 14%|█▍        | 57801/400000 [00:07<00:49, 6866.17it/s] 15%|█▍        | 58567/400000 [00:07<00:48, 7086.41it/s] 15%|█▍        | 59338/400000 [00:07<00:46, 7262.12it/s] 15%|█▌        | 60079/400000 [00:07<00:46, 7304.09it/s] 15%|█▌        | 60845/400000 [00:07<00:45, 7404.75it/s] 15%|█▌        | 61631/400000 [00:08<00:44, 7534.84it/s] 16%|█▌        | 62405/400000 [00:08<00:44, 7594.33it/s] 16%|█▌        | 63166/400000 [00:08<00:44, 7595.46it/s] 16%|█▌        | 63927/400000 [00:08<00:44, 7575.51it/s] 16%|█▌        | 64717/400000 [00:08<00:43, 7666.85it/s] 16%|█▋        | 65486/400000 [00:08<00:43, 7673.71it/s] 17%|█▋        | 66273/400000 [00:08<00:43, 7730.46it/s] 17%|█▋        | 67047/400000 [00:08<00:45, 7350.24it/s] 17%|█▋        | 67833/400000 [00:08<00:44, 7494.19it/s] 17%|█▋        | 68617/400000 [00:08<00:43, 7592.80it/s] 17%|█▋        | 69422/400000 [00:09<00:42, 7723.17it/s] 18%|█▊        | 70211/400000 [00:09<00:42, 7770.53it/s] 18%|█▊        | 70990/400000 [00:09<00:42, 7729.46it/s] 18%|█▊        | 71765/400000 [00:09<00:42, 7646.61it/s] 18%|█▊        | 72567/400000 [00:09<00:42, 7752.84it/s] 18%|█▊        | 73344/400000 [00:09<00:42, 7742.01it/s] 19%|█▊        | 74120/400000 [00:09<00:42, 7732.63it/s] 19%|█▊        | 74894/400000 [00:09<00:42, 7690.28it/s] 19%|█▉        | 75664/400000 [00:09<00:42, 7582.54it/s] 19%|█▉        | 76439/400000 [00:09<00:42, 7628.37it/s] 19%|█▉        | 77203/400000 [00:10<00:42, 7607.31it/s] 19%|█▉        | 77965/400000 [00:10<00:42, 7608.81it/s] 20%|█▉        | 78727/400000 [00:10<00:44, 7176.44it/s] 20%|█▉        | 79450/400000 [00:10<00:45, 7108.19it/s] 20%|██        | 80198/400000 [00:10<00:44, 7215.59it/s] 20%|██        | 80973/400000 [00:10<00:43, 7366.43it/s] 20%|██        | 81747/400000 [00:10<00:42, 7473.22it/s] 21%|██        | 82567/400000 [00:10<00:41, 7677.34it/s] 21%|██        | 83338/400000 [00:10<00:41, 7613.12it/s] 21%|██        | 84117/400000 [00:11<00:41, 7663.36it/s] 21%|██        | 84895/400000 [00:11<00:40, 7696.46it/s] 21%|██▏       | 85669/400000 [00:11<00:40, 7709.34it/s] 22%|██▏       | 86441/400000 [00:11<00:41, 7595.21it/s] 22%|██▏       | 87202/400000 [00:11<00:41, 7577.45it/s] 22%|██▏       | 87993/400000 [00:11<00:40, 7673.72it/s] 22%|██▏       | 88808/400000 [00:11<00:39, 7810.14it/s] 22%|██▏       | 89647/400000 [00:11<00:38, 7973.84it/s] 23%|██▎       | 90447/400000 [00:11<00:40, 7702.51it/s] 23%|██▎       | 91221/400000 [00:11<00:41, 7479.20it/s] 23%|██▎       | 91973/400000 [00:12<00:42, 7312.25it/s] 23%|██▎       | 92728/400000 [00:12<00:41, 7381.95it/s] 23%|██▎       | 93506/400000 [00:12<00:40, 7494.50it/s] 24%|██▎       | 94258/400000 [00:12<00:40, 7480.33it/s] 24%|██▍       | 95039/400000 [00:12<00:40, 7575.35it/s] 24%|██▍       | 95850/400000 [00:12<00:39, 7725.70it/s] 24%|██▍       | 96649/400000 [00:12<00:38, 7800.76it/s] 24%|██▍       | 97431/400000 [00:12<00:38, 7800.57it/s] 25%|██▍       | 98213/400000 [00:12<00:38, 7752.00it/s] 25%|██▍       | 98989/400000 [00:12<00:39, 7708.72it/s] 25%|██▍       | 99796/400000 [00:13<00:38, 7811.21it/s] 25%|██▌       | 100578/400000 [00:13<00:38, 7813.43it/s] 25%|██▌       | 101360/400000 [00:13<00:38, 7774.75it/s] 26%|██▌       | 102142/400000 [00:13<00:38, 7787.28it/s] 26%|██▌       | 102922/400000 [00:13<00:38, 7724.87it/s] 26%|██▌       | 103695/400000 [00:13<00:39, 7498.91it/s] 26%|██▌       | 104447/400000 [00:13<00:39, 7497.82it/s] 26%|██▋       | 105231/400000 [00:13<00:38, 7595.54it/s] 27%|██▋       | 106020/400000 [00:13<00:38, 7678.83it/s] 27%|██▋       | 106789/400000 [00:13<00:39, 7472.25it/s] 27%|██▋       | 107597/400000 [00:14<00:38, 7643.40it/s] 27%|██▋       | 108421/400000 [00:14<00:37, 7811.54it/s] 27%|██▋       | 109219/400000 [00:14<00:36, 7861.12it/s] 28%|██▊       | 110007/400000 [00:14<00:36, 7838.16it/s] 28%|██▊       | 110793/400000 [00:14<00:37, 7799.66it/s] 28%|██▊       | 111603/400000 [00:14<00:36, 7885.68it/s] 28%|██▊       | 112393/400000 [00:14<00:37, 7735.90it/s] 28%|██▊       | 113203/400000 [00:14<00:36, 7839.44it/s] 28%|██▊       | 113989/400000 [00:14<00:36, 7837.98it/s] 29%|██▊       | 114774/400000 [00:14<00:36, 7774.54it/s] 29%|██▉       | 115597/400000 [00:15<00:35, 7904.04it/s] 29%|██▉       | 116411/400000 [00:15<00:35, 7971.30it/s] 29%|██▉       | 117226/400000 [00:15<00:35, 8022.55it/s] 30%|██▉       | 118029/400000 [00:15<00:35, 7955.05it/s] 30%|██▉       | 118826/400000 [00:15<00:36, 7659.71it/s] 30%|██▉       | 119615/400000 [00:15<00:36, 7726.42it/s] 30%|███       | 120413/400000 [00:15<00:35, 7800.13it/s] 30%|███       | 121234/400000 [00:15<00:35, 7917.21it/s] 31%|███       | 122034/400000 [00:15<00:35, 7940.96it/s] 31%|███       | 122830/400000 [00:16<00:35, 7858.28it/s] 31%|███       | 123617/400000 [00:16<00:35, 7847.09it/s] 31%|███       | 124403/400000 [00:16<00:35, 7836.17it/s] 31%|███▏      | 125208/400000 [00:16<00:34, 7898.57it/s] 32%|███▏      | 126039/400000 [00:16<00:34, 8017.23it/s] 32%|███▏      | 126842/400000 [00:16<00:34, 7919.04it/s] 32%|███▏      | 127660/400000 [00:16<00:34, 7992.86it/s] 32%|███▏      | 128461/400000 [00:16<00:34, 7911.13it/s] 32%|███▏      | 129274/400000 [00:16<00:33, 7972.59it/s] 33%|███▎      | 130072/400000 [00:16<00:34, 7927.89it/s] 33%|███▎      | 130866/400000 [00:17<00:35, 7580.96it/s] 33%|███▎      | 131629/400000 [00:17<00:35, 7592.24it/s] 33%|███▎      | 132396/400000 [00:17<00:35, 7613.83it/s] 33%|███▎      | 133175/400000 [00:17<00:34, 7663.15it/s] 33%|███▎      | 133949/400000 [00:17<00:34, 7684.58it/s] 34%|███▎      | 134719/400000 [00:17<00:34, 7590.48it/s] 34%|███▍      | 135500/400000 [00:17<00:34, 7653.83it/s] 34%|███▍      | 136337/400000 [00:17<00:33, 7852.81it/s] 34%|███▍      | 137142/400000 [00:17<00:33, 7909.44it/s] 34%|███▍      | 137965/400000 [00:17<00:32, 8001.20it/s] 35%|███▍      | 138767/400000 [00:18<00:32, 7923.54it/s] 35%|███▍      | 139580/400000 [00:18<00:32, 7982.57it/s] 35%|███▌      | 140380/400000 [00:18<00:32, 7944.85it/s] 35%|███▌      | 141207/400000 [00:18<00:32, 8037.58it/s] 36%|███▌      | 142038/400000 [00:18<00:31, 8115.35it/s] 36%|███▌      | 142851/400000 [00:18<00:32, 7972.54it/s] 36%|███▌      | 143650/400000 [00:18<00:32, 7888.57it/s] 36%|███▌      | 144440/400000 [00:18<00:32, 7817.36it/s] 36%|███▋      | 145223/400000 [00:18<00:32, 7798.48it/s] 37%|███▋      | 146036/400000 [00:18<00:32, 7894.89it/s] 37%|███▋      | 146827/400000 [00:19<00:32, 7800.92it/s] 37%|███▋      | 147608/400000 [00:19<00:32, 7772.51it/s] 37%|███▋      | 148400/400000 [00:19<00:32, 7813.90it/s] 37%|███▋      | 149198/400000 [00:19<00:31, 7862.17it/s] 37%|███▋      | 149996/400000 [00:19<00:31, 7896.34it/s] 38%|███▊      | 150786/400000 [00:19<00:31, 7861.02it/s] 38%|███▊      | 151580/400000 [00:19<00:31, 7882.54it/s] 38%|███▊      | 152393/400000 [00:19<00:31, 7954.21it/s] 38%|███▊      | 153201/400000 [00:19<00:30, 7988.98it/s] 39%|███▊      | 154006/400000 [00:19<00:30, 8005.74it/s] 39%|███▊      | 154807/400000 [00:20<00:31, 7837.73it/s] 39%|███▉      | 155640/400000 [00:20<00:30, 7977.80it/s] 39%|███▉      | 156440/400000 [00:20<00:30, 7960.84it/s] 39%|███▉      | 157272/400000 [00:20<00:30, 8065.15it/s] 40%|███▉      | 158080/400000 [00:20<00:30, 8045.73it/s] 40%|███▉      | 158886/400000 [00:20<00:30, 7916.89it/s] 40%|███▉      | 159679/400000 [00:20<00:30, 7913.28it/s] 40%|████      | 160480/400000 [00:20<00:30, 7940.01it/s] 40%|████      | 161312/400000 [00:20<00:29, 8049.45it/s] 41%|████      | 162118/400000 [00:20<00:29, 8019.99it/s] 41%|████      | 162921/400000 [00:21<00:29, 7927.95it/s] 41%|████      | 163715/400000 [00:21<00:29, 7899.52it/s] 41%|████      | 164514/400000 [00:21<00:29, 7924.60it/s] 41%|████▏     | 165334/400000 [00:21<00:29, 8004.21it/s] 42%|████▏     | 166159/400000 [00:21<00:28, 8074.52it/s] 42%|████▏     | 166967/400000 [00:21<00:28, 8062.41it/s] 42%|████▏     | 167774/400000 [00:21<00:29, 7975.20it/s] 42%|████▏     | 168572/400000 [00:21<00:30, 7701.91it/s] 42%|████▏     | 169345/400000 [00:21<00:30, 7637.43it/s] 43%|████▎     | 170111/400000 [00:22<00:30, 7524.76it/s] 43%|████▎     | 170866/400000 [00:22<00:30, 7468.74it/s] 43%|████▎     | 171615/400000 [00:22<00:31, 7193.01it/s] 43%|████▎     | 172396/400000 [00:22<00:30, 7366.29it/s] 43%|████▎     | 173178/400000 [00:22<00:30, 7491.51it/s] 43%|████▎     | 173961/400000 [00:22<00:29, 7586.58it/s] 44%|████▎     | 174722/400000 [00:22<00:30, 7477.30it/s] 44%|████▍     | 175500/400000 [00:22<00:29, 7564.47it/s] 44%|████▍     | 176264/400000 [00:22<00:29, 7585.60it/s] 44%|████▍     | 177024/400000 [00:22<00:29, 7547.52it/s] 44%|████▍     | 177788/400000 [00:23<00:29, 7573.39it/s] 45%|████▍     | 178546/400000 [00:23<00:29, 7480.49it/s] 45%|████▍     | 179295/400000 [00:23<00:29, 7482.56it/s] 45%|████▌     | 180065/400000 [00:23<00:29, 7544.09it/s] 45%|████▌     | 180839/400000 [00:23<00:28, 7600.68it/s] 45%|████▌     | 181600/400000 [00:23<00:28, 7551.17it/s] 46%|████▌     | 182356/400000 [00:23<00:29, 7490.11it/s] 46%|████▌     | 183161/400000 [00:23<00:28, 7649.20it/s] 46%|████▌     | 183945/400000 [00:23<00:28, 7703.62it/s] 46%|████▌     | 184727/400000 [00:23<00:27, 7737.07it/s] 46%|████▋     | 185534/400000 [00:24<00:27, 7832.87it/s] 47%|████▋     | 186319/400000 [00:24<00:27, 7783.87it/s] 47%|████▋     | 187114/400000 [00:24<00:27, 7831.53it/s] 47%|████▋     | 187900/400000 [00:24<00:27, 7837.78it/s] 47%|████▋     | 188703/400000 [00:24<00:26, 7892.22it/s] 47%|████▋     | 189535/400000 [00:24<00:26, 8012.07it/s] 48%|████▊     | 190337/400000 [00:24<00:26, 7913.94it/s] 48%|████▊     | 191146/400000 [00:24<00:26, 7964.50it/s] 48%|████▊     | 191944/400000 [00:24<00:26, 7866.76it/s] 48%|████▊     | 192756/400000 [00:24<00:26, 7938.56it/s] 48%|████▊     | 193571/400000 [00:25<00:25, 7998.46it/s] 49%|████▊     | 194372/400000 [00:25<00:26, 7839.36it/s] 49%|████▉     | 195167/400000 [00:25<00:26, 7869.77it/s] 49%|████▉     | 195968/400000 [00:25<00:25, 7911.15it/s] 49%|████▉     | 196760/400000 [00:25<00:25, 7845.89it/s] 49%|████▉     | 197546/400000 [00:25<00:26, 7770.04it/s] 50%|████▉     | 198339/400000 [00:25<00:25, 7816.19it/s] 50%|████▉     | 199126/400000 [00:25<00:25, 7831.10it/s] 50%|████▉     | 199910/400000 [00:25<00:25, 7727.85it/s] 50%|█████     | 200684/400000 [00:25<00:25, 7696.85it/s] 50%|█████     | 201495/400000 [00:26<00:25, 7815.61it/s] 51%|█████     | 202286/400000 [00:26<00:25, 7842.77it/s] 51%|█████     | 203081/400000 [00:26<00:25, 7874.47it/s] 51%|█████     | 203882/400000 [00:26<00:24, 7912.50it/s] 51%|█████     | 204674/400000 [00:26<00:24, 7892.01it/s] 51%|█████▏    | 205516/400000 [00:26<00:24, 8041.36it/s] 52%|█████▏    | 206321/400000 [00:26<00:24, 7953.90it/s] 52%|█████▏    | 207128/400000 [00:26<00:24, 7986.85it/s] 52%|█████▏    | 207951/400000 [00:26<00:23, 8054.30it/s] 52%|█████▏    | 208769/400000 [00:26<00:23, 8088.40it/s] 52%|█████▏    | 209579/400000 [00:27<00:23, 7958.57it/s] 53%|█████▎    | 210376/400000 [00:27<00:24, 7785.41it/s] 53%|█████▎    | 211176/400000 [00:27<00:24, 7847.19it/s] 53%|█████▎    | 211967/400000 [00:27<00:23, 7864.34it/s] 53%|█████▎    | 212755/400000 [00:27<00:23, 7809.48it/s] 53%|█████▎    | 213539/400000 [00:27<00:23, 7818.41it/s] 54%|█████▎    | 214322/400000 [00:27<00:23, 7755.37it/s] 54%|█████▍    | 215126/400000 [00:27<00:23, 7837.91it/s] 54%|█████▍    | 215921/400000 [00:27<00:23, 7871.00it/s] 54%|█████▍    | 216735/400000 [00:28<00:23, 7949.15it/s] 54%|█████▍    | 217557/400000 [00:28<00:22, 8027.83it/s] 55%|█████▍    | 218361/400000 [00:28<00:22, 7947.60it/s] 55%|█████▍    | 219157/400000 [00:28<00:22, 7919.35it/s] 55%|█████▍    | 219978/400000 [00:28<00:22, 8003.87it/s] 55%|█████▌    | 220805/400000 [00:28<00:22, 8081.02it/s] 55%|█████▌    | 221614/400000 [00:28<00:22, 7987.91it/s] 56%|█████▌    | 222414/400000 [00:28<00:22, 7874.34it/s] 56%|█████▌    | 223203/400000 [00:28<00:22, 7690.48it/s] 56%|█████▌    | 223999/400000 [00:28<00:22, 7767.76it/s] 56%|█████▌    | 224805/400000 [00:29<00:22, 7852.31it/s] 56%|█████▋    | 225618/400000 [00:29<00:21, 7932.81it/s] 57%|█████▋    | 226413/400000 [00:29<00:22, 7743.01it/s] 57%|█████▋    | 227230/400000 [00:29<00:21, 7864.14it/s] 57%|█████▋    | 228037/400000 [00:29<00:21, 7923.99it/s] 57%|█████▋    | 228872/400000 [00:29<00:21, 8045.84it/s] 57%|█████▋    | 229678/400000 [00:29<00:21, 7930.86it/s] 58%|█████▊    | 230473/400000 [00:29<00:21, 7845.97it/s] 58%|█████▊    | 231259/400000 [00:29<00:21, 7840.16it/s] 58%|█████▊    | 232068/400000 [00:29<00:21, 7913.46it/s] 58%|█████▊    | 232916/400000 [00:30<00:20, 8073.01it/s] 58%|█████▊    | 233725/400000 [00:30<00:20, 8052.60it/s] 59%|█████▊    | 234532/400000 [00:30<00:21, 7854.94it/s] 59%|█████▉    | 235363/400000 [00:30<00:20, 7985.35it/s] 59%|█████▉    | 236164/400000 [00:30<00:20, 7808.70it/s] 59%|█████▉    | 236971/400000 [00:30<00:20, 7884.88it/s] 59%|█████▉    | 237820/400000 [00:30<00:20, 8053.44it/s] 60%|█████▉    | 238628/400000 [00:30<00:20, 7913.03it/s] 60%|█████▉    | 239457/400000 [00:30<00:20, 8021.71it/s] 60%|██████    | 240261/400000 [00:30<00:20, 7896.26it/s] 60%|██████    | 241053/400000 [00:31<00:20, 7782.70it/s] 60%|██████    | 241854/400000 [00:31<00:20, 7847.33it/s] 61%|██████    | 242640/400000 [00:31<00:20, 7771.20it/s] 61%|██████    | 243443/400000 [00:31<00:19, 7847.06it/s] 61%|██████    | 244229/400000 [00:31<00:20, 7780.63it/s] 61%|██████▏   | 245055/400000 [00:31<00:19, 7915.88it/s] 61%|██████▏   | 245848/400000 [00:31<00:19, 7885.08it/s] 62%|██████▏   | 246639/400000 [00:31<00:19, 7890.73it/s] 62%|██████▏   | 247460/400000 [00:31<00:19, 7982.77it/s] 62%|██████▏   | 248265/400000 [00:31<00:18, 8000.31it/s] 62%|██████▏   | 249066/400000 [00:32<00:19, 7886.38it/s] 62%|██████▏   | 249856/400000 [00:32<00:19, 7598.64it/s] 63%|██████▎   | 250635/400000 [00:32<00:19, 7654.77it/s] 63%|██████▎   | 251482/400000 [00:32<00:18, 7880.70it/s] 63%|██████▎   | 252299/400000 [00:32<00:18, 7965.00it/s] 63%|██████▎   | 253098/400000 [00:32<00:18, 7967.98it/s] 63%|██████▎   | 253897/400000 [00:32<00:18, 7884.46it/s] 64%|██████▎   | 254687/400000 [00:32<00:18, 7872.09it/s] 64%|██████▍   | 255496/400000 [00:32<00:18, 7936.17it/s] 64%|██████▍   | 256313/400000 [00:33<00:17, 8003.74it/s] 64%|██████▍   | 257140/400000 [00:33<00:17, 8079.75it/s] 64%|██████▍   | 257949/400000 [00:33<00:17, 7944.92it/s] 65%|██████▍   | 258745/400000 [00:33<00:18, 7653.89it/s] 65%|██████▍   | 259514/400000 [00:33<00:18, 7622.50it/s] 65%|██████▌   | 260279/400000 [00:33<00:18, 7533.98it/s] 65%|██████▌   | 261035/400000 [00:33<00:18, 7467.53it/s] 65%|██████▌   | 261784/400000 [00:33<00:18, 7382.87it/s] 66%|██████▌   | 262538/400000 [00:33<00:18, 7427.87it/s] 66%|██████▌   | 263332/400000 [00:33<00:18, 7574.30it/s] 66%|██████▌   | 264124/400000 [00:34<00:17, 7673.60it/s] 66%|██████▌   | 264953/400000 [00:34<00:17, 7845.35it/s] 66%|██████▋   | 265740/400000 [00:34<00:17, 7815.81it/s] 67%|██████▋   | 266532/400000 [00:34<00:17, 7845.11it/s] 67%|██████▋   | 267327/400000 [00:34<00:16, 7876.15it/s] 67%|██████▋   | 268122/400000 [00:34<00:16, 7897.64it/s] 67%|██████▋   | 268979/400000 [00:34<00:16, 8086.53it/s] 67%|██████▋   | 269790/400000 [00:34<00:16, 8058.90it/s] 68%|██████▊   | 270597/400000 [00:34<00:16, 8003.46it/s] 68%|██████▊   | 271399/400000 [00:34<00:16, 7965.03it/s] 68%|██████▊   | 272223/400000 [00:35<00:15, 8044.42it/s] 68%|██████▊   | 273029/400000 [00:35<00:15, 8018.94it/s] 68%|██████▊   | 273841/400000 [00:35<00:15, 8046.76it/s] 69%|██████▊   | 274652/400000 [00:35<00:15, 8064.59it/s] 69%|██████▉   | 275459/400000 [00:35<00:15, 7963.95it/s] 69%|██████▉   | 276258/400000 [00:35<00:15, 7970.37it/s] 69%|██████▉   | 277118/400000 [00:35<00:15, 8147.25it/s] 69%|██████▉   | 277934/400000 [00:35<00:15, 8090.42it/s] 70%|██████▉   | 278744/400000 [00:35<00:15, 8065.26it/s] 70%|██████▉   | 279583/400000 [00:35<00:14, 8158.72it/s] 70%|███████   | 280419/400000 [00:36<00:14, 8215.89it/s] 70%|███████   | 281242/400000 [00:36<00:14, 8122.05it/s] 71%|███████   | 282055/400000 [00:36<00:14, 7962.26it/s] 71%|███████   | 282853/400000 [00:36<00:14, 7959.01it/s] 71%|███████   | 283650/400000 [00:36<00:14, 7908.38it/s] 71%|███████   | 284442/400000 [00:36<00:14, 7849.76it/s] 71%|███████▏  | 285274/400000 [00:36<00:14, 7982.68it/s] 72%|███████▏  | 286074/400000 [00:36<00:14, 7938.74it/s] 72%|███████▏  | 286897/400000 [00:36<00:14, 8023.03it/s] 72%|███████▏  | 287716/400000 [00:36<00:13, 8070.00it/s] 72%|███████▏  | 288524/400000 [00:37<00:13, 8050.12it/s] 72%|███████▏  | 289330/400000 [00:37<00:13, 8036.62it/s] 73%|███████▎  | 290135/400000 [00:37<00:13, 8040.11it/s] 73%|███████▎  | 290940/400000 [00:37<00:13, 7992.59it/s] 73%|███████▎  | 291741/400000 [00:37<00:13, 7995.12it/s] 73%|███████▎  | 292583/400000 [00:37<00:13, 8117.58it/s] 73%|███████▎  | 293396/400000 [00:37<00:13, 8053.61it/s] 74%|███████▎  | 294240/400000 [00:37<00:12, 8163.61it/s] 74%|███████▍  | 295058/400000 [00:37<00:13, 8020.81it/s] 74%|███████▍  | 295862/400000 [00:37<00:13, 7995.21it/s] 74%|███████▍  | 296663/400000 [00:38<00:12, 7972.93it/s] 74%|███████▍  | 297461/400000 [00:38<00:12, 7958.15it/s] 75%|███████▍  | 298258/400000 [00:38<00:12, 7882.13it/s] 75%|███████▍  | 299047/400000 [00:38<00:12, 7798.98it/s] 75%|███████▍  | 299869/400000 [00:38<00:12, 7919.00it/s] 75%|███████▌  | 300674/400000 [00:38<00:12, 7956.21it/s] 75%|███████▌  | 301489/400000 [00:38<00:12, 8010.24it/s] 76%|███████▌  | 302291/400000 [00:38<00:12, 7965.99it/s] 76%|███████▌  | 303089/400000 [00:38<00:12, 7817.26it/s] 76%|███████▌  | 303918/400000 [00:39<00:12, 7950.10it/s] 76%|███████▌  | 304715/400000 [00:39<00:12, 7908.17it/s] 76%|███████▋  | 305507/400000 [00:39<00:12, 7872.62it/s] 77%|███████▋  | 306313/400000 [00:39<00:11, 7925.61it/s] 77%|███████▋  | 307107/400000 [00:39<00:11, 7890.29it/s] 77%|███████▋  | 307930/400000 [00:39<00:11, 7988.85it/s] 77%|███████▋  | 308795/400000 [00:39<00:11, 8175.47it/s] 77%|███████▋  | 309640/400000 [00:39<00:10, 8254.69it/s] 78%|███████▊  | 310491/400000 [00:39<00:10, 8328.36it/s] 78%|███████▊  | 311325/400000 [00:39<00:10, 8156.50it/s] 78%|███████▊  | 312143/400000 [00:40<00:10, 8143.36it/s] 78%|███████▊  | 312961/400000 [00:40<00:10, 8151.67it/s] 78%|███████▊  | 313777/400000 [00:40<00:10, 8148.15it/s] 79%|███████▊  | 314593/400000 [00:40<00:10, 8138.58it/s] 79%|███████▉  | 315408/400000 [00:40<00:10, 7843.27it/s] 79%|███████▉  | 316195/400000 [00:40<00:10, 7780.17it/s] 79%|███████▉  | 316985/400000 [00:40<00:10, 7815.33it/s] 79%|███████▉  | 317819/400000 [00:40<00:10, 7963.63it/s] 80%|███████▉  | 318618/400000 [00:40<00:10, 7942.06it/s] 80%|███████▉  | 319414/400000 [00:40<00:10, 7863.03it/s] 80%|████████  | 320221/400000 [00:41<00:10, 7921.65it/s] 80%|████████  | 321043/400000 [00:41<00:09, 8006.07it/s] 80%|████████  | 321845/400000 [00:41<00:09, 8009.31it/s] 81%|████████  | 322647/400000 [00:41<00:09, 7904.38it/s] 81%|████████  | 323439/400000 [00:41<00:09, 7737.38it/s] 81%|████████  | 324258/400000 [00:41<00:09, 7867.77it/s] 81%|████████▏ | 325095/400000 [00:41<00:09, 8010.32it/s] 81%|████████▏ | 325927/400000 [00:41<00:09, 8097.99it/s] 82%|████████▏ | 326749/400000 [00:41<00:09, 8133.98it/s] 82%|████████▏ | 327564/400000 [00:41<00:09, 8006.53it/s] 82%|████████▏ | 328393/400000 [00:42<00:08, 8088.86it/s] 82%|████████▏ | 329203/400000 [00:42<00:08, 8031.21it/s] 83%|████████▎ | 330007/400000 [00:42<00:08, 7987.02it/s] 83%|████████▎ | 330820/400000 [00:42<00:08, 8028.27it/s] 83%|████████▎ | 331624/400000 [00:42<00:08, 7881.06it/s] 83%|████████▎ | 332432/400000 [00:42<00:08, 7939.47it/s] 83%|████████▎ | 333262/400000 [00:42<00:08, 8042.67it/s] 84%|████████▎ | 334068/400000 [00:42<00:08, 7984.55it/s] 84%|████████▎ | 334892/400000 [00:42<00:08, 8058.32it/s] 84%|████████▍ | 335699/400000 [00:42<00:08, 7877.94it/s] 84%|████████▍ | 336508/400000 [00:43<00:07, 7937.35it/s] 84%|████████▍ | 337303/400000 [00:43<00:07, 7930.53it/s] 85%|████████▍ | 338097/400000 [00:43<00:07, 7856.72it/s] 85%|████████▍ | 338885/400000 [00:43<00:07, 7862.06it/s] 85%|████████▍ | 339672/400000 [00:43<00:07, 7790.58it/s] 85%|████████▌ | 340452/400000 [00:43<00:07, 7480.46it/s] 85%|████████▌ | 341204/400000 [00:43<00:07, 7423.55it/s] 85%|████████▌ | 341949/400000 [00:43<00:07, 7322.13it/s] 86%|████████▌ | 342684/400000 [00:43<00:07, 7205.75it/s] 86%|████████▌ | 343442/400000 [00:44<00:07, 7312.35it/s] 86%|████████▌ | 344228/400000 [00:44<00:07, 7467.36it/s] 86%|████████▋ | 345038/400000 [00:44<00:07, 7645.84it/s] 86%|████████▋ | 345832/400000 [00:44<00:07, 7729.48it/s] 87%|████████▋ | 346610/400000 [00:44<00:06, 7741.62it/s] 87%|████████▋ | 347386/400000 [00:44<00:06, 7734.55it/s] 87%|████████▋ | 348186/400000 [00:44<00:06, 7811.34it/s] 87%|████████▋ | 348968/400000 [00:44<00:06, 7742.64it/s] 87%|████████▋ | 349743/400000 [00:44<00:06, 7739.34it/s] 88%|████████▊ | 350518/400000 [00:44<00:06, 7628.31it/s] 88%|████████▊ | 351282/400000 [00:45<00:06, 7616.09it/s] 88%|████████▊ | 352053/400000 [00:45<00:06, 7641.52it/s] 88%|████████▊ | 352818/400000 [00:45<00:06, 7568.87it/s] 88%|████████▊ | 353576/400000 [00:45<00:06, 7379.26it/s] 89%|████████▊ | 354316/400000 [00:45<00:06, 7303.73it/s] 89%|████████▉ | 355050/400000 [00:45<00:06, 7313.33it/s] 89%|████████▉ | 355791/400000 [00:45<00:06, 7339.43it/s] 89%|████████▉ | 356564/400000 [00:45<00:05, 7450.49it/s] 89%|████████▉ | 357314/400000 [00:45<00:05, 7463.60it/s] 90%|████████▉ | 358061/400000 [00:45<00:05, 7428.36it/s] 90%|████████▉ | 358807/400000 [00:46<00:05, 7435.01it/s] 90%|████████▉ | 359560/400000 [00:46<00:05, 7460.47it/s] 90%|█████████ | 360307/400000 [00:46<00:05, 7448.63it/s] 90%|█████████ | 361053/400000 [00:46<00:05, 7448.39it/s] 90%|█████████ | 361798/400000 [00:46<00:05, 7440.36it/s] 91%|█████████ | 362547/400000 [00:46<00:05, 7453.23it/s] 91%|█████████ | 363293/400000 [00:46<00:04, 7413.49it/s] 91%|█████████ | 364035/400000 [00:46<00:04, 7352.40it/s] 91%|█████████ | 364785/400000 [00:46<00:04, 7393.82it/s] 91%|█████████▏| 365525/400000 [00:46<00:04, 7369.04it/s] 92%|█████████▏| 366270/400000 [00:47<00:04, 7391.86it/s] 92%|█████████▏| 367010/400000 [00:47<00:04, 7361.22it/s] 92%|█████████▏| 367747/400000 [00:47<00:04, 7273.77it/s] 92%|█████████▏| 368478/400000 [00:47<00:04, 7281.98it/s] 92%|█████████▏| 369225/400000 [00:47<00:04, 7335.66it/s] 92%|█████████▏| 369973/400000 [00:47<00:04, 7378.09it/s] 93%|█████████▎| 370720/400000 [00:47<00:03, 7403.60it/s] 93%|█████████▎| 371499/400000 [00:47<00:03, 7514.66it/s] 93%|█████████▎| 372298/400000 [00:47<00:03, 7649.14it/s] 93%|█████████▎| 373064/400000 [00:47<00:03, 7619.02it/s] 93%|█████████▎| 373827/400000 [00:48<00:03, 7544.37it/s] 94%|█████████▎| 374615/400000 [00:48<00:03, 7641.24it/s] 94%|█████████▍| 375380/400000 [00:48<00:03, 7545.07it/s] 94%|█████████▍| 376173/400000 [00:48<00:03, 7655.85it/s] 94%|█████████▍| 376940/400000 [00:48<00:03, 7655.32it/s] 94%|█████████▍| 377707/400000 [00:48<00:02, 7588.39it/s] 95%|█████████▍| 378467/400000 [00:48<00:02, 7581.55it/s] 95%|█████████▍| 379250/400000 [00:48<00:02, 7652.96it/s] 95%|█████████▌| 380016/400000 [00:48<00:02, 7515.34it/s] 95%|█████████▌| 380769/400000 [00:48<00:02, 7414.52it/s] 95%|█████████▌| 381512/400000 [00:49<00:02, 7321.22it/s] 96%|█████████▌| 382278/400000 [00:49<00:02, 7418.16it/s] 96%|█████████▌| 383046/400000 [00:49<00:02, 7494.48it/s] 96%|█████████▌| 383833/400000 [00:49<00:02, 7601.29it/s] 96%|█████████▌| 384613/400000 [00:49<00:02, 7658.44it/s] 96%|█████████▋| 385383/400000 [00:49<00:01, 7668.67it/s] 97%|█████████▋| 386161/400000 [00:49<00:01, 7701.53it/s] 97%|█████████▋| 386932/400000 [00:49<00:01, 7698.84it/s] 97%|█████████▋| 387703/400000 [00:49<00:01, 7669.19it/s] 97%|█████████▋| 388471/400000 [00:49<00:01, 7629.66it/s] 97%|█████████▋| 389235/400000 [00:50<00:01, 7518.81it/s] 98%|█████████▊| 390003/400000 [00:50<00:01, 7564.09it/s] 98%|█████████▊| 390800/400000 [00:50<00:01, 7680.10it/s] 98%|█████████▊| 391569/400000 [00:50<00:01, 7649.14it/s] 98%|█████████▊| 392335/400000 [00:50<00:01, 7608.21it/s] 98%|█████████▊| 393117/400000 [00:50<00:00, 7669.59it/s] 98%|█████████▊| 393916/400000 [00:50<00:00, 7762.10it/s] 99%|█████████▊| 394726/400000 [00:50<00:00, 7858.96it/s] 99%|█████████▉| 395513/400000 [00:50<00:00, 7826.74it/s] 99%|█████████▉| 396297/400000 [00:51<00:00, 7754.37it/s] 99%|█████████▉| 397086/400000 [00:51<00:00, 7792.91it/s] 99%|█████████▉| 397883/400000 [00:51<00:00, 7841.52it/s]100%|█████████▉| 398668/400000 [00:51<00:00, 7796.06it/s]100%|█████████▉| 399448/400000 [00:51<00:00, 7796.23it/s]100%|█████████▉| 399999/400000 [00:51<00:00, 7770.30it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fee56e90c18> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011049311304297059 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.010919200337451437 	 Accuracy: 64

  model saves at 64% accuracy 

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
2020-05-14 21:26:08.856093: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 21:26:08.860233: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-14 21:26:08.860368: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ead8d4d400 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 21:26:08.860382: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fee62a0cfd0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 8.0193 - accuracy: 0.4770
 2000/25000 [=>............................] - ETA: 9s - loss: 7.9043 - accuracy: 0.4845 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8200 - accuracy: 0.4900
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7395 - accuracy: 0.4952
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7218 - accuracy: 0.4964
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6794 - accuracy: 0.4992
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6732 - accuracy: 0.4996
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6685 - accuracy: 0.4999
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6922 - accuracy: 0.4983
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6896 - accuracy: 0.4985
11000/25000 [============>.................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
12000/25000 [=============>................] - ETA: 4s - loss: 7.6768 - accuracy: 0.4993
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6772 - accuracy: 0.4993
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6940 - accuracy: 0.4982
15000/25000 [=================>............] - ETA: 3s - loss: 7.6830 - accuracy: 0.4989
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6705 - accuracy: 0.4997
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6738 - accuracy: 0.4995
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6590 - accuracy: 0.5005
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6569 - accuracy: 0.5006
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6781 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6871 - accuracy: 0.4987
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6729 - accuracy: 0.4996
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6706 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6717 - accuracy: 0.4997
25000/25000 [==============================] - 10s 396us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fedb6d066a0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fee00370748> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 995ms/step - loss: 1.4229 - crf_viterbi_accuracy: 0.1600 - val_loss: 1.3924 - val_crf_viterbi_accuracy: 0.1867

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

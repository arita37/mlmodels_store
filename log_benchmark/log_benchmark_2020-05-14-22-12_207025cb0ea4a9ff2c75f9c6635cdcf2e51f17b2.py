
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f2acb444f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 22:12:54.353360
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 22:12:54.357720
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 22:12:54.361329
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 22:12:54.365210
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f2acb10f4a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354475.9062
Epoch 2/10

1/1 [==============================] - 0s 97ms/step - loss: 246436.0938
Epoch 3/10

1/1 [==============================] - 0s 121ms/step - loss: 144977.7188
Epoch 4/10

1/1 [==============================] - 0s 91ms/step - loss: 70980.7188
Epoch 5/10

1/1 [==============================] - 0s 89ms/step - loss: 34940.3359
Epoch 6/10

1/1 [==============================] - 0s 94ms/step - loss: 19508.2520
Epoch 7/10

1/1 [==============================] - 0s 91ms/step - loss: 12026.7285
Epoch 8/10

1/1 [==============================] - 0s 95ms/step - loss: 8379.7295
Epoch 9/10

1/1 [==============================] - 0s 93ms/step - loss: 5979.6577
Epoch 10/10

1/1 [==============================] - 0s 94ms/step - loss: 4650.9917

  #### Inference Need return ypred, ytrue ######################### 
[[ 5.59658706e-01  9.89703465e+00  1.01534481e+01  1.05157499e+01
   9.80147171e+00  1.12117004e+01  9.64680386e+00  1.06069632e+01
   1.10281782e+01  9.63340664e+00  1.09051981e+01  1.04758549e+01
   1.03192492e+01  8.89228916e+00  1.26089621e+01  1.11162472e+01
   1.02693558e+01  1.06714735e+01  1.03072176e+01  9.18955135e+00
   1.10032139e+01  9.28436184e+00  1.03775454e+01  1.13314924e+01
   1.29877911e+01  1.21774063e+01  1.05001554e+01  9.68477154e+00
   9.62506199e+00  1.01800890e+01  8.32454872e+00  1.07280140e+01
   1.05631580e+01  9.71118736e+00  1.08463850e+01  9.66388798e+00
   9.54364395e+00  8.94338226e+00  8.93807220e+00  1.18467655e+01
   9.09869862e+00  1.10259562e+01  1.01974058e+01  1.07994633e+01
   1.01722517e+01  1.12951508e+01  9.38043213e+00  7.82109118e+00
   1.00373392e+01  9.56560230e+00  9.79401302e+00  1.14985809e+01
   9.30728340e+00  1.10063152e+01  1.21327896e+01  1.04096050e+01
   9.17844296e+00  1.07280674e+01  1.03774853e+01  8.76061630e+00
   1.55097461e+00 -1.11798835e+00  1.95132685e+00  3.10653865e-01
  -6.13377929e-01 -2.88166463e-01 -1.05812788e+00 -3.59999013e+00
   2.99138874e-02 -3.01369429e-02  1.37372911e+00  2.00017929e-01
   3.43084514e-01 -2.68915009e+00 -2.61073828e+00 -1.06847763e+00
   4.86202031e-01 -2.65810609e+00 -1.02268577e-01  8.21235001e-01
   1.32457054e+00  2.46073532e+00 -1.08262682e+00 -1.46094835e+00
   1.13185930e+00 -6.11042380e-01 -2.26293731e+00  2.72286463e+00
   1.14897823e+00 -2.00396061e-01  7.65555322e-01  8.50762546e-01
  -1.21311712e+00 -2.14355254e+00  1.17192733e+00 -4.05849159e-01
   6.93654597e-01  7.80912220e-01 -1.58785415e+00  6.41123414e-01
  -2.11887062e-01  1.07811534e+00  7.82671690e-01 -3.80776882e-01
   9.07580018e-01  1.30399084e+00 -5.21396399e-01  2.76035070e-03
   1.55756640e+00 -5.42027652e-01 -2.05194283e+00 -2.00579453e+00
  -8.20358455e-01  2.73586655e+00  1.44849849e+00 -1.45855761e+00
   1.27993584e+00  1.08435428e+00 -1.41249239e-01  1.17283165e+00
   4.57393885e-01  4.81655717e-01 -6.54419899e-01 -4.95942473e-01
  -2.32926941e+00 -6.61762953e-01  1.80693865e+00 -1.56218433e+00
   7.58802772e-01  1.61167276e+00  1.07759333e+00  1.44586968e+00
   1.19871283e+00 -3.07309318e+00 -1.04093134e-01  2.14463520e+00
  -9.26431537e-01  1.85542858e+00  4.16627556e-01  3.05758774e-01
  -9.95692372e-01  4.23915327e-01 -1.05134583e+00  1.62741050e-01
  -9.76943791e-01 -1.59728146e+00  7.85427094e-01 -4.74721134e-01
  -1.42244780e+00  1.43614173e-01 -2.74490833e-01  1.04914081e+00
   1.02382064e-01 -3.10761762e+00 -4.41251397e-02  4.33954835e-01
   1.49754786e+00  6.23681068e-01  2.42013752e-01 -3.21650505e-02
  -3.81036997e-02 -1.35589540e+00  1.00940084e+00 -2.37146592e+00
   1.19828904e+00  7.43756294e-01  5.76814175e-01  3.58661801e-01
   1.06463945e+00  2.16079044e+00  1.84850407e+00 -6.99643493e-01
   2.08921939e-01  7.28344858e-01 -9.51795101e-01 -4.50670779e-01
  -2.40717959e+00 -1.86801255e-02  1.96180356e+00 -3.53777528e-01
   7.97560811e-02  9.33764076e+00  1.12165346e+01  9.43726540e+00
   1.02511454e+01  9.60751629e+00  1.15953579e+01  9.34507179e+00
   1.17708960e+01  9.21752167e+00  9.51648426e+00  1.02519016e+01
   1.06035471e+01  7.89644623e+00  9.27162838e+00  8.44306564e+00
   1.24233637e+01  8.38906193e+00  1.03834791e+01  8.85374737e+00
   8.87065315e+00  1.15138311e+01  9.15112495e+00  1.16275272e+01
   1.12253914e+01  8.92820454e+00  8.15517807e+00  9.97347164e+00
   1.07584867e+01  1.15930204e+01  1.02129564e+01  7.50693703e+00
   9.12964058e+00  1.16927547e+01  8.60123253e+00  9.43369293e+00
   1.27430830e+01  1.13313723e+01  1.09503345e+01  7.94209051e+00
   8.81841373e+00  1.01842232e+01  1.03936911e+01  8.82040596e+00
   1.17370939e+01  1.11620245e+01  1.16512527e+01  1.03950691e+01
   1.09965582e+01  9.39017200e+00  9.02281761e+00  9.68988228e+00
   1.10397959e+01  9.55482388e+00  1.17667160e+01  9.95027637e+00
   1.10280466e+01  1.21252842e+01  1.05596638e+01  1.02060204e+01
   1.80033028e+00  5.18947423e-01  4.43452358e-01  3.30748224e+00
   1.44376397e-01  9.19916511e-01  7.94731855e-01  9.70996678e-01
   2.75675631e+00  6.69967175e-01  7.50839889e-01  3.01735306e+00
   1.97912097e-01  3.33186209e-01  7.91411996e-02  2.90140152e-01
   1.14914691e+00  1.03868222e+00  2.53358722e-01  1.27096844e+00
   5.06752014e-01  8.31120610e-02  1.16378403e+00  2.77883649e-01
   1.22107339e+00  2.10524130e+00  1.04761219e+00  2.74214435e+00
   3.09581101e-01  1.11990333e+00  4.94738460e-01  1.52030706e-01
   2.92966533e+00  3.41140890e+00  5.49811661e-01  3.01147699e+00
   5.44460058e-01  1.30182326e+00  2.25240517e+00  1.70941234e-01
   2.54964924e+00  3.63405287e-01  8.68953586e-01  1.49663949e+00
   2.57428694e+00  8.17038774e-01  1.16949236e+00  3.18454027e-01
   2.46642733e+00  1.75882328e+00  1.89108849e-01  2.07249701e-01
   2.75971270e+00  1.89812613e+00  1.85703802e+00  1.16013896e+00
   2.25876713e+00  2.27261007e-01  9.49283063e-01  3.09032559e-01
   2.37842858e-01  4.10624027e-01  2.28108931e+00  3.27179909e+00
   1.58694160e+00  4.34985816e-01  4.88887668e-01  5.67991197e-01
   1.45504594e+00  1.58080041e+00  8.06518555e-01  1.34149778e+00
   5.72725594e-01  1.82416201e-01  2.25506258e+00  1.57343507e+00
   3.16984057e-01  1.58490849e+00  8.04311633e-01  1.66125011e+00
   1.76226735e+00  2.19338298e-01  2.08504391e+00  3.10040188e+00
   8.43991041e-01  3.71069133e-01  1.56879544e-01  4.90119040e-01
   1.50382662e+00  3.17317247e+00  6.73862636e-01  3.97574902e-01
   2.14415836e+00  1.00617278e+00  7.46445537e-01  1.40928853e+00
   2.06658745e+00  3.66223097e+00  2.55048227e+00  1.71007848e+00
   2.07177019e+00  6.66819811e-02  1.91454244e+00  1.09909630e+00
   1.89636731e+00  1.78006029e+00  4.24284279e-01  1.16853046e+00
   1.53296244e+00  5.19865751e-01  1.45561874e-01  1.04397345e+00
   3.74041200e-01  2.38254666e+00  1.28874731e+00  1.35873878e+00
   3.19509029e-01  1.75764740e-01  3.36515903e-01  6.40932322e-01
   9.20883369e+00 -8.65202808e+00 -9.32573509e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 22:13:04.104410
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.4063
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 22:13:04.108539
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8560.02
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 22:13:04.112173
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.4081
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 22:13:04.116053
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -765.61
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139821416428880
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139820340875896
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139820340876400
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139820340876904
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139820340877408
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139820340877912

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f2ad720e400> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.530451
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.492836
grad_step = 000002, loss = 0.470362
grad_step = 000003, loss = 0.447209
grad_step = 000004, loss = 0.423437
grad_step = 000005, loss = 0.403888
grad_step = 000006, loss = 0.396215
grad_step = 000007, loss = 0.389163
grad_step = 000008, loss = 0.373427
grad_step = 000009, loss = 0.356793
grad_step = 000010, loss = 0.345066
grad_step = 000011, loss = 0.335310
grad_step = 000012, loss = 0.324911
grad_step = 000013, loss = 0.313464
grad_step = 000014, loss = 0.301690
grad_step = 000015, loss = 0.290737
grad_step = 000016, loss = 0.280734
grad_step = 000017, loss = 0.271116
grad_step = 000018, loss = 0.260880
grad_step = 000019, loss = 0.249967
grad_step = 000020, loss = 0.239639
grad_step = 000021, loss = 0.230718
grad_step = 000022, loss = 0.222887
grad_step = 000023, loss = 0.215531
grad_step = 000024, loss = 0.208225
grad_step = 000025, loss = 0.200664
grad_step = 000026, loss = 0.192810
grad_step = 000027, loss = 0.184978
grad_step = 000028, loss = 0.177587
grad_step = 000029, loss = 0.170673
grad_step = 000030, loss = 0.163913
grad_step = 000031, loss = 0.157125
grad_step = 000032, loss = 0.150440
grad_step = 000033, loss = 0.144128
grad_step = 000034, loss = 0.138276
grad_step = 000035, loss = 0.132596
grad_step = 000036, loss = 0.126752
grad_step = 000037, loss = 0.120896
grad_step = 000038, loss = 0.115374
grad_step = 000039, loss = 0.110243
grad_step = 000040, loss = 0.105332
grad_step = 000041, loss = 0.100452
grad_step = 000042, loss = 0.095595
grad_step = 000043, loss = 0.090920
grad_step = 000044, loss = 0.086528
grad_step = 000045, loss = 0.082369
grad_step = 000046, loss = 0.078339
grad_step = 000047, loss = 0.074387
grad_step = 000048, loss = 0.070563
grad_step = 000049, loss = 0.066949
grad_step = 000050, loss = 0.063530
grad_step = 000051, loss = 0.060201
grad_step = 000052, loss = 0.056938
grad_step = 000053, loss = 0.053822
grad_step = 000054, loss = 0.050899
grad_step = 000055, loss = 0.048135
grad_step = 000056, loss = 0.045491
grad_step = 000057, loss = 0.042951
grad_step = 000058, loss = 0.040509
grad_step = 000059, loss = 0.038176
grad_step = 000060, loss = 0.035969
grad_step = 000061, loss = 0.033885
grad_step = 000062, loss = 0.031900
grad_step = 000063, loss = 0.030007
grad_step = 000064, loss = 0.028214
grad_step = 000065, loss = 0.026532
grad_step = 000066, loss = 0.024951
grad_step = 000067, loss = 0.023441
grad_step = 000068, loss = 0.022001
grad_step = 000069, loss = 0.020650
grad_step = 000070, loss = 0.019391
grad_step = 000071, loss = 0.018210
grad_step = 000072, loss = 0.017101
grad_step = 000073, loss = 0.016059
grad_step = 000074, loss = 0.015082
grad_step = 000075, loss = 0.014171
grad_step = 000076, loss = 0.013325
grad_step = 000077, loss = 0.012532
grad_step = 000078, loss = 0.011788
grad_step = 000079, loss = 0.011095
grad_step = 000080, loss = 0.010450
grad_step = 000081, loss = 0.009850
grad_step = 000082, loss = 0.009289
grad_step = 000083, loss = 0.008763
grad_step = 000084, loss = 0.008272
grad_step = 000085, loss = 0.007814
grad_step = 000086, loss = 0.007387
grad_step = 000087, loss = 0.006990
grad_step = 000088, loss = 0.006619
grad_step = 000089, loss = 0.006272
grad_step = 000090, loss = 0.005949
grad_step = 000091, loss = 0.005647
grad_step = 000092, loss = 0.005367
grad_step = 000093, loss = 0.005106
grad_step = 000094, loss = 0.004863
grad_step = 000095, loss = 0.004637
grad_step = 000096, loss = 0.004427
grad_step = 000097, loss = 0.004232
grad_step = 000098, loss = 0.004051
grad_step = 000099, loss = 0.003883
grad_step = 000100, loss = 0.003728
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003585
grad_step = 000102, loss = 0.003452
grad_step = 000103, loss = 0.003330
grad_step = 000104, loss = 0.003217
grad_step = 000105, loss = 0.003113
grad_step = 000106, loss = 0.003018
grad_step = 000107, loss = 0.002930
grad_step = 000108, loss = 0.002850
grad_step = 000109, loss = 0.002776
grad_step = 000110, loss = 0.002709
grad_step = 000111, loss = 0.002647
grad_step = 000112, loss = 0.002590
grad_step = 000113, loss = 0.002539
grad_step = 000114, loss = 0.002491
grad_step = 000115, loss = 0.002448
grad_step = 000116, loss = 0.002409
grad_step = 000117, loss = 0.002373
grad_step = 000118, loss = 0.002340
grad_step = 000119, loss = 0.002310
grad_step = 000120, loss = 0.002282
grad_step = 000121, loss = 0.002257
grad_step = 000122, loss = 0.002233
grad_step = 000123, loss = 0.002212
grad_step = 000124, loss = 0.002191
grad_step = 000125, loss = 0.002173
grad_step = 000126, loss = 0.002155
grad_step = 000127, loss = 0.002139
grad_step = 000128, loss = 0.002124
grad_step = 000129, loss = 0.002109
grad_step = 000130, loss = 0.002095
grad_step = 000131, loss = 0.002082
grad_step = 000132, loss = 0.002069
grad_step = 000133, loss = 0.002057
grad_step = 000134, loss = 0.002046
grad_step = 000135, loss = 0.002037
grad_step = 000136, loss = 0.002034
grad_step = 000137, loss = 0.002030
grad_step = 000138, loss = 0.002015
grad_step = 000139, loss = 0.001996
grad_step = 000140, loss = 0.001984
grad_step = 000141, loss = 0.001979
grad_step = 000142, loss = 0.001987
grad_step = 000143, loss = 0.002024
grad_step = 000144, loss = 0.002103
grad_step = 000145, loss = 0.002038
grad_step = 000146, loss = 0.001953
grad_step = 000147, loss = 0.001954
grad_step = 000148, loss = 0.001992
grad_step = 000149, loss = 0.001968
grad_step = 000150, loss = 0.001921
grad_step = 000151, loss = 0.001960
grad_step = 000152, loss = 0.001996
grad_step = 000153, loss = 0.001920
grad_step = 000154, loss = 0.001908
grad_step = 000155, loss = 0.001957
grad_step = 000156, loss = 0.001934
grad_step = 000157, loss = 0.001896
grad_step = 000158, loss = 0.001886
grad_step = 000159, loss = 0.001909
grad_step = 000160, loss = 0.001928
grad_step = 000161, loss = 0.001886
grad_step = 000162, loss = 0.001867
grad_step = 000163, loss = 0.001882
grad_step = 000164, loss = 0.001887
grad_step = 000165, loss = 0.001880
grad_step = 000166, loss = 0.001856
grad_step = 000167, loss = 0.001846
grad_step = 000168, loss = 0.001851
grad_step = 000169, loss = 0.001861
grad_step = 000170, loss = 0.001879
grad_step = 000171, loss = 0.001870
grad_step = 000172, loss = 0.001860
grad_step = 000173, loss = 0.001834
grad_step = 000174, loss = 0.001824
grad_step = 000175, loss = 0.001835
grad_step = 000176, loss = 0.001842
grad_step = 000177, loss = 0.001838
grad_step = 000178, loss = 0.001819
grad_step = 000179, loss = 0.001806
grad_step = 000180, loss = 0.001800
grad_step = 000181, loss = 0.001803
grad_step = 000182, loss = 0.001816
grad_step = 000183, loss = 0.001832
grad_step = 000184, loss = 0.001870
grad_step = 000185, loss = 0.001858
grad_step = 000186, loss = 0.001816
grad_step = 000187, loss = 0.001779
grad_step = 000188, loss = 0.001796
grad_step = 000189, loss = 0.001818
grad_step = 000190, loss = 0.001792
grad_step = 000191, loss = 0.001761
grad_step = 000192, loss = 0.001764
grad_step = 000193, loss = 0.001785
grad_step = 000194, loss = 0.001804
grad_step = 000195, loss = 0.001787
grad_step = 000196, loss = 0.001756
grad_step = 000197, loss = 0.001732
grad_step = 000198, loss = 0.001736
grad_step = 000199, loss = 0.001747
grad_step = 000200, loss = 0.001737
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001719
grad_step = 000202, loss = 0.001707
grad_step = 000203, loss = 0.001710
grad_step = 000204, loss = 0.001715
grad_step = 000205, loss = 0.001709
grad_step = 000206, loss = 0.001697
grad_step = 000207, loss = 0.001683
grad_step = 000208, loss = 0.001670
grad_step = 000209, loss = 0.001662
grad_step = 000210, loss = 0.001659
grad_step = 000211, loss = 0.001658
grad_step = 000212, loss = 0.001657
grad_step = 000213, loss = 0.001666
grad_step = 000214, loss = 0.001673
grad_step = 000215, loss = 0.001692
grad_step = 000216, loss = 0.001680
grad_step = 000217, loss = 0.001652
grad_step = 000218, loss = 0.001610
grad_step = 000219, loss = 0.001590
grad_step = 000220, loss = 0.001594
grad_step = 000221, loss = 0.001614
grad_step = 000222, loss = 0.001642
grad_step = 000223, loss = 0.001664
grad_step = 000224, loss = 0.001662
grad_step = 000225, loss = 0.001617
grad_step = 000226, loss = 0.001555
grad_step = 000227, loss = 0.001541
grad_step = 000228, loss = 0.001568
grad_step = 000229, loss = 0.001582
grad_step = 000230, loss = 0.001576
grad_step = 000231, loss = 0.001545
grad_step = 000232, loss = 0.001534
grad_step = 000233, loss = 0.001533
grad_step = 000234, loss = 0.001537
grad_step = 000235, loss = 0.001550
grad_step = 000236, loss = 0.001531
grad_step = 000237, loss = 0.001503
grad_step = 000238, loss = 0.001505
grad_step = 000239, loss = 0.001525
grad_step = 000240, loss = 0.001525
grad_step = 000241, loss = 0.001498
grad_step = 000242, loss = 0.001481
grad_step = 000243, loss = 0.001488
grad_step = 000244, loss = 0.001506
grad_step = 000245, loss = 0.001527
grad_step = 000246, loss = 0.001550
grad_step = 000247, loss = 0.001620
grad_step = 000248, loss = 0.001602
grad_step = 000249, loss = 0.001604
grad_step = 000250, loss = 0.001584
grad_step = 000251, loss = 0.001552
grad_step = 000252, loss = 0.001508
grad_step = 000253, loss = 0.001473
grad_step = 000254, loss = 0.001506
grad_step = 000255, loss = 0.001534
grad_step = 000256, loss = 0.001488
grad_step = 000257, loss = 0.001455
grad_step = 000258, loss = 0.001485
grad_step = 000259, loss = 0.001524
grad_step = 000260, loss = 0.001517
grad_step = 000261, loss = 0.001475
grad_step = 000262, loss = 0.001456
grad_step = 000263, loss = 0.001466
grad_step = 000264, loss = 0.001463
grad_step = 000265, loss = 0.001460
grad_step = 000266, loss = 0.001458
grad_step = 000267, loss = 0.001446
grad_step = 000268, loss = 0.001442
grad_step = 000269, loss = 0.001430
grad_step = 000270, loss = 0.001422
grad_step = 000271, loss = 0.001425
grad_step = 000272, loss = 0.001438
grad_step = 000273, loss = 0.001450
grad_step = 000274, loss = 0.001425
grad_step = 000275, loss = 0.001410
grad_step = 000276, loss = 0.001418
grad_step = 000277, loss = 0.001425
grad_step = 000278, loss = 0.001424
grad_step = 000279, loss = 0.001407
grad_step = 000280, loss = 0.001399
grad_step = 000281, loss = 0.001397
grad_step = 000282, loss = 0.001400
grad_step = 000283, loss = 0.001405
grad_step = 000284, loss = 0.001403
grad_step = 000285, loss = 0.001399
grad_step = 000286, loss = 0.001389
grad_step = 000287, loss = 0.001385
grad_step = 000288, loss = 0.001388
grad_step = 000289, loss = 0.001396
grad_step = 000290, loss = 0.001410
grad_step = 000291, loss = 0.001424
grad_step = 000292, loss = 0.001450
grad_step = 000293, loss = 0.001434
grad_step = 000294, loss = 0.001410
grad_step = 000295, loss = 0.001389
grad_step = 000296, loss = 0.001399
grad_step = 000297, loss = 0.001426
grad_step = 000298, loss = 0.001440
grad_step = 000299, loss = 0.001467
grad_step = 000300, loss = 0.001471
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001483
grad_step = 000302, loss = 0.001461
grad_step = 000303, loss = 0.001427
grad_step = 000304, loss = 0.001389
grad_step = 000305, loss = 0.001374
grad_step = 000306, loss = 0.001389
grad_step = 000307, loss = 0.001413
grad_step = 000308, loss = 0.001419
grad_step = 000309, loss = 0.001397
grad_step = 000310, loss = 0.001371
grad_step = 000311, loss = 0.001366
grad_step = 000312, loss = 0.001378
grad_step = 000313, loss = 0.001394
grad_step = 000314, loss = 0.001390
grad_step = 000315, loss = 0.001373
grad_step = 000316, loss = 0.001355
grad_step = 000317, loss = 0.001350
grad_step = 000318, loss = 0.001362
grad_step = 000319, loss = 0.001384
grad_step = 000320, loss = 0.001425
grad_step = 000321, loss = 0.001441
grad_step = 000322, loss = 0.001481
grad_step = 000323, loss = 0.001391
grad_step = 000324, loss = 0.001362
grad_step = 000325, loss = 0.001415
grad_step = 000326, loss = 0.001406
grad_step = 000327, loss = 0.001373
grad_step = 000328, loss = 0.001351
grad_step = 000329, loss = 0.001358
grad_step = 000330, loss = 0.001407
grad_step = 000331, loss = 0.001477
grad_step = 000332, loss = 0.001601
grad_step = 000333, loss = 0.001432
grad_step = 000334, loss = 0.001364
grad_step = 000335, loss = 0.001437
grad_step = 000336, loss = 0.001405
grad_step = 000337, loss = 0.001355
grad_step = 000338, loss = 0.001368
grad_step = 000339, loss = 0.001367
grad_step = 000340, loss = 0.001341
grad_step = 000341, loss = 0.001346
grad_step = 000342, loss = 0.001365
grad_step = 000343, loss = 0.001350
grad_step = 000344, loss = 0.001341
grad_step = 000345, loss = 0.001350
grad_step = 000346, loss = 0.001343
grad_step = 000347, loss = 0.001324
grad_step = 000348, loss = 0.001327
grad_step = 000349, loss = 0.001339
grad_step = 000350, loss = 0.001324
grad_step = 000351, loss = 0.001315
grad_step = 000352, loss = 0.001322
grad_step = 000353, loss = 0.001327
grad_step = 000354, loss = 0.001330
grad_step = 000355, loss = 0.001330
grad_step = 000356, loss = 0.001331
grad_step = 000357, loss = 0.001333
grad_step = 000358, loss = 0.001350
grad_step = 000359, loss = 0.001370
grad_step = 000360, loss = 0.001393
grad_step = 000361, loss = 0.001407
grad_step = 000362, loss = 0.001418
grad_step = 000363, loss = 0.001396
grad_step = 000364, loss = 0.001368
grad_step = 000365, loss = 0.001334
grad_step = 000366, loss = 0.001316
grad_step = 000367, loss = 0.001312
grad_step = 000368, loss = 0.001319
grad_step = 000369, loss = 0.001326
grad_step = 000370, loss = 0.001327
grad_step = 000371, loss = 0.001326
grad_step = 000372, loss = 0.001318
grad_step = 000373, loss = 0.001308
grad_step = 000374, loss = 0.001299
grad_step = 000375, loss = 0.001295
grad_step = 000376, loss = 0.001296
grad_step = 000377, loss = 0.001300
grad_step = 000378, loss = 0.001305
grad_step = 000379, loss = 0.001305
grad_step = 000380, loss = 0.001301
grad_step = 000381, loss = 0.001294
grad_step = 000382, loss = 0.001286
grad_step = 000383, loss = 0.001281
grad_step = 000384, loss = 0.001280
grad_step = 000385, loss = 0.001282
grad_step = 000386, loss = 0.001285
grad_step = 000387, loss = 0.001289
grad_step = 000388, loss = 0.001292
grad_step = 000389, loss = 0.001294
grad_step = 000390, loss = 0.001294
grad_step = 000391, loss = 0.001293
grad_step = 000392, loss = 0.001291
grad_step = 000393, loss = 0.001289
grad_step = 000394, loss = 0.001287
grad_step = 000395, loss = 0.001285
grad_step = 000396, loss = 0.001282
grad_step = 000397, loss = 0.001279
grad_step = 000398, loss = 0.001275
grad_step = 000399, loss = 0.001271
grad_step = 000400, loss = 0.001269
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001267
grad_step = 000402, loss = 0.001267
grad_step = 000403, loss = 0.001268
grad_step = 000404, loss = 0.001271
grad_step = 000405, loss = 0.001277
grad_step = 000406, loss = 0.001285
grad_step = 000407, loss = 0.001300
grad_step = 000408, loss = 0.001319
grad_step = 000409, loss = 0.001350
grad_step = 000410, loss = 0.001369
grad_step = 000411, loss = 0.001385
grad_step = 000412, loss = 0.001396
grad_step = 000413, loss = 0.001425
grad_step = 000414, loss = 0.001434
grad_step = 000415, loss = 0.001413
grad_step = 000416, loss = 0.001338
grad_step = 000417, loss = 0.001267
grad_step = 000418, loss = 0.001254
grad_step = 000419, loss = 0.001291
grad_step = 000420, loss = 0.001329
grad_step = 000421, loss = 0.001327
grad_step = 000422, loss = 0.001297
grad_step = 000423, loss = 0.001268
grad_step = 000424, loss = 0.001261
grad_step = 000425, loss = 0.001272
grad_step = 000426, loss = 0.001278
grad_step = 000427, loss = 0.001270
grad_step = 000428, loss = 0.001254
grad_step = 000429, loss = 0.001247
grad_step = 000430, loss = 0.001254
grad_step = 000431, loss = 0.001266
grad_step = 000432, loss = 0.001269
grad_step = 000433, loss = 0.001260
grad_step = 000434, loss = 0.001243
grad_step = 000435, loss = 0.001233
grad_step = 000436, loss = 0.001235
grad_step = 000437, loss = 0.001243
grad_step = 000438, loss = 0.001244
grad_step = 000439, loss = 0.001238
grad_step = 000440, loss = 0.001229
grad_step = 000441, loss = 0.001227
grad_step = 000442, loss = 0.001230
grad_step = 000443, loss = 0.001234
grad_step = 000444, loss = 0.001234
grad_step = 000445, loss = 0.001228
grad_step = 000446, loss = 0.001222
grad_step = 000447, loss = 0.001220
grad_step = 000448, loss = 0.001222
grad_step = 000449, loss = 0.001223
grad_step = 000450, loss = 0.001223
grad_step = 000451, loss = 0.001221
grad_step = 000452, loss = 0.001217
grad_step = 000453, loss = 0.001215
grad_step = 000454, loss = 0.001215
grad_step = 000455, loss = 0.001216
grad_step = 000456, loss = 0.001217
grad_step = 000457, loss = 0.001220
grad_step = 000458, loss = 0.001224
grad_step = 000459, loss = 0.001231
grad_step = 000460, loss = 0.001241
grad_step = 000461, loss = 0.001264
grad_step = 000462, loss = 0.001293
grad_step = 000463, loss = 0.001347
grad_step = 000464, loss = 0.001378
grad_step = 000465, loss = 0.001393
grad_step = 000466, loss = 0.001368
grad_step = 000467, loss = 0.001337
grad_step = 000468, loss = 0.001284
grad_step = 000469, loss = 0.001233
grad_step = 000470, loss = 0.001211
grad_step = 000471, loss = 0.001233
grad_step = 000472, loss = 0.001272
grad_step = 000473, loss = 0.001279
grad_step = 000474, loss = 0.001246
grad_step = 000475, loss = 0.001209
grad_step = 000476, loss = 0.001199
grad_step = 000477, loss = 0.001216
grad_step = 000478, loss = 0.001237
grad_step = 000479, loss = 0.001238
grad_step = 000480, loss = 0.001220
grad_step = 000481, loss = 0.001197
grad_step = 000482, loss = 0.001189
grad_step = 000483, loss = 0.001199
grad_step = 000484, loss = 0.001211
grad_step = 000485, loss = 0.001211
grad_step = 000486, loss = 0.001199
grad_step = 000487, loss = 0.001187
grad_step = 000488, loss = 0.001183
grad_step = 000489, loss = 0.001189
grad_step = 000490, loss = 0.001197
grad_step = 000491, loss = 0.001200
grad_step = 000492, loss = 0.001200
grad_step = 000493, loss = 0.001199
grad_step = 000494, loss = 0.001207
grad_step = 000495, loss = 0.001221
grad_step = 000496, loss = 0.001241
grad_step = 000497, loss = 0.001257
grad_step = 000498, loss = 0.001262
grad_step = 000499, loss = 0.001245
grad_step = 000500, loss = 0.001215
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001184
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

  date_run                              2020-05-14 22:13:27.424025
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.210399
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 22:13:27.430548
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.11067
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 22:13:27.438987
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.126222
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 22:13:27.445508
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.681668
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
0   2020-05-14 22:12:54.353360  ...    mean_absolute_error
1   2020-05-14 22:12:54.357720  ...     mean_squared_error
2   2020-05-14 22:12:54.361329  ...  median_absolute_error
3   2020-05-14 22:12:54.365210  ...               r2_score
4   2020-05-14 22:13:04.104410  ...    mean_absolute_error
5   2020-05-14 22:13:04.108539  ...     mean_squared_error
6   2020-05-14 22:13:04.112173  ...  median_absolute_error
7   2020-05-14 22:13:04.116053  ...               r2_score
8   2020-05-14 22:13:27.424025  ...    mean_absolute_error
9   2020-05-14 22:13:27.430548  ...     mean_squared_error
10  2020-05-14 22:13:27.438987  ...  median_absolute_error
11  2020-05-14 22:13:27.445508  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc3ce246be0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:32, 306564.40it/s]  2%|▏         | 212992/9912422 [00:00<00:24, 396150.03it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 549000.25it/s] 30%|███       | 3014656/9912422 [00:00<00:08, 773374.19it/s] 58%|█████▊    | 5709824/9912422 [00:00<00:03, 1087205.27it/s] 88%|████████▊ | 8675328/9912422 [00:01<00:00, 1522733.33it/s]9920512it [00:01, 9707986.39it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 144422.96it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 306169.67it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 395552.95it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 547396.47it/s]1654784it [00:00, 2695859.09it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 50696.18it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc380c00e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc3802300b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc380c00e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc3801860b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc37d9c14a8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc37d9acc18> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc380c00e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc3801456d8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc37d9c14a8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc3ce209eb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc20f80b1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=a9d9d47cea862e068545b649eacdc94454681d69d15ecba6c6d732dfc024591a
  Stored in directory: /tmp/pip-ephem-wheel-cache-d96a75n0/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc1a7606748> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 42s
   49152/17464789 [..............................] - ETA: 42s
  114688/17464789 [..............................] - ETA: 27s
  253952/17464789 [..............................] - ETA: 16s
  557056/17464789 [..............................] - ETA: 9s 
 1114112/17464789 [>.............................] - ETA: 5s
 2260992/17464789 [==>...........................] - ETA: 2s
 4521984/17464789 [======>.......................] - ETA: 1s
 7634944/17464789 [============>.................] - ETA: 0s
10698752/17464789 [=================>............] - ETA: 0s
13746176/17464789 [======================>.......] - ETA: 0s
16711680/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 22:14:59.332043: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 22:14:59.336481: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-05-14 22:14:59.336632: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f16b749860 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 22:14:59.336649: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.9580 - accuracy: 0.4810
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7280 - accuracy: 0.4960 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7126 - accuracy: 0.4970
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7625 - accuracy: 0.4938
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6912 - accuracy: 0.4984
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7382 - accuracy: 0.4953
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7192 - accuracy: 0.4966
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7069 - accuracy: 0.4974
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7245 - accuracy: 0.4962
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7203 - accuracy: 0.4965
11000/25000 [============>.................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
12000/25000 [=============>................] - ETA: 3s - loss: 7.6743 - accuracy: 0.4995
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6749 - accuracy: 0.4995
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6568 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 3s - loss: 7.6544 - accuracy: 0.5008
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6637 - accuracy: 0.5002
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6774 - accuracy: 0.4993
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6803 - accuracy: 0.4991
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6924 - accuracy: 0.4983
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6812 - accuracy: 0.4990
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6856 - accuracy: 0.4988
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6694 - accuracy: 0.4998
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6486 - accuracy: 0.5012
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6545 - accuracy: 0.5008
25000/25000 [==============================] - 9s 357us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 22:15:15.585889
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 22:15:15.585889  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<11:14:57, 21.3kB/s].vector_cache/glove.6B.zip:   0%|          | 352k/862M [00:00<7:53:48, 30.3kB/s]  .vector_cache/glove.6B.zip:   1%|          | 6.32M/862M [00:00<5:29:26, 43.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.0M/862M [00:00<3:47:45, 61.8kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.6M/862M [00:00<2:37:28, 88.3kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.4M/862M [00:00<1:48:59, 126kB/s] .vector_cache/glove.6B.zip:   5%|▌         | 47.2M/862M [00:01<1:15:25, 180kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:01<52:58, 255kB/s]  .vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:01<37:21, 360kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<9:10:55, 24.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:03<6:25:16, 34.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:04<10:02:58, 22.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:04<7:01:34, 31.7kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<10:36:46, 21.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:05<7:25:10, 30.0kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<10:51:07, 20.5kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:06<7:35:13, 29.2kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:07<10:47:18, 20.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:07<7:32:30, 29.3kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:08<10:56:22, 20.2kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:08<7:38:56, 28.8kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<10:32:56, 20.9kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:09<7:22:26, 29.8kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:10<10:52:03, 20.2kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:10<7:35:55, 28.9kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:11<10:27:32, 21.0kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:11<7:18:41, 29.9kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:12<10:43:57, 20.4kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:12<7:30:14, 29.1kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:13<10:22:10, 21.0kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:13<7:14:55, 30.0kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<10:43:00, 20.3kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:14<7:29:28, 29.0kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:15<10:44:33, 20.2kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:15<7:30:33, 28.8kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:16<10:44:24, 20.2kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:16<7:30:28, 28.8kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<10:38:00, 20.3kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:17<7:25:57, 29.0kB/s] .vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:18<10:43:48, 20.1kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:18<7:30:00, 28.6kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:19<10:42:52, 20.0kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:19<7:29:24, 28.6kB/s] .vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:20<10:31:20, 20.4kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:20<7:21:17, 29.0kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:21<10:38:22, 20.1kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:21<7:26:13, 28.6kB/s] .vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<10:33:30, 20.2kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:22<7:22:52, 28.8kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:23<10:15:42, 20.7kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:23<7:10:20, 29.5kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:24<10:30:22, 20.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<7:20:36, 28.8kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<10:29:11, 20.1kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:25<7:19:48, 28.7kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<10:22:52, 20.3kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<7:15:26, 29.0kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<10:07:57, 20.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:27<7:05:00, 29.6kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<9:59:51, 21.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<6:59:21, 29.9kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<9:58:26, 21.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:29<6:58:16, 29.9kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<10:15:34, 20.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<7:10:15, 29.0kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<10:17:55, 20.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:31<7:11:55, 28.8kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<10:10:00, 20.4kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<7:06:20, 29.1kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<10:16:26, 20.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:33<7:10:50, 28.7kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<10:17:07, 20.0kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:34<7:11:25, 28.6kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<9:47:35, 21.0kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:35<6:50:45, 29.9kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<9:44:33, 21.0kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<6:48:34, 30.0kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<9:57:16, 20.5kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:37<6:57:31, 29.3kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<9:40:36, 21.1kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<6:45:52, 30.0kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<9:34:26, 21.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:39<6:41:28, 30.3kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<9:52:34, 20.5kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<6:54:09, 29.3kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<9:55:31, 20.3kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<6:56:11, 29.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<9:59:33, 20.2kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<6:59:01, 28.8kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<9:57:00, 20.2kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<6:57:13, 28.8kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<9:58:12, 20.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<6:58:05, 28.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<9:49:31, 20.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:45<6:52:03, 29.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<9:33:30, 20.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:46<6:40:47, 29.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<9:47:24, 20.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<6:50:31, 28.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<9:46:29, 20.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:48<6:49:51, 28.9kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<9:50:14, 20.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<6:52:29, 28.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<9:46:12, 20.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<6:49:43, 28.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<9:27:05, 20.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<6:36:22, 29.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<9:17:31, 21.0kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<6:29:36, 30.0kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<9:34:28, 20.4kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<6:41:26, 29.1kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<9:38:09, 20.2kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<6:44:02, 28.8kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<9:32:43, 20.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:55<6:40:12, 29.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<9:38:26, 20.0kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:56<6:44:12, 28.6kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<9:35:29, 20.1kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:57<6:42:09, 28.7kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<9:28:27, 20.3kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:58<6:37:12, 28.9kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<9:32:44, 20.1kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<6:40:12, 28.6kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<9:30:47, 20.1kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:00<6:38:52, 28.6kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<9:22:12, 20.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<6:32:49, 29.0kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<9:27:42, 20.1kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<6:36:41, 28.6kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<9:22:33, 20.2kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:03<6:33:04, 28.8kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<9:25:35, 20.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:04<6:35:11, 28.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<9:22:59, 20.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:05<6:33:24, 28.6kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<9:14:19, 20.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:06<6:27:22, 28.9kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<8:59:17, 20.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<6:16:53, 29.6kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<8:53:02, 21.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:08<6:12:26, 29.9kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<9:06:58, 20.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<6:22:10, 29.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<9:07:56, 20.3kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:10<6:22:52, 28.9kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<9:02:47, 20.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<6:19:16, 29.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<8:58:53, 20.5kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<6:16:34, 29.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<8:50:29, 20.7kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<6:10:38, 29.6kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<9:01:33, 20.2kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<6:18:22, 28.9kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<9:03:01, 20.1kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<6:19:24, 28.7kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<8:56:23, 20.3kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:16<6:14:45, 29.0kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<8:57:17, 20.2kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<6:15:22, 28.8kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<8:56:48, 20.2kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:18<6:15:03, 28.8kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<8:52:07, 20.3kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:19<6:11:46, 28.9kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<8:48:21, 20.4kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:20<6:09:06, 29.0kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<8:54:45, 20.0kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<6:13:35, 28.6kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<8:49:53, 20.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:22<6:10:10, 28.8kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<8:51:13, 20.0kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<6:11:06, 28.6kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<8:48:11, 20.1kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:24<6:09:00, 28.7kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<8:41:27, 20.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<6:04:16, 28.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<8:46:33, 20.0kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:26<6:07:50, 28.6kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<8:46:31, 20.0kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:27<6:07:49, 28.5kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<8:39:16, 20.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:28<6:02:46, 28.8kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<8:33:22, 20.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<5:58:36, 29.0kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<8:39:27, 20.0kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:30<6:02:56, 28.6kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<8:16:55, 20.9kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<5:47:08, 29.8kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<8:26:23, 20.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:32<5:53:48, 29.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<8:12:26, 20.9kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<5:43:59, 29.8kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<8:25:03, 20.3kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:34<5:52:48, 29.0kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<8:22:41, 20.4kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<5:51:08, 29.0kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<8:23:39, 20.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:36<5:51:51, 28.9kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<8:10:12, 20.7kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:37<5:42:24, 29.6kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<8:19:12, 20.3kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:38<5:48:41, 28.9kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<8:18:19, 20.2kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:39<5:48:03, 28.9kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<8:21:18, 20.1kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:40<5:50:10, 28.6kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<8:14:26, 20.3kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:41<5:45:20, 28.9kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<8:17:05, 20.1kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<5:47:11, 28.7kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<8:13:08, 20.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<5:44:25, 28.8kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<8:14:46, 20.0kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<5:45:33, 28.6kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<8:13:20, 20.0kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<5:44:34, 28.6kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<8:10:27, 20.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<5:42:33, 28.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<8:03:46, 20.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:47<5:37:52, 28.9kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<8:07:29, 20.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<5:40:27, 28.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<8:03:54, 20.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:49<5:37:57, 28.7kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<7:59:43, 20.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<5:35:05, 28.9kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<7:46:28, 20.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<5:25:46, 29.6kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<7:57:57, 20.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<5:33:47, 28.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<7:56:49, 20.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<5:33:00, 28.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<7:49:58, 20.4kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<5:28:11, 29.0kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<7:53:59, 20.1kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<5:30:59, 28.7kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<7:53:53, 20.0kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<5:30:56, 28.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<7:47:24, 20.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<5:26:25, 28.9kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<7:41:58, 20.4kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:58<5:22:37, 29.1kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<7:37:19, 20.5kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [01:59<5:19:21, 29.3kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<7:40:27, 20.3kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:00<5:21:36, 29.0kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<7:24:00, 21.0kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:01<5:10:06, 29.9kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<7:18:47, 21.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<5:06:23, 30.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<7:33:53, 20.4kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<5:17:00, 29.1kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<7:18:48, 21.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<5:06:25, 30.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<7:23:55, 20.7kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<5:09:58, 29.5kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<7:29:41, 20.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<5:13:58, 29.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<7:35:55, 20.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:07<5:18:19, 28.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<7:33:03, 20.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:08<5:16:24, 28.6kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<7:09:47, 21.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:09<5:00:04, 30.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<7:21:39, 20.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<5:08:25, 29.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<7:08:21, 20.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<4:59:04, 29.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<7:21:25, 20.2kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<5:08:11, 28.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<7:19:09, 20.3kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:13<5:06:35, 28.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<7:22:25, 20.0kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:14<5:08:53, 28.6kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<7:15:31, 20.3kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:15<5:04:02, 29.0kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:15<3:34:26, 41.0kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<5:55:32, 24.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<4:08:20, 35.3kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<6:22:14, 22.9kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:17<4:26:56, 32.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<6:37:26, 22.0kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:18<4:37:35, 31.3kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<6:26:27, 22.5kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:19<4:29:38, 32.1kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:19<3:13:36, 44.7kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<5:44:08, 25.2kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:20<4:00:03, 35.9kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:20<3:14:19, 44.4kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<5:43:49, 25.1kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:21<4:00:07, 35.8kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<6:13:58, 23.0kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:22<4:21:08, 32.8kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<6:29:05, 22.0kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<4:31:43, 31.3kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<6:20:48, 22.4kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<4:25:53, 31.9kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<6:31:04, 21.7kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<4:33:02, 30.9kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<6:34:05, 21.4kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<4:35:12, 30.6kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<6:20:03, 22.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<4:25:21, 31.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<6:27:37, 21.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:28<4:30:36, 30.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<6:34:16, 21.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:29<4:35:18, 30.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<6:18:19, 22.0kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:30<4:24:06, 31.3kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<6:26:41, 21.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:31<4:29:57, 30.5kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<6:26:00, 21.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:32<4:29:27, 30.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<6:30:10, 21.0kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:33<4:32:21, 30.0kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<6:27:10, 21.1kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<4:30:14, 30.1kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<6:29:06, 20.9kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<4:31:34, 29.8kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<6:31:32, 20.7kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<4:33:15, 29.5kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<3:12:43, 41.8kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<5:26:49, 24.7kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<3:48:10, 35.2kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<5:56:22, 22.5kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:38<4:08:48, 32.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<5:58:12, 22.3kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:39<4:10:01, 31.8kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<6:10:06, 21.5kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:40<4:18:19, 30.7kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<6:12:51, 21.3kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<4:20:14, 30.3kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<6:11:58, 21.2kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:42<4:19:36, 30.3kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<6:16:23, 20.9kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<4:22:40, 29.8kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<6:16:06, 20.8kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<4:22:31, 29.7kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<6:02:01, 21.5kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<4:12:41, 30.7kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<5:55:17, 21.8kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:46<4:08:00, 31.1kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<5:50:16, 22.0kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<4:04:26, 31.4kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<6:00:46, 21.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:48<4:11:45, 30.4kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<6:05:02, 20.9kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:49<4:14:43, 29.9kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<6:06:12, 20.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:50<4:15:32, 29.6kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<6:04:14, 20.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:51<4:14:09, 29.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<6:02:19, 20.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<4:12:51, 29.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<5:47:08, 21.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:53<4:02:17, 30.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<5:38:47, 22.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:54<3:56:24, 31.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<5:47:16, 21.4kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:55<4:02:18, 30.5kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<5:49:00, 21.2kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<4:03:33, 30.2kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<5:40:14, 21.6kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<3:57:23, 30.9kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<5:45:29, 21.2kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<4:01:03, 30.3kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<5:45:58, 21.1kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [02:59<4:01:23, 30.1kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<5:44:44, 21.1kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:00<4:00:29, 30.0kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<5:49:41, 20.7kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:01<4:03:57, 29.5kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<5:40:52, 21.1kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:02<3:57:47, 30.1kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<5:42:23, 20.9kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:03<3:58:51, 29.8kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<5:40:29, 20.9kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:04<3:57:30, 29.8kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<5:39:37, 20.9kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:05<3:56:54, 29.8kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<5:37:26, 20.9kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<3:55:22, 29.8kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<5:36:37, 20.8kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<3:54:47, 29.7kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<5:36:26, 20.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<3:54:42, 29.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<5:24:29, 21.4kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:09<3:46:19, 30.5kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<5:29:04, 21.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:10<3:49:33, 30.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<5:17:54, 21.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:11<3:41:44, 30.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<5:23:10, 21.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:12<3:45:24, 30.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<5:23:35, 21.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:13<3:45:40, 30.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<5:22:23, 21.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:14<3:44:50, 30.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<5:21:34, 20.9kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:15<3:44:15, 29.9kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<5:20:24, 20.9kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:16<3:43:25, 29.8kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<5:21:13, 20.8kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:17<3:44:00, 29.6kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<5:17:53, 20.9kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<3:41:39, 29.8kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<5:18:27, 20.7kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:19<3:42:06, 29.5kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<5:01:52, 21.7kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:20<3:30:33, 31.0kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<4:55:21, 22.1kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:21<3:26:00, 31.5kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<4:51:38, 22.3kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:22<3:23:24, 31.7kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<4:50:32, 22.2kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:23<3:22:35, 31.7kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<5:01:33, 21.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:24<3:30:18, 30.4kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<4:48:32, 22.1kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:25<3:21:10, 31.6kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<4:58:22, 21.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:26<3:27:50, 30.4kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:26<2:34:05, 41.0kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<4:07:04, 25.6kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:27<2:52:17, 36.5kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<4:34:57, 22.8kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:28<3:11:41, 32.6kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<4:48:53, 21.6kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:29<3:21:24, 30.8kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<4:49:21, 21.5kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:30<3:21:43, 30.6kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<4:50:03, 21.3kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:31<3:22:11, 30.4kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<4:49:17, 21.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:32<3:21:38, 30.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<4:50:08, 21.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:33<3:22:13, 30.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<4:53:08, 20.7kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:34<3:24:21, 29.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<4:37:04, 21.8kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<3:13:06, 31.1kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<4:42:34, 21.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<3:16:56, 30.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<4:44:35, 21.0kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<3:18:22, 29.9kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<4:35:34, 21.5kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<3:12:05, 30.7kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<4:28:23, 22.0kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<3:07:02, 31.3kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<4:33:10, 21.5kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:40<3:10:22, 30.6kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<4:35:09, 21.2kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:41<3:11:44, 30.2kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<4:38:04, 20.8kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:42<3:13:47, 29.7kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<4:25:52, 21.7kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:43<3:05:15, 30.9kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<4:30:43, 21.1kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:44<3:08:37, 30.1kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<4:32:45, 20.8kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<3:10:01, 29.7kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<4:32:36, 20.7kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:46<3:09:57, 29.6kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<4:20:17, 21.6kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<3:01:20, 30.8kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<4:24:03, 21.1kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<3:03:56, 30.2kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<4:25:50, 20.9kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<3:05:11, 29.8kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<4:26:19, 20.7kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<3:05:36, 29.6kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:50<2:10:41, 41.9kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<3:39:53, 24.9kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:51<2:33:12, 35.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<3:59:53, 22.7kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:52<2:47:06, 32.4kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<4:10:25, 21.6kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:53<2:54:25, 30.8kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<4:14:51, 21.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:54<2:57:30, 30.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<4:13:46, 21.0kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:55<2:56:44, 30.0kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<4:11:43, 21.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<2:55:20, 30.0kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<4:01:34, 21.8kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:57<2:48:14, 31.1kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<4:06:05, 21.3kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:58<2:51:21, 30.3kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<4:08:17, 20.9kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<2:52:55, 29.9kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<3:59:17, 21.6kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:00<2:46:38, 30.8kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<3:54:16, 21.9kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:01<2:43:07, 31.2kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<3:59:14, 21.3kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:02<2:46:33, 30.4kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<4:02:12, 20.9kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:03<2:48:37, 29.8kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<3:59:13, 21.0kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:04<2:46:32, 30.0kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<3:59:30, 20.8kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:05<2:46:38, 29.7kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:05<1:59:19, 41.5kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<3:20:37, 24.7kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<2:19:41, 35.2kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<3:38:32, 22.5kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<2:32:07, 32.1kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<3:47:44, 21.4kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:08<2:38:32, 30.6kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<3:40:39, 22.0kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:09<2:33:34, 31.3kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:10<3:45:00, 21.4kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<2:36:35, 30.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<3:47:40, 21.0kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:11<2:38:25, 29.9kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<3:47:40, 20.8kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:12<2:38:27, 29.7kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<3:38:03, 21.6kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:13<2:31:43, 30.8kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<3:42:00, 21.1kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:14<2:34:27, 30.0kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<3:42:53, 20.8kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:15<2:35:04, 29.7kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<3:41:51, 20.7kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:16<2:34:22, 29.6kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<3:32:42, 21.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:17<2:27:58, 30.6kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<3:35:39, 21.0kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:18<2:30:01, 30.0kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<3:35:43, 20.9kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<2:30:04, 29.7kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<3:27:39, 21.5kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<2:24:26, 30.7kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<3:29:45, 21.1kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:21<2:25:53, 30.1kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<3:31:46, 20.7kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<2:27:24, 29.6kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:22<1:43:44, 42.0kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<2:54:17, 25.0kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:23<2:01:13, 35.7kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<3:12:08, 22.5kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:24<2:13:37, 32.1kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<3:20:03, 21.4kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<2:19:17, 30.6kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:25<1:37:53, 43.5kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<2:49:20, 25.1kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<1:57:46, 35.8kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<3:03:06, 23.0kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<2:07:16, 32.9kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<1:30:45, 46.1kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<2:41:37, 25.9kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:28<1:52:23, 37.0kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:28<1:19:41, 52.1kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<2:34:23, 26.9kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<1:47:22, 38.4kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<1:16:09, 54.0kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<2:28:17, 27.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<1:43:24, 39.6kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<1:12:30, 56.3kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<2:21:29, 28.8kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<1:38:39, 41.1kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:31<1:09:11, 58.4kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<2:19:40, 29.0kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:32<1:37:22, 41.3kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:32<1:08:18, 58.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<2:19:15, 28.8kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<1:37:00, 41.1kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:33<1:08:10, 58.3kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<2:17:45, 28.8kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:34<1:35:49, 41.2kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:34<1:07:55, 58.0kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<2:10:11, 30.3kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<1:30:38, 43.2kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:35<1:03:49, 61.2kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<2:13:57, 29.1kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<1:33:16, 41.6kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<1:05:42, 58.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<2:06:33, 30.6kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<1:28:09, 43.6kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<1:01:57, 61.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<2:09:19, 29.6kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:38<1:30:05, 42.3kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:38<1:03:16, 60.1kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<2:11:45, 28.8kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:39<1:31:37, 41.2kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:39<1:04:51, 58.0kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<2:05:38, 30.0kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:40<1:27:32, 42.8kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:40<1:01:27, 60.7kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<2:04:51, 29.9kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<1:26:48, 42.6kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<1:01:22, 60.2kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<2:08:19, 28.8kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:42<1:29:14, 41.1kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:42<1:02:57, 58.1kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<2:07:31, 28.7kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<1:28:45, 41.0kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:43<1:02:25, 58.1kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<2:05:28, 28.9kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<1:27:19, 41.2kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:44<1:01:24, 58.5kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<2:04:24, 28.9kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<1:26:37, 41.2kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:45<1:00:49, 58.4kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<2:03:42, 28.7kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:46<1:26:02, 41.0kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:46<1:00:36, 58.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<2:02:21, 28.8kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<1:25:02, 41.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<1:00:03, 58.0kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<2:02:11, 28.5kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<1:25:00, 40.7kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<59:46, 57.7kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<1:59:43, 28.8kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<1:23:14, 41.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<58:39, 58.2kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<1:59:56, 28.5kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:50<1:23:20, 40.6kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:50<58:57, 57.3kB/s]  .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<1:53:20, 29.8kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<1:18:45, 42.6kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<55:35, 60.2kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<1:56:13, 28.8kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:52<1:20:40, 41.1kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:52<57:38, 57.4kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<1:51:15, 29.8kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<1:17:17, 42.5kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:53<54:37, 60.0kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<1:53:19, 28.9kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<1:18:48, 41.2kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:54<55:25, 58.5kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<1:51:55, 28.9kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<1:17:46, 41.3kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:55<54:56, 58.3kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<1:44:24, 30.7kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:56<1:12:29, 43.8kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:56<51:17, 61.8kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<1:47:19, 29.5kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<1:14:36, 42.2kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:57<52:28, 59.8kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<1:48:55, 28.8kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<1:15:41, 41.1kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<53:15, 58.2kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<1:49:53, 28.2kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<1:16:26, 40.3kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<53:38, 57.1kB/s]  .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<1:45:27, 29.1kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<1:13:14, 41.5kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<51:38, 58.7kB/s]  .vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<1:46:08, 28.6kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<1:13:42, 40.8kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<51:56, 57.7kB/s]  .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<1:45:22, 28.4kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<1:13:11, 40.6kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:02<51:30, 57.5kB/s]  .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<1:44:18, 28.4kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<1:12:26, 40.5kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:03<50:58, 57.4kB/s]  .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<1:42:50, 28.4kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<1:11:20, 40.6kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:04<50:33, 57.2kB/s]  .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<1:36:42, 29.9kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<1:07:07, 42.7kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:05<47:21, 60.3kB/s]  .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<1:38:47, 28.9kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<1:08:36, 41.3kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:06<48:14, 58.5kB/s]  .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<1:38:35, 28.6kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:07<1:08:27, 40.8kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<48:08, 57.9kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<1:37:21, 28.6kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<1:07:38, 40.8kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<47:28, 57.9kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<1:36:42, 28.4kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<1:06:56, 40.6kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<48:14, 56.3kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<1:35:58, 28.3kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<1:06:25, 40.4kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<47:43, 56.2kB/s]  .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<1:36:04, 27.9kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:11<1:06:36, 39.8kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:11<46:59, 56.3kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<1:33:18, 28.4kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<1:04:38, 40.5kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:12<45:44, 57.1kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<1:31:55, 28.4kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<1:03:41, 40.5kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:13<44:58, 57.3kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<1:31:25, 28.2kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<1:03:24, 40.2kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:14<44:35, 57.0kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<1:29:40, 28.3kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:15<1:02:13, 40.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:15<43:42, 57.3kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<1:28:36, 28.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<1:01:29, 40.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:16<43:09, 57.3kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<1:26:42, 28.5kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<59:58, 40.7kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<42:46, 57.0kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<1:25:39, 28.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:18<59:19, 40.6kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:18<41:51, 57.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<1:24:29, 28.4kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:19<58:25, 40.6kB/s]  .vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:19<41:47, 56.6kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<1:20:10, 29.5kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:20<55:27, 42.1kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:20<39:22, 59.2kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<1:21:18, 28.7kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:21<56:09, 40.9kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<39:17, 57.6kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<27:36, 81.7kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:23<19:11, 116kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<1:01:51, 36.0kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<42:45, 51.4kB/s]  .vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<29:55, 72.1kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<21:03, 102kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:26<14:39, 145kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<55:54, 38.0kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:27<38:33, 54.2kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<27:04, 75.8kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<19:03, 107kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:29<13:15, 152kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<53:59, 37.4kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<37:10, 53.3kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<26:06, 74.6kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:32<18:21, 106kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<12:46, 150kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<47:21, 40.4kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:33<32:40, 57.6kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<22:49, 80.7kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<16:08, 114kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:35<11:13, 161kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<43:51, 41.2kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<30:16, 58.8kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<21:05, 82.4kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<14:50, 117kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:38<10:18, 165kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<44:52, 37.9kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:39<30:46, 54.2kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<21:34, 75.7kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:41<15:10, 107kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:41<10:31, 152kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<42:42, 37.4kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<29:20, 53.4kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<20:25, 74.9kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<14:20, 106kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:44<09:56, 150kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<36:46, 40.6kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:45<25:09, 57.9kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<17:36, 80.9kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<12:22, 114kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:47<08:33, 162kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<36:36, 37.9kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:48<25:04, 54.1kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<17:23, 75.8kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<12:13, 107kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:50<08:26, 152kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<33:52, 37.9kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:51<23:02, 54.1kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<16:05, 75.5kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<11:17, 107kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<07:46, 151kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<31:18, 37.6kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<21:19, 53.7kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<14:45, 75.2kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<10:21, 107kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:56<07:06, 151kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<28:47, 37.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:57<19:36, 53.2kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<13:28, 74.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<09:26, 106kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [05:59<06:28, 150kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<26:08, 37.0kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:00<17:39, 52.9kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<12:09, 74.0kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<08:30, 105kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<05:49, 148kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<22:28, 38.4kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<15:08, 54.8kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<10:21, 76.7kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<07:15, 109kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:05<04:55, 154kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<20:02, 37.8kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:06<13:26, 54.0kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<09:07, 75.6kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<06:25, 107kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:08<04:19, 151kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<17:26, 37.5kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:09<11:29, 53.5kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<07:49, 74.7kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<05:28, 106kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<03:39, 150kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<14:06, 38.9kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:12<09:12, 55.5kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<06:11, 77.5kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<04:18, 110kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:14<02:51, 155kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<11:46, 37.7kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:15<07:38, 53.8kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<04:58, 75.4kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<03:27, 107kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:17<02:14, 151kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<08:54, 38.1kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:18<05:25, 54.4kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<03:34, 75.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<02:28, 107kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:20<01:33, 151kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<05:45, 40.6kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<03:23, 58.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<02:02, 80.8kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<01:23, 115kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:47, 162kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<03:29, 37.1kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:24<01:39, 53.0kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:48, 73.9kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:32, 104kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:26<00:04, 148kB/s].vector_cache/glove.6B.zip: 862MB [06:26, 2.23MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 773/400000 [00:00<00:51, 7721.54it/s]  0%|          | 1558/400000 [00:00<00:51, 7759.33it/s]  1%|          | 2340/400000 [00:00<00:51, 7775.27it/s]  1%|          | 3097/400000 [00:00<00:51, 7712.49it/s]  1%|          | 3849/400000 [00:00<00:51, 7653.50it/s]  1%|          | 4566/400000 [00:00<00:52, 7500.97it/s]  1%|▏         | 5356/400000 [00:00<00:51, 7613.96it/s]  2%|▏         | 6132/400000 [00:00<00:51, 7655.95it/s]  2%|▏         | 6909/400000 [00:00<00:51, 7687.98it/s]  2%|▏         | 7663/400000 [00:01<00:51, 7641.18it/s]  2%|▏         | 8435/400000 [00:01<00:51, 7662.81it/s]  2%|▏         | 9208/400000 [00:01<00:50, 7681.31it/s]  2%|▏         | 9997/400000 [00:01<00:50, 7742.33it/s]  3%|▎         | 10778/400000 [00:01<00:50, 7759.89it/s]  3%|▎         | 11551/400000 [00:01<00:50, 7750.83it/s]  3%|▎         | 12343/400000 [00:01<00:49, 7799.31it/s]  3%|▎         | 13133/400000 [00:01<00:49, 7826.81it/s]  3%|▎         | 13915/400000 [00:01<00:50, 7693.47it/s]  4%|▎         | 14684/400000 [00:01<00:50, 7625.80it/s]  4%|▍         | 15447/400000 [00:02<00:50, 7603.55it/s]  4%|▍         | 16226/400000 [00:02<00:50, 7657.22it/s]  4%|▍         | 17000/400000 [00:02<00:49, 7681.52it/s]  4%|▍         | 17769/400000 [00:02<00:50, 7587.90it/s]  5%|▍         | 18543/400000 [00:02<00:49, 7630.04it/s]  5%|▍         | 19307/400000 [00:02<00:51, 7444.75it/s]  5%|▌         | 20089/400000 [00:02<00:50, 7553.16it/s]  5%|▌         | 20852/400000 [00:02<00:50, 7575.19it/s]  5%|▌         | 21611/400000 [00:02<00:50, 7449.46it/s]  6%|▌         | 22396/400000 [00:02<00:49, 7564.01it/s]  6%|▌         | 23154/400000 [00:03<00:50, 7533.89it/s]  6%|▌         | 23909/400000 [00:03<00:49, 7534.53it/s]  6%|▌         | 24677/400000 [00:03<00:49, 7574.12it/s]  6%|▋         | 25448/400000 [00:03<00:49, 7613.46it/s]  7%|▋         | 26224/400000 [00:03<00:48, 7655.23it/s]  7%|▋         | 26992/400000 [00:03<00:48, 7620.96it/s]  7%|▋         | 27769/400000 [00:03<00:48, 7662.58it/s]  7%|▋         | 28558/400000 [00:03<00:48, 7726.82it/s]  7%|▋         | 29340/400000 [00:03<00:47, 7753.96it/s]  8%|▊         | 30136/400000 [00:03<00:47, 7814.48it/s]  8%|▊         | 30918/400000 [00:04<00:47, 7731.88it/s]  8%|▊         | 31693/400000 [00:04<00:47, 7734.64it/s]  8%|▊         | 32474/400000 [00:04<00:47, 7756.16it/s]  8%|▊         | 33265/400000 [00:04<00:47, 7801.57it/s]  9%|▊         | 34046/400000 [00:04<00:46, 7792.26it/s]  9%|▊         | 34826/400000 [00:04<00:48, 7554.34it/s]  9%|▉         | 35598/400000 [00:04<00:47, 7601.02it/s]  9%|▉         | 36371/400000 [00:04<00:47, 7637.30it/s]  9%|▉         | 37136/400000 [00:04<00:47, 7620.24it/s]  9%|▉         | 37928/400000 [00:04<00:46, 7705.88it/s] 10%|▉         | 38700/400000 [00:05<00:46, 7695.71it/s] 10%|▉         | 39488/400000 [00:05<00:46, 7748.50it/s] 10%|█         | 40281/400000 [00:05<00:46, 7799.59it/s] 10%|█         | 41069/400000 [00:05<00:45, 7823.47it/s] 10%|█         | 41852/400000 [00:05<00:46, 7769.11it/s] 11%|█         | 42630/400000 [00:05<00:46, 7729.02it/s] 11%|█         | 43409/400000 [00:05<00:46, 7743.91it/s] 11%|█         | 44184/400000 [00:05<00:45, 7735.44it/s] 11%|█         | 44958/400000 [00:05<00:46, 7632.85it/s] 11%|█▏        | 45744/400000 [00:05<00:46, 7698.04it/s] 12%|█▏        | 46516/400000 [00:06<00:45, 7704.26it/s] 12%|█▏        | 47287/400000 [00:06<00:46, 7598.10it/s] 12%|█▏        | 48049/400000 [00:06<00:46, 7603.16it/s] 12%|█▏        | 48867/400000 [00:06<00:45, 7766.86it/s] 12%|█▏        | 49657/400000 [00:06<00:44, 7804.89it/s] 13%|█▎        | 50439/400000 [00:06<00:45, 7622.31it/s] 13%|█▎        | 51230/400000 [00:06<00:45, 7704.40it/s] 13%|█▎        | 52020/400000 [00:06<00:44, 7761.59it/s] 13%|█▎        | 52798/400000 [00:06<00:45, 7695.15it/s] 13%|█▎        | 53576/400000 [00:06<00:44, 7719.13it/s] 14%|█▎        | 54349/400000 [00:07<00:45, 7648.31it/s] 14%|█▍        | 55139/400000 [00:07<00:44, 7719.68it/s] 14%|█▍        | 55936/400000 [00:07<00:44, 7791.45it/s] 14%|█▍        | 56741/400000 [00:07<00:43, 7866.70it/s] 14%|█▍        | 57545/400000 [00:07<00:43, 7915.56it/s] 15%|█▍        | 58338/400000 [00:07<00:43, 7812.93it/s] 15%|█▍        | 59120/400000 [00:07<00:44, 7732.43it/s] 15%|█▍        | 59894/400000 [00:07<00:44, 7691.24it/s] 15%|█▌        | 60664/400000 [00:07<00:44, 7584.31it/s] 15%|█▌        | 61464/400000 [00:07<00:43, 7701.69it/s] 16%|█▌        | 62236/400000 [00:08<00:43, 7706.62it/s] 16%|█▌        | 63030/400000 [00:08<00:43, 7773.75it/s] 16%|█▌        | 63817/400000 [00:08<00:43, 7801.14it/s] 16%|█▌        | 64613/400000 [00:08<00:42, 7845.58it/s] 16%|█▋        | 65427/400000 [00:08<00:42, 7926.45it/s] 17%|█▋        | 66221/400000 [00:08<00:42, 7885.66it/s] 17%|█▋        | 67018/400000 [00:08<00:42, 7908.19it/s] 17%|█▋        | 67810/400000 [00:08<00:42, 7865.41it/s] 17%|█▋        | 68601/400000 [00:08<00:42, 7876.22it/s] 17%|█▋        | 69396/400000 [00:08<00:41, 7896.92it/s] 18%|█▊        | 70186/400000 [00:09<00:42, 7847.90it/s] 18%|█▊        | 70971/400000 [00:09<00:42, 7707.17it/s] 18%|█▊        | 71755/400000 [00:09<00:42, 7744.39it/s] 18%|█▊        | 72530/400000 [00:09<00:42, 7732.06it/s] 18%|█▊        | 73304/400000 [00:09<00:42, 7600.52it/s] 19%|█▊        | 74065/400000 [00:09<00:42, 7590.60it/s] 19%|█▊        | 74846/400000 [00:09<00:42, 7655.11it/s] 19%|█▉        | 75613/400000 [00:09<00:42, 7618.98it/s] 19%|█▉        | 76383/400000 [00:09<00:42, 7643.08it/s] 19%|█▉        | 77148/400000 [00:10<00:42, 7574.30it/s] 19%|█▉        | 77913/400000 [00:10<00:42, 7595.71it/s] 20%|█▉        | 78698/400000 [00:10<00:41, 7669.73it/s] 20%|█▉        | 79477/400000 [00:10<00:41, 7703.34it/s] 20%|██        | 80253/400000 [00:10<00:41, 7719.94it/s] 20%|██        | 81033/400000 [00:10<00:41, 7742.28it/s] 20%|██        | 81808/400000 [00:10<00:41, 7663.27it/s] 21%|██        | 82575/400000 [00:10<00:41, 7639.78it/s] 21%|██        | 83340/400000 [00:10<00:41, 7631.03it/s] 21%|██        | 84122/400000 [00:10<00:41, 7685.66it/s] 21%|██        | 84891/400000 [00:11<00:41, 7660.81it/s] 21%|██▏       | 85658/400000 [00:11<00:41, 7589.56it/s] 22%|██▏       | 86437/400000 [00:11<00:40, 7647.90it/s] 22%|██▏       | 87217/400000 [00:11<00:40, 7691.59it/s] 22%|██▏       | 88002/400000 [00:11<00:40, 7736.69it/s] 22%|██▏       | 88778/400000 [00:11<00:40, 7742.04it/s] 22%|██▏       | 89553/400000 [00:11<00:40, 7685.01it/s] 23%|██▎       | 90323/400000 [00:11<00:40, 7688.84it/s] 23%|██▎       | 91093/400000 [00:11<00:40, 7665.77it/s] 23%|██▎       | 91863/400000 [00:11<00:40, 7666.51it/s] 23%|██▎       | 92662/400000 [00:12<00:39, 7760.56it/s] 23%|██▎       | 93439/400000 [00:12<00:39, 7762.55it/s] 24%|██▎       | 94246/400000 [00:12<00:38, 7851.52it/s] 24%|██▍       | 95041/400000 [00:12<00:38, 7877.16it/s] 24%|██▍       | 95830/400000 [00:12<00:38, 7811.56it/s] 24%|██▍       | 96612/400000 [00:12<00:38, 7808.11it/s] 24%|██▍       | 97394/400000 [00:12<00:38, 7761.33it/s] 25%|██▍       | 98171/400000 [00:12<00:39, 7717.10it/s] 25%|██▍       | 98943/400000 [00:12<00:40, 7480.77it/s] 25%|██▍       | 99712/400000 [00:12<00:39, 7542.24it/s] 25%|██▌       | 100468/400000 [00:13<00:39, 7535.86it/s] 25%|██▌       | 101228/400000 [00:13<00:39, 7554.00it/s] 26%|██▌       | 102010/400000 [00:13<00:39, 7629.66it/s] 26%|██▌       | 102792/400000 [00:13<00:38, 7685.37it/s] 26%|██▌       | 103571/400000 [00:13<00:38, 7714.27it/s] 26%|██▌       | 104364/400000 [00:13<00:38, 7777.06it/s] 26%|██▋       | 105143/400000 [00:13<00:37, 7777.33it/s] 26%|██▋       | 105922/400000 [00:13<00:37, 7780.38it/s] 27%|██▋       | 106701/400000 [00:13<00:37, 7737.45it/s] 27%|██▋       | 107476/400000 [00:13<00:37, 7738.98it/s] 27%|██▋       | 108264/400000 [00:14<00:37, 7778.41it/s] 27%|██▋       | 109042/400000 [00:14<00:37, 7693.63it/s] 27%|██▋       | 109812/400000 [00:14<00:37, 7669.99it/s] 28%|██▊       | 110580/400000 [00:14<00:37, 7626.47it/s] 28%|██▊       | 111343/400000 [00:14<00:38, 7544.97it/s] 28%|██▊       | 112119/400000 [00:14<00:37, 7606.87it/s] 28%|██▊       | 112881/400000 [00:14<00:37, 7607.57it/s] 28%|██▊       | 113671/400000 [00:14<00:37, 7691.17it/s] 29%|██▊       | 114441/400000 [00:14<00:37, 7626.70it/s] 29%|██▉       | 115214/400000 [00:14<00:37, 7655.34it/s] 29%|██▉       | 115980/400000 [00:15<00:37, 7631.19it/s] 29%|██▉       | 116750/400000 [00:15<00:37, 7648.52it/s] 29%|██▉       | 117527/400000 [00:15<00:36, 7682.83it/s] 30%|██▉       | 118296/400000 [00:15<00:36, 7646.41it/s] 30%|██▉       | 119096/400000 [00:15<00:36, 7748.33it/s] 30%|██▉       | 119887/400000 [00:15<00:35, 7793.62it/s] 30%|███       | 120667/400000 [00:15<00:36, 7749.79it/s] 30%|███       | 121448/400000 [00:15<00:35, 7766.52it/s] 31%|███       | 122225/400000 [00:15<00:35, 7740.93it/s] 31%|███       | 123002/400000 [00:15<00:35, 7749.31it/s] 31%|███       | 123793/400000 [00:16<00:35, 7796.11it/s] 31%|███       | 124573/400000 [00:16<00:36, 7610.49it/s] 31%|███▏      | 125336/400000 [00:16<00:36, 7469.37it/s] 32%|███▏      | 126127/400000 [00:16<00:36, 7594.42it/s] 32%|███▏      | 126928/400000 [00:16<00:35, 7713.70it/s] 32%|███▏      | 127719/400000 [00:16<00:35, 7769.61it/s] 32%|███▏      | 128508/400000 [00:16<00:34, 7803.15it/s] 32%|███▏      | 129290/400000 [00:16<00:34, 7740.07it/s] 33%|███▎      | 130065/400000 [00:16<00:34, 7731.20it/s] 33%|███▎      | 130839/400000 [00:16<00:35, 7673.99it/s] 33%|███▎      | 131607/400000 [00:17<00:35, 7637.59it/s] 33%|███▎      | 132375/400000 [00:17<00:34, 7650.17it/s] 33%|███▎      | 133174/400000 [00:17<00:34, 7748.68it/s] 33%|███▎      | 133956/400000 [00:17<00:34, 7767.69it/s] 34%|███▎      | 134747/400000 [00:17<00:33, 7807.89it/s] 34%|███▍      | 135541/400000 [00:17<00:33, 7845.67it/s] 34%|███▍      | 136326/400000 [00:17<00:34, 7739.99it/s] 34%|███▍      | 137101/400000 [00:17<00:34, 7725.36it/s] 34%|███▍      | 137874/400000 [00:17<00:35, 7371.25it/s] 35%|███▍      | 138633/400000 [00:18<00:35, 7434.76it/s] 35%|███▍      | 139380/400000 [00:18<00:35, 7436.95it/s] 35%|███▌      | 140163/400000 [00:18<00:34, 7550.55it/s] 35%|███▌      | 140965/400000 [00:18<00:33, 7683.37it/s] 35%|███▌      | 141754/400000 [00:18<00:33, 7743.26it/s] 36%|███▌      | 142558/400000 [00:18<00:32, 7829.73it/s] 36%|███▌      | 143349/400000 [00:18<00:32, 7852.16it/s] 36%|███▌      | 144137/400000 [00:18<00:32, 7858.45it/s] 36%|███▌      | 144925/400000 [00:18<00:32, 7864.59it/s] 36%|███▋      | 145715/400000 [00:18<00:32, 7873.54it/s] 37%|███▋      | 146508/400000 [00:19<00:32, 7888.81it/s] 37%|███▋      | 147313/400000 [00:19<00:31, 7935.48it/s] 37%|███▋      | 148107/400000 [00:19<00:32, 7841.15it/s] 37%|███▋      | 148897/400000 [00:19<00:31, 7855.58it/s] 37%|███▋      | 149683/400000 [00:19<00:31, 7851.26it/s] 38%|███▊      | 150469/400000 [00:19<00:32, 7793.97it/s] 38%|███▊      | 151249/400000 [00:19<00:32, 7769.78it/s] 38%|███▊      | 152028/400000 [00:19<00:31, 7773.85it/s] 38%|███▊      | 152806/400000 [00:19<00:31, 7762.58it/s] 38%|███▊      | 153608/400000 [00:19<00:31, 7835.75it/s] 39%|███▊      | 154414/400000 [00:20<00:31, 7900.74it/s] 39%|███▉      | 155228/400000 [00:20<00:30, 7967.86it/s] 39%|███▉      | 156026/400000 [00:20<00:30, 7965.80it/s] 39%|███▉      | 156850/400000 [00:20<00:30, 8045.80it/s] 39%|███▉      | 157673/400000 [00:20<00:29, 8098.98it/s] 40%|███▉      | 158484/400000 [00:20<00:29, 8101.10it/s] 40%|███▉      | 159295/400000 [00:20<00:29, 8026.03it/s] 40%|████      | 160098/400000 [00:20<00:30, 7887.61it/s] 40%|████      | 160888/400000 [00:20<00:30, 7813.10it/s] 40%|████      | 161671/400000 [00:20<00:30, 7801.81it/s] 41%|████      | 162452/400000 [00:21<00:30, 7740.72it/s] 41%|████      | 163227/400000 [00:21<00:30, 7646.22it/s] 41%|████      | 163993/400000 [00:21<00:31, 7546.37it/s] 41%|████      | 164783/400000 [00:21<00:30, 7648.08it/s] 41%|████▏     | 165566/400000 [00:21<00:30, 7700.81it/s] 42%|████▏     | 166361/400000 [00:21<00:30, 7773.23it/s] 42%|████▏     | 167152/400000 [00:21<00:29, 7813.70it/s] 42%|████▏     | 167934/400000 [00:21<00:29, 7762.25it/s] 42%|████▏     | 168719/400000 [00:21<00:29, 7786.11it/s] 42%|████▏     | 169514/400000 [00:21<00:29, 7832.24it/s] 43%|████▎     | 170298/400000 [00:22<00:29, 7790.98it/s] 43%|████▎     | 171081/400000 [00:22<00:29, 7800.42it/s] 43%|████▎     | 171862/400000 [00:22<00:29, 7739.32it/s] 43%|████▎     | 172660/400000 [00:22<00:29, 7807.39it/s] 43%|████▎     | 173451/400000 [00:22<00:28, 7836.71it/s] 44%|████▎     | 174235/400000 [00:22<00:28, 7835.68it/s] 44%|████▍     | 175019/400000 [00:22<00:28, 7815.48it/s] 44%|████▍     | 175801/400000 [00:22<00:28, 7792.14it/s] 44%|████▍     | 176581/400000 [00:22<00:28, 7760.07it/s] 44%|████▍     | 177358/400000 [00:22<00:28, 7697.20it/s] 45%|████▍     | 178128/400000 [00:23<00:29, 7639.34it/s] 45%|████▍     | 178917/400000 [00:23<00:28, 7710.85it/s] 45%|████▍     | 179689/400000 [00:23<00:28, 7669.42it/s] 45%|████▌     | 180470/400000 [00:23<00:28, 7710.35it/s] 45%|████▌     | 181267/400000 [00:23<00:28, 7783.62it/s] 46%|████▌     | 182046/400000 [00:23<00:28, 7775.56it/s] 46%|████▌     | 182839/400000 [00:23<00:27, 7820.92it/s] 46%|████▌     | 183622/400000 [00:23<00:27, 7781.10it/s] 46%|████▌     | 184415/400000 [00:23<00:27, 7825.01it/s] 46%|████▋     | 185210/400000 [00:23<00:27, 7860.09it/s] 47%|████▋     | 186014/400000 [00:24<00:27, 7912.65it/s] 47%|████▋     | 186824/400000 [00:24<00:26, 7965.96it/s] 47%|████▋     | 187621/400000 [00:24<00:26, 7908.57it/s] 47%|████▋     | 188435/400000 [00:24<00:26, 7975.39it/s] 47%|████▋     | 189233/400000 [00:24<00:26, 7941.07it/s] 48%|████▊     | 190028/400000 [00:24<00:27, 7662.01it/s] 48%|████▊     | 190797/400000 [00:24<00:28, 7457.97it/s] 48%|████▊     | 191561/400000 [00:24<00:27, 7509.52it/s] 48%|████▊     | 192363/400000 [00:24<00:27, 7654.86it/s] 48%|████▊     | 193164/400000 [00:25<00:26, 7757.25it/s] 48%|████▊     | 193977/400000 [00:25<00:26, 7863.13it/s] 49%|████▊     | 194769/400000 [00:25<00:26, 7879.61it/s] 49%|████▉     | 195559/400000 [00:25<00:26, 7857.43it/s] 49%|████▉     | 196347/400000 [00:25<00:25, 7863.50it/s] 49%|████▉     | 197134/400000 [00:25<00:25, 7836.38it/s] 49%|████▉     | 197919/400000 [00:25<00:25, 7781.30it/s] 50%|████▉     | 198700/400000 [00:25<00:25, 7788.04it/s] 50%|████▉     | 199480/400000 [00:25<00:25, 7712.43it/s] 50%|█████     | 200275/400000 [00:25<00:25, 7781.65it/s] 50%|█████     | 201054/400000 [00:26<00:26, 7534.51it/s] 50%|█████     | 201810/400000 [00:26<00:26, 7521.05it/s] 51%|█████     | 202582/400000 [00:26<00:26, 7578.68it/s] 51%|█████     | 203341/400000 [00:26<00:26, 7509.23it/s] 51%|█████     | 204129/400000 [00:26<00:25, 7615.84it/s] 51%|█████     | 204925/400000 [00:26<00:25, 7714.10it/s] 51%|█████▏    | 205727/400000 [00:26<00:24, 7800.75it/s] 52%|█████▏    | 206510/400000 [00:26<00:24, 7808.24it/s] 52%|█████▏    | 207292/400000 [00:26<00:25, 7696.14it/s] 52%|█████▏    | 208078/400000 [00:26<00:24, 7744.13it/s] 52%|█████▏    | 208854/400000 [00:27<00:24, 7693.72it/s] 52%|█████▏    | 209642/400000 [00:27<00:24, 7747.52it/s] 53%|█████▎    | 210425/400000 [00:27<00:24, 7770.40it/s] 53%|█████▎    | 211203/400000 [00:27<00:24, 7745.03it/s] 53%|█████▎    | 211992/400000 [00:27<00:24, 7785.88it/s] 53%|█████▎    | 212771/400000 [00:27<00:24, 7742.79it/s] 53%|█████▎    | 213550/400000 [00:27<00:24, 7756.25it/s] 54%|█████▎    | 214326/400000 [00:27<00:23, 7744.83it/s] 54%|█████▍    | 215101/400000 [00:27<00:24, 7671.03it/s] 54%|█████▍    | 215869/400000 [00:27<00:24, 7605.85it/s] 54%|█████▍    | 216632/400000 [00:28<00:24, 7611.85it/s] 54%|█████▍    | 217418/400000 [00:28<00:23, 7681.88it/s] 55%|█████▍    | 218216/400000 [00:28<00:23, 7767.22it/s] 55%|█████▍    | 218994/400000 [00:28<00:23, 7707.11it/s] 55%|█████▍    | 219787/400000 [00:28<00:23, 7769.95it/s] 55%|█████▌    | 220582/400000 [00:28<00:22, 7821.17it/s] 55%|█████▌    | 221385/400000 [00:28<00:22, 7881.36it/s] 56%|█████▌    | 222174/400000 [00:28<00:22, 7856.30it/s] 56%|█████▌    | 222960/400000 [00:28<00:22, 7844.55it/s] 56%|█████▌    | 223762/400000 [00:28<00:22, 7893.91it/s] 56%|█████▌    | 224567/400000 [00:29<00:22, 7937.44it/s] 56%|█████▋    | 225361/400000 [00:29<00:22, 7923.29it/s] 57%|█████▋    | 226154/400000 [00:29<00:22, 7805.17it/s] 57%|█████▋    | 226936/400000 [00:29<00:22, 7705.24it/s] 57%|█████▋    | 227708/400000 [00:29<00:22, 7697.53it/s] 57%|█████▋    | 228479/400000 [00:29<00:22, 7678.94it/s] 57%|█████▋    | 229248/400000 [00:29<00:22, 7446.80it/s] 58%|█████▊    | 230037/400000 [00:29<00:22, 7573.26it/s] 58%|█████▊    | 230806/400000 [00:29<00:22, 7606.86it/s] 58%|█████▊    | 231594/400000 [00:29<00:21, 7685.63it/s] 58%|█████▊    | 232371/400000 [00:30<00:21, 7710.41it/s] 58%|█████▊    | 233179/400000 [00:30<00:21, 7816.70it/s] 58%|█████▊    | 233993/400000 [00:30<00:20, 7909.95it/s] 59%|█████▊    | 234792/400000 [00:30<00:20, 7931.41it/s] 59%|█████▉    | 235617/400000 [00:30<00:20, 8021.68it/s] 59%|█████▉    | 236420/400000 [00:30<00:20, 8022.20it/s] 59%|█████▉    | 237224/400000 [00:30<00:20, 8025.64it/s] 60%|█████▉    | 238038/400000 [00:30<00:20, 8058.38it/s] 60%|█████▉    | 238845/400000 [00:30<00:20, 7950.56it/s] 60%|█████▉    | 239649/400000 [00:30<00:20, 7975.53it/s] 60%|██████    | 240447/400000 [00:31<00:20, 7939.77it/s] 60%|██████    | 241242/400000 [00:31<00:20, 7839.79it/s] 61%|██████    | 242027/400000 [00:31<00:20, 7774.31it/s] 61%|██████    | 242805/400000 [00:31<00:20, 7711.36it/s] 61%|██████    | 243594/400000 [00:31<00:20, 7762.47it/s] 61%|██████    | 244394/400000 [00:31<00:19, 7830.53it/s] 61%|██████▏   | 245195/400000 [00:31<00:19, 7881.80it/s] 61%|██████▏   | 245984/400000 [00:31<00:19, 7860.27it/s] 62%|██████▏   | 246771/400000 [00:31<00:19, 7802.63it/s] 62%|██████▏   | 247573/400000 [00:31<00:19, 7865.54it/s] 62%|██████▏   | 248375/400000 [00:32<00:19, 7909.94it/s] 62%|██████▏   | 249179/400000 [00:32<00:18, 7946.70it/s] 62%|██████▏   | 249990/400000 [00:32<00:18, 7993.98it/s] 63%|██████▎   | 250798/400000 [00:32<00:18, 8018.73it/s] 63%|██████▎   | 251601/400000 [00:32<00:18, 7906.15it/s] 63%|██████▎   | 252393/400000 [00:32<00:18, 7881.08it/s] 63%|██████▎   | 253186/400000 [00:32<00:18, 7895.00it/s] 63%|██████▎   | 253981/400000 [00:32<00:18, 7911.15it/s] 64%|██████▎   | 254779/400000 [00:32<00:18, 7929.57it/s] 64%|██████▍   | 255573/400000 [00:33<00:18, 7842.64it/s] 64%|██████▍   | 256358/400000 [00:33<00:18, 7830.58it/s] 64%|██████▍   | 257156/400000 [00:33<00:18, 7872.55it/s] 64%|██████▍   | 257964/400000 [00:33<00:17, 7931.72it/s] 65%|██████▍   | 258781/400000 [00:33<00:17, 7999.65it/s] 65%|██████▍   | 259582/400000 [00:33<00:17, 7946.56it/s] 65%|██████▌   | 260377/400000 [00:33<00:17, 7862.91it/s] 65%|██████▌   | 261164/400000 [00:33<00:17, 7836.08it/s] 65%|██████▌   | 261964/400000 [00:33<00:17, 7884.10it/s] 66%|██████▌   | 262770/400000 [00:33<00:17, 7935.95it/s] 66%|██████▌   | 263564/400000 [00:34<00:17, 7884.33it/s] 66%|██████▌   | 264377/400000 [00:34<00:17, 7955.16it/s] 66%|██████▋   | 265173/400000 [00:34<00:16, 7946.16it/s] 66%|██████▋   | 265984/400000 [00:34<00:16, 7994.57it/s] 67%|██████▋   | 266791/400000 [00:34<00:16, 8014.79it/s] 67%|██████▋   | 267593/400000 [00:34<00:16, 7938.70it/s] 67%|██████▋   | 268388/400000 [00:34<00:16, 7788.18it/s] 67%|██████▋   | 269179/400000 [00:34<00:16, 7823.71it/s] 67%|██████▋   | 269962/400000 [00:34<00:16, 7781.39it/s] 68%|██████▊   | 270758/400000 [00:34<00:16, 7833.69it/s] 68%|██████▊   | 271542/400000 [00:35<00:16, 7758.07it/s] 68%|██████▊   | 272341/400000 [00:35<00:16, 7824.80it/s] 68%|██████▊   | 273124/400000 [00:35<00:16, 7816.36it/s] 68%|██████▊   | 273931/400000 [00:35<00:15, 7890.54it/s] 69%|██████▊   | 274731/400000 [00:35<00:15, 7922.24it/s] 69%|██████▉   | 275524/400000 [00:35<00:15, 7861.27it/s] 69%|██████▉   | 276324/400000 [00:35<00:15, 7900.24it/s] 69%|██████▉   | 277115/400000 [00:35<00:15, 7888.50it/s] 69%|██████▉   | 277905/400000 [00:35<00:15, 7858.06it/s] 70%|██████▉   | 278706/400000 [00:35<00:15, 7901.21it/s] 70%|██████▉   | 279497/400000 [00:36<00:15, 7854.57it/s] 70%|███████   | 280307/400000 [00:36<00:15, 7926.46it/s] 70%|███████   | 281100/400000 [00:36<00:15, 7857.53it/s] 70%|███████   | 281887/400000 [00:36<00:15, 7549.54it/s] 71%|███████   | 282680/400000 [00:36<00:15, 7657.16it/s] 71%|███████   | 283471/400000 [00:36<00:15, 7730.62it/s] 71%|███████   | 284259/400000 [00:36<00:14, 7774.55it/s] 71%|███████▏  | 285048/400000 [00:36<00:14, 7806.98it/s] 71%|███████▏  | 285830/400000 [00:36<00:14, 7763.17it/s] 72%|███████▏  | 286617/400000 [00:36<00:14, 7788.17it/s] 72%|███████▏  | 287397/400000 [00:37<00:14, 7545.90it/s] 72%|███████▏  | 288159/400000 [00:37<00:14, 7567.49it/s] 72%|███████▏  | 288952/400000 [00:37<00:14, 7672.51it/s] 72%|███████▏  | 289731/400000 [00:37<00:14, 7705.52it/s] 73%|███████▎  | 290526/400000 [00:37<00:14, 7776.13it/s] 73%|███████▎  | 291315/400000 [00:37<00:13, 7808.82it/s] 73%|███████▎  | 292115/400000 [00:37<00:13, 7863.96it/s] 73%|███████▎  | 292911/400000 [00:37<00:13, 7891.69it/s] 73%|███████▎  | 293701/400000 [00:37<00:13, 7690.79it/s] 74%|███████▎  | 294472/400000 [00:37<00:14, 7492.92it/s] 74%|███████▍  | 295224/400000 [00:38<00:14, 7385.02it/s] 74%|███████▍  | 295987/400000 [00:38<00:13, 7454.87it/s] 74%|███████▍  | 296780/400000 [00:38<00:13, 7589.12it/s] 74%|███████▍  | 297584/400000 [00:38<00:13, 7717.42it/s] 75%|███████▍  | 298358/400000 [00:38<00:13, 7711.41it/s] 75%|███████▍  | 299157/400000 [00:38<00:12, 7791.74it/s] 75%|███████▍  | 299980/400000 [00:38<00:12, 7916.77it/s] 75%|███████▌  | 300773/400000 [00:38<00:12, 7781.88it/s] 75%|███████▌  | 301557/400000 [00:38<00:12, 7799.02it/s] 76%|███████▌  | 302350/400000 [00:38<00:12, 7836.65it/s] 76%|███████▌  | 303149/400000 [00:39<00:12, 7881.52it/s] 76%|███████▌  | 303955/400000 [00:39<00:12, 7931.97it/s] 76%|███████▌  | 304767/400000 [00:39<00:11, 7987.28it/s] 76%|███████▋  | 305573/400000 [00:39<00:11, 8008.63it/s] 77%|███████▋  | 306375/400000 [00:39<00:12, 7792.06it/s] 77%|███████▋  | 307156/400000 [00:39<00:11, 7784.62it/s] 77%|███████▋  | 307963/400000 [00:39<00:11, 7866.15it/s] 77%|███████▋  | 308751/400000 [00:39<00:11, 7828.21it/s] 77%|███████▋  | 309545/400000 [00:39<00:11, 7859.68it/s] 78%|███████▊  | 310335/400000 [00:40<00:11, 7866.44it/s] 78%|███████▊  | 311123/400000 [00:40<00:11, 7734.23it/s] 78%|███████▊  | 311935/400000 [00:40<00:11, 7844.11it/s] 78%|███████▊  | 312747/400000 [00:40<00:11, 7923.53it/s] 78%|███████▊  | 313567/400000 [00:40<00:10, 7999.40it/s] 79%|███████▊  | 314368/400000 [00:40<00:10, 7946.72it/s] 79%|███████▉  | 315164/400000 [00:40<00:10, 7902.82it/s] 79%|███████▉  | 315975/400000 [00:40<00:10, 7962.79it/s] 79%|███████▉  | 316772/400000 [00:40<00:10, 7954.95it/s] 79%|███████▉  | 317569/400000 [00:40<00:10, 7956.01it/s] 80%|███████▉  | 318365/400000 [00:41<00:10, 7868.21it/s] 80%|███████▉  | 319172/400000 [00:41<00:10, 7926.41it/s] 80%|███████▉  | 319966/400000 [00:41<00:10, 7800.74it/s] 80%|████████  | 320747/400000 [00:41<00:10, 7789.52it/s] 80%|████████  | 321552/400000 [00:41<00:09, 7863.89it/s] 81%|████████  | 322339/400000 [00:41<00:09, 7859.69it/s] 81%|████████  | 323126/400000 [00:41<00:09, 7844.54it/s] 81%|████████  | 323911/400000 [00:41<00:09, 7805.10it/s] 81%|████████  | 324692/400000 [00:41<00:09, 7733.83it/s] 81%|████████▏ | 325466/400000 [00:41<00:09, 7714.45it/s] 82%|████████▏ | 326238/400000 [00:42<00:09, 7713.69it/s] 82%|████████▏ | 327010/400000 [00:42<00:09, 7708.61it/s] 82%|████████▏ | 327817/400000 [00:42<00:09, 7812.35it/s] 82%|████████▏ | 328611/400000 [00:42<00:09, 7848.28it/s] 82%|████████▏ | 329398/400000 [00:42<00:08, 7852.71it/s] 83%|████████▎ | 330184/400000 [00:42<00:08, 7760.44it/s] 83%|████████▎ | 330982/400000 [00:42<00:08, 7823.52it/s] 83%|████████▎ | 331776/400000 [00:42<00:08, 7855.71it/s] 83%|████████▎ | 332562/400000 [00:42<00:08, 7784.27it/s] 83%|████████▎ | 333341/400000 [00:42<00:08, 7751.74it/s] 84%|████████▎ | 334117/400000 [00:43<00:08, 7705.94it/s] 84%|████████▎ | 334888/400000 [00:43<00:08, 7681.99it/s] 84%|████████▍ | 335693/400000 [00:43<00:08, 7787.00it/s] 84%|████████▍ | 336476/400000 [00:43<00:08, 7797.97it/s] 84%|████████▍ | 337264/400000 [00:43<00:08, 7820.83it/s] 85%|████████▍ | 338047/400000 [00:43<00:07, 7821.89it/s] 85%|████████▍ | 338830/400000 [00:43<00:07, 7760.15it/s] 85%|████████▍ | 339607/400000 [00:43<00:07, 7728.09it/s] 85%|████████▌ | 340381/400000 [00:43<00:07, 7593.76it/s] 85%|████████▌ | 341142/400000 [00:43<00:08, 7252.67it/s] 85%|████████▌ | 341871/400000 [00:44<00:08, 7263.15it/s] 86%|████████▌ | 342623/400000 [00:44<00:07, 7338.07it/s] 86%|████████▌ | 343388/400000 [00:44<00:07, 7427.97it/s] 86%|████████▌ | 344186/400000 [00:44<00:07, 7582.98it/s] 86%|████████▌ | 344987/400000 [00:44<00:07, 7704.04it/s] 86%|████████▋ | 345760/400000 [00:44<00:07, 7396.42it/s] 87%|████████▋ | 346513/400000 [00:44<00:07, 7435.19it/s] 87%|████████▋ | 347316/400000 [00:44<00:06, 7602.18it/s] 87%|████████▋ | 348103/400000 [00:44<00:06, 7680.46it/s] 87%|████████▋ | 348879/400000 [00:44<00:06, 7702.62it/s] 87%|████████▋ | 349663/400000 [00:45<00:06, 7742.41it/s] 88%|████████▊ | 350457/400000 [00:45<00:06, 7798.61it/s] 88%|████████▊ | 351251/400000 [00:45<00:06, 7839.25it/s] 88%|████████▊ | 352036/400000 [00:45<00:06, 7787.46it/s] 88%|████████▊ | 352822/400000 [00:45<00:06, 7807.30it/s] 88%|████████▊ | 353604/400000 [00:45<00:05, 7772.77it/s] 89%|████████▊ | 354390/400000 [00:45<00:05, 7798.58it/s] 89%|████████▉ | 355175/400000 [00:45<00:05, 7812.98it/s] 89%|████████▉ | 355965/400000 [00:45<00:05, 7837.47it/s] 89%|████████▉ | 356755/400000 [00:45<00:05, 7853.76it/s] 89%|████████▉ | 357541/400000 [00:46<00:05, 7719.79it/s] 90%|████████▉ | 358314/400000 [00:46<00:05, 7607.81it/s] 90%|████████▉ | 359092/400000 [00:46<00:05, 7655.84it/s] 90%|████████▉ | 359859/400000 [00:46<00:05, 7633.36it/s] 90%|█████████ | 360643/400000 [00:46<00:05, 7693.27it/s] 90%|█████████ | 361413/400000 [00:46<00:05, 7667.83it/s] 91%|█████████ | 362186/400000 [00:46<00:04, 7684.27it/s] 91%|█████████ | 362955/400000 [00:46<00:04, 7610.98it/s] 91%|█████████ | 363732/400000 [00:46<00:04, 7655.48it/s] 91%|█████████ | 364510/400000 [00:47<00:04, 7691.68it/s] 91%|█████████▏| 365280/400000 [00:47<00:04, 7658.38it/s] 92%|█████████▏| 366058/400000 [00:47<00:04, 7693.34it/s] 92%|█████████▏| 366839/400000 [00:47<00:04, 7727.42it/s] 92%|█████████▏| 367628/400000 [00:47<00:04, 7774.76it/s] 92%|█████████▏| 368406/400000 [00:47<00:04, 7756.61it/s] 92%|█████████▏| 369182/400000 [00:47<00:03, 7720.62it/s] 92%|█████████▏| 369973/400000 [00:47<00:03, 7774.01it/s] 93%|█████████▎| 370757/400000 [00:47<00:03, 7793.46it/s] 93%|█████████▎| 371537/400000 [00:47<00:03, 7774.75it/s] 93%|█████████▎| 372315/400000 [00:48<00:03, 7775.59it/s] 93%|█████████▎| 373094/400000 [00:48<00:03, 7779.24it/s] 93%|█████████▎| 373914/400000 [00:48<00:03, 7898.74it/s] 94%|█████████▎| 374714/400000 [00:48<00:03, 7927.27it/s] 94%|█████████▍| 375520/400000 [00:48<00:03, 7966.62it/s] 94%|█████████▍| 376317/400000 [00:48<00:02, 7921.84it/s] 94%|█████████▍| 377112/400000 [00:48<00:02, 7927.60it/s] 94%|█████████▍| 377905/400000 [00:48<00:02, 7898.74it/s] 95%|█████████▍| 378696/400000 [00:48<00:02, 7872.83it/s] 95%|█████████▍| 379502/400000 [00:48<00:02, 7925.35it/s] 95%|█████████▌| 380301/400000 [00:49<00:02, 7943.92it/s] 95%|█████████▌| 381096/400000 [00:49<00:02, 7828.19it/s] 95%|█████████▌| 381880/400000 [00:49<00:02, 7775.19it/s] 96%|█████████▌| 382658/400000 [00:49<00:02, 7755.77it/s] 96%|█████████▌| 383456/400000 [00:49<00:02, 7814.75it/s] 96%|█████████▌| 384238/400000 [00:49<00:02, 7778.17it/s] 96%|█████████▋| 385017/400000 [00:49<00:01, 7780.75it/s] 96%|█████████▋| 385796/400000 [00:49<00:01, 7749.71it/s] 97%|█████████▋| 386572/400000 [00:49<00:01, 7708.65it/s] 97%|█████████▋| 387373/400000 [00:49<00:01, 7793.81it/s] 97%|█████████▋| 388165/400000 [00:50<00:01, 7830.38it/s] 97%|█████████▋| 388949/400000 [00:50<00:01, 7765.64it/s] 97%|█████████▋| 389726/400000 [00:50<00:01, 7742.54it/s] 98%|█████████▊| 390513/400000 [00:50<00:01, 7780.03it/s] 98%|█████████▊| 391310/400000 [00:50<00:01, 7835.95it/s] 98%|█████████▊| 392102/400000 [00:50<00:01, 7860.84it/s] 98%|█████████▊| 392917/400000 [00:50<00:00, 7944.08it/s] 98%|█████████▊| 393712/400000 [00:50<00:00, 7909.71it/s] 99%|█████████▊| 394504/400000 [00:50<00:00, 7908.19it/s] 99%|█████████▉| 395301/400000 [00:50<00:00, 7923.97it/s] 99%|█████████▉| 396094/400000 [00:51<00:00, 7902.33it/s] 99%|█████████▉| 396885/400000 [00:51<00:00, 7895.77it/s] 99%|█████████▉| 397675/400000 [00:51<00:00, 7641.11it/s]100%|█████████▉| 398444/400000 [00:51<00:00, 7653.95it/s]100%|█████████▉| 399255/400000 [00:51<00:00, 7783.06it/s]100%|█████████▉| 399999/400000 [00:51<00:00, 7759.68it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f995c50e940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011086500765428012 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.0111496460477644 	 Accuracy: 54

  model saves at 54% accuracy 

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
2020-05-14 22:24:19.475385: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 22:24:19.479930: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-05-14 22:24:19.480096: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55605bbbe620 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 22:24:19.480113: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f99059d7550> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8813 - accuracy: 0.4860
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6820 - accuracy: 0.4990 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6308 - accuracy: 0.5023
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7510 - accuracy: 0.4945
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8138 - accuracy: 0.4904
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7995 - accuracy: 0.4913
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7893 - accuracy: 0.4920
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.8219 - accuracy: 0.4899
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.8063 - accuracy: 0.4909
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7939 - accuracy: 0.4917
11000/25000 [============>.................] - ETA: 4s - loss: 7.7712 - accuracy: 0.4932
12000/25000 [=============>................] - ETA: 3s - loss: 7.7446 - accuracy: 0.4949
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7268 - accuracy: 0.4961
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7225 - accuracy: 0.4964
15000/25000 [=================>............] - ETA: 2s - loss: 7.6952 - accuracy: 0.4981
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6887 - accuracy: 0.4986
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6901 - accuracy: 0.4985
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6905 - accuracy: 0.4984
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6957 - accuracy: 0.4981
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6651 - accuracy: 0.5001
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6659 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6708 - accuracy: 0.4997
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6653 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 9s 350us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f98bc2c66a0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f9905975a58> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4397 - crf_viterbi_accuracy: 0.6800 - val_loss: 1.4636 - val_crf_viterbi_accuracy: 0.6533

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

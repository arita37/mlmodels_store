
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f47e9097fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 23:11:41.082211
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-14 23:11:41.085509
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-14 23:11:41.088398
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-14 23:11:41.091184
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f47f50af400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 361683.5000
Epoch 2/10

1/1 [==============================] - 0s 92ms/step - loss: 246018.3438
Epoch 3/10

1/1 [==============================] - 0s 95ms/step - loss: 138916.8281
Epoch 4/10

1/1 [==============================] - 0s 88ms/step - loss: 67646.9609
Epoch 5/10

1/1 [==============================] - 0s 93ms/step - loss: 32586.8105
Epoch 6/10

1/1 [==============================] - 0s 94ms/step - loss: 17575.2656
Epoch 7/10

1/1 [==============================] - 0s 89ms/step - loss: 10802.5127
Epoch 8/10

1/1 [==============================] - 0s 90ms/step - loss: 7352.7144
Epoch 9/10

1/1 [==============================] - 0s 91ms/step - loss: 5425.8916
Epoch 10/10

1/1 [==============================] - 0s 93ms/step - loss: 4262.0571

  #### Inference Need return ypred, ytrue ######################### 
[[ 2.06795365e-01  9.72051907e+00  1.15436230e+01  1.22991753e+01
   1.06721973e+01  1.01570082e+01  1.09623489e+01  1.10730667e+01
   9.11695385e+00  1.01049623e+01  1.10341778e+01  1.17146549e+01
   1.27488232e+01  8.76433754e+00  1.01124153e+01  1.01374311e+01
   1.05788250e+01  1.18213243e+01  1.10553970e+01  1.10886106e+01
   1.15513153e+01  1.10098648e+01  1.01247702e+01  1.10426712e+01
   1.05638523e+01  9.72138023e+00  1.15625820e+01  8.72785378e+00
   1.10384550e+01  1.11659460e+01  9.70840359e+00  1.00913782e+01
   1.12720337e+01  1.23467855e+01  1.23960457e+01  1.07135248e+01
   1.01969414e+01  1.05431595e+01  1.16040354e+01  1.11758270e+01
   1.24501286e+01  1.17946987e+01  8.72114182e+00  1.33952160e+01
   1.28438797e+01  1.39387817e+01  9.89905930e+00  1.01439533e+01
   1.34103918e+01  1.07732639e+01  1.05307980e+01  1.00361700e+01
   9.97719193e+00  1.19793501e+01  1.34807644e+01  1.35858135e+01
   7.65639162e+00  1.06394863e+01  1.08119097e+01  1.06989193e+01
  -5.73923826e-01 -4.79964733e-01  2.74376297e+00 -6.47968292e-01
   2.28603721e-01 -2.05606389e+00 -1.13272458e-01 -2.85611928e-01
  -1.63153791e+00 -1.30125666e+00  4.78347242e-02  3.35055768e-01
   2.30746627e-01  8.53958353e-02 -1.38057661e+00 -7.78117180e-02
   5.73711932e-01  2.19444704e+00 -3.65279168e-01 -5.07457614e-01
   2.46987090e-01 -6.12083554e-01  1.79800677e+00  3.98818254e-01
   1.27816188e+00 -2.12133241e+00 -7.47721374e-01  1.98136306e+00
  -1.42389476e-01 -1.23836565e+00  1.13814175e+00 -3.72965038e-01
  -1.91149831e-01  1.44156218e+00  1.20367599e+00  8.47329855e-01
  -1.30593204e+00  1.51961851e+00  7.34582543e-04  6.59206986e-01
   6.63807392e-02  6.98771238e-01  1.54734343e-01 -6.29570305e-01
   9.56309259e-01  2.21281576e+00  4.41913128e-01  5.01613379e-01
  -1.72687149e+00 -2.05292034e+00  3.60443413e-01  1.65962386e+00
   1.17466581e+00  1.49011827e+00  1.75484324e+00 -3.67026806e-01
   9.73945677e-01 -1.32538724e+00 -1.10078263e+00  1.05872536e+00
  -3.43704343e-01 -1.10099649e+00 -2.83701032e-01 -3.65431637e-01
   5.44700742e-01 -6.69448316e-01 -4.66510952e-02 -1.45493960e+00
   7.73990870e-01 -1.45144299e-01  7.38442123e-01  1.34076804e-01
   6.43792212e-01  1.05897069e+00  1.62568557e+00  2.01792288e+00
  -1.07529032e+00  5.19012362e-02  1.08689845e-01 -3.03735733e-01
  -1.27778566e+00  3.47297490e-01 -9.75717425e-01 -2.73452699e-01
  -1.32034540e+00  7.29364336e-01 -4.74842817e-01  4.24496770e-01
   1.77467155e+00  2.68714857e+00  1.36401761e+00  1.84577107e-01
  -5.64985514e-01 -6.71148539e-01 -3.58029246e-01 -1.01507962e-01
  -7.16896534e-01 -2.21865916e+00  4.35878336e-01  1.33087134e+00
   1.01395011e+00  9.77978051e-01  7.65987813e-01 -1.07035160e-01
  -4.02947485e-01  3.48295480e-01 -3.89485151e-01  1.91760778e+00
   1.56315279e+00 -4.53975558e-01  1.73682943e-01 -2.22646451e+00
   2.70067883e+00 -1.15130424e+00  1.47294864e-01  1.11296415e+00
  -2.16698915e-01 -8.13720345e-01  1.17030632e+00 -2.78783154e+00
   3.87422323e-01  1.03285742e+01  1.00636940e+01  1.12991724e+01
   9.52279568e+00  1.02395773e+01  1.00732603e+01  1.02941408e+01
   1.04380407e+01  1.06004028e+01  1.11181221e+01  1.17705135e+01
   1.08676434e+01  1.02659397e+01  1.09482250e+01  1.09805384e+01
   8.67266846e+00  1.18741255e+01  1.22515602e+01  1.18643484e+01
   1.00610609e+01  9.96658516e+00  1.16318092e+01  9.98587513e+00
   1.10696926e+01  1.03586435e+01  1.06478748e+01  1.04807320e+01
   1.11598253e+01  1.06579800e+01  1.15198765e+01  9.77027798e+00
   1.10472088e+01  1.03821602e+01  1.04352188e+01  1.05022993e+01
   1.00193701e+01  1.11538181e+01  1.21078634e+01  1.07424088e+01
   1.10422029e+01  1.00761805e+01  1.00857210e+01  9.91071510e+00
   1.09547100e+01  1.03681803e+01  1.13230171e+01  1.06742182e+01
   9.06995201e+00  1.03500624e+01  1.03637800e+01  1.11109238e+01
   1.08986168e+01  1.08181734e+01  1.19566126e+01  7.49481964e+00
   1.10885677e+01  7.43670273e+00  1.17433071e+01  1.14672976e+01
   1.08888924e+00  6.90281391e-02  2.13573408e+00  1.02135324e+00
   4.78918195e-01  2.72688627e-01  7.02860177e-01  1.41916394e+00
   3.35954475e+00  2.64907217e+00  1.04591668e-01  4.93291378e-01
   2.44927812e+00  1.50007677e+00  2.70401239e-02  1.16022873e+00
   1.72765923e+00  1.22956359e+00  4.07742620e-01  2.13598084e+00
   2.94251919e-01  2.35558319e+00  3.26313114e+00  5.52095592e-01
   8.16894710e-01  3.33233953e-01  5.67222774e-01  3.53995514e+00
   8.34011555e-01  1.74423695e-01  3.95525992e-01  5.67685246e-01
   6.81023240e-01  1.84064901e+00  2.32473278e+00  2.22934198e+00
   2.36038566e-01  2.94472694e+00  2.69687295e+00  2.47071028e+00
   2.18830013e+00  2.12057614e+00  2.61431694e-01  5.76834798e-01
   8.13723087e-01  6.84695363e-01  1.59073424e+00  3.14974427e-01
   1.74597692e+00  6.94720507e-01  7.57041574e-01  2.13880134e+00
   4.49075401e-01  5.02428770e-01  3.61633837e-01  1.26970410e-01
   1.11959219e+00  1.10557175e+00  1.10137558e+00  9.59577322e-01
   4.50158417e-01  2.31591702e+00  2.14926958e+00  1.98339176e+00
   1.31045461e-01  1.08294022e+00  4.73491132e-01  2.31664360e-01
   1.50514317e+00  3.11774063e+00  2.48903036e+00  4.29483950e-01
   3.84385347e-01  5.74260235e-01  1.21277809e+00  7.15180635e-02
   1.15021086e+00  1.31535888e-01  7.30665445e-01  2.73605871e+00
   1.26668954e+00  5.36130428e-01  2.29412079e+00  2.06157982e-01
   4.23895299e-01  8.32775831e-02  1.92373097e-01  6.56798482e-01
   1.83404851e+00  4.30306017e-01  1.38170433e+00  2.31330872e-01
   1.93666327e+00  2.16138792e+00  3.17563915e+00  1.43434632e+00
   1.61057711e-01  1.20217967e+00  7.50079870e-01  2.20923853e+00
   1.11248612e-01  1.18610489e+00  8.07402253e-01  1.80021334e+00
   5.08446097e-01  1.71090603e-01  6.01140499e-01  2.10856581e+00
   3.69272661e+00  1.71705389e+00  2.78695631e+00  1.13909066e+00
   6.59727752e-01  4.11097944e-01  3.56195152e-01  2.45742941e+00
   2.18771398e-01  3.91302705e-01  1.61911607e-01  5.14597774e-01
   6.77211332e+00 -1.21697512e+01 -1.15115242e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 23:11:50.346721
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    91.034
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-14 23:11:50.350009
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8312.62
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-14 23:11:50.352581
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   90.8329
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-14 23:11:50.355253
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -743.454
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139946472388144
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139945262228088
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139945262228592
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139945262229096
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139945262229600
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139945262230104

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f47e8fa5be0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.429629
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.401508
grad_step = 000002, loss = 0.379896
grad_step = 000003, loss = 0.356958
grad_step = 000004, loss = 0.334779
grad_step = 000005, loss = 0.319543
grad_step = 000006, loss = 0.315033
grad_step = 000007, loss = 0.302320
grad_step = 000008, loss = 0.283710
grad_step = 000009, loss = 0.268676
grad_step = 000010, loss = 0.259433
grad_step = 000011, loss = 0.252433
grad_step = 000012, loss = 0.244018
grad_step = 000013, loss = 0.234105
grad_step = 000014, loss = 0.225416
grad_step = 000015, loss = 0.218220
grad_step = 000016, loss = 0.210463
grad_step = 000017, loss = 0.201544
grad_step = 000018, loss = 0.193515
grad_step = 000019, loss = 0.185880
grad_step = 000020, loss = 0.177172
grad_step = 000021, loss = 0.167407
grad_step = 000022, loss = 0.157861
grad_step = 000023, loss = 0.149746
grad_step = 000024, loss = 0.143159
grad_step = 000025, loss = 0.136686
grad_step = 000026, loss = 0.130038
grad_step = 000027, loss = 0.123551
grad_step = 000028, loss = 0.117560
grad_step = 000029, loss = 0.112193
grad_step = 000030, loss = 0.107117
grad_step = 000031, loss = 0.101913
grad_step = 000032, loss = 0.096672
grad_step = 000033, loss = 0.091763
grad_step = 000034, loss = 0.087183
grad_step = 000035, loss = 0.082589
grad_step = 000036, loss = 0.077986
grad_step = 000037, loss = 0.073741
grad_step = 000038, loss = 0.069905
grad_step = 000039, loss = 0.066270
grad_step = 000040, loss = 0.062747
grad_step = 000041, loss = 0.059338
grad_step = 000042, loss = 0.056123
grad_step = 000043, loss = 0.053062
grad_step = 000044, loss = 0.050017
grad_step = 000045, loss = 0.047061
grad_step = 000046, loss = 0.044271
grad_step = 000047, loss = 0.041621
grad_step = 000048, loss = 0.039141
grad_step = 000049, loss = 0.036827
grad_step = 000050, loss = 0.034645
grad_step = 000051, loss = 0.032608
grad_step = 000052, loss = 0.030666
grad_step = 000053, loss = 0.028793
grad_step = 000054, loss = 0.027003
grad_step = 000055, loss = 0.025299
grad_step = 000056, loss = 0.023695
grad_step = 000057, loss = 0.022171
grad_step = 000058, loss = 0.020743
grad_step = 000059, loss = 0.019419
grad_step = 000060, loss = 0.018180
grad_step = 000061, loss = 0.017019
grad_step = 000062, loss = 0.015919
grad_step = 000063, loss = 0.014886
grad_step = 000064, loss = 0.013912
grad_step = 000065, loss = 0.012986
grad_step = 000066, loss = 0.012108
grad_step = 000067, loss = 0.011296
grad_step = 000068, loss = 0.010542
grad_step = 000069, loss = 0.009832
grad_step = 000070, loss = 0.009181
grad_step = 000071, loss = 0.008582
grad_step = 000072, loss = 0.008020
grad_step = 000073, loss = 0.007492
grad_step = 000074, loss = 0.007005
grad_step = 000075, loss = 0.006551
grad_step = 000076, loss = 0.006128
grad_step = 000077, loss = 0.005741
grad_step = 000078, loss = 0.005393
grad_step = 000079, loss = 0.005072
grad_step = 000080, loss = 0.004781
grad_step = 000081, loss = 0.004519
grad_step = 000082, loss = 0.004277
grad_step = 000083, loss = 0.004055
grad_step = 000084, loss = 0.003855
grad_step = 000085, loss = 0.003673
grad_step = 000086, loss = 0.003508
grad_step = 000087, loss = 0.003361
grad_step = 000088, loss = 0.003230
grad_step = 000089, loss = 0.003112
grad_step = 000090, loss = 0.003007
grad_step = 000091, loss = 0.002912
grad_step = 000092, loss = 0.002826
grad_step = 000093, loss = 0.002750
grad_step = 000094, loss = 0.002682
grad_step = 000095, loss = 0.002621
grad_step = 000096, loss = 0.002568
grad_step = 000097, loss = 0.002519
grad_step = 000098, loss = 0.002477
grad_step = 000099, loss = 0.002439
grad_step = 000100, loss = 0.002405
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002375
grad_step = 000102, loss = 0.002349
grad_step = 000103, loss = 0.002324
grad_step = 000104, loss = 0.002302
grad_step = 000105, loss = 0.002282
grad_step = 000106, loss = 0.002263
grad_step = 000107, loss = 0.002246
grad_step = 000108, loss = 0.002231
grad_step = 000109, loss = 0.002216
grad_step = 000110, loss = 0.002202
grad_step = 000111, loss = 0.002189
grad_step = 000112, loss = 0.002177
grad_step = 000113, loss = 0.002165
grad_step = 000114, loss = 0.002153
grad_step = 000115, loss = 0.002143
grad_step = 000116, loss = 0.002134
grad_step = 000117, loss = 0.002123
grad_step = 000118, loss = 0.002113
grad_step = 000119, loss = 0.002104
grad_step = 000120, loss = 0.002095
grad_step = 000121, loss = 0.002086
grad_step = 000122, loss = 0.002077
grad_step = 000123, loss = 0.002069
grad_step = 000124, loss = 0.002061
grad_step = 000125, loss = 0.002054
grad_step = 000126, loss = 0.002045
grad_step = 000127, loss = 0.002038
grad_step = 000128, loss = 0.002031
grad_step = 000129, loss = 0.002022
grad_step = 000130, loss = 0.002016
grad_step = 000131, loss = 0.002008
grad_step = 000132, loss = 0.002001
grad_step = 000133, loss = 0.001993
grad_step = 000134, loss = 0.001986
grad_step = 000135, loss = 0.001979
grad_step = 000136, loss = 0.001973
grad_step = 000137, loss = 0.001966
grad_step = 000138, loss = 0.001959
grad_step = 000139, loss = 0.001953
grad_step = 000140, loss = 0.001946
grad_step = 000141, loss = 0.001941
grad_step = 000142, loss = 0.001936
grad_step = 000143, loss = 0.001930
grad_step = 000144, loss = 0.001923
grad_step = 000145, loss = 0.001916
grad_step = 000146, loss = 0.001910
grad_step = 000147, loss = 0.001906
grad_step = 000148, loss = 0.001902
grad_step = 000149, loss = 0.001897
grad_step = 000150, loss = 0.001896
grad_step = 000151, loss = 0.001891
grad_step = 000152, loss = 0.001881
grad_step = 000153, loss = 0.001874
grad_step = 000154, loss = 0.001872
grad_step = 000155, loss = 0.001870
grad_step = 000156, loss = 0.001864
grad_step = 000157, loss = 0.001856
grad_step = 000158, loss = 0.001851
grad_step = 000159, loss = 0.001849
grad_step = 000160, loss = 0.001847
grad_step = 000161, loss = 0.001843
grad_step = 000162, loss = 0.001839
grad_step = 000163, loss = 0.001834
grad_step = 000164, loss = 0.001829
grad_step = 000165, loss = 0.001824
grad_step = 000166, loss = 0.001820
grad_step = 000167, loss = 0.001817
grad_step = 000168, loss = 0.001814
grad_step = 000169, loss = 0.001810
grad_step = 000170, loss = 0.001807
grad_step = 000171, loss = 0.001805
grad_step = 000172, loss = 0.001806
grad_step = 000173, loss = 0.001812
grad_step = 000174, loss = 0.001820
grad_step = 000175, loss = 0.001829
grad_step = 000176, loss = 0.001837
grad_step = 000177, loss = 0.001841
grad_step = 000178, loss = 0.001842
grad_step = 000179, loss = 0.001820
grad_step = 000180, loss = 0.001786
grad_step = 000181, loss = 0.001761
grad_step = 000182, loss = 0.001759
grad_step = 000183, loss = 0.001771
grad_step = 000184, loss = 0.001777
grad_step = 000185, loss = 0.001771
grad_step = 000186, loss = 0.001759
grad_step = 000187, loss = 0.001749
grad_step = 000188, loss = 0.001740
grad_step = 000189, loss = 0.001729
grad_step = 000190, loss = 0.001720
grad_step = 000191, loss = 0.001716
grad_step = 000192, loss = 0.001717
grad_step = 000193, loss = 0.001716
grad_step = 000194, loss = 0.001708
grad_step = 000195, loss = 0.001695
grad_step = 000196, loss = 0.001685
grad_step = 000197, loss = 0.001681
grad_step = 000198, loss = 0.001675
grad_step = 000199, loss = 0.001667
grad_step = 000200, loss = 0.001660
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001657
grad_step = 000202, loss = 0.001658
grad_step = 000203, loss = 0.001660
grad_step = 000204, loss = 0.001667
grad_step = 000205, loss = 0.001686
grad_step = 000206, loss = 0.001713
grad_step = 000207, loss = 0.001738
grad_step = 000208, loss = 0.001705
grad_step = 000209, loss = 0.001666
grad_step = 000210, loss = 0.001629
grad_step = 000211, loss = 0.001604
grad_step = 000212, loss = 0.001599
grad_step = 000213, loss = 0.001618
grad_step = 000214, loss = 0.001645
grad_step = 000215, loss = 0.001648
grad_step = 000216, loss = 0.001641
grad_step = 000217, loss = 0.001627
grad_step = 000218, loss = 0.001621
grad_step = 000219, loss = 0.001589
grad_step = 000220, loss = 0.001564
grad_step = 000221, loss = 0.001561
grad_step = 000222, loss = 0.001581
grad_step = 000223, loss = 0.001606
grad_step = 000224, loss = 0.001587
grad_step = 000225, loss = 0.001563
grad_step = 000226, loss = 0.001550
grad_step = 000227, loss = 0.001554
grad_step = 000228, loss = 0.001560
grad_step = 000229, loss = 0.001554
grad_step = 000230, loss = 0.001553
grad_step = 000231, loss = 0.001554
grad_step = 000232, loss = 0.001549
grad_step = 000233, loss = 0.001537
grad_step = 000234, loss = 0.001527
grad_step = 000235, loss = 0.001529
grad_step = 000236, loss = 0.001537
grad_step = 000237, loss = 0.001541
grad_step = 000238, loss = 0.001539
grad_step = 000239, loss = 0.001541
grad_step = 000240, loss = 0.001552
grad_step = 000241, loss = 0.001574
grad_step = 000242, loss = 0.001565
grad_step = 000243, loss = 0.001545
grad_step = 000244, loss = 0.001522
grad_step = 000245, loss = 0.001519
grad_step = 000246, loss = 0.001526
grad_step = 000247, loss = 0.001522
grad_step = 000248, loss = 0.001511
grad_step = 000249, loss = 0.001510
grad_step = 000250, loss = 0.001518
grad_step = 000251, loss = 0.001520
grad_step = 000252, loss = 0.001511
grad_step = 000253, loss = 0.001502
grad_step = 000254, loss = 0.001502
grad_step = 000255, loss = 0.001507
grad_step = 000256, loss = 0.001509
grad_step = 000257, loss = 0.001504
grad_step = 000258, loss = 0.001498
grad_step = 000259, loss = 0.001495
grad_step = 000260, loss = 0.001497
grad_step = 000261, loss = 0.001500
grad_step = 000262, loss = 0.001499
grad_step = 000263, loss = 0.001497
grad_step = 000264, loss = 0.001494
grad_step = 000265, loss = 0.001493
grad_step = 000266, loss = 0.001496
grad_step = 000267, loss = 0.001501
grad_step = 000268, loss = 0.001508
grad_step = 000269, loss = 0.001521
grad_step = 000270, loss = 0.001541
grad_step = 000271, loss = 0.001582
grad_step = 000272, loss = 0.001612
grad_step = 000273, loss = 0.001643
grad_step = 000274, loss = 0.001598
grad_step = 000275, loss = 0.001553
grad_step = 000276, loss = 0.001521
grad_step = 000277, loss = 0.001503
grad_step = 000278, loss = 0.001495
grad_step = 000279, loss = 0.001505
grad_step = 000280, loss = 0.001530
grad_step = 000281, loss = 0.001535
grad_step = 000282, loss = 0.001508
grad_step = 000283, loss = 0.001479
grad_step = 000284, loss = 0.001479
grad_step = 000285, loss = 0.001494
grad_step = 000286, loss = 0.001500
grad_step = 000287, loss = 0.001494
grad_step = 000288, loss = 0.001488
grad_step = 000289, loss = 0.001485
grad_step = 000290, loss = 0.001478
grad_step = 000291, loss = 0.001472
grad_step = 000292, loss = 0.001472
grad_step = 000293, loss = 0.001479
grad_step = 000294, loss = 0.001485
grad_step = 000295, loss = 0.001480
grad_step = 000296, loss = 0.001471
grad_step = 000297, loss = 0.001464
grad_step = 000298, loss = 0.001464
grad_step = 000299, loss = 0.001466
grad_step = 000300, loss = 0.001468
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001468
grad_step = 000302, loss = 0.001468
grad_step = 000303, loss = 0.001468
grad_step = 000304, loss = 0.001466
grad_step = 000305, loss = 0.001462
grad_step = 000306, loss = 0.001458
grad_step = 000307, loss = 0.001455
grad_step = 000308, loss = 0.001454
grad_step = 000309, loss = 0.001454
grad_step = 000310, loss = 0.001453
grad_step = 000311, loss = 0.001452
grad_step = 000312, loss = 0.001451
grad_step = 000313, loss = 0.001451
grad_step = 000314, loss = 0.001452
grad_step = 000315, loss = 0.001454
grad_step = 000316, loss = 0.001456
grad_step = 000317, loss = 0.001459
grad_step = 000318, loss = 0.001464
grad_step = 000319, loss = 0.001471
grad_step = 000320, loss = 0.001484
grad_step = 000321, loss = 0.001496
grad_step = 000322, loss = 0.001517
grad_step = 000323, loss = 0.001526
grad_step = 000324, loss = 0.001537
grad_step = 000325, loss = 0.001521
grad_step = 000326, loss = 0.001501
grad_step = 000327, loss = 0.001469
grad_step = 000328, loss = 0.001446
grad_step = 000329, loss = 0.001438
grad_step = 000330, loss = 0.001445
grad_step = 000331, loss = 0.001459
grad_step = 000332, loss = 0.001469
grad_step = 000333, loss = 0.001472
grad_step = 000334, loss = 0.001463
grad_step = 000335, loss = 0.001450
grad_step = 000336, loss = 0.001438
grad_step = 000337, loss = 0.001432
grad_step = 000338, loss = 0.001432
grad_step = 000339, loss = 0.001435
grad_step = 000340, loss = 0.001439
grad_step = 000341, loss = 0.001442
grad_step = 000342, loss = 0.001444
grad_step = 000343, loss = 0.001444
grad_step = 000344, loss = 0.001444
grad_step = 000345, loss = 0.001440
grad_step = 000346, loss = 0.001436
grad_step = 000347, loss = 0.001430
grad_step = 000348, loss = 0.001426
grad_step = 000349, loss = 0.001423
grad_step = 000350, loss = 0.001421
grad_step = 000351, loss = 0.001421
grad_step = 000352, loss = 0.001420
grad_step = 000353, loss = 0.001420
grad_step = 000354, loss = 0.001420
grad_step = 000355, loss = 0.001421
grad_step = 000356, loss = 0.001422
grad_step = 000357, loss = 0.001424
grad_step = 000358, loss = 0.001426
grad_step = 000359, loss = 0.001430
grad_step = 000360, loss = 0.001434
grad_step = 000361, loss = 0.001444
grad_step = 000362, loss = 0.001454
grad_step = 000363, loss = 0.001473
grad_step = 000364, loss = 0.001485
grad_step = 000365, loss = 0.001505
grad_step = 000366, loss = 0.001504
grad_step = 000367, loss = 0.001502
grad_step = 000368, loss = 0.001483
grad_step = 000369, loss = 0.001461
grad_step = 000370, loss = 0.001436
grad_step = 000371, loss = 0.001415
grad_step = 000372, loss = 0.001407
grad_step = 000373, loss = 0.001411
grad_step = 000374, loss = 0.001424
grad_step = 000375, loss = 0.001436
grad_step = 000376, loss = 0.001439
grad_step = 000377, loss = 0.001429
grad_step = 000378, loss = 0.001415
grad_step = 000379, loss = 0.001404
grad_step = 000380, loss = 0.001401
grad_step = 000381, loss = 0.001403
grad_step = 000382, loss = 0.001405
grad_step = 000383, loss = 0.001406
grad_step = 000384, loss = 0.001403
grad_step = 000385, loss = 0.001401
grad_step = 000386, loss = 0.001401
grad_step = 000387, loss = 0.001403
grad_step = 000388, loss = 0.001405
grad_step = 000389, loss = 0.001409
grad_step = 000390, loss = 0.001406
grad_step = 000391, loss = 0.001404
grad_step = 000392, loss = 0.001399
grad_step = 000393, loss = 0.001396
grad_step = 000394, loss = 0.001395
grad_step = 000395, loss = 0.001394
grad_step = 000396, loss = 0.001393
grad_step = 000397, loss = 0.001392
grad_step = 000398, loss = 0.001389
grad_step = 000399, loss = 0.001386
grad_step = 000400, loss = 0.001384
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001382
grad_step = 000402, loss = 0.001382
grad_step = 000403, loss = 0.001381
grad_step = 000404, loss = 0.001380
grad_step = 000405, loss = 0.001379
grad_step = 000406, loss = 0.001377
grad_step = 000407, loss = 0.001376
grad_step = 000408, loss = 0.001375
grad_step = 000409, loss = 0.001375
grad_step = 000410, loss = 0.001374
grad_step = 000411, loss = 0.001374
grad_step = 000412, loss = 0.001373
grad_step = 000413, loss = 0.001372
grad_step = 000414, loss = 0.001371
grad_step = 000415, loss = 0.001370
grad_step = 000416, loss = 0.001369
grad_step = 000417, loss = 0.001369
grad_step = 000418, loss = 0.001370
grad_step = 000419, loss = 0.001372
grad_step = 000420, loss = 0.001377
grad_step = 000421, loss = 0.001386
grad_step = 000422, loss = 0.001408
grad_step = 000423, loss = 0.001445
grad_step = 000424, loss = 0.001526
grad_step = 000425, loss = 0.001600
grad_step = 000426, loss = 0.001719
grad_step = 000427, loss = 0.001702
grad_step = 000428, loss = 0.001638
grad_step = 000429, loss = 0.001504
grad_step = 000430, loss = 0.001414
grad_step = 000431, loss = 0.001413
grad_step = 000432, loss = 0.001459
grad_step = 000433, loss = 0.001500
grad_step = 000434, loss = 0.001474
grad_step = 000435, loss = 0.001400
grad_step = 000436, loss = 0.001370
grad_step = 000437, loss = 0.001406
grad_step = 000438, loss = 0.001442
grad_step = 000439, loss = 0.001422
grad_step = 000440, loss = 0.001369
grad_step = 000441, loss = 0.001365
grad_step = 000442, loss = 0.001397
grad_step = 000443, loss = 0.001397
grad_step = 000444, loss = 0.001368
grad_step = 000445, loss = 0.001359
grad_step = 000446, loss = 0.001376
grad_step = 000447, loss = 0.001380
grad_step = 000448, loss = 0.001355
grad_step = 000449, loss = 0.001349
grad_step = 000450, loss = 0.001368
grad_step = 000451, loss = 0.001368
grad_step = 000452, loss = 0.001352
grad_step = 000453, loss = 0.001342
grad_step = 000454, loss = 0.001348
grad_step = 000455, loss = 0.001355
grad_step = 000456, loss = 0.001348
grad_step = 000457, loss = 0.001343
grad_step = 000458, loss = 0.001343
grad_step = 000459, loss = 0.001343
grad_step = 000460, loss = 0.001340
grad_step = 000461, loss = 0.001337
grad_step = 000462, loss = 0.001337
grad_step = 000463, loss = 0.001337
grad_step = 000464, loss = 0.001337
grad_step = 000465, loss = 0.001335
grad_step = 000466, loss = 0.001331
grad_step = 000467, loss = 0.001329
grad_step = 000468, loss = 0.001330
grad_step = 000469, loss = 0.001330
grad_step = 000470, loss = 0.001328
grad_step = 000471, loss = 0.001325
grad_step = 000472, loss = 0.001326
grad_step = 000473, loss = 0.001327
grad_step = 000474, loss = 0.001325
grad_step = 000475, loss = 0.001322
grad_step = 000476, loss = 0.001321
grad_step = 000477, loss = 0.001321
grad_step = 000478, loss = 0.001321
grad_step = 000479, loss = 0.001319
grad_step = 000480, loss = 0.001317
grad_step = 000481, loss = 0.001317
grad_step = 000482, loss = 0.001316
grad_step = 000483, loss = 0.001315
grad_step = 000484, loss = 0.001314
grad_step = 000485, loss = 0.001313
grad_step = 000486, loss = 0.001312
grad_step = 000487, loss = 0.001312
grad_step = 000488, loss = 0.001311
grad_step = 000489, loss = 0.001310
grad_step = 000490, loss = 0.001310
grad_step = 000491, loss = 0.001309
grad_step = 000492, loss = 0.001310
grad_step = 000493, loss = 0.001311
grad_step = 000494, loss = 0.001313
grad_step = 000495, loss = 0.001318
grad_step = 000496, loss = 0.001328
grad_step = 000497, loss = 0.001343
grad_step = 000498, loss = 0.001369
grad_step = 000499, loss = 0.001391
grad_step = 000500, loss = 0.001425
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001424
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

  date_run                              2020-05-14 23:12:07.168497
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.284466
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-14 23:12:07.174184
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.20776
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-14 23:12:07.180962
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.157731
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-14 23:12:07.185753
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -2.15699
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
0   2020-05-14 23:11:41.082211  ...    mean_absolute_error
1   2020-05-14 23:11:41.085509  ...     mean_squared_error
2   2020-05-14 23:11:41.088398  ...  median_absolute_error
3   2020-05-14 23:11:41.091184  ...               r2_score
4   2020-05-14 23:11:50.346721  ...    mean_absolute_error
5   2020-05-14 23:11:50.350009  ...     mean_squared_error
6   2020-05-14 23:11:50.352581  ...  median_absolute_error
7   2020-05-14 23:11:50.355253  ...               r2_score
8   2020-05-14 23:12:07.168497  ...    mean_absolute_error
9   2020-05-14 23:12:07.174184  ...     mean_squared_error
10  2020-05-14 23:12:07.180962  ...  median_absolute_error
11  2020-05-14 23:12:07.185753  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f424b12eba8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 23%|██▎       | 2260992/9912422 [00:00<00:00, 22338953.33it/s] 78%|███████▊  | 7700480/9912422 [00:00<00:00, 27132327.08it/s]9920512it [00:00, 24018773.69it/s]                             
0it [00:00, ?it/s]32768it [00:00, 644941.43it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1036173.08it/s]1654784it [00:00, 12825559.86it/s]                           
0it [00:00, ?it/s]8192it [00:00, 178397.62it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f41fdae9e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f41fd1190b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f41fdae9e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f41fd06e0f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f41fa8aa4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f41fa894c50> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f41fdae9e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f41fd02c710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f41fa8aa4e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f41fd119128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f9be3c751d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=a3b8ee738f4b823ad9c8b93da9a3996231d1911bb0504a6acf76aa4ed9e06f35
  Stored in directory: /tmp/pip-ephem-wheel-cache-4zi1znl7/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f9b7ba71710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 3s
  753664/17464789 [>.............................] - ETA: 1s
 4792320/17464789 [=======>......................] - ETA: 0s
10895360/17464789 [=================>............] - ETA: 0s
15179776/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 23:13:31.715628: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 23:13:31.720197: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-14 23:13:31.720492: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5605759a4210 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 23:13:31.720506: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.6206 - accuracy: 0.5030
 2000/25000 [=>............................] - ETA: 7s - loss: 7.6130 - accuracy: 0.5035 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6717 - accuracy: 0.4997
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7241 - accuracy: 0.4963
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7678 - accuracy: 0.4934
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7867 - accuracy: 0.4922
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.8287 - accuracy: 0.4894
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7797 - accuracy: 0.4926
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7876 - accuracy: 0.4921
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7908 - accuracy: 0.4919
11000/25000 [============>.................] - ETA: 3s - loss: 7.7461 - accuracy: 0.4948
12000/25000 [=============>................] - ETA: 2s - loss: 7.7318 - accuracy: 0.4958
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7197 - accuracy: 0.4965
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7356 - accuracy: 0.4955
15000/25000 [=================>............] - ETA: 2s - loss: 7.7259 - accuracy: 0.4961
16000/25000 [==================>...........] - ETA: 1s - loss: 7.6973 - accuracy: 0.4980
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6856 - accuracy: 0.4988
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6871 - accuracy: 0.4987
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7021 - accuracy: 0.4977
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7011 - accuracy: 0.4978
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6783 - accuracy: 0.4992
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6729 - accuracy: 0.4996
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6833 - accuracy: 0.4989
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6717 - accuracy: 0.4997
25000/25000 [==============================] - 6s 258us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-14 23:13:44.217947
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-14 23:13:44.217947  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<16:35:23, 14.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<11:51:01, 20.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<8:21:00, 28.7kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:00<5:51:17, 40.9kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:05:20, 58.3kB/s].vector_cache/glove.6B.zip:   1%|          | 9.59M/862M [00:01<2:50:36, 83.3kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:01<1:59:10, 119kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.2M/862M [00:01<1:22:58, 170kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.9M/862M [00:01<57:47, 242kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.2M/862M [00:01<40:16, 345kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.5M/862M [00:01<28:13, 490kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.9M/862M [00:01<19:41, 697kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.0M/862M [00:02<13:52, 986kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.6M/862M [00:02<09:43, 1.40MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.7M/862M [00:02<06:55, 1.96MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:02<05:45, 2.35MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:04<05:55, 2.27MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:04<06:13, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<04:52, 2.75MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<05:48, 2.30MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:06<05:38, 2.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<04:16, 3.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<05:46, 2.30MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:08<06:55, 1.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.0M/862M [00:09<05:33, 2.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:09<04:02, 3.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<34:15, 386kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:10<25:22, 521kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:11<18:01, 732kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<15:39, 840kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:12<13:43, 958kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:13<10:16, 1.28MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:13<07:17, 1.79MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<12:41:14, 17.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<8:53:57, 24.5kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:14<6:13:21, 34.9kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<4:23:41, 49.3kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:16<3:05:50, 70.0kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:16<2:10:09, 99.7kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<1:33:55, 138kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:18<1:07:03, 193kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.5M/862M [00:18<47:08, 274kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<35:58, 358kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<27:50, 462kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<20:08, 639kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<16:06, 795kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<12:35, 1.02MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:22<09:04, 1.41MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<09:19, 1.36MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<07:50, 1.62MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:24<05:48, 2.19MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:02, 1.80MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:14, 2.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:41, 2.69MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:15, 2.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:41, 2.21MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:17, 2.92MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:57, 2.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:27, 2.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:08, 3.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:50, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:22, 2.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:04, 3.05MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:47, 2.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:17, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:01, 3.08MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:43, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:16, 2.33MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<04:00, 3.07MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:41, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:15, 2.33MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<03:59, 3.07MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:39, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:29, 1.87MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<05:04, 2.40MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<03:46, 3.21MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:15, 1.94MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:39, 2.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<04:13, 2.86MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:46, 2.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:16, 2.28MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<03:57, 3.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:36, 2.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:09, 2.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<03:52, 3.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:32, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:29, 1.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:57, 2.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:33, 2.13MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:58, 1.70MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:39, 2.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<04:08, 2.85MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:58, 1.31MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:28, 1.58MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<05:28, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:36, 1.77MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:49, 2.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<04:19, 2.70MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:47, 2.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:15, 2.21MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:55, 2.96MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:29, 2.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<06:14, 1.85MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:53, 2.36MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<03:31, 3.26MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<20:45, 554kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<15:44, 730kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<11:17, 1.02MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:01<10:33, 1.08MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<08:41, 1.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<06:21, 1.79MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<06:51, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<07:28, 1.52MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:49, 1.95MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<04:13, 2.67MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<11:52, 950kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<09:38, 1.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<07:03, 1.59MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<07:19, 1.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:24, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<04:45, 2.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:42, 1.95MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:37, 1.68MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:11, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:46, 2.94MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<07:25, 1.49MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:27, 1.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:50, 2.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:44, 1.92MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:16, 2.08MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<04:00, 2.74MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<05:08, 2.13MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:53, 2.24MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<03:43, 2.92MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:55, 2.21MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<06:02, 1.80MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<04:50, 2.24MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<03:31, 3.06MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<13:04, 826kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<10:10, 1.06MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<07:23, 1.46MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<05:19, 2.02MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<12:03, 891kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<10:57, 979kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<08:12, 1.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<05:51, 1.82MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<09:48, 1.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<08:05, 1.32MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<05:57, 1.78MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<06:23, 1.66MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<06:51, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:19, 1.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<03:51, 2.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<07:00, 1.50MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<06:08, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:35, 2.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<05:26, 1.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:16, 1.67MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:54, 2.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<03:33, 2.92MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<07:04, 1.47MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:08, 1.69MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<04:32, 2.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:23, 1.92MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:12, 1.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<04:55, 2.09MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<03:35, 2.86MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<12:35, 814kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<09:59, 1.03MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<07:14, 1.41MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:13, 1.41MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<07:21, 1.38MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:39, 1.80MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:06, 2.47MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<06:30, 1.55MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:45, 1.75MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<04:19, 2.33MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:09, 1.95MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:58, 1.68MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<04:45, 2.11MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<03:26, 2.90MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<11:39, 855kB/s] .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<09:18, 1.07MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<06:45, 1.47MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:49, 1.45MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<07:06, 1.39MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<05:32, 1.78MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<03:59, 2.47MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<11:48, 833kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<09:23, 1.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<06:48, 1.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:50, 1.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<07:09, 1.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<05:29, 1.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:49<03:57, 2.45MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<07:04, 1.37MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<06:03, 1.60MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<04:30, 2.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:12, 1.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:48, 1.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<04:31, 2.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<03:18, 2.90MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:47, 1.65MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:58, 1.92MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<03:43, 2.56MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<02:44, 3.47MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<09:16, 1.02MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<08:43, 1.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<06:34, 1.44MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<04:41, 2.01MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<09:45, 964kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<07:55, 1.19MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<05:45, 1.63MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:01, 1.55MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<05:18, 1.76MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<03:58, 2.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:44, 1.96MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:31, 1.68MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<04:23, 2.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<03:11, 2.89MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<10:37, 867kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<08:29, 1.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<06:10, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<06:14, 1.46MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<06:25, 1.42MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<04:57, 1.84MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:07<03:34, 2.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<14:33, 623kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<11:13, 807kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<08:03, 1.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<07:33, 1.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<06:19, 1.42MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<04:40, 1.92MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:10, 1.73MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:40, 1.91MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:29, 2.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:19, 2.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:08, 1.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:02, 2.19MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:15<02:55, 3.01MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<06:47, 1.29MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:45, 1.52MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<04:16, 2.05MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:51, 1.79MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:23, 1.98MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<03:17, 2.64MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:08, 2.09MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:58, 1.74MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:54, 2.21MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<02:49, 3.04MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:57, 1.23MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:49, 1.47MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<04:17, 1.99MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:53, 1.74MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:15, 2.00MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:13, 2.63MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<04:02, 2.09MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:48, 1.75MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:47, 2.22MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<02:44, 3.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<07:08, 1.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:59, 1.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<04:23, 1.90MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:48, 1.73MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:08, 1.61MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:58, 2.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<02:52, 2.87MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<06:52, 1.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<05:42, 1.44MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<04:13, 1.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:44, 1.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<05:04, 1.61MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:55, 2.07MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<02:50, 2.86MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<07:10, 1.13MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<05:56, 1.36MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<04:23, 1.84MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:46, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:04, 1.58MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:58, 2.02MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<02:51, 2.79MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<26:04, 305kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<19:00, 418kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<13:35, 584kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:41<09:34, 825kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<12:43, 620kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<10:36, 744kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<07:50, 1.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<05:32, 1.41MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<22:36, 346kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<16:40, 469kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<11:49, 659kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<09:57, 779kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<08:48, 880kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<06:33, 1.18MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:47<04:40, 1.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<10:04, 762kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<07:56, 968kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<05:45, 1.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:38, 1.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:44, 1.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:23, 1.73MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<03:10, 2.38MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 410M/862M [02:53<04:58, 1.52MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:21, 1.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:15, 2.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:51, 1.93MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:24, 1.70MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:30, 2.12MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<02:33, 2.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<09:10, 807kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<07:16, 1.02MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<05:15, 1.40MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:14, 1.40MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:15, 1.39MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:04, 1.80MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<02:55, 2.48MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<23:48, 305kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<17:29, 415kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<12:22, 585kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<10:09, 708kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<08:43, 824kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<06:26, 1.11MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<04:39, 1.54MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<05:04, 1.40MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:19, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:12, 2.21MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:48, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:25, 2.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:33, 2.74MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:21, 2.08MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:55, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:04, 2.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:15, 3.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:45, 1.84MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:26, 2.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:34, 2.68MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:15, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:53, 1.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:06, 2.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<02:14, 3.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<07:47, 869kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<06:14, 1.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<04:31, 1.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<04:34, 1.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:38, 1.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:33, 1.88MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<02:34, 2.59MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<04:52, 1.36MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:11, 1.58MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:05, 2.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:33, 1.84MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:03, 1.62MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:10, 2.07MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<02:16, 2.86MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<07:23, 880kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<05:55, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<04:18, 1.50MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:22, 1.47MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:34, 1.40MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:30, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:32, 2.51MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:14, 1.50MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<03:41, 1.72MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:45, 2.29MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:16, 1.93MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<03:42, 1.69MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<02:57, 2.12MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:08, 2.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<07:42, 808kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<06:06, 1.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:24, 1.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:23, 1.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:24, 1.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:20, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:25, 2.51MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:50, 1.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:21, 1.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:28, 2.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:04, 1.96MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:33, 1.69MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:46, 2.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<02:01, 2.94MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:23, 1.75MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<03:03, 1.94MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:18, 2.57MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:51, 2.06MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:40, 2.20MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<02:02, 2.88MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:39, 2.19MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<03:10, 1.83MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:30, 2.31MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<01:48, 3.18MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<05:21, 1.07MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<04:24, 1.30MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<03:14, 1.76MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:27, 1.64MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:45, 1.51MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:57, 1.91MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<02:07, 2.63MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<06:59, 801kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<05:32, 1.01MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<03:59, 1.39MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:58, 1.39MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<03:24, 1.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<02:32, 2.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:55, 1.87MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:19, 1.64MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:36, 2.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:52<01:52, 2.89MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<05:47, 932kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<04:40, 1.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<03:24, 1.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:30, 1.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:35, 1.48MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:47, 1.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<02:00, 2.61MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<18:01, 292kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<13:12, 398kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<09:21, 559kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<07:36, 682kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<06:32, 793kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<04:52, 1.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:00<03:27, 1.48MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<07:25, 689kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<05:46, 884kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<04:09, 1.22MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:58, 1.27MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<03:58, 1.27MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<03:02, 1.66MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:04<02:10, 2.29MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<05:34, 894kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<04:28, 1.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<03:15, 1.52MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:18, 1.48MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:50, 1.73MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<02:06, 2.31MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:33, 1.90MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:55, 1.66MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:17, 2.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<01:39, 2.87MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<05:30, 865kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:23, 1.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<03:12, 1.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:13, 1.46MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:18, 1.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:33, 1.84MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:14<01:49, 2.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<04:01, 1.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:20, 1.38MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<02:26, 1.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:40, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:53, 1.58MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:15, 2.01MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:18<01:38, 2.75MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<05:14, 856kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<04:10, 1.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<03:01, 1.47MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:02, 1.45MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:10, 1.39MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:25, 1.81MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<01:44, 2.51MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:54, 1.11MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:13, 1.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<02:21, 1.84MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:33, 1.68MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:44, 1.56MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:08, 1.99MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<01:32, 2.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<04:56, 852kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<03:56, 1.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<02:50, 1.47MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:51, 1.45MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:55, 1.41MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<02:16, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<01:36, 2.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<04:35, 886kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<03:40, 1.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<02:40, 1.51MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:42, 1.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:50, 1.41MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:10, 1.83MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:34<01:33, 2.53MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:50, 1.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:26, 1.61MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:48, 2.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:09, 1.79MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:22, 1.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:50, 2.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:38<01:21, 2.83MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:59, 1.91MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:48, 2.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:21, 2.78MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:46, 2.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:40, 2.23MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:15, 2.96MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:39, 2.21MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:58, 1.85MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:34, 2.30MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<01:08, 3.14MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<04:05, 877kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<03:42, 968kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:45, 1.30MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:46<01:58, 1.80MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:30, 1.40MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:09, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:35, 2.18MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:50, 1.87MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:41, 2.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:16, 2.68MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:36, 2.11MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:52<01:53, 1.79MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:29, 2.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:52<01:04, 3.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:03, 1.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:30, 1.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:50, 1.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:58, 1.64MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:45, 1.85MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:18, 2.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:34, 2.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:26, 2.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:04, 2.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:26, 2.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:42, 1.82MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:22, 2.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<00:59, 3.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<03:40, 826kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<02:55, 1.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<02:06, 1.43MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:05, 1.42MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:09, 1.37MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:38, 1.79MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:11, 2.46MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:48, 1.60MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:35, 1.81MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:10, 2.43MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:24, 2.00MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:39, 1.71MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:18, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<00:56, 2.94MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:08, 875kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:30, 1.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:48, 1.51MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:50, 1.45MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:53, 1.42MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:27, 1.84MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<01:02, 2.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:13, 1.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:51, 1.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:21, 1.92MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:28, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:37, 1.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:15, 2.00MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<00:54, 2.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:57, 1.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:39, 1.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:13, 2.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:21, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:29, 1.62MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:10, 2.04MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<00:50, 2.81MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<02:45, 846kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<02:12, 1.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:35, 1.46MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:34, 1.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:35, 1.43MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:12, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<00:51, 2.57MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:46, 1.24MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:29, 1.47MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:05, 2.00MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:12, 1.78MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:19, 1.62MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:01, 2.07MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<00:44, 2.83MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:14, 1.65MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:06, 1.85MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:49, 2.45MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:59, 2.00MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:53, 2.24MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<00:43, 2.75MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<00:31, 3.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:52, 1.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:43, 1.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:18, 1.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<00:54, 2.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<06:15, 297kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<04:32, 407kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<03:11, 574kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<02:13, 810kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<02:55, 609kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<02:25, 733kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:46, 1.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:14, 1.40MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:30, 1.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:15, 1.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:54, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:58, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:02, 1.57MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:48, 2.03MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:34, 2.76MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:56, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:49, 1.89MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:37, 2.51MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:44, 2.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:52, 1.72MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:40, 2.19MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:29, 3.00MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:58, 1.48MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:50, 1.69MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:37, 2.28MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:42, 1.92MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:49, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:38, 2.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:27, 2.90MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:30, 866kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:22, 940kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:04, 1.19MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:45, 1.65MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:50, 1.45MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:52, 1.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:40, 1.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:28, 2.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:27, 789kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:08, 998kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:48, 1.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:46, 1.38MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:40, 1.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:29, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:32, 1.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:35, 1.69MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:27, 2.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:19, 2.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:17, 725kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:00, 926kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:43, 1.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:39, 1.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:13, 712kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:56, 926kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:39, 1.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:27, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:56, 856kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:44, 1.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:31, 1.47MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:30, 1.45MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:31, 1.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:23, 1.83MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:16, 2.54MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:36, 1.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:30, 1.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:21, 1.78MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:21, 1.64MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:22, 1.56MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:17, 2.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:11, 2.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:20, 1.56MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:17, 1.77MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:12, 2.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:13, 1.96MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:12, 2.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<00:08, 2.94MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:28, 808kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:22, 1.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:15, 1.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:13, 1.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:13, 1.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:10, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:06, 2.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:12, 1.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:10, 1.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:06, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:23<00:06, 1.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:06, 1.56MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 1.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:02, 2.71MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:08, 805kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:06, 1.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.41MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.62MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 2.19MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 905/400000 [00:00<00:44, 9045.22it/s]  0%|          | 1878/400000 [00:00<00:43, 9239.99it/s]  1%|          | 2829/400000 [00:00<00:42, 9319.21it/s]  1%|          | 3779/400000 [00:00<00:42, 9370.77it/s]  1%|          | 4748/400000 [00:00<00:41, 9462.24it/s]  1%|▏         | 5741/400000 [00:00<00:41, 9595.37it/s]  2%|▏         | 6698/400000 [00:00<00:41, 9585.82it/s]  2%|▏         | 7622/400000 [00:00<00:41, 9476.90it/s]  2%|▏         | 8555/400000 [00:00<00:41, 9430.37it/s]  2%|▏         | 9522/400000 [00:01<00:41, 9499.70it/s]  3%|▎         | 10501/400000 [00:01<00:40, 9582.31it/s]  3%|▎         | 11442/400000 [00:01<00:40, 9526.94it/s]  3%|▎         | 12383/400000 [00:01<00:41, 9416.89it/s]  3%|▎         | 13317/400000 [00:01<00:42, 9120.58it/s]  4%|▎         | 14226/400000 [00:01<00:42, 9043.34it/s]  4%|▍         | 15172/400000 [00:01<00:42, 9161.85it/s]  4%|▍         | 16133/400000 [00:01<00:41, 9289.56it/s]  4%|▍         | 17069/400000 [00:01<00:41, 9308.64it/s]  4%|▍         | 18000/400000 [00:01<00:41, 9265.39it/s]  5%|▍         | 18971/400000 [00:02<00:40, 9393.06it/s]  5%|▍         | 19953/400000 [00:02<00:39, 9516.71it/s]  5%|▌         | 20934/400000 [00:02<00:39, 9602.75it/s]  5%|▌         | 21895/400000 [00:02<00:39, 9539.90it/s]  6%|▌         | 22850/400000 [00:02<00:40, 9362.56it/s]  6%|▌         | 23836/400000 [00:02<00:39, 9504.56it/s]  6%|▌         | 24798/400000 [00:02<00:39, 9537.00it/s]  6%|▋         | 25766/400000 [00:02<00:39, 9577.16it/s]  7%|▋         | 26725/400000 [00:02<00:39, 9556.78it/s]  7%|▋         | 27682/400000 [00:02<00:39, 9503.18it/s]  7%|▋         | 28633/400000 [00:03<00:40, 9265.43it/s]  7%|▋         | 29562/400000 [00:03<00:40, 9181.85it/s]  8%|▊         | 30513/400000 [00:03<00:39, 9277.37it/s]  8%|▊         | 31472/400000 [00:03<00:39, 9367.48it/s]  8%|▊         | 32410/400000 [00:03<00:39, 9341.13it/s]  8%|▊         | 33401/400000 [00:03<00:38, 9501.88it/s]  9%|▊         | 34379/400000 [00:03<00:38, 9582.13it/s]  9%|▉         | 35339/400000 [00:03<00:38, 9507.42it/s]  9%|▉         | 36291/400000 [00:03<00:38, 9357.88it/s]  9%|▉         | 37241/400000 [00:03<00:38, 9397.71it/s] 10%|▉         | 38208/400000 [00:04<00:38, 9476.18it/s] 10%|▉         | 39190/400000 [00:04<00:37, 9575.06it/s] 10%|█         | 40149/400000 [00:04<00:37, 9550.76it/s] 10%|█         | 41119/400000 [00:04<00:37, 9592.98it/s] 11%|█         | 42079/400000 [00:04<00:37, 9586.06it/s] 11%|█         | 43042/400000 [00:04<00:37, 9597.60it/s] 11%|█         | 44015/400000 [00:04<00:36, 9634.14it/s] 11%|█         | 44979/400000 [00:04<00:37, 9499.89it/s] 11%|█▏        | 45930/400000 [00:04<00:37, 9480.31it/s] 12%|█▏        | 46883/400000 [00:04<00:37, 9492.68it/s] 12%|█▏        | 47863/400000 [00:05<00:36, 9581.77it/s] 12%|█▏        | 48863/400000 [00:05<00:36, 9700.55it/s] 12%|█▏        | 49834/400000 [00:05<00:36, 9621.89it/s] 13%|█▎        | 50798/400000 [00:05<00:36, 9625.82it/s] 13%|█▎        | 51761/400000 [00:05<00:36, 9538.26it/s] 13%|█▎        | 52741/400000 [00:05<00:36, 9612.36it/s] 13%|█▎        | 53707/400000 [00:05<00:35, 9624.41it/s] 14%|█▎        | 54670/400000 [00:05<00:36, 9411.60it/s] 14%|█▍        | 55613/400000 [00:05<00:37, 9111.00it/s] 14%|█▍        | 56528/400000 [00:05<00:37, 9112.48it/s] 14%|█▍        | 57442/400000 [00:06<00:37, 9111.93it/s] 15%|█▍        | 58407/400000 [00:06<00:36, 9264.71it/s] 15%|█▍        | 59360/400000 [00:06<00:36, 9340.55it/s] 15%|█▌        | 60300/400000 [00:06<00:36, 9355.68it/s] 15%|█▌        | 61237/400000 [00:06<00:36, 9310.08it/s] 16%|█▌        | 62192/400000 [00:06<00:36, 9379.16it/s] 16%|█▌        | 63131/400000 [00:06<00:37, 8990.53it/s] 16%|█▌        | 64034/400000 [00:06<00:37, 8901.80it/s] 16%|█▌        | 64928/400000 [00:06<00:37, 8901.05it/s] 16%|█▋        | 65847/400000 [00:07<00:37, 8985.01it/s] 17%|█▋        | 66816/400000 [00:07<00:36, 9185.06it/s] 17%|█▋        | 67746/400000 [00:07<00:36, 9218.81it/s] 17%|█▋        | 68708/400000 [00:07<00:35, 9334.19it/s] 17%|█▋        | 69691/400000 [00:07<00:34, 9475.41it/s] 18%|█▊        | 70641/400000 [00:07<00:35, 9364.82it/s] 18%|█▊        | 71580/400000 [00:07<00:35, 9371.92it/s] 18%|█▊        | 72519/400000 [00:07<00:35, 9243.59it/s] 18%|█▊        | 73468/400000 [00:07<00:35, 9315.95it/s] 19%|█▊        | 74434/400000 [00:07<00:34, 9416.18it/s] 19%|█▉        | 75377/400000 [00:08<00:34, 9337.89it/s] 19%|█▉        | 76312/400000 [00:08<00:35, 9134.24it/s] 19%|█▉        | 77248/400000 [00:08<00:35, 9200.10it/s] 20%|█▉        | 78170/400000 [00:08<00:35, 9188.72it/s] 20%|█▉        | 79090/400000 [00:08<00:35, 8925.04it/s] 20%|█▉        | 79991/400000 [00:08<00:35, 8950.28it/s] 20%|██        | 80934/400000 [00:08<00:35, 9086.14it/s] 20%|██        | 81923/400000 [00:08<00:34, 9312.06it/s] 21%|██        | 82892/400000 [00:08<00:33, 9421.14it/s] 21%|██        | 83874/400000 [00:08<00:33, 9537.02it/s] 21%|██        | 84830/400000 [00:09<00:33, 9466.34it/s] 21%|██▏       | 85779/400000 [00:09<00:33, 9377.54it/s] 22%|██▏       | 86718/400000 [00:09<00:33, 9315.83it/s] 22%|██▏       | 87652/400000 [00:09<00:33, 9321.61it/s] 22%|██▏       | 88625/400000 [00:09<00:32, 9437.67it/s] 22%|██▏       | 89570/400000 [00:09<00:33, 9237.12it/s] 23%|██▎       | 90525/400000 [00:09<00:33, 9328.14it/s] 23%|██▎       | 91487/400000 [00:09<00:32, 9412.24it/s] 23%|██▎       | 92430/400000 [00:09<00:33, 9220.09it/s] 23%|██▎       | 93408/400000 [00:09<00:32, 9379.99it/s] 24%|██▎       | 94348/400000 [00:10<00:32, 9311.49it/s] 24%|██▍       | 95281/400000 [00:10<00:33, 9221.73it/s] 24%|██▍       | 96205/400000 [00:10<00:32, 9206.44it/s] 24%|██▍       | 97128/400000 [00:10<00:32, 9212.98it/s] 25%|██▍       | 98094/400000 [00:10<00:32, 9340.79it/s] 25%|██▍       | 99054/400000 [00:10<00:31, 9414.96it/s] 25%|██▌       | 100045/400000 [00:10<00:31, 9555.71it/s] 25%|██▌       | 101002/400000 [00:10<00:31, 9433.25it/s] 25%|██▌       | 101991/400000 [00:10<00:31, 9564.96it/s] 26%|██▌       | 102973/400000 [00:10<00:30, 9637.70it/s] 26%|██▌       | 103938/400000 [00:11<00:30, 9613.10it/s] 26%|██▌       | 104900/400000 [00:11<00:30, 9597.67it/s] 26%|██▋       | 105875/400000 [00:11<00:30, 9641.70it/s] 27%|██▋       | 106840/400000 [00:11<00:30, 9643.24it/s] 27%|██▋       | 107831/400000 [00:11<00:30, 9719.79it/s] 27%|██▋       | 108811/400000 [00:11<00:29, 9741.84it/s] 27%|██▋       | 109786/400000 [00:11<00:30, 9579.90it/s] 28%|██▊       | 110745/400000 [00:11<00:30, 9341.67it/s] 28%|██▊       | 111682/400000 [00:11<00:31, 9100.61it/s] 28%|██▊       | 112625/400000 [00:11<00:31, 9196.18it/s] 28%|██▊       | 113547/400000 [00:12<00:31, 9183.31it/s] 29%|██▊       | 114481/400000 [00:12<00:30, 9227.55it/s] 29%|██▉       | 115423/400000 [00:12<00:30, 9283.58it/s] 29%|██▉       | 116399/400000 [00:12<00:30, 9419.16it/s] 29%|██▉       | 117390/400000 [00:12<00:29, 9558.75it/s] 30%|██▉       | 118365/400000 [00:12<00:29, 9613.09it/s] 30%|██▉       | 119347/400000 [00:12<00:29, 9673.11it/s] 30%|███       | 120333/400000 [00:12<00:28, 9727.73it/s] 30%|███       | 121324/400000 [00:12<00:28, 9781.36it/s] 31%|███       | 122303/400000 [00:13<00:28, 9589.27it/s] 31%|███       | 123264/400000 [00:13<00:29, 9522.67it/s] 31%|███       | 124280/400000 [00:13<00:28, 9703.97it/s] 31%|███▏      | 125252/400000 [00:13<00:29, 9419.23it/s] 32%|███▏      | 126197/400000 [00:13<00:29, 9323.14it/s] 32%|███▏      | 127132/400000 [00:13<00:30, 8868.29it/s] 32%|███▏      | 128026/400000 [00:13<00:31, 8751.31it/s] 32%|███▏      | 128992/400000 [00:13<00:30, 9004.29it/s] 32%|███▏      | 129963/400000 [00:13<00:29, 9202.82it/s] 33%|███▎      | 130971/400000 [00:13<00:28, 9447.47it/s] 33%|███▎      | 131974/400000 [00:14<00:27, 9614.66it/s] 33%|███▎      | 132948/400000 [00:14<00:27, 9651.63it/s] 33%|███▎      | 133941/400000 [00:14<00:27, 9732.52it/s] 34%|███▎      | 134917/400000 [00:14<00:27, 9731.66it/s] 34%|███▍      | 135892/400000 [00:14<00:27, 9647.92it/s] 34%|███▍      | 136859/400000 [00:14<00:27, 9446.16it/s] 34%|███▍      | 137806/400000 [00:14<00:28, 9305.72it/s] 35%|███▍      | 138772/400000 [00:14<00:27, 9407.41it/s] 35%|███▍      | 139733/400000 [00:14<00:27, 9466.32it/s] 35%|███▌      | 140695/400000 [00:14<00:27, 9509.20it/s] 35%|███▌      | 141647/400000 [00:15<00:27, 9399.80it/s] 36%|███▌      | 142588/400000 [00:15<00:27, 9305.26it/s] 36%|███▌      | 143520/400000 [00:15<00:27, 9298.62it/s] 36%|███▌      | 144451/400000 [00:15<00:27, 9136.94it/s] 36%|███▋      | 145419/400000 [00:15<00:27, 9292.20it/s] 37%|███▋      | 146350/400000 [00:15<00:27, 9185.73it/s] 37%|███▋      | 147275/400000 [00:15<00:27, 9203.83it/s] 37%|███▋      | 148242/400000 [00:15<00:26, 9336.41it/s] 37%|███▋      | 149177/400000 [00:15<00:27, 9077.48it/s] 38%|███▊      | 150099/400000 [00:15<00:27, 9119.00it/s] 38%|███▊      | 151073/400000 [00:16<00:26, 9294.67it/s] 38%|███▊      | 152017/400000 [00:16<00:26, 9336.37it/s] 38%|███▊      | 152953/400000 [00:16<00:26, 9300.43it/s] 38%|███▊      | 153914/400000 [00:16<00:26, 9389.12it/s] 39%|███▊      | 154854/400000 [00:16<00:26, 9290.50it/s] 39%|███▉      | 155829/400000 [00:16<00:25, 9421.20it/s] 39%|███▉      | 156773/400000 [00:16<00:25, 9410.74it/s] 39%|███▉      | 157730/400000 [00:16<00:25, 9456.50it/s] 40%|███▉      | 158677/400000 [00:16<00:26, 9273.80it/s] 40%|███▉      | 159606/400000 [00:17<00:26, 9206.68it/s] 40%|████      | 160590/400000 [00:17<00:25, 9385.97it/s] 40%|████      | 161534/400000 [00:17<00:25, 9399.79it/s] 41%|████      | 162504/400000 [00:17<00:25, 9485.50it/s] 41%|████      | 163454/400000 [00:17<00:25, 9417.23it/s] 41%|████      | 164428/400000 [00:17<00:24, 9510.85it/s] 41%|████▏     | 165394/400000 [00:17<00:24, 9552.86it/s] 42%|████▏     | 166350/400000 [00:17<00:24, 9470.59it/s] 42%|████▏     | 167336/400000 [00:17<00:24, 9581.83it/s] 42%|████▏     | 168302/400000 [00:17<00:24, 9602.80it/s] 42%|████▏     | 169263/400000 [00:18<00:24, 9295.75it/s] 43%|████▎     | 170268/400000 [00:18<00:24, 9509.76it/s] 43%|████▎     | 171222/400000 [00:18<00:24, 9467.99it/s] 43%|████▎     | 172192/400000 [00:18<00:23, 9534.96it/s] 43%|████▎     | 173148/400000 [00:18<00:23, 9516.89it/s] 44%|████▎     | 174101/400000 [00:18<00:24, 9375.22it/s] 44%|████▍     | 175040/400000 [00:18<00:24, 9189.28it/s] 44%|████▍     | 175964/400000 [00:18<00:24, 9201.80it/s] 44%|████▍     | 176946/400000 [00:18<00:23, 9377.03it/s] 44%|████▍     | 177934/400000 [00:18<00:23, 9522.27it/s] 45%|████▍     | 178910/400000 [00:19<00:23, 9591.19it/s] 45%|████▍     | 179910/400000 [00:19<00:22, 9708.79it/s] 45%|████▌     | 180883/400000 [00:19<00:22, 9649.02it/s] 45%|████▌     | 181877/400000 [00:19<00:22, 9734.45it/s] 46%|████▌     | 182869/400000 [00:19<00:22, 9788.92it/s] 46%|████▌     | 183866/400000 [00:19<00:21, 9839.88it/s] 46%|████▌     | 184851/400000 [00:19<00:21, 9826.26it/s] 46%|████▋     | 185835/400000 [00:19<00:22, 9714.42it/s] 47%|████▋     | 186807/400000 [00:19<00:21, 9697.15it/s] 47%|████▋     | 187789/400000 [00:19<00:21, 9731.69it/s] 47%|████▋     | 188763/400000 [00:20<00:21, 9720.03it/s] 47%|████▋     | 189736/400000 [00:20<00:21, 9708.89it/s] 48%|████▊     | 190708/400000 [00:20<00:22, 9490.63it/s] 48%|████▊     | 191659/400000 [00:20<00:22, 9450.73it/s] 48%|████▊     | 192659/400000 [00:20<00:21, 9606.43it/s] 48%|████▊     | 193661/400000 [00:20<00:21, 9725.29it/s] 49%|████▊     | 194667/400000 [00:20<00:20, 9820.83it/s] 49%|████▉     | 195651/400000 [00:20<00:20, 9780.40it/s] 49%|████▉     | 196630/400000 [00:20<00:20, 9771.02it/s] 49%|████▉     | 197617/400000 [00:20<00:20, 9798.79it/s] 50%|████▉     | 198626/400000 [00:21<00:20, 9882.95it/s] 50%|████▉     | 199615/400000 [00:21<00:20, 9797.67it/s] 50%|█████     | 200596/400000 [00:21<00:21, 9488.91it/s] 50%|█████     | 201548/400000 [00:21<00:20, 9487.09it/s] 51%|█████     | 202513/400000 [00:21<00:20, 9532.58it/s] 51%|█████     | 203495/400000 [00:21<00:20, 9615.60it/s] 51%|█████     | 204477/400000 [00:21<00:20, 9674.63it/s] 51%|█████▏    | 205446/400000 [00:21<00:20, 9633.72it/s] 52%|█████▏    | 206410/400000 [00:21<00:20, 9293.03it/s] 52%|█████▏    | 207343/400000 [00:21<00:20, 9202.42it/s] 52%|█████▏    | 208313/400000 [00:22<00:20, 9346.15it/s] 52%|█████▏    | 209316/400000 [00:22<00:19, 9539.72it/s] 53%|█████▎    | 210273/400000 [00:22<00:19, 9516.42it/s] 53%|█████▎    | 211236/400000 [00:22<00:19, 9547.44it/s] 53%|█████▎    | 212219/400000 [00:22<00:19, 9630.17it/s] 53%|█████▎    | 213207/400000 [00:22<00:19, 9702.55it/s] 54%|█████▎    | 214194/400000 [00:22<00:19, 9750.92it/s] 54%|█████▍    | 215170/400000 [00:22<00:19, 9602.86it/s] 54%|█████▍    | 216169/400000 [00:22<00:18, 9713.12it/s] 54%|█████▍    | 217183/400000 [00:22<00:18, 9837.16it/s] 55%|█████▍    | 218168/400000 [00:23<00:18, 9800.85it/s] 55%|█████▍    | 219186/400000 [00:23<00:18, 9910.79it/s] 55%|█████▌    | 220178/400000 [00:23<00:18, 9834.24it/s] 55%|█████▌    | 221163/400000 [00:23<00:18, 9721.49it/s] 56%|█████▌    | 222136/400000 [00:23<00:18, 9583.58it/s] 56%|█████▌    | 223096/400000 [00:23<00:19, 9212.60it/s] 56%|█████▌    | 224035/400000 [00:23<00:18, 9263.42it/s] 56%|█████▌    | 224965/400000 [00:23<00:19, 9202.39it/s] 56%|█████▋    | 225896/400000 [00:23<00:18, 9233.41it/s] 57%|█████▋    | 226888/400000 [00:24<00:18, 9427.29it/s] 57%|█████▋    | 227837/400000 [00:24<00:18, 9445.37it/s] 57%|█████▋    | 228783/400000 [00:24<00:18, 9406.59it/s] 57%|█████▋    | 229725/400000 [00:24<00:18, 9182.26it/s] 58%|█████▊    | 230707/400000 [00:24<00:18, 9364.65it/s] 58%|█████▊    | 231692/400000 [00:24<00:17, 9503.79it/s] 58%|█████▊    | 232648/400000 [00:24<00:17, 9519.05it/s] 58%|█████▊    | 233602/400000 [00:24<00:17, 9486.66it/s] 59%|█████▊    | 234552/400000 [00:24<00:17, 9370.78it/s] 59%|█████▉    | 235500/400000 [00:24<00:17, 9400.60it/s] 59%|█████▉    | 236487/400000 [00:25<00:17, 9534.86it/s] 59%|█████▉    | 237455/400000 [00:25<00:16, 9577.06it/s] 60%|█████▉    | 238414/400000 [00:25<00:17, 9327.72it/s] 60%|█████▉    | 239349/400000 [00:25<00:17, 9199.06it/s] 60%|██████    | 240272/400000 [00:25<00:17, 9207.55it/s] 60%|██████    | 241194/400000 [00:25<00:17, 9195.87it/s] 61%|██████    | 242160/400000 [00:25<00:16, 9329.77it/s] 61%|██████    | 243115/400000 [00:25<00:16, 9393.74it/s] 61%|██████    | 244056/400000 [00:25<00:16, 9391.70it/s] 61%|██████    | 244996/400000 [00:25<00:17, 8904.44it/s] 61%|██████▏   | 245893/400000 [00:26<00:17, 8800.05it/s] 62%|██████▏   | 246778/400000 [00:26<00:17, 8721.78it/s] 62%|██████▏   | 247658/400000 [00:26<00:17, 8742.67it/s] 62%|██████▏   | 248535/400000 [00:26<00:17, 8712.93it/s] 62%|██████▏   | 249408/400000 [00:26<00:17, 8716.20it/s] 63%|██████▎   | 250285/400000 [00:26<00:17, 8729.67it/s] 63%|██████▎   | 251159/400000 [00:26<00:17, 8702.77it/s] 63%|██████▎   | 252034/400000 [00:26<00:16, 8714.78it/s] 63%|██████▎   | 252906/400000 [00:26<00:17, 8642.81it/s] 63%|██████▎   | 253771/400000 [00:26<00:17, 8591.04it/s] 64%|██████▎   | 254631/400000 [00:27<00:17, 8547.34it/s] 64%|██████▍   | 255496/400000 [00:27<00:16, 8576.43it/s] 64%|██████▍   | 256354/400000 [00:27<00:16, 8575.67it/s] 64%|██████▍   | 257212/400000 [00:27<00:16, 8475.48it/s] 65%|██████▍   | 258072/400000 [00:27<00:16, 8511.21it/s] 65%|██████▍   | 258943/400000 [00:27<00:16, 8568.25it/s] 65%|██████▍   | 259805/400000 [00:27<00:16, 8579.23it/s] 65%|██████▌   | 260670/400000 [00:27<00:16, 8600.18it/s] 65%|██████▌   | 261531/400000 [00:27<00:16, 8585.52it/s] 66%|██████▌   | 262390/400000 [00:27<00:16, 8567.47it/s] 66%|██████▌   | 263265/400000 [00:28<00:15, 8618.87it/s] 66%|██████▌   | 264130/400000 [00:28<00:15, 8626.25it/s] 66%|██████▌   | 264993/400000 [00:28<00:15, 8562.55it/s] 66%|██████▋   | 265864/400000 [00:28<00:15, 8605.79it/s] 67%|██████▋   | 266736/400000 [00:28<00:15, 8637.41it/s] 67%|██████▋   | 267606/400000 [00:28<00:15, 8653.88it/s] 67%|██████▋   | 268472/400000 [00:28<00:15, 8654.48it/s] 67%|██████▋   | 269349/400000 [00:28<00:15, 8686.98it/s] 68%|██████▊   | 270224/400000 [00:28<00:14, 8705.49it/s] 68%|██████▊   | 271095/400000 [00:29<00:14, 8682.46it/s] 68%|██████▊   | 271964/400000 [00:29<00:14, 8646.29it/s] 68%|██████▊   | 272835/400000 [00:29<00:14, 8664.02it/s] 68%|██████▊   | 273702/400000 [00:29<00:14, 8663.44it/s] 69%|██████▊   | 274569/400000 [00:29<00:14, 8663.93it/s] 69%|██████▉   | 275436/400000 [00:29<00:14, 8629.25it/s] 69%|██████▉   | 276299/400000 [00:29<00:14, 8581.55it/s] 69%|██████▉   | 277158/400000 [00:29<00:14, 8565.04it/s] 70%|██████▉   | 278023/400000 [00:29<00:14, 8588.27it/s] 70%|██████▉   | 278882/400000 [00:29<00:14, 8548.24it/s] 70%|██████▉   | 279737/400000 [00:30<00:14, 8478.23it/s] 70%|███████   | 280614/400000 [00:30<00:13, 8562.15it/s] 70%|███████   | 281478/400000 [00:30<00:13, 8584.82it/s] 71%|███████   | 282337/400000 [00:30<00:13, 8573.22it/s] 71%|███████   | 283195/400000 [00:30<00:13, 8460.74it/s] 71%|███████   | 284069/400000 [00:30<00:13, 8539.96it/s] 71%|███████   | 284934/400000 [00:30<00:13, 8570.26it/s] 71%|███████▏  | 285792/400000 [00:30<00:13, 8559.82it/s] 72%|███████▏  | 286650/400000 [00:30<00:13, 8563.17it/s] 72%|███████▏  | 287507/400000 [00:30<00:13, 8547.15it/s] 72%|███████▏  | 288370/400000 [00:31<00:13, 8570.88it/s] 72%|███████▏  | 289231/400000 [00:31<00:12, 8579.92it/s] 73%|███████▎  | 290090/400000 [00:31<00:12, 8558.36it/s] 73%|███████▎  | 290946/400000 [00:31<00:13, 8229.51it/s] 73%|███████▎  | 291797/400000 [00:31<00:13, 8310.60it/s] 73%|███████▎  | 292631/400000 [00:31<00:13, 8188.42it/s] 73%|███████▎  | 293489/400000 [00:31<00:12, 8301.38it/s] 74%|███████▎  | 294344/400000 [00:31<00:12, 8372.59it/s] 74%|███████▍  | 295206/400000 [00:31<00:12, 8442.87it/s] 74%|███████▍  | 296052/400000 [00:31<00:12, 8404.23it/s] 74%|███████▍  | 296894/400000 [00:32<00:12, 8400.07it/s] 74%|███████▍  | 297766/400000 [00:32<00:12, 8493.22it/s] 75%|███████▍  | 298631/400000 [00:32<00:11, 8539.56it/s] 75%|███████▍  | 299499/400000 [00:32<00:11, 8579.96it/s] 75%|███████▌  | 300377/400000 [00:32<00:11, 8637.12it/s] 75%|███████▌  | 301255/400000 [00:32<00:11, 8677.32it/s] 76%|███████▌  | 302124/400000 [00:32<00:11, 8679.98it/s] 76%|███████▌  | 302996/400000 [00:32<00:11, 8689.47it/s] 76%|███████▌  | 303869/400000 [00:32<00:11, 8700.49it/s] 76%|███████▌  | 304747/400000 [00:32<00:10, 8722.99it/s] 76%|███████▋  | 305632/400000 [00:33<00:10, 8760.51it/s] 77%|███████▋  | 306518/400000 [00:33<00:10, 8789.61it/s] 77%|███████▋  | 307398/400000 [00:33<00:10, 8773.31it/s] 77%|███████▋  | 308279/400000 [00:33<00:10, 8783.87it/s] 77%|███████▋  | 309158/400000 [00:33<00:10, 8719.54it/s] 78%|███████▊  | 310031/400000 [00:33<00:10, 8711.67it/s] 78%|███████▊  | 310903/400000 [00:33<00:10, 8674.93it/s] 78%|███████▊  | 311772/400000 [00:33<00:10, 8678.60it/s] 78%|███████▊  | 312640/400000 [00:33<00:10, 8651.41it/s] 78%|███████▊  | 313506/400000 [00:33<00:09, 8652.80it/s] 79%|███████▊  | 314372/400000 [00:34<00:09, 8652.21it/s] 79%|███████▉  | 315238/400000 [00:34<00:09, 8645.58it/s] 79%|███████▉  | 316113/400000 [00:34<00:09, 8675.71it/s] 79%|███████▉  | 316981/400000 [00:34<00:09, 8641.58it/s] 79%|███████▉  | 317863/400000 [00:34<00:09, 8692.22it/s] 80%|███████▉  | 318733/400000 [00:34<00:09, 8647.35it/s] 80%|███████▉  | 319603/400000 [00:34<00:09, 8662.18it/s] 80%|████████  | 320482/400000 [00:34<00:09, 8697.49it/s] 80%|████████  | 321352/400000 [00:34<00:09, 8691.84it/s] 81%|████████  | 322231/400000 [00:34<00:08, 8721.02it/s] 81%|████████  | 323104/400000 [00:35<00:08, 8702.88it/s] 81%|████████  | 323984/400000 [00:35<00:08, 8731.16it/s] 81%|████████  | 324860/400000 [00:35<00:08, 8738.40it/s] 81%|████████▏ | 325734/400000 [00:35<00:08, 8684.51it/s] 82%|████████▏ | 326603/400000 [00:35<00:08, 8518.45it/s] 82%|████████▏ | 327467/400000 [00:35<00:08, 8553.50it/s] 82%|████████▏ | 328343/400000 [00:35<00:08, 8613.53it/s] 82%|████████▏ | 329212/400000 [00:35<00:08, 8635.65it/s] 83%|████████▎ | 330090/400000 [00:35<00:08, 8675.83it/s] 83%|████████▎ | 330958/400000 [00:35<00:08, 8310.36it/s] 83%|████████▎ | 331835/400000 [00:36<00:08, 8442.24it/s] 83%|████████▎ | 332725/400000 [00:36<00:07, 8572.21it/s] 83%|████████▎ | 333606/400000 [00:36<00:07, 8640.05it/s] 84%|████████▎ | 334479/400000 [00:36<00:07, 8664.91it/s] 84%|████████▍ | 335349/400000 [00:36<00:07, 8673.81it/s] 84%|████████▍ | 336226/400000 [00:36<00:07, 8700.96it/s] 84%|████████▍ | 337097/400000 [00:36<00:07, 8700.21it/s] 84%|████████▍ | 337968/400000 [00:36<00:07, 8700.29it/s] 85%|████████▍ | 338845/400000 [00:36<00:07, 8721.07it/s] 85%|████████▍ | 339718/400000 [00:36<00:06, 8692.77it/s] 85%|████████▌ | 340588/400000 [00:37<00:06, 8637.07it/s] 85%|████████▌ | 341452/400000 [00:37<00:06, 8634.42it/s] 86%|████████▌ | 342316/400000 [00:37<00:06, 8625.58it/s] 86%|████████▌ | 343192/400000 [00:37<00:06, 8664.50it/s] 86%|████████▌ | 344065/400000 [00:37<00:06, 8683.28it/s] 86%|████████▌ | 344940/400000 [00:37<00:06, 8702.94it/s] 86%|████████▋ | 345821/400000 [00:37<00:06, 8734.00it/s] 87%|████████▋ | 346695/400000 [00:37<00:06, 8734.08it/s] 87%|████████▋ | 347647/400000 [00:37<00:05, 8954.38it/s] 87%|████████▋ | 348591/400000 [00:37<00:05, 9092.73it/s] 87%|████████▋ | 349507/400000 [00:38<00:05, 9110.07it/s] 88%|████████▊ | 350506/400000 [00:38<00:05, 9356.08it/s] 88%|████████▊ | 351462/400000 [00:38<00:05, 9415.07it/s] 88%|████████▊ | 352406/400000 [00:38<00:05, 9277.48it/s] 88%|████████▊ | 353367/400000 [00:38<00:04, 9370.55it/s] 89%|████████▊ | 354306/400000 [00:38<00:04, 9307.87it/s] 89%|████████▉ | 355238/400000 [00:38<00:04, 9215.88it/s] 89%|████████▉ | 356161/400000 [00:38<00:04, 9177.02it/s] 89%|████████▉ | 357136/400000 [00:38<00:04, 9339.15it/s] 90%|████████▉ | 358130/400000 [00:38<00:04, 9510.68it/s] 90%|████████▉ | 359083/400000 [00:39<00:04, 9447.83it/s] 90%|█████████ | 360061/400000 [00:39<00:04, 9541.97it/s] 90%|█████████ | 361028/400000 [00:39<00:04, 9579.65it/s] 90%|█████████ | 361987/400000 [00:39<00:04, 9434.28it/s] 91%|█████████ | 362932/400000 [00:39<00:03, 9353.51it/s] 91%|█████████ | 363869/400000 [00:39<00:03, 9270.36it/s] 91%|█████████ | 364797/400000 [00:39<00:03, 9267.54it/s] 91%|█████████▏| 365725/400000 [00:39<00:03, 9252.82it/s] 92%|█████████▏| 366678/400000 [00:39<00:03, 9332.66it/s] 92%|█████████▏| 367634/400000 [00:40<00:03, 9399.27it/s] 92%|█████████▏| 368575/400000 [00:40<00:03, 9391.39it/s] 92%|█████████▏| 369529/400000 [00:40<00:03, 9435.03it/s] 93%|█████████▎| 370507/400000 [00:40<00:03, 9533.95it/s] 93%|█████████▎| 371461/400000 [00:40<00:03, 9097.29it/s] 93%|█████████▎| 372388/400000 [00:40<00:03, 9145.60it/s] 93%|█████████▎| 373306/400000 [00:40<00:02, 9050.26it/s] 94%|█████████▎| 374230/400000 [00:40<00:02, 9105.80it/s] 94%|█████████▍| 375179/400000 [00:40<00:02, 9216.42it/s] 94%|█████████▍| 376189/400000 [00:40<00:02, 9462.39it/s] 94%|█████████▍| 377151/400000 [00:41<00:02, 9508.46it/s] 95%|█████████▍| 378104/400000 [00:41<00:02, 9364.26it/s] 95%|█████████▍| 379084/400000 [00:41<00:02, 9486.28it/s] 95%|█████████▌| 380037/400000 [00:41<00:02, 9496.84it/s] 95%|█████████▌| 380999/400000 [00:41<00:01, 9532.55it/s] 95%|█████████▌| 381954/400000 [00:41<00:01, 9444.68it/s] 96%|█████████▌| 382900/400000 [00:41<00:01, 9431.06it/s] 96%|█████████▌| 383844/400000 [00:41<00:01, 9260.46it/s] 96%|█████████▌| 384772/400000 [00:41<00:01, 9256.41it/s] 96%|█████████▋| 385761/400000 [00:41<00:01, 9435.21it/s] 97%|█████████▋| 386706/400000 [00:42<00:01, 9299.52it/s] 97%|█████████▋| 387638/400000 [00:42<00:01, 8826.11it/s] 97%|█████████▋| 388577/400000 [00:42<00:01, 8985.38it/s] 97%|█████████▋| 389481/400000 [00:42<00:01, 8931.04it/s] 98%|█████████▊| 390456/400000 [00:42<00:01, 9161.40it/s] 98%|█████████▊| 391430/400000 [00:42<00:00, 9326.76it/s] 98%|█████████▊| 392378/400000 [00:42<00:00, 9369.51it/s] 98%|█████████▊| 393318/400000 [00:42<00:00, 9289.27it/s] 99%|█████████▊| 394249/400000 [00:42<00:00, 9288.94it/s] 99%|█████████▉| 395209/400000 [00:42<00:00, 9378.92it/s] 99%|█████████▉| 396149/400000 [00:43<00:00, 9374.02it/s] 99%|█████████▉| 397118/400000 [00:43<00:00, 9465.89it/s]100%|█████████▉| 398098/400000 [00:43<00:00, 9560.69it/s]100%|█████████▉| 399055/400000 [00:43<00:00, 9299.72it/s]100%|█████████▉| 399995/400000 [00:43<00:00, 9329.03it/s]100%|█████████▉| 399999/400000 [00:43<00:00, 9198.32it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f57c3c74a58> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011251521468674164 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.010890906471073826 	 Accuracy: 69

  model saves at 69% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 16190 out of table with 16170 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 16190 out of table with 16170 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-14 23:22:40.659722: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 23:22:40.664043: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-14 23:22:40.664183: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5616b04940b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 23:22:40.664197: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f57771ab160> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.9580 - accuracy: 0.4810
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7510 - accuracy: 0.4945 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7331 - accuracy: 0.4957
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7088 - accuracy: 0.4972
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6820 - accuracy: 0.4990
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6863 - accuracy: 0.4987
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6839 - accuracy: 0.4989
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6837 - accuracy: 0.4989
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
11000/25000 [============>.................] - ETA: 3s - loss: 7.6652 - accuracy: 0.5001
12000/25000 [=============>................] - ETA: 3s - loss: 7.6347 - accuracy: 0.5021
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6430 - accuracy: 0.5015
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6283 - accuracy: 0.5025
15000/25000 [=================>............] - ETA: 2s - loss: 7.6278 - accuracy: 0.5025
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6149 - accuracy: 0.5034
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6405 - accuracy: 0.5017
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6743 - accuracy: 0.4995
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6900 - accuracy: 0.4985
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6850 - accuracy: 0.4988
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6761 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6527 - accuracy: 0.5009
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6580 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
25000/25000 [==============================] - 7s 260us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f57403ef6a0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f574043b588> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.1199 - crf_viterbi_accuracy: 0.6800 - val_loss: 1.1075 - val_crf_viterbi_accuracy: 0.6533

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

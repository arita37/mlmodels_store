
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f6f52edbf28> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 00:18:23.248692
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 00:18:23.252438
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 00:18:23.255330
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 00:18:23.258390
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f6f5eca53c8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356012.1562
Epoch 2/10

1/1 [==============================] - 0s 101ms/step - loss: 201882.5000
Epoch 3/10

1/1 [==============================] - 0s 93ms/step - loss: 98395.0625
Epoch 4/10

1/1 [==============================] - 0s 91ms/step - loss: 46301.3164
Epoch 5/10

1/1 [==============================] - 0s 96ms/step - loss: 23743.2344
Epoch 6/10

1/1 [==============================] - 0s 95ms/step - loss: 13659.0166
Epoch 7/10

1/1 [==============================] - 0s 93ms/step - loss: 8763.0459
Epoch 8/10

1/1 [==============================] - 0s 88ms/step - loss: 6151.5713
Epoch 9/10

1/1 [==============================] - 0s 86ms/step - loss: 4637.9624
Epoch 10/10

1/1 [==============================] - 0s 97ms/step - loss: 3709.5535

  #### Inference Need return ypred, ytrue ######################### 
[[-5.78585386e-01 -3.36311311e-01 -5.43859541e-01 -1.78087986e+00
  -2.40903783e+00 -6.15095735e-01 -1.41264617e-01  5.30255020e-01
   1.79951906e+00  1.86745238e+00  6.54814482e-01 -9.29916501e-01
  -2.71056980e-01 -8.28157902e-01 -2.08567429e+00  2.71936595e-01
  -4.95343208e-02 -2.68623948e-01 -1.45680320e+00  2.06981802e+00
  -5.04477143e-01 -1.12102127e+00  5.88749170e-01  8.04558694e-01
   3.46810818e-02  2.01734948e+00  3.61442655e-01  8.13876748e-01
   1.86528444e-01 -1.05757928e+00  1.15859604e+00  1.25865626e+00
  -2.39653730e+00  1.35126233e+00  5.08340836e-01  1.07274854e+00
  -1.64630628e+00  1.44017339e-02 -2.67780840e-01 -6.98130071e-01
  -1.53417969e+00 -1.36150074e+00 -2.70814776e+00 -9.38127697e-01
   9.64950323e-02  2.39923060e-01 -1.22725058e+00 -1.65171421e+00
   3.83668095e-01 -9.03751194e-01  6.84737027e-01 -4.49274480e-01
   1.15634513e+00 -7.05046296e-01 -2.36810088e+00  7.87822366e-01
  -2.05243498e-01  1.32234347e+00  1.21965611e+00  1.50027499e-01
   6.02775276e-01  9.31059182e-01 -3.43673348e-01  3.47020894e-01
   1.87005103e-02  7.42691934e-01 -1.13344997e-01  1.02106309e+00
  -2.06764549e-01 -2.19103742e+00 -4.58483160e-01  3.81389469e-01
  -2.23235965e-01  9.39093888e-01  4.03172195e-01  1.19851017e+00
   1.30758858e+00  6.52310014e-01  1.03359556e+00  1.14404702e+00
  -2.17023802e+00  4.76468951e-01  1.05697083e+00 -9.71806586e-01
   1.16534412e-01  2.02612591e+00 -6.00067496e-01  2.07488641e-01
   5.94958842e-01 -7.90245175e-01  8.93561304e-01 -3.07604134e-01
  -2.50374019e-01  7.21543491e-01 -7.04175532e-01  1.70405233e+00
   5.92111826e-01 -4.08481717e-01 -7.77849197e-01 -2.07299757e+00
  -1.65692687e-01  1.90638363e-01  1.19928241e+00  2.94244021e-01
   7.31299520e-01  5.81853867e-01  4.41478848e-01  5.77455461e-01
  -1.48001683e+00  7.56964236e-02 -1.39797544e+00  1.81496406e+00
   2.09313035e+00 -2.70148635e-01 -5.77451944e-01 -6.06562734e-01
   2.34520388e+00 -1.21821773e+00 -2.19413102e-01  8.24064136e-01
   6.42590880e-01  1.05983324e+01  9.32357597e+00  1.19449282e+01
   1.26206608e+01  1.03280716e+01  1.08772488e+01  1.08844862e+01
   1.18999681e+01  1.37890615e+01  1.08822041e+01  1.21497927e+01
   9.98370171e+00  1.25264235e+01  1.30824566e+01  1.31498728e+01
   1.18021498e+01  1.13185863e+01  1.04862165e+01  9.97155857e+00
   1.03034687e+01  1.18281345e+01  1.37464266e+01  1.07153568e+01
   1.12362061e+01  1.34646339e+01  1.12077742e+01  1.37282925e+01
   1.35452909e+01  1.10834379e+01  1.15409775e+01  1.25334587e+01
   9.89908886e+00  1.05754690e+01  1.20977936e+01  1.29567366e+01
   1.05473700e+01  1.27871399e+01  1.06022387e+01  1.08074636e+01
   1.18140078e+01  1.07439938e+01  1.30329428e+01  1.05651264e+01
   1.39403820e+01  1.11079893e+01  1.05391731e+01  1.36590843e+01
   1.12476206e+01  1.10474358e+01  1.09655695e+01  1.38350439e+01
   1.15649948e+01  1.16367731e+01  1.08515730e+01  1.42352152e+01
   1.18979292e+01  1.10628719e+01  1.01129923e+01  1.16731615e+01
   5.16592443e-01  2.38792944e+00  1.64366651e+00  6.66985095e-01
   2.45139623e+00  1.18763041e+00  3.43042898e+00  7.35018015e-01
   1.84681773e+00  1.63880610e+00  1.13273072e+00  2.82704711e-01
   8.04017782e-02  1.25150919e+00  3.04664993e+00  2.45920229e+00
   7.90510774e-02  6.24335647e-01  1.62364817e+00  1.55376375e+00
   1.83717430e-01  1.33750772e+00  1.53186202e+00  4.89848256e-01
   1.63993597e+00  1.24279535e+00  2.27679396e+00  3.22186232e+00
   5.44836998e-01  1.43938899e+00  1.24351752e+00  8.15064311e-01
   1.85749388e+00  3.75549912e-01  1.80118752e+00  1.31108725e+00
   7.05835104e-01  1.42659843e+00  1.23740327e+00  2.85301447e-01
   2.11034870e+00  6.07140839e-01  5.27171195e-01  1.99086714e+00
   9.27507639e-01  2.26738167e+00  1.47758913e+00  4.22317266e-01
   2.45696139e+00  1.16567278e+00  7.83391118e-01  8.12869847e-01
   1.18775308e-01  2.18715906e+00  1.38332415e+00  9.64580417e-01
   1.25068188e+00  5.05936563e-01  3.36811900e-01  2.89065695e+00
   1.15646124e+00  3.66773605e-01  1.74559641e+00  1.03042126e+00
   7.64870644e-02  4.56541896e-01  7.01102376e-01  8.22204888e-01
   7.69791245e-01  1.55733478e+00  2.16614723e-01  3.18965137e-01
   1.15133452e+00  6.69853449e-01  2.03004360e+00  2.25141573e+00
   3.35316718e-01  9.23433781e-01  1.28708291e+00  1.38912141e+00
   2.10240006e+00  3.99990559e-01  2.93325424e+00  5.14924049e-01
   5.15988886e-01  1.84483457e+00  3.38648605e+00  2.06163025e+00
   6.22739434e-01  1.43500376e+00  2.79637694e-01  6.47800207e-01
   1.94623125e+00  4.44216132e-01  5.45333982e-01  6.57278121e-01
   1.42777026e+00  1.64456308e-01  1.63672996e+00  1.92786312e+00
   2.47769260e+00  8.70551705e-01  1.60581756e+00  1.81430197e+00
   1.85224879e+00  2.61461830e+00  1.05477166e+00  1.97967052e+00
   1.26334691e+00  1.09251332e+00  8.06395352e-01  1.53833389e-01
   8.61303568e-01  6.54740810e-01  1.82350850e+00  2.29830933e+00
   1.46359813e+00  8.99229586e-01  1.48410273e+00  3.58923495e-01
   1.99326408e+00  1.05439482e+01  1.24701538e+01  1.11140213e+01
   1.31117544e+01  1.07979727e+01  1.07276688e+01  1.13720703e+01
   1.18574848e+01  1.12493696e+01  1.07107258e+01  1.17686968e+01
   1.09080019e+01  1.19360790e+01  1.03266239e+01  1.45467825e+01
   1.16495771e+01  1.26379442e+01  1.35884724e+01  1.10028400e+01
   1.08718023e+01  1.26254864e+01  1.17543869e+01  1.18201914e+01
   1.04575300e+01  8.77445984e+00  1.14581232e+01  1.16299629e+01
   1.06116171e+01  1.05648870e+01  1.16627541e+01  1.19968805e+01
   1.23542404e+01  1.17107878e+01  1.15296469e+01  1.19988918e+01
   1.26175518e+01  1.16915932e+01  1.15027637e+01  1.00693207e+01
   1.21482811e+01  1.21777916e+01  1.09488544e+01  9.68036938e+00
   1.07867451e+01  1.33296881e+01  9.58057499e+00  1.22891350e+01
   9.07366467e+00  1.14721413e+01  1.03086290e+01  1.08158789e+01
   1.30526829e+01  1.17955694e+01  1.26091738e+01  1.30124207e+01
   1.13028355e+01  9.91660023e+00  9.16893959e+00  1.43040438e+01
  -8.86359692e+00 -6.33013868e+00  1.37079000e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 00:18:33.089946
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   90.6411
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 00:18:33.093549
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8251.33
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 00:18:33.097149
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.1216
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 00:18:33.100257
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -737.965
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140115750314280
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140114674614792
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140114674615296
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140114674615800
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140114674616304
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140114674616808

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f6f3e8b6e10> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.547039
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.521291
grad_step = 000002, loss = 0.510566
grad_step = 000003, loss = 0.499720
grad_step = 000004, loss = 0.489612
grad_step = 000005, loss = 0.477592
grad_step = 000006, loss = 0.463764
grad_step = 000007, loss = 0.447936
grad_step = 000008, loss = 0.432026
grad_step = 000009, loss = 0.417173
grad_step = 000010, loss = 0.404388
grad_step = 000011, loss = 0.393827
grad_step = 000012, loss = 0.383333
grad_step = 000013, loss = 0.372799
grad_step = 000014, loss = 0.362294
grad_step = 000015, loss = 0.352490
grad_step = 000016, loss = 0.343049
grad_step = 000017, loss = 0.333694
grad_step = 000018, loss = 0.323979
grad_step = 000019, loss = 0.313801
grad_step = 000020, loss = 0.303908
grad_step = 000021, loss = 0.294515
grad_step = 000022, loss = 0.285268
grad_step = 000023, loss = 0.276125
grad_step = 000024, loss = 0.266312
grad_step = 000025, loss = 0.256567
grad_step = 000026, loss = 0.246687
grad_step = 000027, loss = 0.236657
grad_step = 000028, loss = 0.226983
grad_step = 000029, loss = 0.214871
grad_step = 000030, loss = 0.202503
grad_step = 000031, loss = 0.190775
grad_step = 000032, loss = 0.180705
grad_step = 000033, loss = 0.172161
grad_step = 000034, loss = 0.164420
grad_step = 000035, loss = 0.157627
grad_step = 000036, loss = 0.151714
grad_step = 000037, loss = 0.145697
grad_step = 000038, loss = 0.139571
grad_step = 000039, loss = 0.133678
grad_step = 000040, loss = 0.127659
grad_step = 000041, loss = 0.121678
grad_step = 000042, loss = 0.115928
grad_step = 000043, loss = 0.110465
grad_step = 000044, loss = 0.105204
grad_step = 000045, loss = 0.100204
grad_step = 000046, loss = 0.095354
grad_step = 000047, loss = 0.090703
grad_step = 000048, loss = 0.086282
grad_step = 000049, loss = 0.082068
grad_step = 000050, loss = 0.078055
grad_step = 000051, loss = 0.074226
grad_step = 000052, loss = 0.070440
grad_step = 000053, loss = 0.066811
grad_step = 000054, loss = 0.063286
grad_step = 000055, loss = 0.059883
grad_step = 000056, loss = 0.056641
grad_step = 000057, loss = 0.053544
grad_step = 000058, loss = 0.050576
grad_step = 000059, loss = 0.047798
grad_step = 000060, loss = 0.045140
grad_step = 000061, loss = 0.042621
grad_step = 000062, loss = 0.040202
grad_step = 000063, loss = 0.037899
grad_step = 000064, loss = 0.035687
grad_step = 000065, loss = 0.033573
grad_step = 000066, loss = 0.031553
grad_step = 000067, loss = 0.029615
grad_step = 000068, loss = 0.027794
grad_step = 000069, loss = 0.026056
grad_step = 000070, loss = 0.024420
grad_step = 000071, loss = 0.022864
grad_step = 000072, loss = 0.021385
grad_step = 000073, loss = 0.019986
grad_step = 000074, loss = 0.018653
grad_step = 000075, loss = 0.017404
grad_step = 000076, loss = 0.016227
grad_step = 000077, loss = 0.015122
grad_step = 000078, loss = 0.014078
grad_step = 000079, loss = 0.013100
grad_step = 000080, loss = 0.012184
grad_step = 000081, loss = 0.011329
grad_step = 000082, loss = 0.010535
grad_step = 000083, loss = 0.009796
grad_step = 000084, loss = 0.009106
grad_step = 000085, loss = 0.008463
grad_step = 000086, loss = 0.007871
grad_step = 000087, loss = 0.007323
grad_step = 000088, loss = 0.006824
grad_step = 000089, loss = 0.006372
grad_step = 000090, loss = 0.005972
grad_step = 000091, loss = 0.005593
grad_step = 000092, loss = 0.005213
grad_step = 000093, loss = 0.004828
grad_step = 000094, loss = 0.004515
grad_step = 000095, loss = 0.004274
grad_step = 000096, loss = 0.004042
grad_step = 000097, loss = 0.003782
grad_step = 000098, loss = 0.003525
grad_step = 000099, loss = 0.003337
grad_step = 000100, loss = 0.003197
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003053
grad_step = 000102, loss = 0.002884
grad_step = 000103, loss = 0.002736
grad_step = 000104, loss = 0.002632
grad_step = 000105, loss = 0.002547
grad_step = 000106, loss = 0.002441
grad_step = 000107, loss = 0.002324
grad_step = 000108, loss = 0.002230
grad_step = 000109, loss = 0.002173
grad_step = 000110, loss = 0.002133
grad_step = 000111, loss = 0.002088
grad_step = 000112, loss = 0.002024
grad_step = 000113, loss = 0.001963
grad_step = 000114, loss = 0.001921
grad_step = 000115, loss = 0.001903
grad_step = 000116, loss = 0.001894
grad_step = 000117, loss = 0.001878
grad_step = 000118, loss = 0.001858
grad_step = 000119, loss = 0.001857
grad_step = 000120, loss = 0.001914
grad_step = 000121, loss = 0.002032
grad_step = 000122, loss = 0.002068
grad_step = 000123, loss = 0.001908
grad_step = 000124, loss = 0.001774
grad_step = 000125, loss = 0.001833
grad_step = 000126, loss = 0.001864
grad_step = 000127, loss = 0.001773
grad_step = 000128, loss = 0.001761
grad_step = 000129, loss = 0.001830
grad_step = 000130, loss = 0.001768
grad_step = 000131, loss = 0.001694
grad_step = 000132, loss = 0.001747
grad_step = 000133, loss = 0.001767
grad_step = 000134, loss = 0.001691
grad_step = 000135, loss = 0.001676
grad_step = 000136, loss = 0.001722
grad_step = 000137, loss = 0.001698
grad_step = 000138, loss = 0.001649
grad_step = 000139, loss = 0.001671
grad_step = 000140, loss = 0.001685
grad_step = 000141, loss = 0.001641
grad_step = 000142, loss = 0.001627
grad_step = 000143, loss = 0.001649
grad_step = 000144, loss = 0.001639
grad_step = 000145, loss = 0.001609
grad_step = 000146, loss = 0.001606
grad_step = 000147, loss = 0.001617
grad_step = 000148, loss = 0.001604
grad_step = 000149, loss = 0.001582
grad_step = 000150, loss = 0.001580
grad_step = 000151, loss = 0.001587
grad_step = 000152, loss = 0.001576
grad_step = 000153, loss = 0.001559
grad_step = 000154, loss = 0.001555
grad_step = 000155, loss = 0.001560
grad_step = 000156, loss = 0.001557
grad_step = 000157, loss = 0.001553
grad_step = 000158, loss = 0.001574
grad_step = 000159, loss = 0.001669
grad_step = 000160, loss = 0.001955
grad_step = 000161, loss = 0.002361
grad_step = 000162, loss = 0.002489
grad_step = 000163, loss = 0.001761
grad_step = 000164, loss = 0.001679
grad_step = 000165, loss = 0.002139
grad_step = 000166, loss = 0.001931
grad_step = 000167, loss = 0.001575
grad_step = 000168, loss = 0.002007
grad_step = 000169, loss = 0.001791
grad_step = 000170, loss = 0.001620
grad_step = 000171, loss = 0.001812
grad_step = 000172, loss = 0.001711
grad_step = 000173, loss = 0.001557
grad_step = 000174, loss = 0.001747
grad_step = 000175, loss = 0.001588
grad_step = 000176, loss = 0.001559
grad_step = 000177, loss = 0.001686
grad_step = 000178, loss = 0.001510
grad_step = 000179, loss = 0.001597
grad_step = 000180, loss = 0.001606
grad_step = 000181, loss = 0.001508
grad_step = 000182, loss = 0.001568
grad_step = 000183, loss = 0.001561
grad_step = 000184, loss = 0.001482
grad_step = 000185, loss = 0.001558
grad_step = 000186, loss = 0.001503
grad_step = 000187, loss = 0.001493
grad_step = 000188, loss = 0.001526
grad_step = 000189, loss = 0.001486
grad_step = 000190, loss = 0.001490
grad_step = 000191, loss = 0.001502
grad_step = 000192, loss = 0.001481
grad_step = 000193, loss = 0.001474
grad_step = 000194, loss = 0.001493
grad_step = 000195, loss = 0.001466
grad_step = 000196, loss = 0.001468
grad_step = 000197, loss = 0.001477
grad_step = 000198, loss = 0.001457
grad_step = 000199, loss = 0.001461
grad_step = 000200, loss = 0.001462
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001455
grad_step = 000202, loss = 0.001451
grad_step = 000203, loss = 0.001454
grad_step = 000204, loss = 0.001449
grad_step = 000205, loss = 0.001443
grad_step = 000206, loss = 0.001448
grad_step = 000207, loss = 0.001442
grad_step = 000208, loss = 0.001439
grad_step = 000209, loss = 0.001441
grad_step = 000210, loss = 0.001438
grad_step = 000211, loss = 0.001434
grad_step = 000212, loss = 0.001434
grad_step = 000213, loss = 0.001434
grad_step = 000214, loss = 0.001430
grad_step = 000215, loss = 0.001429
grad_step = 000216, loss = 0.001430
grad_step = 000217, loss = 0.001426
grad_step = 000218, loss = 0.001424
grad_step = 000219, loss = 0.001425
grad_step = 000220, loss = 0.001423
grad_step = 000221, loss = 0.001421
grad_step = 000222, loss = 0.001420
grad_step = 000223, loss = 0.001419
grad_step = 000224, loss = 0.001418
grad_step = 000225, loss = 0.001416
grad_step = 000226, loss = 0.001415
grad_step = 000227, loss = 0.001414
grad_step = 000228, loss = 0.001413
grad_step = 000229, loss = 0.001412
grad_step = 000230, loss = 0.001411
grad_step = 000231, loss = 0.001409
grad_step = 000232, loss = 0.001408
grad_step = 000233, loss = 0.001407
grad_step = 000234, loss = 0.001406
grad_step = 000235, loss = 0.001405
grad_step = 000236, loss = 0.001403
grad_step = 000237, loss = 0.001402
grad_step = 000238, loss = 0.001401
grad_step = 000239, loss = 0.001400
grad_step = 000240, loss = 0.001399
grad_step = 000241, loss = 0.001398
grad_step = 000242, loss = 0.001397
grad_step = 000243, loss = 0.001395
grad_step = 000244, loss = 0.001394
grad_step = 000245, loss = 0.001393
grad_step = 000246, loss = 0.001392
grad_step = 000247, loss = 0.001391
grad_step = 000248, loss = 0.001389
grad_step = 000249, loss = 0.001388
grad_step = 000250, loss = 0.001387
grad_step = 000251, loss = 0.001386
grad_step = 000252, loss = 0.001385
grad_step = 000253, loss = 0.001384
grad_step = 000254, loss = 0.001382
grad_step = 000255, loss = 0.001381
grad_step = 000256, loss = 0.001380
grad_step = 000257, loss = 0.001379
grad_step = 000258, loss = 0.001379
grad_step = 000259, loss = 0.001379
grad_step = 000260, loss = 0.001381
grad_step = 000261, loss = 0.001385
grad_step = 000262, loss = 0.001395
grad_step = 000263, loss = 0.001409
grad_step = 000264, loss = 0.001429
grad_step = 000265, loss = 0.001434
grad_step = 000266, loss = 0.001426
grad_step = 000267, loss = 0.001395
grad_step = 000268, loss = 0.001372
grad_step = 000269, loss = 0.001371
grad_step = 000270, loss = 0.001383
grad_step = 000271, loss = 0.001391
grad_step = 000272, loss = 0.001385
grad_step = 000273, loss = 0.001374
grad_step = 000274, loss = 0.001368
grad_step = 000275, loss = 0.001368
grad_step = 000276, loss = 0.001368
grad_step = 000277, loss = 0.001366
grad_step = 000278, loss = 0.001364
grad_step = 000279, loss = 0.001363
grad_step = 000280, loss = 0.001364
grad_step = 000281, loss = 0.001361
grad_step = 000282, loss = 0.001357
grad_step = 000283, loss = 0.001351
grad_step = 000284, loss = 0.001348
grad_step = 000285, loss = 0.001349
grad_step = 000286, loss = 0.001351
grad_step = 000287, loss = 0.001352
grad_step = 000288, loss = 0.001350
grad_step = 000289, loss = 0.001347
grad_step = 000290, loss = 0.001342
grad_step = 000291, loss = 0.001339
grad_step = 000292, loss = 0.001337
grad_step = 000293, loss = 0.001336
grad_step = 000294, loss = 0.001337
grad_step = 000295, loss = 0.001337
grad_step = 000296, loss = 0.001336
grad_step = 000297, loss = 0.001336
grad_step = 000298, loss = 0.001336
grad_step = 000299, loss = 0.001338
grad_step = 000300, loss = 0.001343
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001358
grad_step = 000302, loss = 0.001381
grad_step = 000303, loss = 0.001427
grad_step = 000304, loss = 0.001455
grad_step = 000305, loss = 0.001482
grad_step = 000306, loss = 0.001443
grad_step = 000307, loss = 0.001423
grad_step = 000308, loss = 0.001427
grad_step = 000309, loss = 0.001434
grad_step = 000310, loss = 0.001407
grad_step = 000311, loss = 0.001343
grad_step = 000312, loss = 0.001311
grad_step = 000313, loss = 0.001332
grad_step = 000314, loss = 0.001368
grad_step = 000315, loss = 0.001378
grad_step = 000316, loss = 0.001354
grad_step = 000317, loss = 0.001330
grad_step = 000318, loss = 0.001323
grad_step = 000319, loss = 0.001322
grad_step = 000320, loss = 0.001317
grad_step = 000321, loss = 0.001311
grad_step = 000322, loss = 0.001316
grad_step = 000323, loss = 0.001327
grad_step = 000324, loss = 0.001331
grad_step = 000325, loss = 0.001323
grad_step = 000326, loss = 0.001307
grad_step = 000327, loss = 0.001297
grad_step = 000328, loss = 0.001295
grad_step = 000329, loss = 0.001295
grad_step = 000330, loss = 0.001293
grad_step = 000331, loss = 0.001290
grad_step = 000332, loss = 0.001290
grad_step = 000333, loss = 0.001293
grad_step = 000334, loss = 0.001296
grad_step = 000335, loss = 0.001296
grad_step = 000336, loss = 0.001292
grad_step = 000337, loss = 0.001286
grad_step = 000338, loss = 0.001281
grad_step = 000339, loss = 0.001278
grad_step = 000340, loss = 0.001277
grad_step = 000341, loss = 0.001275
grad_step = 000342, loss = 0.001273
grad_step = 000343, loss = 0.001270
grad_step = 000344, loss = 0.001267
grad_step = 000345, loss = 0.001265
grad_step = 000346, loss = 0.001264
grad_step = 000347, loss = 0.001264
grad_step = 000348, loss = 0.001264
grad_step = 000349, loss = 0.001264
grad_step = 000350, loss = 0.001266
grad_step = 000351, loss = 0.001269
grad_step = 000352, loss = 0.001274
grad_step = 000353, loss = 0.001284
grad_step = 000354, loss = 0.001296
grad_step = 000355, loss = 0.001322
grad_step = 000356, loss = 0.001349
grad_step = 000357, loss = 0.001398
grad_step = 000358, loss = 0.001430
grad_step = 000359, loss = 0.001462
grad_step = 000360, loss = 0.001444
grad_step = 000361, loss = 0.001395
grad_step = 000362, loss = 0.001350
grad_step = 000363, loss = 0.001332
grad_step = 000364, loss = 0.001370
grad_step = 000365, loss = 0.001336
grad_step = 000366, loss = 0.001291
grad_step = 000367, loss = 0.001266
grad_step = 000368, loss = 0.001295
grad_step = 000369, loss = 0.001329
grad_step = 000370, loss = 0.001279
grad_step = 000371, loss = 0.001237
grad_step = 000372, loss = 0.001257
grad_step = 000373, loss = 0.001288
grad_step = 000374, loss = 0.001286
grad_step = 000375, loss = 0.001247
grad_step = 000376, loss = 0.001246
grad_step = 000377, loss = 0.001270
grad_step = 000378, loss = 0.001261
grad_step = 000379, loss = 0.001237
grad_step = 000380, loss = 0.001229
grad_step = 000381, loss = 0.001241
grad_step = 000382, loss = 0.001248
grad_step = 000383, loss = 0.001234
grad_step = 000384, loss = 0.001228
grad_step = 000385, loss = 0.001233
grad_step = 000386, loss = 0.001236
grad_step = 000387, loss = 0.001231
grad_step = 000388, loss = 0.001225
grad_step = 000389, loss = 0.001225
grad_step = 000390, loss = 0.001227
grad_step = 000391, loss = 0.001224
grad_step = 000392, loss = 0.001220
grad_step = 000393, loss = 0.001217
grad_step = 000394, loss = 0.001217
grad_step = 000395, loss = 0.001218
grad_step = 000396, loss = 0.001216
grad_step = 000397, loss = 0.001212
grad_step = 000398, loss = 0.001211
grad_step = 000399, loss = 0.001211
grad_step = 000400, loss = 0.001213
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001211
grad_step = 000402, loss = 0.001208
grad_step = 000403, loss = 0.001207
grad_step = 000404, loss = 0.001210
grad_step = 000405, loss = 0.001213
grad_step = 000406, loss = 0.001215
grad_step = 000407, loss = 0.001220
grad_step = 000408, loss = 0.001230
grad_step = 000409, loss = 0.001248
grad_step = 000410, loss = 0.001283
grad_step = 000411, loss = 0.001324
grad_step = 000412, loss = 0.001391
grad_step = 000413, loss = 0.001443
grad_step = 000414, loss = 0.001482
grad_step = 000415, loss = 0.001452
grad_step = 000416, loss = 0.001358
grad_step = 000417, loss = 0.001253
grad_step = 000418, loss = 0.001200
grad_step = 000419, loss = 0.001221
grad_step = 000420, loss = 0.001273
grad_step = 000421, loss = 0.001303
grad_step = 000422, loss = 0.001282
grad_step = 000423, loss = 0.001234
grad_step = 000424, loss = 0.001199
grad_step = 000425, loss = 0.001201
grad_step = 000426, loss = 0.001227
grad_step = 000427, loss = 0.001241
grad_step = 000428, loss = 0.001232
grad_step = 000429, loss = 0.001206
grad_step = 000430, loss = 0.001187
grad_step = 000431, loss = 0.001190
grad_step = 000432, loss = 0.001204
grad_step = 000433, loss = 0.001214
grad_step = 000434, loss = 0.001208
grad_step = 000435, loss = 0.001191
grad_step = 000436, loss = 0.001177
grad_step = 000437, loss = 0.001175
grad_step = 000438, loss = 0.001183
grad_step = 000439, loss = 0.001191
grad_step = 000440, loss = 0.001191
grad_step = 000441, loss = 0.001183
grad_step = 000442, loss = 0.001174
grad_step = 000443, loss = 0.001170
grad_step = 000444, loss = 0.001171
grad_step = 000445, loss = 0.001174
grad_step = 000446, loss = 0.001175
grad_step = 000447, loss = 0.001172
grad_step = 000448, loss = 0.001167
grad_step = 000449, loss = 0.001163
grad_step = 000450, loss = 0.001162
grad_step = 000451, loss = 0.001164
grad_step = 000452, loss = 0.001165
grad_step = 000453, loss = 0.001165
grad_step = 000454, loss = 0.001164
grad_step = 000455, loss = 0.001162
grad_step = 000456, loss = 0.001160
grad_step = 000457, loss = 0.001161
grad_step = 000458, loss = 0.001164
grad_step = 000459, loss = 0.001169
grad_step = 000460, loss = 0.001176
grad_step = 000461, loss = 0.001188
grad_step = 000462, loss = 0.001198
grad_step = 000463, loss = 0.001218
grad_step = 000464, loss = 0.001222
grad_step = 000465, loss = 0.001233
grad_step = 000466, loss = 0.001206
grad_step = 000467, loss = 0.001176
grad_step = 000468, loss = 0.001151
grad_step = 000469, loss = 0.001149
grad_step = 000470, loss = 0.001163
grad_step = 000471, loss = 0.001173
grad_step = 000472, loss = 0.001174
grad_step = 000473, loss = 0.001158
grad_step = 000474, loss = 0.001146
grad_step = 000475, loss = 0.001144
grad_step = 000476, loss = 0.001151
grad_step = 000477, loss = 0.001157
grad_step = 000478, loss = 0.001154
grad_step = 000479, loss = 0.001146
grad_step = 000480, loss = 0.001137
grad_step = 000481, loss = 0.001134
grad_step = 000482, loss = 0.001137
grad_step = 000483, loss = 0.001141
grad_step = 000484, loss = 0.001141
grad_step = 000485, loss = 0.001137
grad_step = 000486, loss = 0.001132
grad_step = 000487, loss = 0.001130
grad_step = 000488, loss = 0.001131
grad_step = 000489, loss = 0.001133
grad_step = 000490, loss = 0.001133
grad_step = 000491, loss = 0.001131
grad_step = 000492, loss = 0.001128
grad_step = 000493, loss = 0.001126
grad_step = 000494, loss = 0.001125
grad_step = 000495, loss = 0.001126
grad_step = 000496, loss = 0.001127
grad_step = 000497, loss = 0.001127
grad_step = 000498, loss = 0.001127
grad_step = 000499, loss = 0.001128
grad_step = 000500, loss = 0.001129
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001132
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

  date_run                              2020-05-15 00:18:50.883009
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.308845
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 00:18:50.888585
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.291345
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 00:18:50.895588
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.133636
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 00:18:50.900494
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -3.4271
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
0   2020-05-15 00:18:23.248692  ...    mean_absolute_error
1   2020-05-15 00:18:23.252438  ...     mean_squared_error
2   2020-05-15 00:18:23.255330  ...  median_absolute_error
3   2020-05-15 00:18:23.258390  ...               r2_score
4   2020-05-15 00:18:33.089946  ...    mean_absolute_error
5   2020-05-15 00:18:33.093549  ...     mean_squared_error
6   2020-05-15 00:18:33.097149  ...  median_absolute_error
7   2020-05-15 00:18:33.100257  ...               r2_score
8   2020-05-15 00:18:50.883009  ...    mean_absolute_error
9   2020-05-15 00:18:50.888585  ...     mean_squared_error
10  2020-05-15 00:18:50.895588  ...  median_absolute_error
11  2020-05-15 00:18:50.900494  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc2bd1e6898> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▍      | 3448832/9912422 [00:00<00:00, 34445811.99it/s]9920512it [00:00, 34039998.50it/s]                             
0it [00:00, ?it/s]32768it [00:00, 581624.17it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 141505.83it/s]1654784it [00:00, 11138286.27it/s]                         
0it [00:00, ?it/s]8192it [00:00, 163015.42it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc26fb94e10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc26c9e20b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc26fb94e10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc26f11d080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc26c956470> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc26c941be0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc26fb94e10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc26f0db6a0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc26c956470> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc2bd19ee80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f69ebaaf1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=8e360f9d69f93f12db7ede0da9f4749ad8395002968657370f8954ac98adf7fc
  Stored in directory: /tmp/pip-ephem-wheel-cache-6qcun0sk/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f69e1c1a048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 11s
 3178496/17464789 [====>.........................] - ETA: 0s 
10305536/17464789 [================>.............] - ETA: 0s
14786560/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 00:20:16.459281: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 00:20:16.463927: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-15 00:20:16.464080: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555cb273da90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 00:20:16.464094: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8813 - accuracy: 0.4860
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7433 - accuracy: 0.4950 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7995 - accuracy: 0.4913
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8430 - accuracy: 0.4885
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.8077 - accuracy: 0.4908
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7740 - accuracy: 0.4930
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7652 - accuracy: 0.4936
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7529 - accuracy: 0.4944
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7160 - accuracy: 0.4968
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6927 - accuracy: 0.4983
11000/25000 [============>.................] - ETA: 3s - loss: 7.6694 - accuracy: 0.4998
12000/25000 [=============>................] - ETA: 3s - loss: 7.6832 - accuracy: 0.4989
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6796 - accuracy: 0.4992
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6579 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 2s - loss: 7.6564 - accuracy: 0.5007
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6475 - accuracy: 0.5013
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6531 - accuracy: 0.5009
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6683 - accuracy: 0.4999
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6779 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6743 - accuracy: 0.4995
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6776 - accuracy: 0.4993
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6694 - accuracy: 0.4998
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6564 - accuracy: 0.5007
25000/25000 [==============================] - 7s 269us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 00:20:29.637777
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 00:20:29.637777  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:03<97:08:44, 2.47kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:03<68:13:57, 3.51kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:03<47:48:09, 5.01kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:03<33:26:58, 7.15kB/s].vector_cache/glove.6B.zip:   0%|          | 2.79M/862M [00:03<23:22:01, 10.2kB/s].vector_cache/glove.6B.zip:   1%|          | 6.66M/862M [00:03<16:17:06, 14.6kB/s].vector_cache/glove.6B.zip:   1%|          | 10.3M/862M [00:04<11:21:09, 20.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.6M/862M [00:04<7:53:56, 29.8kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.4M/862M [00:04<5:29:36, 42.5kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.0M/862M [00:04<3:49:16, 60.7kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.8M/862M [00:04<2:39:29, 86.7kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.4M/862M [00:04<1:50:58, 124kB/s] .vector_cache/glove.6B.zip:   5%|▌         | 44.1M/862M [00:04<1:17:14, 177kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.8M/862M [00:04<53:47, 252kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:05<38:30, 350kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:07<28:44, 467kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:07<22:22, 600kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:07<16:12, 827kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:09<13:37, 981kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:09<11:12, 1.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:09<08:15, 1.61MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:11<08:24, 1.58MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:11<08:58, 1.48MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:11<07:02, 1.88MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:11<05:06, 2.59MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:13<14:33, 908kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:13<11:33, 1.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:13<08:24, 1.57MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:15<08:55, 1.47MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:15<08:54, 1.47MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:15<06:54, 1.90MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:15<04:58, 2.63MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:17<12:37:08, 17.3kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:17<8:51:03, 24.6kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:17<6:11:20, 35.1kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:19<4:22:16, 49.6kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:19<3:04:49, 70.4kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:19<2:09:27, 100kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:21<1:33:23, 139kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:21<1:06:39, 194kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:21<46:50, 276kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:23<35:46, 360kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:23<26:24, 487kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:23<18:43, 686kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:25<15:59, 801kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:25<13:55, 919kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:25<10:18, 1.24MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:25<07:22, 1.73MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:27<12:03, 1.06MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:27<09:47, 1.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:27<07:07, 1.78MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<07:51, 1.61MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:29<06:50, 1.85MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:29<05:06, 2.47MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<06:25, 1.96MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:31<07:09, 1.76MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:31<05:40, 2.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<04:06, 3.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<34:12, 366kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:33<25:16, 495kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:33<17:59, 695kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:35<15:21, 811kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:35<13:23, 930kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:35<10:01, 1.24MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<07:09, 1.73MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:37<42:14, 293kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:37<30:52, 401kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:37<21:50, 566kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<18:02, 683kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<15:14, 808kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:39<11:18, 1.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<08:01, 1.53MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:41<35:17, 347kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:41<26:00, 471kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:41<18:26, 663kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<15:38, 779kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<13:31, 901kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:43<09:59, 1.22MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:43<07:08, 1.70MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<06:29, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<8:37:44, 23.4kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:45<6:02:43, 33.4kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<4:13:03, 47.7kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<3:08:59, 63.8kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<2:14:51, 89.3kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:47<1:34:56, 127kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<1:06:20, 181kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<1:24:24, 142kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:49<1:00:19, 199kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:49<42:24, 282kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<32:17, 369kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:51<25:07, 474kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:51<18:05, 658kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:51<12:52, 922kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<12:24, 954kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<09:58, 1.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:53<07:13, 1.63MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<07:42, 1.53MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<06:39, 1.77MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:55<04:54, 2.39MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<06:05, 1.92MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<06:44, 1.74MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:57<05:14, 2.23MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:57<03:53, 3.00MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<06:13, 1.87MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<05:35, 2.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:59<04:09, 2.79MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:00<05:32, 2.09MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:00<06:21, 1.82MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:01<04:58, 2.32MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:01<03:37, 3.18MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<08:32, 1.34MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:02<07:12, 1.59MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:02<05:17, 2.17MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:04<06:17, 1.82MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:04<06:49, 1.68MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:04<05:18, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<03:49, 2.97MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<18:14, 623kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<13:58, 813kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:06<10:03, 1.13MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:08<09:34, 1.18MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:08<09:05, 1.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:08<06:57, 1.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<04:59, 2.25MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<37:18, 301kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<27:18, 411kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:10<19:19, 579kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<16:01, 696kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<13:34, 821kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:12<10:05, 1.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<07:09, 1.55MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<32:50, 337kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:14<24:09, 459kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:14<17:09, 644kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<14:27, 762kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<12:27, 884kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<09:17, 1.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<06:37, 1.65MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<37:32, 292kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:18<27:26, 399kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:18<19:25, 562kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:20<16:00, 680kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:20<12:21, 880kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:20<08:54, 1.22MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<08:39, 1.25MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<13:34, 796kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<10:55, 989kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:22<07:57, 1.36MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<05:50, 1.84MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<07:31, 1.43MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<08:22, 1.28MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:24<06:38, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<04:49, 2.22MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<07:18, 1.46MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<08:10, 1.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:26<06:20, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<04:36, 2.30MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<06:15, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:28<07:25, 1.43MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:28<05:48, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<04:13, 2.50MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<06:11, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:30<07:11, 1.46MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:30<05:42, 1.84MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<04:08, 2.53MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:32<07:23, 1.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:32<07:51, 1.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:32<06:08, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<04:28, 2.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<07:45, 1.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<07:58, 1.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:34<06:08, 1.69MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<04:25, 2.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:36<07:15, 1.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:36<07:52, 1.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:36<06:11, 1.66MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<04:30, 2.28MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<07:21, 1.39MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<07:44, 1.32MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:38<06:01, 1.70MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<04:22, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:40<08:01, 1.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:40<08:05, 1.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:40<06:10, 1.65MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<04:27, 2.27MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:42<07:00, 1.44MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:42<07:15, 1.39MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:42<05:39, 1.78MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:42<04:05, 2.46MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:44<07:52, 1.28MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:44<07:50, 1.28MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:44<05:58, 1.68MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<04:22, 2.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:46<06:26, 1.55MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:46<09:55, 1.01MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:46<08:13, 1.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:46<06:01, 1.65MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<04:25, 2.24MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:48<08:15, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:48<11:09, 888kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:48<09:04, 1.09MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:48<06:39, 1.48MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<04:50, 2.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:50<08:54, 1.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:50<11:03, 890kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:50<08:56, 1.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:50<06:33, 1.50MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<04:47, 2.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:52<09:37, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:52<11:06, 879kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:52<08:44, 1.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:52<06:24, 1.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:52<04:41, 2.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:54<10:18, 941kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:54<11:57, 811kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:54<09:29, 1.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:54<06:52, 1.41MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<05:00, 1.92MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:56<10:40, 902kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:56<11:44, 820kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:56<09:10, 1.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:56<06:42, 1.43MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<04:53, 1.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:58<10:14, 933kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:58<11:24, 838kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:58<09:02, 1.06MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:58<06:36, 1.44MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<04:48, 1.97MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:00<11:19, 838kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:00<12:07, 782kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:00<09:32, 993kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:00<06:55, 1.37MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<05:01, 1.87MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:02<11:39, 809kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:02<12:19, 764kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:02<09:40, 973kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:02<07:02, 1.33MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<05:07, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:04<08:52, 1.05MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:04<11:17, 828kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:04<09:05, 1.03MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:04<06:36, 1.41MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<04:47, 1.94MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<03:40, 2.53MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:06<1:01:47, 150kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:06<47:46, 194kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:06<34:38, 268kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:06<24:27, 379kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<17:22, 532kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:07<13:01, 707kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:07<5:32:56, 27.7kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:08<3:54:09, 39.3kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:08<2:43:50, 56.1kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:08<1:54:33, 80.0kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<1:24:37, 108kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:10<1:03:42, 144kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:10<45:46, 200kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:10<32:16, 283kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<22:44, 400kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:11<20:27, 444kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:12<18:45, 484kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:12<14:19, 633kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:12<10:14, 883kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:12<07:21, 1.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<05:26, 1.66MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:13<1:34:16, 95.6kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:14<1:10:52, 127kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:14<50:44, 177kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:14<35:43, 251kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<25:08, 356kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:15<22:00, 406kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:16<20:17, 440kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:16<15:21, 582kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:16<11:00, 809kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<07:53, 1.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:17<10:01, 885kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:18<11:49, 750kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:18<09:24, 942kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:18<06:51, 1.29MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:18<04:59, 1.76MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:19<07:59, 1.10MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:19<10:21, 849kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:20<08:22, 1.05MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:20<06:05, 1.44MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:20<04:26, 1.97MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:20<03:20, 2.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<3:45:14, 38.8kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<2:41:54, 53.9kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:22<1:54:25, 76.3kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:22<1:20:08, 109kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<56:07, 155kB/s]  .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:23<42:48, 202kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:23<34:13, 253kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:24<25:05, 345kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:24<17:45, 487kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:24<12:36, 683kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<09:00, 955kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:25<7:26:39, 19.2kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:25<5:17:16, 27.1kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:26<3:42:54, 38.5kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:26<2:35:56, 55.0kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:26<1:49:01, 78.4kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:27<1:19:45, 107kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:27<59:58, 142kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:28<43:01, 198kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:28<30:15, 281kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<21:19, 397kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:29<18:03, 469kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:29<16:46, 504kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:29<12:46, 662kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:30<09:11, 918kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:30<06:34, 1.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:31<09:43, 863kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:31<10:54, 769kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:31<08:39, 968kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:32<06:18, 1.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<04:35, 1.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:33<08:32, 973kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:33<10:02, 829kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:33<07:59, 1.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:34<05:48, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<04:11, 1.97MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:35<07:50, 1.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:35<09:08, 902kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:35<07:18, 1.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:36<05:21, 1.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:36<03:54, 2.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:37<10:10, 805kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:37<10:44, 762kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:37<08:25, 970kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:38<06:08, 1.33MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:38<04:25, 1.83MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:39<11:33, 701kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:39<11:42, 693kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:39<09:02, 896kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:39<06:31, 1.24MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:40<04:41, 1.72MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:41<09:11, 875kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:41<09:42, 829kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:41<07:30, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:41<05:28, 1.46MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<03:58, 2.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:43<19:50, 402kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:43<16:52, 472kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:43<12:25, 641kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:43<08:54, 892kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<06:21, 1.24MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:45<08:18, 952kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:45<08:34, 922kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:45<06:43, 1.17MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<04:52, 1.62MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:46<03:30, 2.23MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:47<27:58, 280kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:47<22:18, 351kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:47<16:09, 484kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<11:29, 680kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:48<08:12, 948kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:49<09:56, 781kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:49<09:39, 803kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:49<07:18, 1.06MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:49<05:16, 1.47MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<03:49, 2.01MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:51<07:50, 983kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:51<07:49, 984kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:51<06:04, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<04:23, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:53<05:07, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:53<05:46, 1.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:53<04:30, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:53<03:20, 2.28MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<02:25, 3.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:55<3:34:07, 35.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:55<2:32:00, 49.7kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:55<1:46:49, 70.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<1:14:38, 101kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:57<54:15, 138kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:57<39:59, 187kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:57<28:28, 263kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<19:57, 373kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:59<16:29, 450kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:59<13:48, 537kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:59<10:06, 733kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<07:09, 1.03MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<07:18, 1.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<07:00, 1.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:01<05:20, 1.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<03:51, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<05:11, 1.40MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<05:15, 1.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:03<04:01, 1.81MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<02:55, 2.48MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<04:27, 1.62MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:05<04:38, 1.55MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:05<03:34, 2.02MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<02:35, 2.77MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:07<04:52, 1.47MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:07<04:51, 1.47MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<03:42, 1.92MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<02:40, 2.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:09<06:50, 1.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:09<06:36, 1.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:09<05:01, 1.41MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:09<03:36, 1.95MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:11<04:54, 1.43MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:11<05:08, 1.36MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:11<03:57, 1.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<02:50, 2.44MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:13<05:22, 1.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:13<05:00, 1.38MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:13<03:48, 1.82MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<03:49, 1.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<04:17, 1.60MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:15<03:21, 2.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:15<02:25, 2.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:17<04:53, 1.39MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:17<03:53, 1.74MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:17<02:55, 2.31MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<03:24, 1.97MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:19<03:24, 1.97MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<02:38, 2.54MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:21<03:01, 2.20MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:21<03:28, 1.92MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:21<02:43, 2.44MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:21<01:58, 3.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:23<05:51, 1.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:23<04:53, 1.35MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<03:39, 1.79MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:23<02:37, 2.48MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:25<10:08, 643kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:25<08:22, 778kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:25<06:07, 1.06MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<04:19, 1.49MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:27<14:15, 453kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:27<11:11, 576kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:27<08:07, 792kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<05:56, 1.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:29<4:46:57, 22.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:29<3:21:16, 31.7kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<2:20:25, 45.2kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:31<1:39:20, 63.6kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:31<1:12:05, 87.6kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:31<51:02, 124kB/s]   .vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:31<35:42, 176kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:32<26:19, 237kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:33<19:35, 319kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:33<13:58, 446kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:34<10:43, 576kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:35<08:42, 709kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:35<06:19, 974kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<04:28, 1.37MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:36<07:15, 843kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:37<06:12, 983kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:37<04:38, 1.31MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:38<04:12, 1.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:38<03:55, 1.54MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:39<03:02, 1.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:39<02:11, 2.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:40<05:41, 1.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:40<06:24, 931kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:41<05:04, 1.18MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<03:39, 1.62MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<03:56, 1.50MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<03:52, 1.53MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:43<02:56, 2.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:43<02:07, 2.75MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<04:25, 1.32MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<04:14, 1.37MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:45<03:14, 1.80MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<03:12, 1.80MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<03:21, 1.71MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:46<02:34, 2.23MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<01:53, 3.03MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:48<03:49, 1.49MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:48<03:44, 1.52MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:48<02:53, 1.97MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<02:56, 1.92MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:50<03:06, 1.81MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:50<02:26, 2.30MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:52<02:36, 2.13MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:52<02:54, 1.91MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:52<02:15, 2.46MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<01:40, 3.29MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:54<02:59, 1.84MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:54<03:07, 1.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:54<02:26, 2.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:56<02:35, 2.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:56<02:49, 1.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<02:14, 2.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<02:26, 2.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<03:53, 1.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<03:14, 1.65MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<02:21, 2.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:00<02:53, 1.82MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:00<03:01, 1.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:00<02:19, 2.27MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<01:40, 3.11MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:02<04:27, 1.17MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:02<04:06, 1.27MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:02<03:04, 1.69MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<02:12, 2.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:04<05:04, 1.01MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:04<04:33, 1.13MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:04<03:25, 1.50MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:06<03:13, 1.58MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:06<03:12, 1.59MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:06<02:28, 2.04MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:08<02:32, 1.97MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:08<02:43, 1.84MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<02:08, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:10<02:18, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:10<02:34, 1.92MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<01:59, 2.47MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<01:26, 3.38MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:12<04:22, 1.12MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:12<04:00, 1.22MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:12<03:01, 1.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:14<02:53, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:14<03:49, 1.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:14<03:04, 1.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:14<02:15, 2.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<02:40, 1.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<02:45, 1.72MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<02:06, 2.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<01:32, 3.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:18<03:17, 1.42MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:18<03:10, 1.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:18<02:26, 1.91MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<02:26, 1.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<02:34, 1.78MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:20<02:01, 2.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<02:08, 2.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:22<02:23, 1.90MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:22<01:52, 2.40MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<01:29, 2.99MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<3:19:59, 22.3kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:24<2:20:12, 31.8kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<1:37:37, 45.4kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:26<1:08:54, 63.8kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:26<50:00, 87.8kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:26<35:22, 124kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<24:42, 176kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:27<18:11, 238kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:28<13:32, 319kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<09:38, 446kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:29<07:22, 577kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:30<05:59, 709kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:30<04:23, 966kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:31<03:43, 1.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:32<03:23, 1.23MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<02:32, 1.64MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:32<01:48, 2.28MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:33<04:12, 980kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:34<03:42, 1.11MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:34<02:47, 1.47MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:35<02:35, 1.56MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:35<02:34, 1.57MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:36<01:59, 2.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<02:02, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<02:12, 1.81MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:38<01:43, 2.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:39<01:50, 2.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:39<02:01, 1.93MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:40<01:35, 2.44MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:41<01:44, 2.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:41<01:56, 1.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:41<01:32, 2.49MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:43<01:41, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:43<01:54, 1.97MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:43<01:29, 2.51MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:44<01:05, 3.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:45<03:35, 1.03MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:45<03:14, 1.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:45<02:26, 1.51MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:47<02:17, 1.59MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:47<02:17, 1.59MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:47<01:46, 2.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:49<01:48, 1.97MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:49<01:56, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:49<01:31, 2.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:51<01:37, 2.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:51<02:33, 1.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:51<02:04, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:51<01:31, 2.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:53<01:52, 1.83MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:53<01:57, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:53<01:31, 2.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:55<01:36, 2.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:55<01:47, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:55<01:22, 2.43MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<00:59, 3.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<02:37, 1.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:57<02:27, 1.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:57<01:51, 1.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:59<01:49, 1.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:59<01:52, 1.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<01:27, 2.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:01<01:31, 2.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:01<01:39, 1.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:01<01:17, 2.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<00:55, 3.34MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:03<02:49, 1.09MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:03<02:34, 1.20MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:03<01:56, 1.58MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<01:50, 1.64MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<01:51, 1.63MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<01:26, 2.09MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:07<01:28, 2.00MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:07<01:35, 1.86MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:07<01:14, 2.35MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<01:20, 2.16MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<01:30, 1.92MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:09<01:09, 2.46MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<00:50, 3.36MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<02:20, 1.20MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:11<02:11, 1.29MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:11<01:39, 1.69MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<01:35, 1.72MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:13<01:37, 1.69MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:13<01:15, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:15<01:18, 2.04MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:15<01:25, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:15<01:06, 2.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<00:52, 2.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<1:54:50, 22.7kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<1:20:25, 32.4kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:17<55:45, 46.2kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:18<39:08, 64.9kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<28:24, 89.4kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<20:05, 126kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<13:57, 180kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:20<10:13, 242kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<07:36, 325kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:21<05:24, 454kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:22<04:06, 585kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<03:20, 718kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:23<02:26, 977kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:24<02:03, 1.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:25<01:52, 1.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<01:24, 1.64MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:26<01:20, 1.68MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:26<01:21, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:27<01:03, 2.13MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:28<01:05, 2.02MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:28<01:10, 1.87MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<00:55, 2.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<00:39, 3.27MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:30<05:52, 362kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:30<04:31, 470kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:31<03:13, 654kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:31<02:14, 923kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:32<03:03, 675kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:32<02:54, 709kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:32<02:11, 934kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:33<01:33, 1.30MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:34<01:32, 1.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:34<01:27, 1.36MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:34<01:05, 1.80MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:35<00:47, 2.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:36<01:17, 1.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:36<01:15, 1.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:36<00:58, 1.96MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<00:40, 2.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:38<1:48:51, 17.0kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:38<1:16:24, 24.2kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:38<53:10, 34.5kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:40<36:28, 49.0kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:40<25:49, 69.0kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:40<18:01, 98.1kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:42<12:32, 137kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:42<09:05, 188kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:42<06:23, 266kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:44<04:36, 357kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:44<03:31, 466kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:44<02:31, 644kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:46<01:58, 799kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<01:40, 940kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<01:13, 1.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<00:54, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:48<01:00, 1.51MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:48<01:07, 1.33MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:48<00:53, 1.68MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<00:37, 2.31MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:50<01:03, 1.37MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:50<01:07, 1.28MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:50<00:52, 1.62MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:50<00:37, 2.25MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:52<01:03, 1.29MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:52<01:05, 1.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<00:50, 1.62MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:52<00:35, 2.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<00:53, 1.46MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<00:57, 1.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:44, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:54<00:31, 2.40MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:42, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:48, 1.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<00:38, 1.91MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:56<00:26, 2.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:55, 1.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:55, 1.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:41, 1.65MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:58<00:29, 2.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<00:38, 1.70MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:44, 1.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:34, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:00<00:25, 2.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<00:31, 1.92MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<00:37, 1.65MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<00:28, 2.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<00:20, 2.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<00:33, 1.72MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<00:35, 1.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:04<00:28, 2.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:04<00:19, 2.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<00:46, 1.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<00:44, 1.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<00:33, 1.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:06<00:23, 2.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:29, 1.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:31, 1.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<00:24, 1.99MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:16, 2.74MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:43, 1.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<00:39, 1.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<00:29, 1.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:20, 2.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:31, 1.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:30, 1.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:22, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:15, 2.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:38, 943kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:34, 1.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:14<00:25, 1.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:17, 1.95MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:16<00:25, 1.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:16<00:24, 1.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:18, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:12, 2.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<00:36, 784kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<00:31, 889kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:23, 1.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:18<00:15, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<00:18, 1.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<00:17, 1.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:20<00:13, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:20<00:08, 2.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<00:17, 1.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<00:16, 1.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<00:11, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:07, 2.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:19, 837kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:16, 962kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:11, 1.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:06, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:07, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:25<08:31, 23.4kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<05:50, 33.2kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:26<03:38, 47.4kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:27<01:57, 66.8kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<01:22, 93.2kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<00:53, 132kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:28<00:23, 188kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:29<00:15, 233kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:11, 309kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:06, 429kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:30<00:00, 607kB/s].vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 922/400000 [00:00<00:43, 9215.57it/s]  0%|          | 1808/400000 [00:00<00:43, 9105.87it/s]  1%|          | 2691/400000 [00:00<00:44, 8943.15it/s]  1%|          | 3598/400000 [00:00<00:44, 8978.69it/s]  1%|          | 4497/400000 [00:00<00:44, 8980.27it/s]  1%|▏         | 5410/400000 [00:00<00:43, 9021.43it/s]  2%|▏         | 6284/400000 [00:00<00:44, 8934.66it/s]  2%|▏         | 7181/400000 [00:00<00:43, 8944.83it/s]  2%|▏         | 8112/400000 [00:00<00:43, 9048.67it/s]  2%|▏         | 9046/400000 [00:01<00:42, 9132.20it/s]  2%|▏         | 9968/400000 [00:01<00:42, 9158.05it/s]  3%|▎         | 10866/400000 [00:01<00:42, 9073.83it/s]  3%|▎         | 11774/400000 [00:01<00:42, 9075.06it/s]  3%|▎         | 12697/400000 [00:01<00:42, 9119.41it/s]  3%|▎         | 13643/400000 [00:01<00:41, 9216.33it/s]  4%|▎         | 14561/400000 [00:01<00:42, 9158.34it/s]  4%|▍         | 15475/400000 [00:01<00:42, 9142.88it/s]  4%|▍         | 16388/400000 [00:01<00:42, 9125.84it/s]  4%|▍         | 17315/400000 [00:01<00:41, 9168.59it/s]  5%|▍         | 18239/400000 [00:02<00:41, 9189.55it/s]  5%|▍         | 19159/400000 [00:02<00:41, 9191.88it/s]  5%|▌         | 20078/400000 [00:02<00:41, 9132.95it/s]  5%|▌         | 20992/400000 [00:02<00:41, 9089.27it/s]  5%|▌         | 21903/400000 [00:02<00:41, 9093.74it/s]  6%|▌         | 22813/400000 [00:02<00:41, 9054.86it/s]  6%|▌         | 23719/400000 [00:02<00:41, 9011.50it/s]  6%|▌         | 24621/400000 [00:02<00:41, 8982.96it/s]  6%|▋         | 25520/400000 [00:02<00:41, 8967.31it/s]  7%|▋         | 26423/400000 [00:02<00:41, 8983.82it/s]  7%|▋         | 27325/400000 [00:03<00:41, 8993.82it/s]  7%|▋         | 28246/400000 [00:03<00:41, 9054.84it/s]  7%|▋         | 29152/400000 [00:03<00:41, 9044.34it/s]  8%|▊         | 30057/400000 [00:03<00:41, 9003.14it/s]  8%|▊         | 30977/400000 [00:03<00:40, 9057.98it/s]  8%|▊         | 31886/400000 [00:03<00:40, 9066.40it/s]  8%|▊         | 32815/400000 [00:03<00:40, 9130.57it/s]  8%|▊         | 33733/400000 [00:03<00:40, 9143.50it/s]  9%|▊         | 34648/400000 [00:03<00:40, 9045.67it/s]  9%|▉         | 35553/400000 [00:03<00:41, 8824.93it/s]  9%|▉         | 36469/400000 [00:04<00:40, 8922.57it/s]  9%|▉         | 37387/400000 [00:04<00:40, 8996.17it/s] 10%|▉         | 38288/400000 [00:04<00:40, 8970.13it/s] 10%|▉         | 39186/400000 [00:04<00:40, 8918.61it/s] 10%|█         | 40115/400000 [00:04<00:39, 9026.03it/s] 10%|█         | 41036/400000 [00:04<00:39, 9079.07it/s] 10%|█         | 41956/400000 [00:04<00:39, 9113.58it/s] 11%|█         | 42873/400000 [00:04<00:39, 9128.14it/s] 11%|█         | 43787/400000 [00:04<00:39, 9040.66it/s] 11%|█         | 44694/400000 [00:04<00:39, 9049.29it/s] 11%|█▏        | 45610/400000 [00:05<00:39, 9080.22it/s] 12%|█▏        | 46531/400000 [00:05<00:38, 9115.84it/s] 12%|█▏        | 47443/400000 [00:05<00:38, 9108.75it/s] 12%|█▏        | 48355/400000 [00:05<00:38, 9047.85it/s] 12%|█▏        | 49278/400000 [00:05<00:38, 9101.09it/s] 13%|█▎        | 50189/400000 [00:05<00:38, 9047.02it/s] 13%|█▎        | 51094/400000 [00:05<00:38, 9020.91it/s] 13%|█▎        | 52014/400000 [00:05<00:38, 9073.11it/s] 13%|█▎        | 52922/400000 [00:05<00:38, 9035.96it/s] 13%|█▎        | 53826/400000 [00:05<00:38, 9028.33it/s] 14%|█▎        | 54743/400000 [00:06<00:38, 9070.31it/s] 14%|█▍        | 55660/400000 [00:06<00:37, 9098.63it/s] 14%|█▍        | 56587/400000 [00:06<00:37, 9146.36it/s] 14%|█▍        | 57502/400000 [00:06<00:37, 9126.08it/s] 15%|█▍        | 58423/400000 [00:06<00:37, 9149.66it/s] 15%|█▍        | 59339/400000 [00:06<00:37, 9010.84it/s] 15%|█▌        | 60260/400000 [00:06<00:37, 9068.41it/s] 15%|█▌        | 61181/400000 [00:06<00:37, 9110.05it/s] 16%|█▌        | 62093/400000 [00:06<00:37, 9052.29it/s] 16%|█▌        | 63006/400000 [00:06<00:37, 9074.01it/s] 16%|█▌        | 63914/400000 [00:07<00:37, 9034.20it/s] 16%|█▌        | 64818/400000 [00:07<00:38, 8805.59it/s] 16%|█▋        | 65725/400000 [00:07<00:37, 8881.51it/s] 17%|█▋        | 66615/400000 [00:07<00:37, 8849.41it/s] 17%|█▋        | 67536/400000 [00:07<00:37, 8952.37it/s] 17%|█▋        | 68470/400000 [00:07<00:36, 9062.59it/s] 17%|█▋        | 69397/400000 [00:07<00:36, 9122.62it/s] 18%|█▊        | 70313/400000 [00:07<00:36, 9133.67it/s] 18%|█▊        | 71227/400000 [00:07<00:36, 9045.45it/s] 18%|█▊        | 72133/400000 [00:07<00:36, 9028.83it/s] 18%|█▊        | 73037/400000 [00:08<00:36, 8878.75it/s] 18%|█▊        | 73960/400000 [00:08<00:36, 8981.27it/s] 19%|█▊        | 74875/400000 [00:08<00:36, 9029.90it/s] 19%|█▉        | 75786/400000 [00:08<00:35, 9050.85it/s] 19%|█▉        | 76692/400000 [00:08<00:36, 8913.35it/s] 19%|█▉        | 77599/400000 [00:08<00:35, 8957.13it/s] 20%|█▉        | 78496/400000 [00:08<00:35, 8935.69it/s] 20%|█▉        | 79391/400000 [00:08<00:35, 8906.38it/s] 20%|██        | 80282/400000 [00:08<00:36, 8794.22it/s] 20%|██        | 81184/400000 [00:08<00:35, 8859.57it/s] 21%|██        | 82089/400000 [00:09<00:35, 8914.26it/s] 21%|██        | 83011/400000 [00:09<00:35, 9001.16it/s] 21%|██        | 83912/400000 [00:09<00:35, 8960.86it/s] 21%|██        | 84809/400000 [00:09<00:35, 8793.25it/s] 21%|██▏       | 85718/400000 [00:09<00:35, 8878.54it/s] 22%|██▏       | 86610/400000 [00:09<00:35, 8888.12it/s] 22%|██▏       | 87520/400000 [00:09<00:34, 8949.89it/s] 22%|██▏       | 88438/400000 [00:09<00:34, 9016.65it/s] 22%|██▏       | 89341/400000 [00:09<00:34, 8961.69it/s] 23%|██▎       | 90238/400000 [00:10<00:35, 8837.79it/s] 23%|██▎       | 91133/400000 [00:10<00:34, 8870.28it/s] 23%|██▎       | 92021/400000 [00:10<00:35, 8709.91it/s] 23%|██▎       | 92909/400000 [00:10<00:35, 8759.75it/s] 23%|██▎       | 93786/400000 [00:10<00:34, 8749.69it/s] 24%|██▎       | 94697/400000 [00:10<00:34, 8853.46it/s] 24%|██▍       | 95601/400000 [00:10<00:34, 8907.07it/s] 24%|██▍       | 96493/400000 [00:10<00:34, 8778.56it/s] 24%|██▍       | 97409/400000 [00:10<00:34, 8887.12it/s] 25%|██▍       | 98303/400000 [00:10<00:33, 8902.45it/s] 25%|██▍       | 99205/400000 [00:11<00:33, 8935.17it/s] 25%|██▌       | 100128/400000 [00:11<00:33, 9020.92it/s] 25%|██▌       | 101056/400000 [00:11<00:32, 9095.28it/s] 25%|██▌       | 101975/400000 [00:11<00:32, 9121.71it/s] 26%|██▌       | 102888/400000 [00:11<00:32, 9013.65it/s] 26%|██▌       | 103804/400000 [00:11<00:32, 9056.97it/s] 26%|██▌       | 104719/400000 [00:11<00:32, 9082.98it/s] 26%|██▋       | 105628/400000 [00:11<00:32, 9081.66it/s] 27%|██▋       | 106537/400000 [00:11<00:32, 9076.95it/s] 27%|██▋       | 107445/400000 [00:11<00:32, 9025.71it/s] 27%|██▋       | 108348/400000 [00:12<00:32, 9007.48it/s] 27%|██▋       | 109260/400000 [00:12<00:32, 9040.92it/s] 28%|██▊       | 110165/400000 [00:12<00:32, 9026.33it/s] 28%|██▊       | 111068/400000 [00:12<00:32, 9023.40it/s] 28%|██▊       | 111971/400000 [00:12<00:32, 8754.08it/s] 28%|██▊       | 112884/400000 [00:12<00:32, 8861.04it/s] 28%|██▊       | 113806/400000 [00:12<00:31, 8963.26it/s] 29%|██▊       | 114728/400000 [00:12<00:31, 9037.65it/s] 29%|██▉       | 115651/400000 [00:12<00:31, 9094.26it/s] 29%|██▉       | 116562/400000 [00:12<00:31, 8998.39it/s] 29%|██▉       | 117463/400000 [00:13<00:31, 8967.46it/s] 30%|██▉       | 118361/400000 [00:13<00:31, 8913.64it/s] 30%|██▉       | 119259/400000 [00:13<00:31, 8933.30it/s] 30%|███       | 120166/400000 [00:13<00:31, 8972.00it/s] 30%|███       | 121064/400000 [00:13<00:31, 8919.18it/s] 30%|███       | 121971/400000 [00:13<00:31, 8963.80it/s] 31%|███       | 122885/400000 [00:13<00:30, 9013.61it/s] 31%|███       | 123787/400000 [00:13<00:30, 8971.50it/s] 31%|███       | 124685/400000 [00:13<00:30, 8967.79it/s] 31%|███▏      | 125582/400000 [00:13<00:30, 8898.08it/s] 32%|███▏      | 126473/400000 [00:14<00:30, 8824.72it/s] 32%|███▏      | 127376/400000 [00:14<00:30, 8883.95it/s] 32%|███▏      | 128298/400000 [00:14<00:30, 8980.33it/s] 32%|███▏      | 129211/400000 [00:14<00:30, 9022.31it/s] 33%|███▎      | 130114/400000 [00:14<00:29, 9000.92it/s] 33%|███▎      | 131032/400000 [00:14<00:29, 9052.93it/s] 33%|███▎      | 131966/400000 [00:14<00:29, 9137.17it/s] 33%|███▎      | 132881/400000 [00:14<00:29, 9134.49it/s] 33%|███▎      | 133799/400000 [00:14<00:29, 9147.18it/s] 34%|███▎      | 134714/400000 [00:14<00:29, 9078.12it/s] 34%|███▍      | 135623/400000 [00:15<00:29, 9045.81it/s] 34%|███▍      | 136528/400000 [00:15<00:29, 9034.99it/s] 34%|███▍      | 137439/400000 [00:15<00:28, 9054.95it/s] 35%|███▍      | 138355/400000 [00:15<00:28, 9085.32it/s] 35%|███▍      | 139264/400000 [00:15<00:29, 8948.43it/s] 35%|███▌      | 140174/400000 [00:15<00:28, 8992.31it/s] 35%|███▌      | 141074/400000 [00:15<00:28, 8929.85it/s] 35%|███▌      | 141971/400000 [00:15<00:28, 8940.71it/s] 36%|███▌      | 142866/400000 [00:15<00:28, 8911.67it/s] 36%|███▌      | 143758/400000 [00:15<00:28, 8858.47it/s] 36%|███▌      | 144661/400000 [00:16<00:28, 8907.81it/s] 36%|███▋      | 145569/400000 [00:16<00:28, 8956.49it/s] 37%|███▋      | 146470/400000 [00:16<00:28, 8969.75it/s] 37%|███▋      | 147389/400000 [00:16<00:27, 9033.89it/s] 37%|███▋      | 148293/400000 [00:16<00:28, 8813.30it/s] 37%|███▋      | 149213/400000 [00:16<00:28, 8923.43it/s] 38%|███▊      | 150119/400000 [00:16<00:27, 8962.89it/s] 38%|███▊      | 151017/400000 [00:16<00:27, 8905.44it/s] 38%|███▊      | 151928/400000 [00:16<00:27, 8963.45it/s] 38%|███▊      | 152825/400000 [00:16<00:27, 8948.54it/s] 38%|███▊      | 153726/400000 [00:17<00:27, 8964.39it/s] 39%|███▊      | 154623/400000 [00:17<00:27, 8914.42it/s] 39%|███▉      | 155522/400000 [00:17<00:27, 8936.72it/s] 39%|███▉      | 156416/400000 [00:17<00:27, 8930.15it/s] 39%|███▉      | 157310/400000 [00:17<00:27, 8698.58it/s] 40%|███▉      | 158235/400000 [00:17<00:27, 8855.03it/s] 40%|███▉      | 159152/400000 [00:17<00:26, 8945.00it/s] 40%|████      | 160062/400000 [00:17<00:26, 8988.61it/s] 40%|████      | 160986/400000 [00:17<00:26, 9062.21it/s] 40%|████      | 161894/400000 [00:17<00:26, 9064.70it/s] 41%|████      | 162802/400000 [00:18<00:26, 9036.62it/s] 41%|████      | 163707/400000 [00:18<00:26, 9006.54it/s] 41%|████      | 164611/400000 [00:18<00:26, 9016.49it/s] 41%|████▏     | 165526/400000 [00:18<00:25, 9055.67it/s] 42%|████▏     | 166432/400000 [00:18<00:25, 9005.72it/s] 42%|████▏     | 167352/400000 [00:18<00:25, 9061.23it/s] 42%|████▏     | 168259/400000 [00:18<00:25, 9060.70it/s] 42%|████▏     | 169166/400000 [00:18<00:25, 8976.04it/s] 43%|████▎     | 170064/400000 [00:18<00:25, 8947.04it/s] 43%|████▎     | 170959/400000 [00:19<00:25, 8820.26it/s] 43%|████▎     | 171870/400000 [00:19<00:25, 8903.82it/s] 43%|████▎     | 172769/400000 [00:19<00:25, 8929.21it/s] 43%|████▎     | 173676/400000 [00:19<00:25, 8969.36it/s] 44%|████▎     | 174579/400000 [00:19<00:25, 8986.85it/s] 44%|████▍     | 175478/400000 [00:19<00:25, 8975.65it/s] 44%|████▍     | 176389/400000 [00:19<00:24, 9013.77it/s] 44%|████▍     | 177292/400000 [00:19<00:24, 9016.83it/s] 45%|████▍     | 178194/400000 [00:19<00:24, 8985.86it/s] 45%|████▍     | 179094/400000 [00:19<00:24, 8988.81it/s] 45%|████▌     | 180011/400000 [00:20<00:24, 9041.53it/s] 45%|████▌     | 180916/400000 [00:20<00:24, 9043.64it/s] 45%|████▌     | 181821/400000 [00:20<00:24, 9016.38it/s] 46%|████▌     | 182723/400000 [00:20<00:24, 8995.86it/s] 46%|████▌     | 183642/400000 [00:20<00:23, 9050.74it/s] 46%|████▌     | 184565/400000 [00:20<00:23, 9102.65it/s] 46%|████▋     | 185476/400000 [00:20<00:23, 8961.88it/s] 47%|████▋     | 186400/400000 [00:20<00:23, 9043.02it/s] 47%|████▋     | 187318/400000 [00:20<00:23, 9081.68it/s] 47%|████▋     | 188227/400000 [00:20<00:23, 9040.51it/s] 47%|████▋     | 189136/400000 [00:21<00:23, 9053.17it/s] 48%|████▊     | 190042/400000 [00:21<00:23, 8843.78it/s] 48%|████▊     | 190956/400000 [00:21<00:23, 8929.37it/s] 48%|████▊     | 191874/400000 [00:21<00:23, 9001.83it/s] 48%|████▊     | 192782/400000 [00:21<00:22, 9022.91it/s] 48%|████▊     | 193685/400000 [00:21<00:22, 9000.67it/s] 49%|████▊     | 194586/400000 [00:21<00:23, 8830.72it/s] 49%|████▉     | 195495/400000 [00:21<00:22, 8904.96it/s] 49%|████▉     | 196406/400000 [00:21<00:22, 8964.86it/s] 49%|████▉     | 197310/400000 [00:21<00:22, 8986.58it/s] 50%|████▉     | 198220/400000 [00:22<00:22, 9020.26it/s] 50%|████▉     | 199123/400000 [00:22<00:22, 8964.57it/s] 50%|█████     | 200045/400000 [00:22<00:22, 9037.80it/s] 50%|█████     | 200982/400000 [00:22<00:21, 9134.75it/s] 50%|█████     | 201896/400000 [00:22<00:21, 9096.62it/s] 51%|█████     | 202807/400000 [00:22<00:21, 9095.37it/s] 51%|█████     | 203717/400000 [00:22<00:21, 8975.98it/s] 51%|█████     | 204626/400000 [00:22<00:21, 9009.50it/s] 51%|█████▏    | 205573/400000 [00:22<00:21, 9140.13it/s] 52%|█████▏    | 206513/400000 [00:22<00:21, 9209.99it/s] 52%|█████▏    | 207435/400000 [00:23<00:21, 9083.77it/s] 52%|█████▏    | 208345/400000 [00:23<00:21, 9032.98it/s] 52%|█████▏    | 209262/400000 [00:23<00:21, 9073.52it/s] 53%|█████▎    | 210181/400000 [00:23<00:20, 9106.27it/s] 53%|█████▎    | 211093/400000 [00:23<00:20, 9047.01it/s] 53%|█████▎    | 211999/400000 [00:23<00:20, 9039.56it/s] 53%|█████▎    | 212904/400000 [00:23<00:20, 8975.51it/s] 53%|█████▎    | 213809/400000 [00:23<00:20, 8994.73it/s] 54%|█████▎    | 214709/400000 [00:23<00:20, 8977.26it/s] 54%|█████▍    | 215636/400000 [00:23<00:20, 9060.73it/s] 54%|█████▍    | 216543/400000 [00:24<00:20, 8925.79it/s] 54%|█████▍    | 217437/400000 [00:24<00:20, 8763.85it/s] 55%|█████▍    | 218315/400000 [00:24<00:21, 8632.02it/s] 55%|█████▍    | 219186/400000 [00:24<00:20, 8652.91it/s] 55%|█████▌    | 220090/400000 [00:24<00:20, 8763.67it/s] 55%|█████▌    | 221026/400000 [00:24<00:20, 8932.15it/s] 55%|█████▌    | 221921/400000 [00:24<00:20, 8779.58it/s] 56%|█████▌    | 222821/400000 [00:24<00:20, 8843.21it/s] 56%|█████▌    | 223707/400000 [00:24<00:20, 8711.89it/s] 56%|█████▌    | 224626/400000 [00:24<00:19, 8847.83it/s] 56%|█████▋    | 225513/400000 [00:25<00:19, 8771.48it/s] 57%|█████▋    | 226392/400000 [00:25<00:19, 8773.71it/s] 57%|█████▋    | 227305/400000 [00:25<00:19, 8876.39it/s] 57%|█████▋    | 228211/400000 [00:25<00:19, 8930.24it/s] 57%|█████▋    | 229110/400000 [00:25<00:19, 8945.09it/s] 58%|█████▊    | 230023/400000 [00:25<00:18, 8997.50it/s] 58%|█████▊    | 230924/400000 [00:25<00:18, 8979.50it/s] 58%|█████▊    | 231839/400000 [00:25<00:18, 9028.26it/s] 58%|█████▊    | 232743/400000 [00:25<00:18, 9007.43it/s] 58%|█████▊    | 233644/400000 [00:25<00:18, 8919.31it/s] 59%|█████▊    | 234537/400000 [00:26<00:18, 8853.87it/s] 59%|█████▉    | 235423/400000 [00:26<00:18, 8854.61it/s] 59%|█████▉    | 236337/400000 [00:26<00:18, 8936.57it/s] 59%|█████▉    | 237238/400000 [00:26<00:18, 8957.44it/s] 60%|█████▉    | 238148/400000 [00:26<00:17, 8997.41it/s] 60%|█████▉    | 239052/400000 [00:26<00:17, 9008.13it/s] 60%|█████▉    | 239953/400000 [00:26<00:17, 8957.53it/s] 60%|██████    | 240849/400000 [00:26<00:17, 8930.45it/s] 60%|██████    | 241743/400000 [00:26<00:17, 8895.71it/s] 61%|██████    | 242660/400000 [00:26<00:17, 8974.17it/s] 61%|██████    | 243580/400000 [00:27<00:17, 9039.56it/s] 61%|██████    | 244485/400000 [00:27<00:17, 8972.07it/s] 61%|██████▏   | 245383/400000 [00:27<00:17, 8846.88it/s] 62%|██████▏   | 246291/400000 [00:27<00:17, 8913.28it/s] 62%|██████▏   | 247186/400000 [00:27<00:17, 8923.84it/s] 62%|██████▏   | 248079/400000 [00:27<00:17, 8909.01it/s] 62%|██████▏   | 248971/400000 [00:27<00:17, 8823.89it/s] 62%|██████▏   | 249854/400000 [00:27<00:17, 8794.85it/s] 63%|██████▎   | 250734/400000 [00:27<00:16, 8780.73it/s] 63%|██████▎   | 251613/400000 [00:28<00:16, 8782.57it/s] 63%|██████▎   | 252492/400000 [00:28<00:16, 8699.58it/s] 63%|██████▎   | 253367/400000 [00:28<00:16, 8713.90it/s] 64%|██████▎   | 254240/400000 [00:28<00:16, 8713.80it/s] 64%|██████▍   | 255156/400000 [00:28<00:16, 8842.80it/s] 64%|██████▍   | 256051/400000 [00:28<00:16, 8874.13it/s] 64%|██████▍   | 256943/400000 [00:28<00:16, 8886.59it/s] 64%|██████▍   | 257843/400000 [00:28<00:15, 8918.14it/s] 65%|██████▍   | 258746/400000 [00:28<00:15, 8949.96it/s] 65%|██████▍   | 259642/400000 [00:28<00:15, 8950.95it/s] 65%|██████▌   | 260552/400000 [00:29<00:15, 8994.37it/s] 65%|██████▌   | 261459/400000 [00:29<00:15, 9014.46it/s] 66%|██████▌   | 262361/400000 [00:29<00:15, 8971.47it/s] 66%|██████▌   | 263259/400000 [00:29<00:15, 8937.70it/s] 66%|██████▌   | 264156/400000 [00:29<00:15, 8945.14it/s] 66%|██████▋   | 265057/400000 [00:29<00:15, 8963.22it/s] 66%|██████▋   | 265954/400000 [00:29<00:15, 8911.78it/s] 67%|██████▋   | 266846/400000 [00:29<00:14, 8892.23it/s] 67%|██████▋   | 267799/400000 [00:29<00:14, 9072.26it/s] 67%|██████▋   | 268737/400000 [00:29<00:14, 9162.34it/s] 67%|██████▋   | 269655/400000 [00:30<00:14, 9167.62it/s] 68%|██████▊   | 270573/400000 [00:30<00:14, 9159.88it/s] 68%|██████▊   | 271490/400000 [00:30<00:14, 9099.10it/s] 68%|██████▊   | 272408/400000 [00:30<00:13, 9122.10it/s] 68%|██████▊   | 273321/400000 [00:30<00:14, 9020.29it/s] 69%|██████▊   | 274229/400000 [00:30<00:13, 9037.77it/s] 69%|██████▉   | 275147/400000 [00:30<00:13, 9077.96it/s] 69%|██████▉   | 276056/400000 [00:30<00:13, 8992.86it/s] 69%|██████▉   | 276984/400000 [00:30<00:13, 9076.64it/s] 69%|██████▉   | 277893/400000 [00:30<00:13, 8949.01it/s] 70%|██████▉   | 278799/400000 [00:31<00:13, 8981.31it/s] 70%|██████▉   | 279711/400000 [00:31<00:13, 9020.55it/s] 70%|███████   | 280614/400000 [00:31<00:13, 8934.20it/s] 70%|███████   | 281508/400000 [00:31<00:13, 8897.21it/s] 71%|███████   | 282436/400000 [00:31<00:13, 9005.86it/s] 71%|███████   | 283377/400000 [00:31<00:12, 9120.69it/s] 71%|███████   | 284290/400000 [00:31<00:12, 8978.63it/s] 71%|███████▏  | 285189/400000 [00:31<00:12, 8842.78it/s] 72%|███████▏  | 286086/400000 [00:31<00:12, 8878.36it/s] 72%|███████▏  | 286983/400000 [00:31<00:12, 8904.25it/s] 72%|███████▏  | 287875/400000 [00:32<00:12, 8774.82it/s] 72%|███████▏  | 288777/400000 [00:32<00:12, 8846.09it/s] 72%|███████▏  | 289663/400000 [00:32<00:12, 8833.38it/s] 73%|███████▎  | 290579/400000 [00:32<00:12, 8927.25it/s] 73%|███████▎  | 291497/400000 [00:32<00:12, 9000.47it/s] 73%|███████▎  | 292398/400000 [00:32<00:12, 8960.84it/s] 73%|███████▎  | 293295/400000 [00:32<00:11, 8946.31it/s] 74%|███████▎  | 294190/400000 [00:32<00:11, 8886.11it/s] 74%|███████▍  | 295105/400000 [00:32<00:11, 8961.73it/s] 74%|███████▍  | 296002/400000 [00:32<00:11, 8917.12it/s] 74%|███████▍  | 296915/400000 [00:33<00:11, 8978.31it/s] 74%|███████▍  | 297814/400000 [00:33<00:11, 8940.42it/s] 75%|███████▍  | 298709/400000 [00:33<00:11, 8893.17it/s] 75%|███████▍  | 299611/400000 [00:33<00:11, 8929.34it/s] 75%|███████▌  | 300517/400000 [00:33<00:11, 8967.11it/s] 75%|███████▌  | 301414/400000 [00:33<00:11, 8948.19it/s] 76%|███████▌  | 302339/400000 [00:33<00:10, 9034.32it/s] 76%|███████▌  | 303243/400000 [00:33<00:10, 8826.47it/s] 76%|███████▌  | 304127/400000 [00:33<00:10, 8799.63it/s] 76%|███████▋  | 305008/400000 [00:33<00:10, 8748.76it/s] 76%|███████▋  | 305927/400000 [00:34<00:10, 8876.23it/s] 77%|███████▋  | 306824/400000 [00:34<00:10, 8902.61it/s] 77%|███████▋  | 307723/400000 [00:34<00:10, 8928.28it/s] 77%|███████▋  | 308637/400000 [00:34<00:10, 8989.87it/s] 77%|███████▋  | 309537/400000 [00:34<00:10, 8989.11it/s] 78%|███████▊  | 310437/400000 [00:34<00:09, 8969.10it/s] 78%|███████▊  | 311335/400000 [00:34<00:09, 8928.77it/s] 78%|███████▊  | 312229/400000 [00:34<00:09, 8891.19it/s] 78%|███████▊  | 313122/400000 [00:34<00:09, 8900.77it/s] 79%|███████▊  | 314013/400000 [00:34<00:09, 8635.95it/s] 79%|███████▊  | 314905/400000 [00:35<00:09, 8718.06it/s] 79%|███████▉  | 315802/400000 [00:35<00:09, 8790.54it/s] 79%|███████▉  | 316683/400000 [00:35<00:09, 8753.17it/s] 79%|███████▉  | 317571/400000 [00:35<00:09, 8788.43it/s] 80%|███████▉  | 318453/400000 [00:35<00:09, 8797.58it/s] 80%|███████▉  | 319370/400000 [00:35<00:09, 8904.39it/s] 80%|████████  | 320269/400000 [00:35<00:08, 8927.74it/s] 80%|████████  | 321163/400000 [00:35<00:08, 8865.76it/s] 81%|████████  | 322067/400000 [00:35<00:08, 8916.15it/s] 81%|████████  | 322959/400000 [00:35<00:08, 8902.95it/s] 81%|████████  | 323850/400000 [00:36<00:08, 8789.58it/s] 81%|████████  | 324751/400000 [00:36<00:08, 8852.91it/s] 81%|████████▏ | 325637/400000 [00:36<00:08, 8789.14it/s] 82%|████████▏ | 326532/400000 [00:36<00:08, 8834.16it/s] 82%|████████▏ | 327420/400000 [00:36<00:08, 8845.77it/s] 82%|████████▏ | 328321/400000 [00:36<00:08, 8894.13it/s] 82%|████████▏ | 329222/400000 [00:36<00:07, 8926.98it/s] 83%|████████▎ | 330117/400000 [00:36<00:07, 8933.78it/s] 83%|████████▎ | 331036/400000 [00:36<00:07, 9007.59it/s] 83%|████████▎ | 331961/400000 [00:37<00:07, 9076.76it/s] 83%|████████▎ | 332882/400000 [00:37<00:07, 9115.79it/s] 83%|████████▎ | 333819/400000 [00:37<00:07, 9188.37it/s] 84%|████████▎ | 334739/400000 [00:37<00:07, 9118.12it/s] 84%|████████▍ | 335652/400000 [00:37<00:07, 9092.29it/s] 84%|████████▍ | 336562/400000 [00:37<00:06, 9088.47it/s] 84%|████████▍ | 337477/400000 [00:37<00:06, 9106.42it/s] 85%|████████▍ | 338388/400000 [00:37<00:06, 9045.66it/s] 85%|████████▍ | 339293/400000 [00:37<00:06, 8963.49it/s] 85%|████████▌ | 340194/400000 [00:37<00:06, 8975.77it/s] 85%|████████▌ | 341112/400000 [00:38<00:06, 9034.10it/s] 86%|████████▌ | 342027/400000 [00:38<00:06, 9066.89it/s] 86%|████████▌ | 342941/400000 [00:38<00:06, 9088.45it/s] 86%|████████▌ | 343850/400000 [00:38<00:06, 9068.45it/s] 86%|████████▌ | 344757/400000 [00:38<00:06, 9030.97it/s] 86%|████████▋ | 345661/400000 [00:38<00:06, 8966.03it/s] 87%|████████▋ | 346581/400000 [00:38<00:05, 9032.84it/s] 87%|████████▋ | 347494/400000 [00:38<00:05, 9059.74it/s] 87%|████████▋ | 348401/400000 [00:38<00:05, 8955.86it/s] 87%|████████▋ | 349311/400000 [00:38<00:05, 8998.37it/s] 88%|████████▊ | 350242/400000 [00:39<00:05, 9088.80it/s] 88%|████████▊ | 351165/400000 [00:39<00:05, 9128.32it/s] 88%|████████▊ | 352079/400000 [00:39<00:05, 9117.23it/s] 88%|████████▊ | 352991/400000 [00:39<00:05, 8966.12it/s] 88%|████████▊ | 353889/400000 [00:39<00:05, 8828.29it/s] 89%|████████▊ | 354799/400000 [00:39<00:05, 8906.40it/s] 89%|████████▉ | 355713/400000 [00:39<00:04, 8974.51it/s] 89%|████████▉ | 356614/400000 [00:39<00:04, 8979.51it/s] 89%|████████▉ | 357521/400000 [00:39<00:04, 9004.93it/s] 90%|████████▉ | 358422/400000 [00:39<00:04, 8936.95it/s] 90%|████████▉ | 359326/400000 [00:40<00:04, 8965.02it/s] 90%|█████████ | 360223/400000 [00:40<00:04, 8960.63it/s] 90%|█████████ | 361120/400000 [00:40<00:04, 8918.88it/s] 91%|█████████ | 362027/400000 [00:40<00:04, 8963.33it/s] 91%|█████████ | 362924/400000 [00:40<00:04, 8810.23it/s] 91%|█████████ | 363840/400000 [00:40<00:04, 8909.75it/s] 91%|█████████ | 364759/400000 [00:40<00:03, 8990.89it/s] 91%|█████████▏| 365659/400000 [00:40<00:03, 8880.54it/s] 92%|█████████▏| 366567/400000 [00:40<00:03, 8937.40it/s] 92%|█████████▏| 367468/400000 [00:40<00:03, 8957.55it/s] 92%|█████████▏| 368372/400000 [00:41<00:03, 8980.33it/s] 92%|█████████▏| 369271/400000 [00:41<00:03, 8966.87it/s] 93%|█████████▎| 370207/400000 [00:41<00:03, 9080.02it/s] 93%|█████████▎| 371136/400000 [00:41<00:03, 9141.85it/s] 93%|█████████▎| 372051/400000 [00:41<00:03, 9116.79it/s] 93%|█████████▎| 372979/400000 [00:41<00:02, 9162.77it/s] 93%|█████████▎| 373896/400000 [00:41<00:02, 9145.67it/s] 94%|█████████▎| 374812/400000 [00:41<00:02, 9147.38it/s] 94%|█████████▍| 375739/400000 [00:41<00:02, 9181.25it/s] 94%|█████████▍| 376658/400000 [00:41<00:02, 9109.20it/s] 94%|█████████▍| 377571/400000 [00:42<00:02, 9114.60it/s] 95%|█████████▍| 378483/400000 [00:42<00:02, 8706.15it/s] 95%|█████████▍| 379398/400000 [00:42<00:02, 8833.41it/s] 95%|█████████▌| 380321/400000 [00:42<00:02, 8946.87it/s] 95%|█████████▌| 381219/400000 [00:42<00:02, 8953.84it/s] 96%|█████████▌| 382117/400000 [00:42<00:01, 8944.32it/s] 96%|█████████▌| 383024/400000 [00:42<00:01, 8981.28it/s] 96%|█████████▌| 383929/400000 [00:42<00:01, 9000.28it/s] 96%|█████████▌| 384830/400000 [00:42<00:01, 8975.68it/s] 96%|█████████▋| 385729/400000 [00:42<00:01, 8887.46it/s] 97%|█████████▋| 386619/400000 [00:43<00:01, 8860.15it/s] 97%|█████████▋| 387522/400000 [00:43<00:01, 8909.90it/s] 97%|█████████▋| 388447/400000 [00:43<00:01, 9008.70it/s] 97%|█████████▋| 389349/400000 [00:43<00:01, 8928.66it/s] 98%|█████████▊| 390260/400000 [00:43<00:01, 8981.99it/s] 98%|█████████▊| 391188/400000 [00:43<00:00, 9069.13it/s] 98%|█████████▊| 392108/400000 [00:43<00:00, 9106.87it/s] 98%|█████████▊| 393024/400000 [00:43<00:00, 9121.12it/s] 98%|█████████▊| 393937/400000 [00:43<00:00, 9061.00it/s] 99%|█████████▊| 394844/400000 [00:43<00:00, 9031.23it/s] 99%|█████████▉| 395748/400000 [00:44<00:00, 9007.15it/s] 99%|█████████▉| 396649/400000 [00:44<00:00, 8827.38it/s] 99%|█████████▉| 397535/400000 [00:44<00:00, 8836.73it/s]100%|█████████▉| 398422/400000 [00:44<00:00, 8846.33it/s]100%|█████████▉| 399310/400000 [00:44<00:00, 8855.55it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8974.58it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fa66f73e940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011126586381287364 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011031937838398094 	 Accuracy: 53

  model saves at 53% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15859 out of table with 15825 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15859 out of table with 15825 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-15 00:29:31.030230: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 00:29:31.033962: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-15 00:29:31.034147: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c60d883d00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 00:29:31.034161: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fa67d9a6198> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.5746 - accuracy: 0.5060
 2000/25000 [=>............................] - ETA: 7s - loss: 7.4213 - accuracy: 0.5160 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5440 - accuracy: 0.5080
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.4903 - accuracy: 0.5115
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5317 - accuracy: 0.5088
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6078 - accuracy: 0.5038
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5549 - accuracy: 0.5073
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5785 - accuracy: 0.5058
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6206 - accuracy: 0.5030
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6697 - accuracy: 0.4998
11000/25000 [============>.................] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
12000/25000 [=============>................] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6820 - accuracy: 0.4990
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6918 - accuracy: 0.4984
15000/25000 [=================>............] - ETA: 2s - loss: 7.6973 - accuracy: 0.4980
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6915 - accuracy: 0.4984
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6964 - accuracy: 0.4981
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7135 - accuracy: 0.4969
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7175 - accuracy: 0.4967
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7011 - accuracy: 0.4978
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6914 - accuracy: 0.4984
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6875 - accuracy: 0.4986
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6773 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6653 - accuracy: 0.5001
25000/25000 [==============================] - 7s 283us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fa5cbdf9860> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fa5ec9f7128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.2573 - crf_viterbi_accuracy: 0.2667 - val_loss: 1.1783 - val_crf_viterbi_accuracy: 0.3200

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

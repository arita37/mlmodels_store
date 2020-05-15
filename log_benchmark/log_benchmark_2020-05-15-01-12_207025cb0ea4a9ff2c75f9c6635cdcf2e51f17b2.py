
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fa12b9dff28> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 01:12:16.441135
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 01:12:16.444893
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 01:12:16.448474
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 01:12:16.452638
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fa1377a9438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352403.8125
Epoch 2/10

1/1 [==============================] - 0s 97ms/step - loss: 264531.8750
Epoch 3/10

1/1 [==============================] - 0s 92ms/step - loss: 161705.5156
Epoch 4/10

1/1 [==============================] - 0s 91ms/step - loss: 85070.8125
Epoch 5/10

1/1 [==============================] - 0s 90ms/step - loss: 43073.5312
Epoch 6/10

1/1 [==============================] - 0s 87ms/step - loss: 22821.8086
Epoch 7/10

1/1 [==============================] - 0s 91ms/step - loss: 13259.3086
Epoch 8/10

1/1 [==============================] - 0s 91ms/step - loss: 8549.7012
Epoch 9/10

1/1 [==============================] - 0s 89ms/step - loss: 5969.0737
Epoch 10/10

1/1 [==============================] - 0s 92ms/step - loss: 4487.7363

  #### Inference Need return ypred, ytrue ######################### 
[[ 3.82002771e-01  1.39269233e-01 -2.00502658e+00  1.35662889e+00
   3.87685955e-01  1.25293183e+00 -1.40523112e+00 -2.12374663e+00
  -3.04382992e+00 -8.82261813e-01  4.66780424e-01 -9.22522306e-01
   6.58053607e-02 -6.96626306e-03 -2.94742703e-01  2.97570288e-01
   8.25316012e-01 -4.72821683e-01  9.43013191e-01  5.42083740e-01
   1.18633378e+00 -2.55503953e-02  1.04430318e+00  1.78074157e+00
   1.45015121e+00 -2.35116690e-01  1.72924256e+00  1.69434357e+00
  -1.28624439e-01  1.22320771e+00 -1.51058301e-01 -2.20785841e-01
   1.43632305e+00 -9.29215252e-01 -9.57216620e-01 -2.87072092e-01
  -7.57583141e-01  1.12941241e+00 -2.05527115e+00  6.23440742e-01
   7.88356066e-01  4.34106022e-01 -9.91777062e-01  1.41253507e+00
  -5.07830739e-01 -5.95906675e-01  5.60360312e-01  4.82703567e-01
   7.93062806e-01  2.15773821e+00  1.18197665e-01 -9.02615488e-02
   1.90152776e+00  9.68527079e-01 -3.18350524e-01  1.42719316e+00
   3.46217155e-01  2.04030824e+00  4.28080857e-01 -1.24100757e+00
  -4.68212515e-01  9.14069557e+00  8.57701302e+00  1.11836185e+01
   1.14183168e+01  9.16416168e+00  7.44140053e+00  9.55540466e+00
   1.18929234e+01  1.09212370e+01  1.05941353e+01  1.16775637e+01
   1.18226767e+01  1.15962687e+01  1.00584240e+01  1.11599722e+01
   1.02687626e+01  9.43621445e+00  8.18247223e+00  1.06332779e+01
   1.12185535e+01  1.04025517e+01  9.78068161e+00  9.71331310e+00
   8.82327652e+00  8.15787411e+00  1.08338842e+01  1.09259510e+01
   9.17307854e+00  8.59978962e+00  9.66778469e+00  1.13563423e+01
   9.44990063e+00  9.59774685e+00  1.10481873e+01  9.51015759e+00
   1.13091440e+01  9.57520866e+00  8.68108749e+00  1.20327921e+01
   1.06474409e+01  1.07496510e+01  9.93859100e+00  1.19880981e+01
   9.41524601e+00  1.04232464e+01  1.29981985e+01  1.23176003e+01
   9.76545334e+00  9.44129753e+00  1.00682030e+01  9.23141670e+00
   8.78164196e+00  1.20092525e+01  9.54765129e+00  8.49973583e+00
   1.16532946e+01  1.11434259e+01  1.17449141e+01  1.09233627e+01
   4.75830436e-01  1.13479614e-01  4.15977836e-01  1.66691256e+00
  -5.31864047e-01  1.10968971e+00 -7.97997117e-01  9.55767930e-01
   7.37504601e-01  2.03476334e+00  1.02288973e+00 -7.31415987e-01
  -5.29948473e-01 -8.56977224e-01  6.07749879e-01  1.36914060e-01
   7.92744756e-03  1.09300637e+00  1.00859785e+00  5.73665142e-01
   2.17464423e+00 -8.26214135e-01  4.33373749e-01 -1.41352284e+00
   1.66004467e+00 -1.16841638e+00  2.56735146e-01 -6.15603924e-01
  -1.09908187e+00 -9.04375732e-01  2.37196779e+00  1.63464260e+00
   5.38483441e-01 -1.56856167e+00 -2.07956314e-01 -1.27574670e+00
  -1.87327993e+00  2.65471220e-01  3.16765606e-02 -1.76130104e+00
  -1.67007637e+00  6.39421642e-01  3.64363790e-01 -6.65771663e-01
  -2.34901905e-03 -2.36069411e-01  5.11616826e-01  1.29584503e+00
   1.61552155e+00 -2.52897501e-01 -1.64917976e-01  1.86367798e+00
  -5.81363201e-01  1.65100193e+00  4.53235924e-01 -1.81575489e+00
   1.09158909e+00  7.64386535e-01  8.22414219e-01 -1.08153629e+00
   6.47684276e-01  2.73862839e+00  2.80492544e+00  2.89813280e+00
   2.71146297e-01  1.33759499e+00  7.14709878e-01  8.19182932e-01
   1.47554207e+00  6.64938390e-01  7.41748333e-01  1.84519887e+00
   7.90718496e-01  1.69868851e+00  1.83798456e+00  6.33294642e-01
   5.21833479e-01  2.20591998e+00  1.36610270e-01  8.33095431e-01
   6.01269186e-01  3.61626863e+00  2.36577344e+00  9.33824778e-02
   1.10193634e+00  2.90731728e-01  1.32616925e+00  6.49639070e-01
   7.92558789e-01  1.95605493e+00  1.17388678e+00  1.23838806e+00
   2.03925323e+00  3.54896545e+00  1.00809455e+00  3.67512763e-01
   2.50437617e-01  3.22375298e-01  1.21100402e+00  3.61177504e-01
   2.46867180e-01  4.71665502e-01  1.26428270e+00  9.51764226e-01
   3.54384601e-01  8.01455259e-01  4.45463836e-01  2.96288633e+00
   8.53435993e-02  3.65044236e-01  1.38784468e-01  2.19028854e+00
   8.27059150e-01  4.20588350e+00  2.87056208e-01  2.34912586e+00
   4.84171391e-01  1.69124484e-01  5.32921076e-01  1.52436614e-01
   2.96401262e-01  9.75281906e+00  1.15704489e+01  1.02520561e+01
   1.20200539e+01  9.14601994e+00  1.02330103e+01  1.14554148e+01
   1.00750294e+01  1.02055731e+01  1.09127111e+01  1.02359276e+01
   1.17020788e+01  1.15893774e+01  1.07184820e+01  9.97741318e+00
   1.07053041e+01  1.04205732e+01  1.00644417e+01  1.02579937e+01
   1.16605349e+01  1.13093576e+01  1.24164143e+01  9.61780357e+00
   1.21053753e+01  8.72669029e+00  1.02380133e+01  9.95209694e+00
   1.14143324e+01  1.17059631e+01  1.13980961e+01  1.01395273e+01
   7.87673998e+00  8.01620674e+00  1.07494669e+01  1.05450134e+01
   9.43709564e+00  1.05335732e+01  1.15866032e+01  8.99790668e+00
   8.93007851e+00  1.10307789e+01  9.23679256e+00  1.09570637e+01
   1.02109480e+01  1.19084682e+01  1.16389141e+01  9.55095100e+00
   1.29974241e+01  8.87162685e+00  1.04915905e+01  1.15222244e+01
   1.15942526e+01  1.13957348e+01  1.07179890e+01  1.02579060e+01
   1.07235975e+01  8.72474766e+00  1.02672443e+01  1.06374426e+01
   1.68941987e+00  5.29528260e-01  2.87834883e+00  1.81481981e+00
   3.03794265e-01  8.12781215e-01  2.60964656e+00  4.58780289e-01
   2.11795855e+00  7.78430462e-01  1.00510621e+00  1.16563106e+00
   1.21164322e+00  4.28010285e-01  1.37450182e+00  1.60515976e+00
   1.17022645e+00  8.86843324e-01  1.37714720e+00  2.29082370e+00
   2.73183918e+00  1.79983509e+00  7.28743732e-01  9.91207361e-02
   9.03803349e-01  2.28727102e-01  1.42523170e-01  9.40603018e-01
   1.30434299e+00  1.04413581e+00  5.14609218e-01  1.19334638e-01
   1.63317537e+00  3.28639507e-01  1.22915697e+00  3.61916900e-01
   4.00862873e-01  6.02076352e-01  2.73329139e-01  1.49659324e+00
   1.48546922e+00  1.77313495e+00  4.14568663e-01  2.09806776e+00
   2.28509378e+00  1.38486934e+00  2.16630793e+00  5.03849030e-01
   5.96169889e-01  1.66648388e+00  7.70204186e-01  1.44899166e+00
   1.44973397e-01  1.39754677e+00  1.84895945e+00  5.55597186e-01
   1.34786391e+00  7.60113597e-02  2.62383604e+00  3.15237820e-01
  -4.16029835e+00  1.03633232e+01 -8.75183582e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 01:12:26.368231
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.0062
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 01:12:26.372001
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8492.92
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 01:12:26.374992
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.0623
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 01:12:26.377790
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -759.601
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140329839092008
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140328629355016
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140328629355520
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140328629356024
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140328629356528
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140328629357032

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fa1173bae10> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.629896
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.596256
grad_step = 000002, loss = 0.566655
grad_step = 000003, loss = 0.535937
grad_step = 000004, loss = 0.502178
grad_step = 000005, loss = 0.468501
grad_step = 000006, loss = 0.445856
grad_step = 000007, loss = 0.437019
grad_step = 000008, loss = 0.419603
grad_step = 000009, loss = 0.403315
grad_step = 000010, loss = 0.381055
grad_step = 000011, loss = 0.359597
grad_step = 000012, loss = 0.342505
grad_step = 000013, loss = 0.329134
grad_step = 000014, loss = 0.317284
grad_step = 000015, loss = 0.307048
grad_step = 000016, loss = 0.299023
grad_step = 000017, loss = 0.290857
grad_step = 000018, loss = 0.281060
grad_step = 000019, loss = 0.269863
grad_step = 000020, loss = 0.258990
grad_step = 000021, loss = 0.249261
grad_step = 000022, loss = 0.239611
grad_step = 000023, loss = 0.229355
grad_step = 000024, loss = 0.219286
grad_step = 000025, loss = 0.210361
grad_step = 000026, loss = 0.202507
grad_step = 000027, loss = 0.194688
grad_step = 000028, loss = 0.186120
grad_step = 000029, loss = 0.177451
grad_step = 000030, loss = 0.169757
grad_step = 000031, loss = 0.162642
grad_step = 000032, loss = 0.155480
grad_step = 000033, loss = 0.148153
grad_step = 000034, loss = 0.140875
grad_step = 000035, loss = 0.134114
grad_step = 000036, loss = 0.127627
grad_step = 000037, loss = 0.121113
grad_step = 000038, loss = 0.114903
grad_step = 000039, loss = 0.109195
grad_step = 000040, loss = 0.103824
grad_step = 000041, loss = 0.098334
grad_step = 000042, loss = 0.092882
grad_step = 000043, loss = 0.087749
grad_step = 000044, loss = 0.082937
grad_step = 000045, loss = 0.078259
grad_step = 000046, loss = 0.073757
grad_step = 000047, loss = 0.069564
grad_step = 000048, loss = 0.065506
grad_step = 000049, loss = 0.061532
grad_step = 000050, loss = 0.057756
grad_step = 000051, loss = 0.054310
grad_step = 000052, loss = 0.050992
grad_step = 000053, loss = 0.047750
grad_step = 000054, loss = 0.044685
grad_step = 000055, loss = 0.041815
grad_step = 000056, loss = 0.039043
grad_step = 000057, loss = 0.036459
grad_step = 000058, loss = 0.034096
grad_step = 000059, loss = 0.031843
grad_step = 000060, loss = 0.029659
grad_step = 000061, loss = 0.027635
grad_step = 000062, loss = 0.025763
grad_step = 000063, loss = 0.023968
grad_step = 000064, loss = 0.022291
grad_step = 000065, loss = 0.020730
grad_step = 000066, loss = 0.019246
grad_step = 000067, loss = 0.017868
grad_step = 000068, loss = 0.016604
grad_step = 000069, loss = 0.015413
grad_step = 000070, loss = 0.014294
grad_step = 000071, loss = 0.013262
grad_step = 000072, loss = 0.012292
grad_step = 000073, loss = 0.011394
grad_step = 000074, loss = 0.010572
grad_step = 000075, loss = 0.009795
grad_step = 000076, loss = 0.009073
grad_step = 000077, loss = 0.008419
grad_step = 000078, loss = 0.007807
grad_step = 000079, loss = 0.007245
grad_step = 000080, loss = 0.006733
grad_step = 000081, loss = 0.006258
grad_step = 000082, loss = 0.005826
grad_step = 000083, loss = 0.005434
grad_step = 000084, loss = 0.005070
grad_step = 000085, loss = 0.004740
grad_step = 000086, loss = 0.004443
grad_step = 000087, loss = 0.004176
grad_step = 000088, loss = 0.003942
grad_step = 000089, loss = 0.003735
grad_step = 000090, loss = 0.003551
grad_step = 000091, loss = 0.003374
grad_step = 000092, loss = 0.003194
grad_step = 000093, loss = 0.003033
grad_step = 000094, loss = 0.002906
grad_step = 000095, loss = 0.002807
grad_step = 000096, loss = 0.002724
grad_step = 000097, loss = 0.002640
grad_step = 000098, loss = 0.002551
grad_step = 000099, loss = 0.002466
grad_step = 000100, loss = 0.002395
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002343
grad_step = 000102, loss = 0.002304
grad_step = 000103, loss = 0.002275
grad_step = 000104, loss = 0.002255
grad_step = 000105, loss = 0.002238
grad_step = 000106, loss = 0.002221
grad_step = 000107, loss = 0.002191
grad_step = 000108, loss = 0.002153
grad_step = 000109, loss = 0.002112
grad_step = 000110, loss = 0.002082
grad_step = 000111, loss = 0.002067
grad_step = 000112, loss = 0.002064
grad_step = 000113, loss = 0.002068
grad_step = 000114, loss = 0.002076
grad_step = 000115, loss = 0.002088
grad_step = 000116, loss = 0.002089
grad_step = 000117, loss = 0.002079
grad_step = 000118, loss = 0.002047
grad_step = 000119, loss = 0.002012
grad_step = 000120, loss = 0.001987
grad_step = 000121, loss = 0.001982
grad_step = 000122, loss = 0.001990
grad_step = 000123, loss = 0.002003
grad_step = 000124, loss = 0.002015
grad_step = 000125, loss = 0.002014
grad_step = 000126, loss = 0.002005
grad_step = 000127, loss = 0.001982
grad_step = 000128, loss = 0.001959
grad_step = 000129, loss = 0.001940
grad_step = 000130, loss = 0.001932
grad_step = 000131, loss = 0.001932
grad_step = 000132, loss = 0.001939
grad_step = 000133, loss = 0.001949
grad_step = 000134, loss = 0.001961
grad_step = 000135, loss = 0.001975
grad_step = 000136, loss = 0.001981
grad_step = 000137, loss = 0.001980
grad_step = 000138, loss = 0.001957
grad_step = 000139, loss = 0.001927
grad_step = 000140, loss = 0.001901
grad_step = 000141, loss = 0.001889
grad_step = 000142, loss = 0.001894
grad_step = 000143, loss = 0.001906
grad_step = 000144, loss = 0.001918
grad_step = 000145, loss = 0.001921
grad_step = 000146, loss = 0.001918
grad_step = 000147, loss = 0.001905
grad_step = 000148, loss = 0.001891
grad_step = 000149, loss = 0.001877
grad_step = 000150, loss = 0.001867
grad_step = 000151, loss = 0.001861
grad_step = 000152, loss = 0.001859
grad_step = 000153, loss = 0.001860
grad_step = 000154, loss = 0.001864
grad_step = 000155, loss = 0.001870
grad_step = 000156, loss = 0.001879
grad_step = 000157, loss = 0.001896
grad_step = 000158, loss = 0.001918
grad_step = 000159, loss = 0.001953
grad_step = 000160, loss = 0.001976
grad_step = 000161, loss = 0.001982
grad_step = 000162, loss = 0.001936
grad_step = 000163, loss = 0.001873
grad_step = 000164, loss = 0.001834
grad_step = 000165, loss = 0.001844
grad_step = 000166, loss = 0.001879
grad_step = 000167, loss = 0.001894
grad_step = 000168, loss = 0.001873
grad_step = 000169, loss = 0.001836
grad_step = 000170, loss = 0.001819
grad_step = 000171, loss = 0.001829
grad_step = 000172, loss = 0.001850
grad_step = 000173, loss = 0.001863
grad_step = 000174, loss = 0.001855
grad_step = 000175, loss = 0.001835
grad_step = 000176, loss = 0.001814
grad_step = 000177, loss = 0.001804
grad_step = 000178, loss = 0.001808
grad_step = 000179, loss = 0.001817
grad_step = 000180, loss = 0.001825
grad_step = 000181, loss = 0.001825
grad_step = 000182, loss = 0.001818
grad_step = 000183, loss = 0.001806
grad_step = 000184, loss = 0.001795
grad_step = 000185, loss = 0.001788
grad_step = 000186, loss = 0.001786
grad_step = 000187, loss = 0.001788
grad_step = 000188, loss = 0.001791
grad_step = 000189, loss = 0.001795
grad_step = 000190, loss = 0.001796
grad_step = 000191, loss = 0.001796
grad_step = 000192, loss = 0.001795
grad_step = 000193, loss = 0.001792
grad_step = 000194, loss = 0.001788
grad_step = 000195, loss = 0.001785
grad_step = 000196, loss = 0.001781
grad_step = 000197, loss = 0.001778
grad_step = 000198, loss = 0.001775
grad_step = 000199, loss = 0.001773
grad_step = 000200, loss = 0.001772
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001772
grad_step = 000202, loss = 0.001773
grad_step = 000203, loss = 0.001778
grad_step = 000204, loss = 0.001786
grad_step = 000205, loss = 0.001801
grad_step = 000206, loss = 0.001824
grad_step = 000207, loss = 0.001854
grad_step = 000208, loss = 0.001881
grad_step = 000209, loss = 0.001889
grad_step = 000210, loss = 0.001859
grad_step = 000211, loss = 0.001800
grad_step = 000212, loss = 0.001746
grad_step = 000213, loss = 0.001731
grad_step = 000214, loss = 0.001754
grad_step = 000215, loss = 0.001786
grad_step = 000216, loss = 0.001795
grad_step = 000217, loss = 0.001775
grad_step = 000218, loss = 0.001741
grad_step = 000219, loss = 0.001718
grad_step = 000220, loss = 0.001716
grad_step = 000221, loss = 0.001730
grad_step = 000222, loss = 0.001746
grad_step = 000223, loss = 0.001752
grad_step = 000224, loss = 0.001744
grad_step = 000225, loss = 0.001728
grad_step = 000226, loss = 0.001710
grad_step = 000227, loss = 0.001698
grad_step = 000228, loss = 0.001694
grad_step = 000229, loss = 0.001697
grad_step = 000230, loss = 0.001703
grad_step = 000231, loss = 0.001709
grad_step = 000232, loss = 0.001713
grad_step = 000233, loss = 0.001713
grad_step = 000234, loss = 0.001707
grad_step = 000235, loss = 0.001696
grad_step = 000236, loss = 0.001685
grad_step = 000237, loss = 0.001676
grad_step = 000238, loss = 0.001669
grad_step = 000239, loss = 0.001665
grad_step = 000240, loss = 0.001663
grad_step = 000241, loss = 0.001665
grad_step = 000242, loss = 0.001669
grad_step = 000243, loss = 0.001677
grad_step = 000244, loss = 0.001695
grad_step = 000245, loss = 0.001723
grad_step = 000246, loss = 0.001764
grad_step = 000247, loss = 0.001810
grad_step = 000248, loss = 0.001835
grad_step = 000249, loss = 0.001807
grad_step = 000250, loss = 0.001727
grad_step = 000251, loss = 0.001652
grad_step = 000252, loss = 0.001638
grad_step = 000253, loss = 0.001678
grad_step = 000254, loss = 0.001723
grad_step = 000255, loss = 0.001729
grad_step = 000256, loss = 0.001686
grad_step = 000257, loss = 0.001638
grad_step = 000258, loss = 0.001623
grad_step = 000259, loss = 0.001644
grad_step = 000260, loss = 0.001670
grad_step = 000261, loss = 0.001674
grad_step = 000262, loss = 0.001651
grad_step = 000263, loss = 0.001623
grad_step = 000264, loss = 0.001611
grad_step = 000265, loss = 0.001619
grad_step = 000266, loss = 0.001633
grad_step = 000267, loss = 0.001640
grad_step = 000268, loss = 0.001633
grad_step = 000269, loss = 0.001617
grad_step = 000270, loss = 0.001604
grad_step = 000271, loss = 0.001601
grad_step = 000272, loss = 0.001606
grad_step = 000273, loss = 0.001612
grad_step = 000274, loss = 0.001615
grad_step = 000275, loss = 0.001610
grad_step = 000276, loss = 0.001602
grad_step = 000277, loss = 0.001595
grad_step = 000278, loss = 0.001591
grad_step = 000279, loss = 0.001591
grad_step = 000280, loss = 0.001593
grad_step = 000281, loss = 0.001595
grad_step = 000282, loss = 0.001595
grad_step = 000283, loss = 0.001593
grad_step = 000284, loss = 0.001590
grad_step = 000285, loss = 0.001586
grad_step = 000286, loss = 0.001582
grad_step = 000287, loss = 0.001579
grad_step = 000288, loss = 0.001577
grad_step = 000289, loss = 0.001576
grad_step = 000290, loss = 0.001576
grad_step = 000291, loss = 0.001576
grad_step = 000292, loss = 0.001576
grad_step = 000293, loss = 0.001576
grad_step = 000294, loss = 0.001575
grad_step = 000295, loss = 0.001575
grad_step = 000296, loss = 0.001574
grad_step = 000297, loss = 0.001573
grad_step = 000298, loss = 0.001572
grad_step = 000299, loss = 0.001570
grad_step = 000300, loss = 0.001568
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001565
grad_step = 000302, loss = 0.001563
grad_step = 000303, loss = 0.001561
grad_step = 000304, loss = 0.001559
grad_step = 000305, loss = 0.001558
grad_step = 000306, loss = 0.001556
grad_step = 000307, loss = 0.001554
grad_step = 000308, loss = 0.001553
grad_step = 000309, loss = 0.001552
grad_step = 000310, loss = 0.001551
grad_step = 000311, loss = 0.001551
grad_step = 000312, loss = 0.001552
grad_step = 000313, loss = 0.001556
grad_step = 000314, loss = 0.001564
grad_step = 000315, loss = 0.001576
grad_step = 000316, loss = 0.001601
grad_step = 000317, loss = 0.001623
grad_step = 000318, loss = 0.001657
grad_step = 000319, loss = 0.001664
grad_step = 000320, loss = 0.001677
grad_step = 000321, loss = 0.001681
grad_step = 000322, loss = 0.001670
grad_step = 000323, loss = 0.001654
grad_step = 000324, loss = 0.001578
grad_step = 000325, loss = 0.001530
grad_step = 000326, loss = 0.001534
grad_step = 000327, loss = 0.001558
grad_step = 000328, loss = 0.001571
grad_step = 000329, loss = 0.001560
grad_step = 000330, loss = 0.001554
grad_step = 000331, loss = 0.001555
grad_step = 000332, loss = 0.001535
grad_step = 000333, loss = 0.001514
grad_step = 000334, loss = 0.001512
grad_step = 000335, loss = 0.001526
grad_step = 000336, loss = 0.001536
grad_step = 000337, loss = 0.001526
grad_step = 000338, loss = 0.001511
grad_step = 000339, loss = 0.001504
grad_step = 000340, loss = 0.001507
grad_step = 000341, loss = 0.001509
grad_step = 000342, loss = 0.001502
grad_step = 000343, loss = 0.001495
grad_step = 000344, loss = 0.001494
grad_step = 000345, loss = 0.001497
grad_step = 000346, loss = 0.001497
grad_step = 000347, loss = 0.001491
grad_step = 000348, loss = 0.001483
grad_step = 000349, loss = 0.001479
grad_step = 000350, loss = 0.001479
grad_step = 000351, loss = 0.001480
grad_step = 000352, loss = 0.001478
grad_step = 000353, loss = 0.001474
grad_step = 000354, loss = 0.001469
grad_step = 000355, loss = 0.001467
grad_step = 000356, loss = 0.001466
grad_step = 000357, loss = 0.001466
grad_step = 000358, loss = 0.001465
grad_step = 000359, loss = 0.001463
grad_step = 000360, loss = 0.001459
grad_step = 000361, loss = 0.001457
grad_step = 000362, loss = 0.001455
grad_step = 000363, loss = 0.001455
grad_step = 000364, loss = 0.001457
grad_step = 000365, loss = 0.001461
grad_step = 000366, loss = 0.001469
grad_step = 000367, loss = 0.001483
grad_step = 000368, loss = 0.001512
grad_step = 000369, loss = 0.001551
grad_step = 000370, loss = 0.001623
grad_step = 000371, loss = 0.001651
grad_step = 000372, loss = 0.001676
grad_step = 000373, loss = 0.001566
grad_step = 000374, loss = 0.001484
grad_step = 000375, loss = 0.001482
grad_step = 000376, loss = 0.001495
grad_step = 000377, loss = 0.001476
grad_step = 000378, loss = 0.001428
grad_step = 000379, loss = 0.001445
grad_step = 000380, loss = 0.001488
grad_step = 000381, loss = 0.001459
grad_step = 000382, loss = 0.001417
grad_step = 000383, loss = 0.001414
grad_step = 000384, loss = 0.001437
grad_step = 000385, loss = 0.001439
grad_step = 000386, loss = 0.001410
grad_step = 000387, loss = 0.001395
grad_step = 000388, loss = 0.001409
grad_step = 000389, loss = 0.001420
grad_step = 000390, loss = 0.001412
grad_step = 000391, loss = 0.001394
grad_step = 000392, loss = 0.001387
grad_step = 000393, loss = 0.001393
grad_step = 000394, loss = 0.001399
grad_step = 000395, loss = 0.001396
grad_step = 000396, loss = 0.001385
grad_step = 000397, loss = 0.001376
grad_step = 000398, loss = 0.001372
grad_step = 000399, loss = 0.001375
grad_step = 000400, loss = 0.001380
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001381
grad_step = 000402, loss = 0.001378
grad_step = 000403, loss = 0.001371
grad_step = 000404, loss = 0.001363
grad_step = 000405, loss = 0.001356
grad_step = 000406, loss = 0.001353
grad_step = 000407, loss = 0.001352
grad_step = 000408, loss = 0.001353
grad_step = 000409, loss = 0.001355
grad_step = 000410, loss = 0.001356
grad_step = 000411, loss = 0.001358
grad_step = 000412, loss = 0.001359
grad_step = 000413, loss = 0.001366
grad_step = 000414, loss = 0.001380
grad_step = 000415, loss = 0.001417
grad_step = 000416, loss = 0.001458
grad_step = 000417, loss = 0.001537
grad_step = 000418, loss = 0.001564
grad_step = 000419, loss = 0.001597
grad_step = 000420, loss = 0.001555
grad_step = 000421, loss = 0.001499
grad_step = 000422, loss = 0.001463
grad_step = 000423, loss = 0.001388
grad_step = 000424, loss = 0.001347
grad_step = 000425, loss = 0.001362
grad_step = 000426, loss = 0.001408
grad_step = 000427, loss = 0.001423
grad_step = 000428, loss = 0.001363
grad_step = 000429, loss = 0.001319
grad_step = 000430, loss = 0.001332
grad_step = 000431, loss = 0.001367
grad_step = 000432, loss = 0.001369
grad_step = 000433, loss = 0.001331
grad_step = 000434, loss = 0.001311
grad_step = 000435, loss = 0.001325
grad_step = 000436, loss = 0.001341
grad_step = 000437, loss = 0.001339
grad_step = 000438, loss = 0.001314
grad_step = 000439, loss = 0.001299
grad_step = 000440, loss = 0.001305
grad_step = 000441, loss = 0.001317
grad_step = 000442, loss = 0.001322
grad_step = 000443, loss = 0.001311
grad_step = 000444, loss = 0.001300
grad_step = 000445, loss = 0.001295
grad_step = 000446, loss = 0.001299
grad_step = 000447, loss = 0.001302
grad_step = 000448, loss = 0.001298
grad_step = 000449, loss = 0.001291
grad_step = 000450, loss = 0.001285
grad_step = 000451, loss = 0.001284
grad_step = 000452, loss = 0.001287
grad_step = 000453, loss = 0.001289
grad_step = 000454, loss = 0.001288
grad_step = 000455, loss = 0.001284
grad_step = 000456, loss = 0.001280
grad_step = 000457, loss = 0.001278
grad_step = 000458, loss = 0.001280
grad_step = 000459, loss = 0.001283
grad_step = 000460, loss = 0.001286
grad_step = 000461, loss = 0.001288
grad_step = 000462, loss = 0.001290
grad_step = 000463, loss = 0.001297
grad_step = 000464, loss = 0.001304
grad_step = 000465, loss = 0.001316
grad_step = 000466, loss = 0.001321
grad_step = 000467, loss = 0.001325
grad_step = 000468, loss = 0.001313
grad_step = 000469, loss = 0.001296
grad_step = 000470, loss = 0.001274
grad_step = 000471, loss = 0.001259
grad_step = 000472, loss = 0.001255
grad_step = 000473, loss = 0.001261
grad_step = 000474, loss = 0.001270
grad_step = 000475, loss = 0.001273
grad_step = 000476, loss = 0.001270
grad_step = 000477, loss = 0.001260
grad_step = 000478, loss = 0.001251
grad_step = 000479, loss = 0.001245
grad_step = 000480, loss = 0.001244
grad_step = 000481, loss = 0.001247
grad_step = 000482, loss = 0.001250
grad_step = 000483, loss = 0.001253
grad_step = 000484, loss = 0.001254
grad_step = 000485, loss = 0.001254
grad_step = 000486, loss = 0.001252
grad_step = 000487, loss = 0.001250
grad_step = 000488, loss = 0.001248
grad_step = 000489, loss = 0.001247
grad_step = 000490, loss = 0.001245
grad_step = 000491, loss = 0.001244
grad_step = 000492, loss = 0.001241
grad_step = 000493, loss = 0.001239
grad_step = 000494, loss = 0.001235
grad_step = 000495, loss = 0.001232
grad_step = 000496, loss = 0.001228
grad_step = 000497, loss = 0.001224
grad_step = 000498, loss = 0.001222
grad_step = 000499, loss = 0.001220
grad_step = 000500, loss = 0.001218
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001217
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

  date_run                              2020-05-15 01:12:43.592648
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.253334
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 01:12:43.598201
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.179937
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 01:12:43.606042
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.139851
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 01:12:43.610828
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.73421
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
0   2020-05-15 01:12:16.441135  ...    mean_absolute_error
1   2020-05-15 01:12:16.444893  ...     mean_squared_error
2   2020-05-15 01:12:16.448474  ...  median_absolute_error
3   2020-05-15 01:12:16.452638  ...               r2_score
4   2020-05-15 01:12:26.368231  ...    mean_absolute_error
5   2020-05-15 01:12:26.372001  ...     mean_squared_error
6   2020-05-15 01:12:26.374992  ...  median_absolute_error
7   2020-05-15 01:12:26.377790  ...               r2_score
8   2020-05-15 01:12:43.592648  ...    mean_absolute_error
9   2020-05-15 01:12:43.598201  ...     mean_squared_error
10  2020-05-15 01:12:43.606042  ...  median_absolute_error
11  2020-05-15 01:12:43.610828  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f61dbc41fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:05, 150166.46it/s] 62%|██████▏   | 6103040/9912422 [00:00<00:17, 214296.89it/s]9920512it [00:00, 43935825.44it/s]                           
0it [00:00, ?it/s]32768it [00:00, 576774.23it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162658.12it/s]1654784it [00:00, 11567123.11it/s]                         
0it [00:00, ?it/s]8192it [00:00, 167700.92it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f61dbc94b00> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f61dbc4cef0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f61dbc94b00> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f61dbc4cef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f61dbc94b00> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f61dbc4cef0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f61dbc94b00> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f61dbc4cef0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f61dbc94b00> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f61dbc4cef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f3e308a91d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=d3545b7974a2f78353c8976cb310ecab851600cf665d6f3c3d7296912f9781fe
  Stored in directory: /tmp/pip-ephem-wheel-cache-hia55xoy/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f3dc86a4710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
  581632/17464789 [..............................] - ETA: 1s
 9388032/17464789 [===============>..............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 01:14:08.826800: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 01:14:08.831239: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-15 01:14:08.831377: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560057668cd0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 01:14:08.831391: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.3906 - accuracy: 0.5180
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4903 - accuracy: 0.5115 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5184 - accuracy: 0.5097
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6436 - accuracy: 0.5015
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6544 - accuracy: 0.5008
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6411 - accuracy: 0.5017
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6776 - accuracy: 0.4993
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7011 - accuracy: 0.4978
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7211 - accuracy: 0.4964
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7356 - accuracy: 0.4955
11000/25000 [============>.................] - ETA: 3s - loss: 7.7433 - accuracy: 0.4950
12000/25000 [=============>................] - ETA: 3s - loss: 7.7318 - accuracy: 0.4958
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7445 - accuracy: 0.4949
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7422 - accuracy: 0.4951
15000/25000 [=================>............] - ETA: 2s - loss: 7.7402 - accuracy: 0.4952
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7385 - accuracy: 0.4953
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7253 - accuracy: 0.4962
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7152 - accuracy: 0.4968
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7296 - accuracy: 0.4959
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7295 - accuracy: 0.4959
21000/25000 [========================>.....] - ETA: 0s - loss: 7.7141 - accuracy: 0.4969
22000/25000 [=========================>....] - ETA: 0s - loss: 7.7064 - accuracy: 0.4974
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6880 - accuracy: 0.4986
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6730 - accuracy: 0.4996
25000/25000 [==============================] - 7s 264us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 01:14:21.686107
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 01:14:21.686107  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:31:31, 10.6kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:00:00, 15.0kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:01<11:15:24, 21.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 885k/862M [00:01<7:53:16, 30.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.54M/862M [00:01<5:30:27, 43.3kB/s].vector_cache/glove.6B.zip:   1%|          | 8.41M/862M [00:01<3:50:05, 61.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.8M/862M [00:01<2:40:21, 88.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.5M/862M [00:01<1:51:35, 126kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 24.4M/862M [00:01<1:17:40, 180kB/s].vector_cache/glove.6B.zip:   3%|▎         | 30.1M/862M [00:01<54:05, 256kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.8M/862M [00:02<37:42, 365kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.6M/862M [00:02<26:16, 520kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.5M/862M [00:02<18:28, 738kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.1M/862M [00:02<12:54, 1.05MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<09:40, 1.40MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<08:39, 1.55MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.0M/862M [00:04<07:53, 1.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:04<05:58, 2.25MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:06<06:43, 1.99MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<06:22, 2.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:06<04:48, 2.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<06:01, 2.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<07:10, 1.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<05:39, 2.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.2M/862M [00:08<04:07, 3.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<09:12, 1.44MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<07:48, 1.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:10<05:45, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<07:04, 1.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<07:37, 1.73MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<05:55, 2.22MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:12<04:16, 3.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<22:16, 588kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<16:54, 774kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:14<12:05, 1.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<11:30, 1.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<09:24, 1.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:16<06:54, 1.88MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<07:52, 1.65MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<08:08, 1.59MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<06:14, 2.07MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.1M/862M [00:18<04:31, 2.86MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<12:14, 1.05MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<09:52, 1.31MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<07:10, 1.79MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<08:02, 1.60MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<06:55, 1.85MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<05:09, 2.48MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.0M/862M [00:24<06:37, 1.92MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<05:56, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:24<04:28, 2.85MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:07, 2.07MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:52, 1.85MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:24, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<03:56, 3.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<2:27:47, 85.4kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<1:44:41, 120kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<1:13:27, 171kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<54:13, 231kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<39:11, 320kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<27:38, 453kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<22:16, 560kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<16:51, 740kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<12:05, 1.03MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<11:22, 1.09MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<10:29, 1.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:52, 1.57MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<05:40, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<09:27, 1.31MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:00, 1.54MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:53, 2.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<06:43, 1.83MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:33, 1.62MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:00, 2.04MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:38<04:22, 2.80MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<13:11, 925kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<10:39, 1.15MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<07:47, 1.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<08:00, 1.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<08:25, 1.44MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:29, 1.86MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<04:41, 2.57MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<09:05, 1.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:45, 1.55MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:42, 2.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<06:33, 1.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:22, 1.63MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:50, 2.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<04:14, 2.82MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<12:41, 940kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<10:13, 1.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:29, 1.59MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<07:44, 1.53MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<08:09, 1.45MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:18, 1.88MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<04:34, 2.59MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<07:53, 1.49MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:53, 1.71MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:05, 2.31MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:03, 1.94MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<06:57, 1.68MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:25, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<03:58, 2.94MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:58, 1.67MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:14, 1.87MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:38, 2.50MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:42, 2.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:18, 2.18MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:00, 2.88MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:15, 2.19MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<06:21, 1.81MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:01, 2.29MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<03:38, 3.15MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<10:15, 1.11MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<08:30, 1.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<06:16, 1.82MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:47, 1.67MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<07:23, 1.54MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:43, 1.98MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<04:09, 2.72MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<07:37, 1.48MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<06:37, 1.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:54, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:49, 1.93MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<06:41, 1.68MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:16, 2.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<03:48, 2.94MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<09:37, 1.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<08:01, 1.39MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:55, 1.88MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:29, 1.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<07:07, 1.56MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:38, 1.96MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<04:04, 2.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<11:50, 930kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<09:34, 1.15MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<07:00, 1.57MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:12, 1.52MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:28, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<05:45, 1.90MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<04:09, 2.62MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<07:43, 1.41MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<06:38, 1.64MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<04:54, 2.21MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:43, 1.89MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:30, 1.66MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<05:09, 2.10MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<03:44, 2.87MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<10:36, 1.01MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:39, 1.24MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<06:20, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:40, 1.60MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<07:09, 1.49MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<05:36, 1.90MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<04:03, 2.61MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<13:17, 798kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<10:28, 1.01MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<07:37, 1.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:34, 1.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:44, 1.36MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:56, 1.77MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:16, 2.45MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<08:19, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<07:02, 1.49MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<05:13, 2.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:50, 1.78MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:30, 1.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:03, 2.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<03:40, 2.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:53, 1.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:59, 1.72MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<04:26, 2.32MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:16, 1.94MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:04, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<04:47, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<03:29, 2.92MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<09:55, 1.03MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<08:05, 1.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<05:54, 1.72MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:15, 1.61MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:38, 1.52MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:07, 1.97MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<03:43, 2.70MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<06:20, 1.59MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:36, 1.79MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<04:12, 2.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<05:03, 1.98MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:36, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:29, 2.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:37, 2.14MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:32, 1.79MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:27, 2.22MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<03:14, 3.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<10:33, 932kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<08:30, 1.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<06:13, 1.58MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:24, 1.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:32, 1.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<04:07, 2.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<05:03, 1.92MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<05:41, 1.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:29, 2.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<03:16, 2.95MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<11:06, 868kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<08:51, 1.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<06:25, 1.50MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<06:32, 1.46MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<05:41, 1.68MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<04:12, 2.26MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:57, 1.92MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:40, 1.67MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:31, 2.10MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<03:17, 2.87MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<10:14, 920kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<08:14, 1.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<06:01, 1.56MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:10, 1.51MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:29, 1.44MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<05:01, 1.86MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<03:37, 2.56MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<11:03, 840kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<08:48, 1.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<06:23, 1.45MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:25, 1.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:37, 1.39MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:06, 1.80MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<03:41, 2.48MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<09:18, 983kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<07:32, 1.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<05:29, 1.66MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:45, 1.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<06:03, 1.50MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:43, 1.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<03:23, 2.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<15:56, 565kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<11:58, 752kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<08:36, 1.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<07:59, 1.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<06:36, 1.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<04:50, 1.84MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:15, 1.69MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:44, 1.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:32, 1.95MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:15<03:16, 2.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<08:51, 994kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<07:11, 1.22MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<05:14, 1.68MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:31, 1.58MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:54, 1.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:33, 1.91MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<03:18, 2.63MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:40, 1.53MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:57, 1.75MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:40, 2.35MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:24, 1.95MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:05, 1.69MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:03, 2.11MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<02:56, 2.90MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<09:05, 937kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<07:21, 1.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<05:20, 1.59MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:31, 1.53MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:50, 1.74MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:37, 2.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:18, 1.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:52, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:50, 2.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<02:47, 2.98MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<09:42, 856kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<07:44, 1.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<05:36, 1.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<05:39, 1.46MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<05:52, 1.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:34, 1.80MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<03:17, 2.49MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<09:49, 832kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<07:49, 1.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:41, 1.43MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:41, 1.42MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:46, 1.40MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:27, 1.81MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<03:13, 2.50MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<09:37, 836kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<07:38, 1.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<05:33, 1.44MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<05:34, 1.43MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<05:40, 1.41MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:23, 1.81MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:41<03:09, 2.50MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<14:38, 540kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<11:04, 713kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<07:54, 995kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<07:16, 1.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<06:54, 1.13MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:17, 1.48MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<03:47, 2.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<09:05, 853kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<07:15, 1.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<05:17, 1.46MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<05:19, 1.45MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<05:30, 1.39MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:13, 1.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<03:03, 2.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:02, 1.51MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:23, 1.74MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:16, 2.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:53, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:29, 1.68MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:31, 2.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<02:32, 2.95MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<06:36, 1.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:28, 1.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:01, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:23, 1.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:47, 1.54MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:46, 1.96MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<02:44, 2.68MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<08:01, 915kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<06:27, 1.13MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:43, 1.55MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:49, 1.50MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<05:04, 1.43MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<03:53, 1.86MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:49, 2.56MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:41, 1.53MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:06, 1.76MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:03, 2.34MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:39, 1.95MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:23, 2.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<02:34, 2.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<03:17, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:52, 1.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:02, 2.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:13, 3.16MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:30, 1.55MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:56, 1.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<02:55, 2.38MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:29, 1.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:03, 1.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:10, 2.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:20, 2.93MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:30, 1.95MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:15, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:13<02:28, 2.77MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:09, 2.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:46, 1.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:02, 2.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<02:12, 3.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<07:07, 943kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<05:45, 1.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<04:11, 1.60MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<04:20, 1.53MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:47, 1.75MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:48, 2.36MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:21, 1.95MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:06, 2.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:21, 2.79MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<03:01, 2.15MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:37, 1.79MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:51, 2.27MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:08, 3.02MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:02, 2.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:51, 2.25MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:10, 2.94MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:51, 2.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:29, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:45, 2.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:00, 3.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:34, 1.38MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:55, 1.61MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:53, 2.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:20, 1.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:43, 1.67MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:54, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:07, 2.91MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:26, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:03, 2.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:17, 2.67MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:56, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:32, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:47, 2.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<02:01, 2.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<04:00, 1.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:30, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:35, 2.32MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:04, 1.94MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:49, 2.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:06, 2.80MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:43, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:12, 1.83MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:32, 2.31MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<01:49, 3.18MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<05:23, 1.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<04:22, 1.33MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:11, 1.81MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:30, 1.63MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:40, 1.56MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:50, 2.01MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<02:02, 2.79MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<07:51, 722kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<06:08, 921kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:26, 1.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:16, 1.31MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:11, 1.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:12, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<02:18, 2.41MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<08:31, 650kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<06:33, 843kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<04:42, 1.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:30, 1.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<04:25, 1.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:24, 1.60MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<02:26, 2.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<06:11, 872kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<04:56, 1.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:34, 1.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:37, 1.47MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:46, 1.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:56, 1.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:07, 2.48MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<05:51, 898kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<04:39, 1.13MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:23, 1.54MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:30, 1.48MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:39, 1.41MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:48, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:01, 2.53MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:40, 1.39MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<03:09, 1.61MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<02:19, 2.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:41, 1.87MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<03:00, 1.67MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:21, 2.14MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<01:42, 2.92MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:08, 1.59MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:45, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<02:03, 2.40MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:28, 1.99MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:17, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:43, 2.82MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:14, 2.16MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:41, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:09, 2.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<01:33, 3.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<05:03, 943kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:04, 1.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<02:58, 1.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:03, 1.53MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:14, 1.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:29, 1.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<01:47, 2.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<03:11, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:46, 1.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<02:02, 2.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:23, 1.90MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:44, 1.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:07, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:32, 2.92MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<02:44, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:25, 1.85MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:49, 2.46MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:12, 2.01MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:33, 1.72MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:00, 2.19MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<01:27, 3.00MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:47, 1.56MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:27, 1.77MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:49, 2.36MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:10, 1.97MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:31, 1.70MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:59, 2.14MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<01:26, 2.93MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<04:08, 1.02MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<03:23, 1.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<02:27, 1.70MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:35, 1.60MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:46, 1.49MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<02:08, 1.93MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<01:32, 2.66MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:46, 1.47MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:24, 1.68MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:47, 2.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:05, 1.91MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:55, 2.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:26, 2.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:49, 2.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:11, 1.79MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:43, 2.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<01:16, 3.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:04, 1.87MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:53, 2.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:25, 2.69MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:47, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:41, 2.25MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:15, 2.98MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:40, 2.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:36, 2.32MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:13, 3.03MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:37, 2.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:59, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:34, 2.30MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<01:08, 3.14MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<03:25, 1.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:48, 1.28MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<02:03, 1.73MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:10, 1.62MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:16, 1.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:46, 1.98MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<01:15, 2.73MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<05:36, 615kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<04:19, 796kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<03:06, 1.10MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:51, 1.18MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:52<02:47, 1.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:08, 1.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:52<01:31, 2.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:51, 859kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:04, 1.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<02:12, 1.48MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:13, 1.46MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:16, 1.42MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:45, 1.83MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:56<01:15, 2.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<05:52, 540kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<04:26, 713kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<03:09, 992kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:52, 1.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:46, 1.12MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<02:05, 1.48MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<01:28, 2.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<03:18, 918kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:39, 1.14MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:55, 1.56MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:57, 1.51MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:01, 1.46MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:33, 1.90MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<01:06, 2.62MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:10, 1.33MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:50, 1.56MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:21, 2.12MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:32, 1.84MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:43, 1.63MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:20, 2.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<00:58, 2.87MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:45, 1.57MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:32, 1.79MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:08, 2.38MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:21, 1.97MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:15, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<00:56, 2.83MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:12, 2.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:25, 1.84MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:07, 2.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<00:48, 3.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<04:34, 556kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<03:27, 733kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<02:27, 1.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:15, 1.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:07, 1.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:36, 1.53MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<01:08, 2.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<04:05, 588kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<03:08, 766kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:14, 1.06MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<02:02, 1.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:38, 1.42MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:12, 1.91MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:18, 1.72MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:10, 1.91MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:53, 2.54MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:04, 2.05MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:15, 1.74MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:00, 2.17MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<00:43, 2.98MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<02:15, 942kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:49, 1.16MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:19, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:20, 1.53MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:08, 1.80MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:49, 2.44MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:36, 3.30MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:01, 981kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:52, 1.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<01:24, 1.41MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:32<00:59, 1.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:22, 1.39MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:11, 1.62MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:52, 2.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:59, 1.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:07, 1.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:52, 2.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:37, 2.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:24, 1.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:11, 1.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:52, 2.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:57, 1.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:04, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:49, 2.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<00:35, 2.79MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:37, 1.02MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:18, 1.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:57, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:59, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:02, 1.50MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:49, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:34, 2.62MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:39, 911kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:18, 1.14MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:56, 1.57MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:57, 1.50MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:58, 1.47MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:44, 1.90MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:31, 2.62MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<02:13, 613kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:41, 800kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:12, 1.11MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:06, 1.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:55, 1.40MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:40, 1.89MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:42, 1.72MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:38, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:28, 2.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:33, 2.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:39, 1.78MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:30, 2.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:21, 3.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:44, 1.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:38, 1.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:28, 2.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:32, 1.91MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:36, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:28, 2.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:19, 2.91MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:35, 1.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:31, 1.82MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:22, 2.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:26, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:24, 2.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:17, 2.84MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:22, 2.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:26, 1.80MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:20, 2.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:14, 3.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:26, 1.65MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:23, 1.86MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:17, 2.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:19, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:23, 1.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:18, 2.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:12, 2.96MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:44, 819kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:34, 1.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:24, 1.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:22, 1.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:23, 1.38MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:17, 1.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:12, 2.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:15, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:13, 1.97MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:09, 2.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:11, 2.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:10, 2.21MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:06, 3.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:11, 1.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:09, 1.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:06, 2.60MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 2.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:08, 1.75MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:03, 3.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:07, 1.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:24<00:06, 1.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 2.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:03, 2.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:01, 2.76MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.13MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.78MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:00, 2.26MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 911/400000 [00:00<00:43, 9102.87it/s]  0%|          | 1818/400000 [00:00<00:43, 9090.55it/s]  1%|          | 2733/400000 [00:00<00:43, 9105.50it/s]  1%|          | 3681/400000 [00:00<00:43, 9212.81it/s]  1%|          | 4653/400000 [00:00<00:42, 9358.47it/s]  1%|▏         | 5609/400000 [00:00<00:41, 9406.47it/s]  2%|▏         | 6558/400000 [00:00<00:41, 9429.95it/s]  2%|▏         | 7499/400000 [00:00<00:41, 9421.27it/s]  2%|▏         | 8486/400000 [00:00<00:40, 9551.35it/s]  2%|▏         | 9467/400000 [00:01<00:40, 9626.20it/s]  3%|▎         | 10426/400000 [00:01<00:40, 9612.18it/s]  3%|▎         | 11410/400000 [00:01<00:40, 9677.98it/s]  3%|▎         | 12430/400000 [00:01<00:39, 9826.83it/s]  3%|▎         | 13405/400000 [00:01<00:39, 9745.05it/s]  4%|▎         | 14375/400000 [00:01<00:40, 9624.08it/s]  4%|▍         | 15334/400000 [00:01<00:40, 9475.05it/s]  4%|▍         | 16280/400000 [00:01<00:41, 9342.21it/s]  4%|▍         | 17214/400000 [00:01<00:41, 9252.48it/s]  5%|▍         | 18139/400000 [00:01<00:41, 9097.65it/s]  5%|▍         | 19074/400000 [00:02<00:41, 9171.47it/s]  5%|▍         | 19992/400000 [00:02<00:42, 9000.33it/s]  5%|▌         | 20900/400000 [00:02<00:42, 9021.26it/s]  5%|▌         | 21815/400000 [00:02<00:41, 9057.87it/s]  6%|▌         | 22783/400000 [00:02<00:40, 9233.35it/s]  6%|▌         | 23716/400000 [00:02<00:40, 9259.69it/s]  6%|▌         | 24648/400000 [00:02<00:40, 9276.40it/s]  6%|▋         | 25626/400000 [00:02<00:39, 9421.37it/s]  7%|▋         | 26599/400000 [00:02<00:39, 9510.37it/s]  7%|▋         | 27617/400000 [00:02<00:38, 9699.58it/s]  7%|▋         | 28623/400000 [00:03<00:37, 9804.09it/s]  7%|▋         | 29605/400000 [00:03<00:37, 9797.02it/s]  8%|▊         | 30586/400000 [00:03<00:37, 9797.39it/s]  8%|▊         | 31567/400000 [00:03<00:38, 9574.56it/s]  8%|▊         | 32527/400000 [00:03<00:38, 9448.78it/s]  8%|▊         | 33474/400000 [00:03<00:39, 9298.87it/s]  9%|▊         | 34406/400000 [00:03<00:40, 9120.29it/s]  9%|▉         | 35320/400000 [00:03<00:40, 9106.51it/s]  9%|▉         | 36232/400000 [00:03<00:40, 9050.04it/s]  9%|▉         | 37141/400000 [00:03<00:40, 9060.97it/s] 10%|▉         | 38061/400000 [00:04<00:39, 9102.17it/s] 10%|▉         | 38972/400000 [00:04<00:40, 8909.46it/s] 10%|▉         | 39908/400000 [00:04<00:39, 9038.98it/s] 10%|█         | 40825/400000 [00:04<00:39, 9076.58it/s] 10%|█         | 41734/400000 [00:04<00:40, 8885.64it/s] 11%|█         | 42625/400000 [00:04<00:40, 8765.12it/s] 11%|█         | 43503/400000 [00:04<00:40, 8754.79it/s] 11%|█         | 44398/400000 [00:04<00:40, 8809.91it/s] 11%|█▏        | 45310/400000 [00:04<00:39, 8899.10it/s] 12%|█▏        | 46201/400000 [00:04<00:39, 8882.02it/s] 12%|█▏        | 47129/400000 [00:05<00:39, 8995.16it/s] 12%|█▏        | 48030/400000 [00:05<00:39, 8882.11it/s] 12%|█▏        | 48966/400000 [00:05<00:38, 9019.28it/s] 12%|█▏        | 49876/400000 [00:05<00:38, 9040.23it/s] 13%|█▎        | 50858/400000 [00:05<00:37, 9260.52it/s] 13%|█▎        | 51838/400000 [00:05<00:36, 9415.07it/s] 13%|█▎        | 52782/400000 [00:05<00:37, 9362.39it/s] 13%|█▎        | 53720/400000 [00:05<00:37, 9314.02it/s] 14%|█▎        | 54660/400000 [00:05<00:36, 9337.12it/s] 14%|█▍        | 55624/400000 [00:05<00:36, 9423.31it/s] 14%|█▍        | 56626/400000 [00:06<00:35, 9593.70it/s] 14%|█▍        | 57587/400000 [00:06<00:36, 9399.07it/s] 15%|█▍        | 58590/400000 [00:06<00:35, 9576.86it/s] 15%|█▍        | 59594/400000 [00:06<00:35, 9709.90it/s] 15%|█▌        | 60567/400000 [00:06<00:35, 9496.31it/s] 15%|█▌        | 61519/400000 [00:06<00:36, 9352.11it/s] 16%|█▌        | 62457/400000 [00:06<00:36, 9246.18it/s] 16%|█▌        | 63384/400000 [00:06<00:36, 9205.96it/s] 16%|█▌        | 64306/400000 [00:06<00:36, 9169.55it/s] 16%|█▋        | 65224/400000 [00:07<00:37, 9041.34it/s] 17%|█▋        | 66130/400000 [00:07<00:37, 8809.27it/s] 17%|█▋        | 67025/400000 [00:07<00:37, 8850.52it/s] 17%|█▋        | 67957/400000 [00:07<00:36, 8984.52it/s] 17%|█▋        | 68868/400000 [00:07<00:36, 9021.32it/s] 17%|█▋        | 69772/400000 [00:07<00:36, 8931.95it/s] 18%|█▊        | 70667/400000 [00:07<00:37, 8759.97it/s] 18%|█▊        | 71545/400000 [00:07<00:37, 8728.15it/s] 18%|█▊        | 72468/400000 [00:07<00:36, 8871.34it/s] 18%|█▊        | 73392/400000 [00:07<00:36, 8978.22it/s] 19%|█▊        | 74318/400000 [00:08<00:35, 9060.18it/s] 19%|█▉        | 75245/400000 [00:08<00:35, 9120.15it/s] 19%|█▉        | 76163/400000 [00:08<00:35, 9137.10it/s] 19%|█▉        | 77088/400000 [00:08<00:35, 9170.26it/s] 20%|█▉        | 78006/400000 [00:08<00:35, 9084.02it/s] 20%|█▉        | 78915/400000 [00:08<00:35, 9042.17it/s] 20%|█▉        | 79820/400000 [00:08<00:35, 9020.35it/s] 20%|██        | 80723/400000 [00:08<00:35, 8905.65it/s] 20%|██        | 81630/400000 [00:08<00:35, 8953.91it/s] 21%|██        | 82539/400000 [00:08<00:35, 8992.94it/s] 21%|██        | 83439/400000 [00:09<00:35, 8986.67it/s] 21%|██        | 84338/400000 [00:09<00:35, 8932.70it/s] 21%|██▏       | 85232/400000 [00:09<00:35, 8904.19it/s] 22%|██▏       | 86123/400000 [00:09<00:35, 8890.46it/s] 22%|██▏       | 87022/400000 [00:09<00:35, 8917.57it/s] 22%|██▏       | 87945/400000 [00:09<00:34, 9008.46it/s] 22%|██▏       | 88847/400000 [00:09<00:36, 8457.47it/s] 22%|██▏       | 89713/400000 [00:09<00:36, 8514.90it/s] 23%|██▎       | 90570/400000 [00:09<00:36, 8444.26it/s] 23%|██▎       | 91428/400000 [00:09<00:36, 8482.18it/s] 23%|██▎       | 92279/400000 [00:10<00:36, 8458.16it/s] 23%|██▎       | 93165/400000 [00:10<00:35, 8573.19it/s] 24%|██▎       | 94069/400000 [00:10<00:35, 8707.60it/s] 24%|██▍       | 95004/400000 [00:10<00:34, 8890.54it/s] 24%|██▍       | 95930/400000 [00:10<00:33, 8998.05it/s] 24%|██▍       | 96832/400000 [00:10<00:34, 8821.37it/s] 24%|██▍       | 97717/400000 [00:10<00:34, 8686.45it/s] 25%|██▍       | 98604/400000 [00:10<00:34, 8738.13it/s] 25%|██▍       | 99524/400000 [00:10<00:33, 8869.24it/s] 25%|██▌       | 100431/400000 [00:10<00:33, 8927.53it/s] 25%|██▌       | 101332/400000 [00:11<00:33, 8951.01it/s] 26%|██▌       | 102228/400000 [00:11<00:33, 8936.32it/s] 26%|██▌       | 103123/400000 [00:11<00:33, 8939.23it/s] 26%|██▌       | 104035/400000 [00:11<00:32, 8991.33it/s] 26%|██▌       | 104957/400000 [00:11<00:32, 9058.35it/s] 26%|██▋       | 105864/400000 [00:11<00:32, 8957.01it/s] 27%|██▋       | 106764/400000 [00:11<00:32, 8967.18it/s] 27%|██▋       | 107662/400000 [00:11<00:32, 8916.95it/s] 27%|██▋       | 108566/400000 [00:11<00:32, 8952.60it/s] 27%|██▋       | 109462/400000 [00:12<00:32, 8887.09it/s] 28%|██▊       | 110358/400000 [00:12<00:32, 8906.34it/s] 28%|██▊       | 111249/400000 [00:12<00:32, 8850.18it/s] 28%|██▊       | 112135/400000 [00:12<00:33, 8567.46it/s] 28%|██▊       | 113052/400000 [00:12<00:32, 8738.56it/s] 28%|██▊       | 113969/400000 [00:12<00:32, 8861.52it/s] 29%|██▊       | 114866/400000 [00:12<00:32, 8892.44it/s] 29%|██▉       | 115757/400000 [00:12<00:31, 8884.99it/s] 29%|██▉       | 116668/400000 [00:12<00:31, 8951.25it/s] 29%|██▉       | 117592/400000 [00:12<00:31, 9033.57it/s] 30%|██▉       | 118497/400000 [00:13<00:31, 9010.15it/s] 30%|██▉       | 119405/400000 [00:13<00:31, 9030.10it/s] 30%|███       | 120309/400000 [00:13<00:31, 8982.89it/s] 30%|███       | 121208/400000 [00:13<00:31, 8899.63it/s] 31%|███       | 122106/400000 [00:13<00:31, 8923.02it/s] 31%|███       | 123029/400000 [00:13<00:30, 9010.93it/s] 31%|███       | 123971/400000 [00:13<00:30, 9128.36it/s] 31%|███       | 124885/400000 [00:13<00:30, 9105.86it/s] 31%|███▏      | 125797/400000 [00:13<00:30, 9069.22it/s] 32%|███▏      | 126705/400000 [00:13<00:30, 9032.88it/s] 32%|███▏      | 127609/400000 [00:14<00:30, 9003.21it/s] 32%|███▏      | 128510/400000 [00:14<00:30, 8972.76it/s] 32%|███▏      | 129414/400000 [00:14<00:30, 8991.32it/s] 33%|███▎      | 130322/400000 [00:14<00:29, 9016.89it/s] 33%|███▎      | 131224/400000 [00:14<00:30, 8946.22it/s] 33%|███▎      | 132128/400000 [00:14<00:29, 8972.63it/s] 33%|███▎      | 133026/400000 [00:14<00:29, 8970.09it/s] 33%|███▎      | 133924/400000 [00:14<00:29, 8942.32it/s] 34%|███▎      | 134819/400000 [00:14<00:30, 8754.89it/s] 34%|███▍      | 135713/400000 [00:14<00:30, 8809.05it/s] 34%|███▍      | 136610/400000 [00:15<00:29, 8855.59it/s] 34%|███▍      | 137497/400000 [00:15<00:29, 8816.77it/s] 35%|███▍      | 138398/400000 [00:15<00:29, 8872.04it/s] 35%|███▍      | 139286/400000 [00:15<00:29, 8846.95it/s] 35%|███▌      | 140179/400000 [00:15<00:29, 8870.69it/s] 35%|███▌      | 141088/400000 [00:15<00:28, 8934.06it/s] 35%|███▌      | 141982/400000 [00:15<00:29, 8895.61it/s] 36%|███▌      | 142892/400000 [00:15<00:28, 8955.63it/s] 36%|███▌      | 143794/400000 [00:15<00:28, 8972.53it/s] 36%|███▌      | 144753/400000 [00:15<00:27, 9147.79it/s] 36%|███▋      | 145690/400000 [00:16<00:27, 9210.91it/s] 37%|███▋      | 146612/400000 [00:16<00:27, 9098.43it/s] 37%|███▋      | 147523/400000 [00:16<00:28, 8984.09it/s] 37%|███▋      | 148423/400000 [00:16<00:28, 8896.91it/s] 37%|███▋      | 149314/400000 [00:16<00:28, 8734.72it/s] 38%|███▊      | 150216/400000 [00:16<00:28, 8816.89it/s] 38%|███▊      | 151099/400000 [00:16<00:28, 8806.71it/s] 38%|███▊      | 151981/400000 [00:16<00:28, 8807.77it/s] 38%|███▊      | 152865/400000 [00:16<00:28, 8815.91it/s] 38%|███▊      | 153763/400000 [00:16<00:27, 8864.05it/s] 39%|███▊      | 154688/400000 [00:17<00:27, 8975.08it/s] 39%|███▉      | 155587/400000 [00:17<00:27, 8905.27it/s] 39%|███▉      | 156493/400000 [00:17<00:27, 8949.55it/s] 39%|███▉      | 157389/400000 [00:17<00:27, 8666.79it/s] 40%|███▉      | 158312/400000 [00:17<00:27, 8826.13it/s] 40%|███▉      | 159214/400000 [00:17<00:27, 8883.15it/s] 40%|████      | 160104/400000 [00:17<00:27, 8831.46it/s] 40%|████      | 160997/400000 [00:17<00:26, 8860.22it/s] 40%|████      | 161887/400000 [00:17<00:26, 8871.62it/s] 41%|████      | 162787/400000 [00:17<00:26, 8909.01it/s] 41%|████      | 163717/400000 [00:18<00:26, 9020.10it/s] 41%|████      | 164620/400000 [00:18<00:26, 9005.40it/s] 41%|████▏     | 165522/400000 [00:18<00:26, 8920.93it/s] 42%|████▏     | 166415/400000 [00:18<00:26, 8661.81it/s] 42%|████▏     | 167293/400000 [00:18<00:26, 8694.26it/s] 42%|████▏     | 168204/400000 [00:18<00:26, 8814.13it/s] 42%|████▏     | 169110/400000 [00:18<00:25, 8885.88it/s] 43%|████▎     | 170009/400000 [00:18<00:25, 8914.50it/s] 43%|████▎     | 170902/400000 [00:18<00:25, 8912.63it/s] 43%|████▎     | 171798/400000 [00:19<00:25, 8924.62it/s] 43%|████▎     | 172720/400000 [00:19<00:25, 9009.44it/s] 43%|████▎     | 173629/400000 [00:19<00:25, 9031.26it/s] 44%|████▎     | 174546/400000 [00:19<00:24, 9070.08it/s] 44%|████▍     | 175454/400000 [00:19<00:25, 8931.52it/s] 44%|████▍     | 176348/400000 [00:19<00:25, 8837.34it/s] 44%|████▍     | 177233/400000 [00:19<00:25, 8746.80it/s] 45%|████▍     | 178109/400000 [00:19<00:25, 8651.35it/s] 45%|████▍     | 178975/400000 [00:19<00:25, 8566.34it/s] 45%|████▍     | 179833/400000 [00:19<00:26, 8413.81it/s] 45%|████▌     | 180676/400000 [00:20<00:26, 8333.43it/s] 45%|████▌     | 181524/400000 [00:20<00:26, 8376.64it/s] 46%|████▌     | 182399/400000 [00:20<00:25, 8482.77it/s] 46%|████▌     | 183313/400000 [00:20<00:24, 8667.49it/s] 46%|████▌     | 184210/400000 [00:20<00:24, 8754.12it/s] 46%|████▋     | 185096/400000 [00:20<00:24, 8782.22it/s] 46%|████▋     | 185976/400000 [00:20<00:24, 8594.03it/s] 47%|████▋     | 186837/400000 [00:20<00:24, 8552.81it/s] 47%|████▋     | 187754/400000 [00:20<00:24, 8728.31it/s] 47%|████▋     | 188629/400000 [00:20<00:24, 8710.03it/s] 47%|████▋     | 189572/400000 [00:21<00:23, 8911.73it/s] 48%|████▊     | 190496/400000 [00:21<00:23, 9006.78it/s] 48%|████▊     | 191399/400000 [00:21<00:23, 8989.84it/s] 48%|████▊     | 192331/400000 [00:21<00:22, 9084.44it/s] 48%|████▊     | 193241/400000 [00:21<00:22, 9060.43it/s] 49%|████▊     | 194150/400000 [00:21<00:22, 9067.61it/s] 49%|████▉     | 195067/400000 [00:21<00:22, 9095.82it/s] 49%|████▉     | 195977/400000 [00:21<00:22, 9083.35it/s] 49%|████▉     | 196886/400000 [00:21<00:22, 8954.68it/s] 49%|████▉     | 197783/400000 [00:21<00:22, 8943.52it/s] 50%|████▉     | 198683/400000 [00:22<00:22, 8958.30it/s] 50%|████▉     | 199580/400000 [00:22<00:22, 8913.40it/s] 50%|█████     | 200472/400000 [00:22<00:22, 8822.15it/s] 50%|█████     | 201369/400000 [00:22<00:22, 8864.19it/s] 51%|█████     | 202279/400000 [00:22<00:22, 8933.27it/s] 51%|█████     | 203202/400000 [00:22<00:21, 9019.48it/s] 51%|█████     | 204105/400000 [00:22<00:22, 8894.50it/s] 51%|█████     | 204996/400000 [00:22<00:22, 8786.63it/s] 51%|█████▏    | 205906/400000 [00:22<00:21, 8875.83it/s] 52%|█████▏    | 206807/400000 [00:22<00:21, 8915.52it/s] 52%|█████▏    | 207714/400000 [00:23<00:21, 8959.63it/s] 52%|█████▏    | 208611/400000 [00:23<00:21, 8903.04it/s] 52%|█████▏    | 209502/400000 [00:23<00:21, 8863.24it/s] 53%|█████▎    | 210413/400000 [00:23<00:21, 8932.52it/s] 53%|█████▎    | 211307/400000 [00:23<00:21, 8914.81it/s] 53%|█████▎    | 212219/400000 [00:23<00:20, 8974.34it/s] 53%|█████▎    | 213117/400000 [00:23<00:20, 8967.29it/s] 54%|█████▎    | 214078/400000 [00:23<00:20, 9150.14it/s] 54%|█████▍    | 215010/400000 [00:23<00:20, 9198.07it/s] 54%|█████▍    | 215931/400000 [00:23<00:20, 9142.49it/s] 54%|█████▍    | 216846/400000 [00:24<00:20, 9060.68it/s] 54%|█████▍    | 217769/400000 [00:24<00:20, 9107.86it/s] 55%|█████▍    | 218683/400000 [00:24<00:19, 9116.19it/s] 55%|█████▍    | 219595/400000 [00:24<00:19, 9108.85it/s] 55%|█████▌    | 220525/400000 [00:24<00:19, 9163.23it/s] 55%|█████▌    | 221442/400000 [00:24<00:19, 9037.96it/s] 56%|█████▌    | 222347/400000 [00:24<00:19, 8967.82it/s] 56%|█████▌    | 223295/400000 [00:24<00:19, 9113.05it/s] 56%|█████▌    | 224232/400000 [00:24<00:19, 9187.57it/s] 56%|█████▋    | 225152/400000 [00:24<00:19, 9011.75it/s] 57%|█████▋    | 226071/400000 [00:25<00:19, 9063.89it/s] 57%|█████▋    | 226979/400000 [00:25<00:19, 8987.37it/s] 57%|█████▋    | 227897/400000 [00:25<00:19, 9041.91it/s] 57%|█████▋    | 228802/400000 [00:25<00:18, 9023.02it/s] 57%|█████▋    | 229734/400000 [00:25<00:18, 9109.72it/s] 58%|█████▊    | 230646/400000 [00:25<00:18, 9036.70it/s] 58%|█████▊    | 231564/400000 [00:25<00:18, 9078.91it/s] 58%|█████▊    | 232497/400000 [00:25<00:18, 9151.75it/s] 58%|█████▊    | 233413/400000 [00:25<00:18, 9107.83it/s] 59%|█████▊    | 234335/400000 [00:25<00:18, 9138.96it/s] 59%|█████▉    | 235250/400000 [00:26<00:18, 9125.90it/s] 59%|█████▉    | 236163/400000 [00:26<00:18, 9093.27it/s] 59%|█████▉    | 237094/400000 [00:26<00:17, 9156.41it/s] 60%|█████▉    | 238010/400000 [00:26<00:17, 9113.79it/s] 60%|█████▉    | 238948/400000 [00:26<00:17, 9191.54it/s] 60%|█████▉    | 239868/400000 [00:26<00:17, 9033.32it/s] 60%|██████    | 240773/400000 [00:26<00:17, 8875.98it/s] 60%|██████    | 241662/400000 [00:26<00:18, 8766.50it/s] 61%|██████    | 242563/400000 [00:26<00:17, 8836.61it/s] 61%|██████    | 243488/400000 [00:27<00:17, 8955.76it/s] 61%|██████    | 244385/400000 [00:27<00:17, 8684.08it/s] 61%|██████▏   | 245305/400000 [00:27<00:17, 8831.83it/s] 62%|██████▏   | 246226/400000 [00:27<00:17, 8939.68it/s] 62%|██████▏   | 247126/400000 [00:27<00:17, 8957.36it/s] 62%|██████▏   | 248034/400000 [00:27<00:16, 8990.94it/s] 62%|██████▏   | 248935/400000 [00:27<00:16, 8934.82it/s] 62%|██████▏   | 249830/400000 [00:27<00:16, 8935.50it/s] 63%|██████▎   | 250725/400000 [00:27<00:16, 8930.61it/s] 63%|██████▎   | 251619/400000 [00:27<00:16, 8916.03it/s] 63%|██████▎   | 252545/400000 [00:28<00:16, 9016.41it/s] 63%|██████▎   | 253450/400000 [00:28<00:16, 9024.48it/s] 64%|██████▎   | 254366/400000 [00:28<00:16, 9063.78it/s] 64%|██████▍   | 255273/400000 [00:28<00:16, 9044.11it/s] 64%|██████▍   | 256178/400000 [00:28<00:16, 8981.88it/s] 64%|██████▍   | 257077/400000 [00:28<00:16, 8774.98it/s] 64%|██████▍   | 257956/400000 [00:28<00:16, 8732.07it/s] 65%|██████▍   | 258860/400000 [00:28<00:15, 8821.77it/s] 65%|██████▍   | 259752/400000 [00:28<00:15, 8849.53it/s] 65%|██████▌   | 260657/400000 [00:28<00:15, 8908.01it/s] 65%|██████▌   | 261561/400000 [00:29<00:15, 8944.72it/s] 66%|██████▌   | 262468/400000 [00:29<00:15, 8980.87it/s] 66%|██████▌   | 263367/400000 [00:29<00:15, 8949.07it/s] 66%|██████▌   | 264279/400000 [00:29<00:15, 8997.43it/s] 66%|██████▋   | 265179/400000 [00:29<00:14, 8989.96it/s] 67%|██████▋   | 266081/400000 [00:29<00:14, 8996.49it/s] 67%|██████▋   | 266981/400000 [00:29<00:14, 8961.61it/s] 67%|██████▋   | 267878/400000 [00:29<00:14, 8934.27it/s] 67%|██████▋   | 268772/400000 [00:29<00:14, 8881.69it/s] 67%|██████▋   | 269661/400000 [00:29<00:14, 8840.14it/s] 68%|██████▊   | 270563/400000 [00:30<00:14, 8891.90it/s] 68%|██████▊   | 271453/400000 [00:30<00:14, 8882.24it/s] 68%|██████▊   | 272342/400000 [00:30<00:14, 8841.48it/s] 68%|██████▊   | 273274/400000 [00:30<00:14, 8976.59it/s] 69%|██████▊   | 274173/400000 [00:30<00:14, 8974.38it/s] 69%|██████▉   | 275122/400000 [00:30<00:13, 9120.73it/s] 69%|██████▉   | 276035/400000 [00:30<00:13, 9023.64it/s] 69%|██████▉   | 276939/400000 [00:30<00:13, 8949.46it/s] 69%|██████▉   | 277835/400000 [00:30<00:13, 8905.41it/s] 70%|██████▉   | 278727/400000 [00:30<00:13, 8877.85it/s] 70%|██████▉   | 279616/400000 [00:31<00:13, 8854.64it/s] 70%|███████   | 280502/400000 [00:31<00:13, 8856.07it/s] 70%|███████   | 281403/400000 [00:31<00:13, 8900.68it/s] 71%|███████   | 282327/400000 [00:31<00:13, 8997.63it/s] 71%|███████   | 283239/400000 [00:31<00:12, 9031.91it/s] 71%|███████   | 284147/400000 [00:31<00:12, 9044.54it/s] 71%|███████▏  | 285055/400000 [00:31<00:12, 9054.30it/s] 71%|███████▏  | 285961/400000 [00:31<00:12, 8977.75it/s] 72%|███████▏  | 286860/400000 [00:31<00:12, 8932.85it/s] 72%|███████▏  | 287764/400000 [00:31<00:12, 8963.77it/s] 72%|███████▏  | 288704/400000 [00:32<00:12, 9087.99it/s] 72%|███████▏  | 289626/400000 [00:32<00:12, 9124.33it/s] 73%|███████▎  | 290557/400000 [00:32<00:11, 9178.92it/s] 73%|███████▎  | 291476/400000 [00:32<00:11, 9164.84it/s] 73%|███████▎  | 292398/400000 [00:32<00:11, 9181.01it/s] 73%|███████▎  | 293320/400000 [00:32<00:11, 9192.57it/s] 74%|███████▎  | 294240/400000 [00:32<00:11, 9105.34it/s] 74%|███████▍  | 295168/400000 [00:32<00:11, 9154.50it/s] 74%|███████▍  | 296084/400000 [00:32<00:11, 9126.99it/s] 74%|███████▍  | 296997/400000 [00:32<00:11, 8992.96it/s] 74%|███████▍  | 297916/400000 [00:33<00:11, 9048.76it/s] 75%|███████▍  | 298822/400000 [00:33<00:11, 9020.84it/s] 75%|███████▍  | 299726/400000 [00:33<00:11, 9023.87it/s] 75%|███████▌  | 300629/400000 [00:33<00:11, 9025.05it/s] 75%|███████▌  | 301532/400000 [00:33<00:11, 8931.97it/s] 76%|███████▌  | 302488/400000 [00:33<00:10, 9110.13it/s] 76%|███████▌  | 303401/400000 [00:33<00:10, 9112.81it/s] 76%|███████▌  | 304339/400000 [00:33<00:10, 9188.58it/s] 76%|███████▋  | 305259/400000 [00:33<00:10, 9152.36it/s] 77%|███████▋  | 306175/400000 [00:33<00:10, 9106.51it/s] 77%|███████▋  | 307101/400000 [00:34<00:10, 9149.38it/s] 77%|███████▋  | 308017/400000 [00:34<00:10, 8994.41it/s] 77%|███████▋  | 308934/400000 [00:34<00:10, 9044.45it/s] 77%|███████▋  | 309840/400000 [00:34<00:09, 9035.82it/s] 78%|███████▊  | 310745/400000 [00:34<00:09, 9035.26it/s] 78%|███████▊  | 311649/400000 [00:34<00:09, 9023.03it/s] 78%|███████▊  | 312552/400000 [00:34<00:09, 8986.12it/s] 78%|███████▊  | 313466/400000 [00:34<00:09, 9031.40it/s] 79%|███████▊  | 314373/400000 [00:34<00:09, 9042.04it/s] 79%|███████▉  | 315278/400000 [00:35<00:09, 9014.95it/s] 79%|███████▉  | 316180/400000 [00:35<00:09, 8932.23it/s] 79%|███████▉  | 317074/400000 [00:35<00:09, 8902.66it/s] 79%|███████▉  | 317991/400000 [00:35<00:09, 8979.02it/s] 80%|███████▉  | 318910/400000 [00:35<00:08, 9038.29it/s] 80%|███████▉  | 319848/400000 [00:35<00:08, 9137.86it/s] 80%|████████  | 320763/400000 [00:35<00:08, 9113.68it/s] 80%|████████  | 321675/400000 [00:35<00:08, 9032.94it/s] 81%|████████  | 322607/400000 [00:35<00:08, 9116.94it/s] 81%|████████  | 323527/400000 [00:35<00:08, 9140.22it/s] 81%|████████  | 324442/400000 [00:36<00:08, 8975.39it/s] 81%|████████▏ | 325354/400000 [00:36<00:08, 9016.34it/s] 82%|████████▏ | 326257/400000 [00:36<00:08, 8940.83it/s] 82%|████████▏ | 327160/400000 [00:36<00:08, 8966.18it/s] 82%|████████▏ | 328091/400000 [00:36<00:07, 9065.22it/s] 82%|████████▏ | 328999/400000 [00:36<00:07, 9003.49it/s] 82%|████████▏ | 329903/400000 [00:36<00:07, 9013.88it/s] 83%|████████▎ | 330805/400000 [00:36<00:07, 8946.10it/s] 83%|████████▎ | 331708/400000 [00:36<00:07, 8969.40it/s] 83%|████████▎ | 332606/400000 [00:36<00:07, 8962.30it/s] 83%|████████▎ | 333507/400000 [00:37<00:07, 8973.92it/s] 84%|████████▎ | 334427/400000 [00:37<00:07, 9040.15it/s] 84%|████████▍ | 335334/400000 [00:37<00:07, 9046.66it/s] 84%|████████▍ | 336239/400000 [00:37<00:07, 9032.19it/s] 84%|████████▍ | 337143/400000 [00:37<00:06, 9022.81it/s] 85%|████████▍ | 338069/400000 [00:37<00:06, 9090.22it/s] 85%|████████▍ | 338979/400000 [00:37<00:06, 9064.19it/s] 85%|████████▍ | 339904/400000 [00:37<00:06, 9118.47it/s] 85%|████████▌ | 340817/400000 [00:37<00:06, 8996.61it/s] 85%|████████▌ | 341718/400000 [00:37<00:06, 8947.06it/s] 86%|████████▌ | 342614/400000 [00:38<00:06, 8945.92it/s] 86%|████████▌ | 343509/400000 [00:38<00:06, 8902.87it/s] 86%|████████▌ | 344412/400000 [00:38<00:06, 8938.21it/s] 86%|████████▋ | 345333/400000 [00:38<00:06, 9015.25it/s] 87%|████████▋ | 346235/400000 [00:38<00:06, 8951.50it/s] 87%|████████▋ | 347163/400000 [00:38<00:05, 9046.38it/s] 87%|████████▋ | 348069/400000 [00:38<00:05, 8995.18it/s] 87%|████████▋ | 348985/400000 [00:38<00:05, 9042.22it/s] 87%|████████▋ | 349890/400000 [00:38<00:05, 8991.55it/s] 88%|████████▊ | 350802/400000 [00:38<00:05, 9028.55it/s] 88%|████████▊ | 351706/400000 [00:39<00:05, 8949.09it/s] 88%|████████▊ | 352602/400000 [00:39<00:05, 8886.49it/s] 88%|████████▊ | 353491/400000 [00:39<00:05, 8851.98it/s] 89%|████████▊ | 354379/400000 [00:39<00:05, 8860.29it/s] 89%|████████▉ | 355266/400000 [00:39<00:05, 8840.51it/s] 89%|████████▉ | 356151/400000 [00:39<00:04, 8822.78it/s] 89%|████████▉ | 357034/400000 [00:39<00:04, 8787.28it/s] 89%|████████▉ | 357913/400000 [00:39<00:04, 8767.25it/s] 90%|████████▉ | 358790/400000 [00:39<00:04, 8673.06it/s] 90%|████████▉ | 359662/400000 [00:39<00:04, 8684.42it/s] 90%|█████████ | 360531/400000 [00:40<00:04, 8541.36it/s] 90%|█████████ | 361419/400000 [00:40<00:04, 8635.78it/s] 91%|█████████ | 362343/400000 [00:40<00:04, 8805.73it/s] 91%|█████████ | 363229/400000 [00:40<00:04, 8820.59it/s] 91%|█████████ | 364135/400000 [00:40<00:04, 8889.55it/s] 91%|█████████▏| 365067/400000 [00:40<00:03, 9012.79it/s] 91%|█████████▏| 365970/400000 [00:40<00:03, 9001.11it/s] 92%|█████████▏| 366876/400000 [00:40<00:03, 9018.03it/s] 92%|█████████▏| 367779/400000 [00:40<00:03, 8949.09it/s] 92%|█████████▏| 368691/400000 [00:40<00:03, 8997.37it/s] 92%|█████████▏| 369599/400000 [00:41<00:03, 9020.39it/s] 93%|█████████▎| 370502/400000 [00:41<00:03, 9011.61it/s] 93%|█████████▎| 371408/400000 [00:41<00:03, 9024.31it/s] 93%|█████████▎| 372311/400000 [00:41<00:03, 8971.40it/s] 93%|█████████▎| 373225/400000 [00:41<00:02, 9019.05it/s] 94%|█████████▎| 374128/400000 [00:41<00:02, 8972.76it/s] 94%|█████████▍| 375026/400000 [00:41<00:02, 8969.78it/s] 94%|█████████▍| 375924/400000 [00:41<00:02, 8920.34it/s] 94%|█████████▍| 376817/400000 [00:41<00:02, 8880.52it/s] 94%|█████████▍| 377706/400000 [00:41<00:02, 8850.88it/s] 95%|█████████▍| 378596/400000 [00:42<00:02, 8865.32it/s] 95%|█████████▍| 379483/400000 [00:42<00:02, 8763.35it/s] 95%|█████████▌| 380360/400000 [00:42<00:02, 8727.08it/s] 95%|█████████▌| 381274/400000 [00:42<00:02, 8845.30it/s] 96%|█████████▌| 382217/400000 [00:42<00:01, 9011.49it/s] 96%|█████████▌| 383159/400000 [00:42<00:01, 9127.62it/s] 96%|█████████▌| 384073/400000 [00:42<00:01, 9071.17it/s] 96%|█████████▋| 385006/400000 [00:42<00:01, 9145.79it/s] 96%|█████████▋| 385926/400000 [00:42<00:01, 9160.39it/s] 97%|█████████▋| 386853/400000 [00:42<00:01, 9192.30it/s] 97%|█████████▋| 387775/400000 [00:43<00:01, 9197.58it/s] 97%|█████████▋| 388696/400000 [00:43<00:01, 9149.86it/s] 97%|█████████▋| 389612/400000 [00:43<00:01, 9138.05it/s] 98%|█████████▊| 390526/400000 [00:43<00:01, 8732.07it/s] 98%|█████████▊| 391461/400000 [00:43<00:00, 8907.80it/s] 98%|█████████▊| 392384/400000 [00:43<00:00, 9000.59it/s] 98%|█████████▊| 393290/400000 [00:43<00:00, 9017.70it/s] 99%|█████████▊| 394200/400000 [00:43<00:00, 9039.97it/s] 99%|█████████▉| 395117/400000 [00:43<00:00, 9076.18it/s] 99%|█████████▉| 396053/400000 [00:44<00:00, 9158.32it/s] 99%|█████████▉| 396970/400000 [00:44<00:00, 9108.76it/s] 99%|█████████▉| 397882/400000 [00:44<00:00, 9004.65it/s]100%|█████████▉| 398784/400000 [00:44<00:00, 8926.82it/s]100%|█████████▉| 399678/400000 [00:44<00:00, 8878.23it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8998.96it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f0b3d59bc88> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011146585658213953 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011168213392978528 	 Accuracy: 52

  model saves at 52% accuracy 

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
2020-05-15 01:23:16.801402: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 01:23:16.805672: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-15 01:23:16.805873: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bc9ab67420 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 01:23:16.805888: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f0ae69fe048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.6053 - accuracy: 0.5040
 2000/25000 [=>............................] - ETA: 7s - loss: 7.6360 - accuracy: 0.5020 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6768 - accuracy: 0.4993
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.5708 - accuracy: 0.5063
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6206 - accuracy: 0.5030
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6692 - accuracy: 0.4998
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6513 - accuracy: 0.5010
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6915 - accuracy: 0.4984
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6922 - accuracy: 0.4983
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6866 - accuracy: 0.4987
11000/25000 [============>.................] - ETA: 3s - loss: 7.6834 - accuracy: 0.4989
12000/25000 [=============>................] - ETA: 3s - loss: 7.6781 - accuracy: 0.4992
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6643 - accuracy: 0.5002
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6524 - accuracy: 0.5009
15000/25000 [=================>............] - ETA: 2s - loss: 7.6799 - accuracy: 0.4991
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6919 - accuracy: 0.4984
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6837 - accuracy: 0.4989
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6771 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6751 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6659 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6687 - accuracy: 0.4999
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6680 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6545 - accuracy: 0.5008
25000/25000 [==============================] - 7s 268us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f0a9a3bb668> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f0af0af7b70> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1000ms/step - loss: 1.7747 - crf_viterbi_accuracy: 0.3333 - val_loss: 1.6662 - val_crf_viterbi_accuracy: 0.5067

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

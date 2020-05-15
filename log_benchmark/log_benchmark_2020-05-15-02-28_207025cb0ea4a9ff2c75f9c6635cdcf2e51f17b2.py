
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f392cf69fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 02:28:47.495607
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 02:28:47.498769
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 02:28:47.501797
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 02:28:47.504499
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f3938f81470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355027.9688
Epoch 2/10

1/1 [==============================] - 0s 96ms/step - loss: 233845.4844
Epoch 3/10

1/1 [==============================] - 0s 96ms/step - loss: 127043.0781
Epoch 4/10

1/1 [==============================] - 0s 87ms/step - loss: 64679.8164
Epoch 5/10

1/1 [==============================] - 0s 97ms/step - loss: 34486.5859
Epoch 6/10

1/1 [==============================] - 0s 86ms/step - loss: 20390.5273
Epoch 7/10

1/1 [==============================] - 0s 93ms/step - loss: 13312.7920
Epoch 8/10

1/1 [==============================] - 0s 87ms/step - loss: 9416.4541
Epoch 9/10

1/1 [==============================] - 0s 93ms/step - loss: 7088.6602
Epoch 10/10

1/1 [==============================] - 0s 94ms/step - loss: 5621.6206

  #### Inference Need return ypred, ytrue ######################### 
[[ 4.1268110e-01 -9.4404292e-01  1.7440715e-01 -4.4313481e-01
   4.9997616e-01  2.8924346e-03  1.8561957e+00 -3.2317707e-01
   1.1917421e+00 -7.2717071e-01  7.9313332e-01 -3.6889488e-01
   4.7487211e-01  2.5015897e-01 -1.9648135e-01 -2.7437420e+00
   7.3506439e-01  1.6046500e-01  5.1139414e-01  7.5507337e-01
   6.0965407e-01 -1.1284311e+00 -7.3122358e-01 -2.0048198e-01
   1.3202727e-01  8.2638186e-01  3.3853760e-01  1.0631970e+00
  -3.0731559e-03  2.7605084e-01  1.0681311e+00  2.4027064e-01
  -3.4883553e-01 -6.1751294e-01  1.0791358e+00 -6.6578311e-01
  -1.0369776e-01  3.1128460e-01 -1.5527284e+00  1.6766459e+00
  -4.7521549e-01 -6.9926822e-01  1.5664924e+00 -1.3434319e+00
  -1.5354216e-01  9.4237524e-01 -4.2866784e-01 -1.1647642e-01
   4.2344713e-01 -1.0787551e+00  6.6953313e-01 -5.8037758e-01
   5.8488733e-01 -1.7934834e+00  2.2926159e+00 -5.4689717e-01
   1.2504615e+00 -6.0336649e-01 -4.4232112e-01 -5.7392347e-01
   7.6574600e-01  3.7379590e-01  7.7271849e-01 -1.1386236e+00
   1.2141428e+00 -9.1070938e-01 -1.0647639e+00 -5.7226336e-01
   5.5841058e-02  1.8613942e+00 -1.5102757e+00  2.0471036e-01
   2.2117175e-01  5.6100786e-01  1.7294793e+00  2.4509606e-01
   1.3056442e+00 -7.5985324e-01 -1.4134419e-01  5.2130818e-03
  -5.7397306e-01 -1.6830275e+00  3.0021134e-01  8.7046647e-01
  -5.6938326e-01  1.9753832e-01  1.5483065e+00 -1.5736612e+00
  -2.1320531e+00  4.0274602e-01  2.6639342e-01 -1.4791039e+00
   1.0962805e+00 -5.8242071e-01 -2.4641520e-01 -1.3970990e+00
   6.4107008e-02 -1.6764739e+00 -6.1587149e-01 -8.2735163e-01
   9.6603370e-01  6.0253531e-01 -1.1600517e+00 -1.5514222e-01
  -4.4453681e-01  4.4059566e-01  1.1574230e+00  4.2902571e-01
   1.2748936e+00  3.1014436e-01 -3.9841169e-01  8.4104133e-01
   8.5730863e-01 -2.7502465e-01  3.0195627e-01 -7.4043429e-01
   3.2718477e-01  1.4178427e+00 -1.7526890e+00  5.2896410e-01
  -3.5589981e-01  8.8614655e+00  9.5563555e+00  8.7805796e+00
   8.0009203e+00  8.3153191e+00  1.0063145e+01  8.0221844e+00
   8.5592070e+00  9.5037813e+00  1.0123734e+01  9.9882326e+00
   8.9313869e+00  9.2013054e+00  1.1171231e+01  8.4244642e+00
   9.7750893e+00  7.7736778e+00  9.6308031e+00  9.7664251e+00
   8.5229540e+00  9.9895163e+00  8.4376659e+00  8.2914305e+00
   9.5036440e+00  9.0867634e+00  9.5179787e+00  9.5212555e+00
   8.8320169e+00  1.0354717e+01  6.7937946e+00  9.0092802e+00
   8.7560759e+00  1.0853111e+01  8.2704506e+00  8.8620691e+00
   9.1000271e+00  9.3768415e+00  9.3468351e+00  7.6394019e+00
   9.0154591e+00  8.3252583e+00  1.0051943e+01  8.4770565e+00
   7.9731460e+00  8.5987673e+00  1.0560025e+01  8.2616224e+00
   7.4684920e+00  9.4634495e+00  8.7466602e+00  9.4881229e+00
   1.0887747e+01  8.8035069e+00  1.0711352e+01  1.0057254e+01
   7.9245200e+00  9.3869123e+00  8.4958620e+00  9.6552753e+00
   5.8548892e-01  1.4665344e+00  1.2714095e+00  1.8260281e+00
   2.0515404e+00  3.2825863e-01  2.6418680e-01  1.2907553e-01
   4.5461923e-01  6.2468255e-01  8.4624827e-01  2.3250046e+00
   1.7214112e+00  1.8303653e+00  2.3024020e+00  2.5331080e-01
   1.3086171e+00  1.0479374e+00  7.5058442e-01  4.6970034e-01
   7.8237766e-01  1.0083048e+00  6.8256760e-01  1.6196401e+00
   3.6642802e-01  1.7633471e+00  3.7472349e-01  3.1564398e+00
   1.0570954e+00  3.9814293e-01  1.6581190e-01  2.0632653e+00
   6.4992839e-01  2.4515381e+00  2.2387054e+00  2.3683327e-01
   4.2199874e-01  8.7128556e-01  1.7141143e+00  2.7930090e+00
   1.2813996e+00  5.3973424e-01  1.5307333e+00  7.2388852e-01
   1.3212734e-01  2.9299283e-01  1.0103135e+00  3.5010093e-01
   6.5902865e-01  2.3644072e-01  5.7089335e-01  3.1737173e-01
   7.7867162e-01  9.3176842e-02  1.2109619e+00  3.0596151e+00
   5.8015174e-01  1.1163888e+00  2.4070168e+00  1.2983383e+00
   3.0553904e+00  1.5222950e+00  7.6915729e-01  6.6195154e-01
   1.7776611e+00  1.6685141e+00  1.5520577e+00  6.1748987e-01
   3.8227654e-01  9.2485160e-01  1.4077320e+00  1.3466723e+00
   4.6603882e-01  9.6596336e-01  4.3271267e-01  1.2650250e+00
   4.1170186e-01  2.9221243e-01  9.7457844e-01  5.3342777e-01
   3.9400119e-01  1.5383494e+00  3.1297693e+00  1.8639346e+00
   1.7065606e+00  3.8917726e-01  7.0848149e-01  3.1712079e-01
   1.7934728e-01  1.0678085e+00  1.6039189e+00  4.9826908e-01
   1.1720850e+00  8.6479807e-01  4.0656638e-01  1.6015410e+00
   1.6923480e+00  3.3447301e-01  2.5632815e+00  2.1113386e+00
   2.1775186e-01  4.6933299e-01  1.2139013e+00  7.1806282e-01
   2.6670706e-01  6.4739650e-01  1.3529928e+00  2.9045188e-01
   4.1330242e-01  3.5620415e-01  1.4567277e+00  1.4360088e+00
   1.1887300e+00  1.2167555e-01  7.2397643e-01  9.3617666e-01
   1.5841824e+00  1.4201521e+00  9.6029985e-01  3.2556438e+00
   3.0407226e-01  9.8638468e+00  9.5034513e+00  9.1627436e+00
   9.4828568e+00  8.7290316e+00  1.0191329e+01  8.3259602e+00
   9.4745312e+00  8.1386480e+00  8.8989496e+00  8.8606691e+00
   9.3862867e+00  9.1006842e+00  8.2974033e+00  8.3774977e+00
   8.2872505e+00  7.4937167e+00  9.5956278e+00  7.5389509e+00
   1.0107212e+01  9.7676735e+00  9.8159389e+00  9.9853802e+00
   9.4525070e+00  8.5461292e+00  8.6890068e+00  7.8253288e+00
   8.3413944e+00  8.3558807e+00  7.9273090e+00  9.1350317e+00
   9.0829306e+00  8.7762661e+00  9.1148939e+00  8.2729120e+00
   9.6226616e+00  8.9468555e+00  1.0421703e+01  8.2603054e+00
   9.8971424e+00  9.8471413e+00  9.3640594e+00  9.3550367e+00
   9.5300808e+00  8.8680735e+00  9.1473732e+00  9.1001997e+00
   9.6464472e+00  8.9634819e+00  7.5113344e+00  8.5126295e+00
   8.7352257e+00  9.9210043e+00  9.2493725e+00  7.7971511e+00
   8.2914381e+00  8.7147512e+00  9.0821476e+00  9.8666000e+00
  -7.1036463e+00 -5.6662874e+00  7.1698885e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 02:28:57.303170
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.4783
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 02:28:57.306239
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8764.33
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 02:28:57.308972
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.1175
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 02:28:57.311329
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -783.907
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139883187507608
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139881977504320
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139881977504824
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139881977505328
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139881977505832
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139881977506336

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f39268ee518> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.510661
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.473228
grad_step = 000002, loss = 0.440728
grad_step = 000003, loss = 0.408097
grad_step = 000004, loss = 0.372857
grad_step = 000005, loss = 0.335544
grad_step = 000006, loss = 0.304697
grad_step = 000007, loss = 0.281801
grad_step = 000008, loss = 0.261917
grad_step = 000009, loss = 0.235235
grad_step = 000010, loss = 0.215929
grad_step = 000011, loss = 0.199989
grad_step = 000012, loss = 0.188098
grad_step = 000013, loss = 0.181086
grad_step = 000014, loss = 0.173108
grad_step = 000015, loss = 0.162572
grad_step = 000016, loss = 0.150837
grad_step = 000017, loss = 0.139339
grad_step = 000018, loss = 0.128419
grad_step = 000019, loss = 0.118764
grad_step = 000020, loss = 0.110451
grad_step = 000021, loss = 0.102729
grad_step = 000022, loss = 0.095620
grad_step = 000023, loss = 0.088464
grad_step = 000024, loss = 0.081046
grad_step = 000025, loss = 0.074383
grad_step = 000026, loss = 0.068690
grad_step = 000027, loss = 0.063121
grad_step = 000028, loss = 0.057309
grad_step = 000029, loss = 0.051669
grad_step = 000030, loss = 0.046786
grad_step = 000031, loss = 0.042774
grad_step = 000032, loss = 0.039081
grad_step = 000033, loss = 0.035188
grad_step = 000034, loss = 0.031340
grad_step = 000035, loss = 0.028081
grad_step = 000036, loss = 0.025526
grad_step = 000037, loss = 0.023287
grad_step = 000038, loss = 0.020921
grad_step = 000039, loss = 0.018452
grad_step = 000040, loss = 0.016319
grad_step = 000041, loss = 0.014695
grad_step = 000042, loss = 0.013290
grad_step = 000043, loss = 0.011923
grad_step = 000044, loss = 0.010690
grad_step = 000045, loss = 0.009670
grad_step = 000046, loss = 0.008816
grad_step = 000047, loss = 0.008015
grad_step = 000048, loss = 0.007211
grad_step = 000049, loss = 0.006490
grad_step = 000050, loss = 0.005950
grad_step = 000051, loss = 0.005539
grad_step = 000052, loss = 0.005160
grad_step = 000053, loss = 0.004789
grad_step = 000054, loss = 0.004436
grad_step = 000055, loss = 0.004140
grad_step = 000056, loss = 0.003924
grad_step = 000057, loss = 0.003744
grad_step = 000058, loss = 0.003567
grad_step = 000059, loss = 0.003414
grad_step = 000060, loss = 0.003294
grad_step = 000061, loss = 0.003181
grad_step = 000062, loss = 0.003060
grad_step = 000063, loss = 0.002952
grad_step = 000064, loss = 0.002887
grad_step = 000065, loss = 0.002850
grad_step = 000066, loss = 0.002801
grad_step = 000067, loss = 0.002737
grad_step = 000068, loss = 0.002688
grad_step = 000069, loss = 0.002651
grad_step = 000070, loss = 0.002613
grad_step = 000071, loss = 0.002573
grad_step = 000072, loss = 0.002543
grad_step = 000073, loss = 0.002525
grad_step = 000074, loss = 0.002507
grad_step = 000075, loss = 0.002482
grad_step = 000076, loss = 0.002460
grad_step = 000077, loss = 0.002445
grad_step = 000078, loss = 0.002431
grad_step = 000079, loss = 0.002416
grad_step = 000080, loss = 0.002398
grad_step = 000081, loss = 0.002385
grad_step = 000082, loss = 0.002377
grad_step = 000083, loss = 0.002368
grad_step = 000084, loss = 0.002357
grad_step = 000085, loss = 0.002348
grad_step = 000086, loss = 0.002342
grad_step = 000087, loss = 0.002334
grad_step = 000088, loss = 0.002324
grad_step = 000089, loss = 0.002316
grad_step = 000090, loss = 0.002312
grad_step = 000091, loss = 0.002308
grad_step = 000092, loss = 0.002301
grad_step = 000093, loss = 0.002295
grad_step = 000094, loss = 0.002291
grad_step = 000095, loss = 0.002286
grad_step = 000096, loss = 0.002281
grad_step = 000097, loss = 0.002277
grad_step = 000098, loss = 0.002273
grad_step = 000099, loss = 0.002269
grad_step = 000100, loss = 0.002265
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002261
grad_step = 000102, loss = 0.002258
grad_step = 000103, loss = 0.002254
grad_step = 000104, loss = 0.002251
grad_step = 000105, loss = 0.002247
grad_step = 000106, loss = 0.002244
grad_step = 000107, loss = 0.002241
grad_step = 000108, loss = 0.002238
grad_step = 000109, loss = 0.002236
grad_step = 000110, loss = 0.002233
grad_step = 000111, loss = 0.002230
grad_step = 000112, loss = 0.002228
grad_step = 000113, loss = 0.002225
grad_step = 000114, loss = 0.002223
grad_step = 000115, loss = 0.002221
grad_step = 000116, loss = 0.002219
grad_step = 000117, loss = 0.002216
grad_step = 000118, loss = 0.002214
grad_step = 000119, loss = 0.002212
grad_step = 000120, loss = 0.002210
grad_step = 000121, loss = 0.002209
grad_step = 000122, loss = 0.002207
grad_step = 000123, loss = 0.002205
grad_step = 000124, loss = 0.002203
grad_step = 000125, loss = 0.002201
grad_step = 000126, loss = 0.002199
grad_step = 000127, loss = 0.002197
grad_step = 000128, loss = 0.002195
grad_step = 000129, loss = 0.002193
grad_step = 000130, loss = 0.002191
grad_step = 000131, loss = 0.002189
grad_step = 000132, loss = 0.002187
grad_step = 000133, loss = 0.002185
grad_step = 000134, loss = 0.002183
grad_step = 000135, loss = 0.002180
grad_step = 000136, loss = 0.002178
grad_step = 000137, loss = 0.002175
grad_step = 000138, loss = 0.002173
grad_step = 000139, loss = 0.002170
grad_step = 000140, loss = 0.002167
grad_step = 000141, loss = 0.002164
grad_step = 000142, loss = 0.002161
grad_step = 000143, loss = 0.002158
grad_step = 000144, loss = 0.002155
grad_step = 000145, loss = 0.002152
grad_step = 000146, loss = 0.002149
grad_step = 000147, loss = 0.002147
grad_step = 000148, loss = 0.002143
grad_step = 000149, loss = 0.002140
grad_step = 000150, loss = 0.002137
grad_step = 000151, loss = 0.002134
grad_step = 000152, loss = 0.002131
grad_step = 000153, loss = 0.002128
grad_step = 000154, loss = 0.002125
grad_step = 000155, loss = 0.002122
grad_step = 000156, loss = 0.002119
grad_step = 000157, loss = 0.002116
grad_step = 000158, loss = 0.002112
grad_step = 000159, loss = 0.002109
grad_step = 000160, loss = 0.002107
grad_step = 000161, loss = 0.002104
grad_step = 000162, loss = 0.002103
grad_step = 000163, loss = 0.002102
grad_step = 000164, loss = 0.002098
grad_step = 000165, loss = 0.002093
grad_step = 000166, loss = 0.002089
grad_step = 000167, loss = 0.002088
grad_step = 000168, loss = 0.002088
grad_step = 000169, loss = 0.002087
grad_step = 000170, loss = 0.002084
grad_step = 000171, loss = 0.002079
grad_step = 000172, loss = 0.002075
grad_step = 000173, loss = 0.002073
grad_step = 000174, loss = 0.002071
grad_step = 000175, loss = 0.002071
grad_step = 000176, loss = 0.002073
grad_step = 000177, loss = 0.002076
grad_step = 000178, loss = 0.002078
grad_step = 000179, loss = 0.002074
grad_step = 000180, loss = 0.002068
grad_step = 000181, loss = 0.002058
grad_step = 000182, loss = 0.002054
grad_step = 000183, loss = 0.002054
grad_step = 000184, loss = 0.002057
grad_step = 000185, loss = 0.002061
grad_step = 000186, loss = 0.002056
grad_step = 000187, loss = 0.002049
grad_step = 000188, loss = 0.002042
grad_step = 000189, loss = 0.002039
grad_step = 000190, loss = 0.002039
grad_step = 000191, loss = 0.002040
grad_step = 000192, loss = 0.002041
grad_step = 000193, loss = 0.002037
grad_step = 000194, loss = 0.002032
grad_step = 000195, loss = 0.002026
grad_step = 000196, loss = 0.002022
grad_step = 000197, loss = 0.002019
grad_step = 000198, loss = 0.002017
grad_step = 000199, loss = 0.002016
grad_step = 000200, loss = 0.002017
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002020
grad_step = 000202, loss = 0.002026
grad_step = 000203, loss = 0.002032
grad_step = 000204, loss = 0.002025
grad_step = 000205, loss = 0.002015
grad_step = 000206, loss = 0.002005
grad_step = 000207, loss = 0.001997
grad_step = 000208, loss = 0.001991
grad_step = 000209, loss = 0.001993
grad_step = 000210, loss = 0.001999
grad_step = 000211, loss = 0.002000
grad_step = 000212, loss = 0.001994
grad_step = 000213, loss = 0.001982
grad_step = 000214, loss = 0.001976
grad_step = 000215, loss = 0.001976
grad_step = 000216, loss = 0.001976
grad_step = 000217, loss = 0.001972
grad_step = 000218, loss = 0.001965
grad_step = 000219, loss = 0.001960
grad_step = 000220, loss = 0.001960
grad_step = 000221, loss = 0.001964
grad_step = 000222, loss = 0.001973
grad_step = 000223, loss = 0.001980
grad_step = 000224, loss = 0.001996
grad_step = 000225, loss = 0.002022
grad_step = 000226, loss = 0.002014
grad_step = 000227, loss = 0.001973
grad_step = 000228, loss = 0.001938
grad_step = 000229, loss = 0.001958
grad_step = 000230, loss = 0.001979
grad_step = 000231, loss = 0.001946
grad_step = 000232, loss = 0.001938
grad_step = 000233, loss = 0.001959
grad_step = 000234, loss = 0.001947
grad_step = 000235, loss = 0.001926
grad_step = 000236, loss = 0.001935
grad_step = 000237, loss = 0.001951
grad_step = 000238, loss = 0.001953
grad_step = 000239, loss = 0.001947
grad_step = 000240, loss = 0.001941
grad_step = 000241, loss = 0.001922
grad_step = 000242, loss = 0.001909
grad_step = 000243, loss = 0.001912
grad_step = 000244, loss = 0.001928
grad_step = 000245, loss = 0.001925
grad_step = 000246, loss = 0.001910
grad_step = 000247, loss = 0.001909
grad_step = 000248, loss = 0.001911
grad_step = 000249, loss = 0.001897
grad_step = 000250, loss = 0.001891
grad_step = 000251, loss = 0.001894
grad_step = 000252, loss = 0.001899
grad_step = 000253, loss = 0.001898
grad_step = 000254, loss = 0.001891
grad_step = 000255, loss = 0.001886
grad_step = 000256, loss = 0.001890
grad_step = 000257, loss = 0.001898
grad_step = 000258, loss = 0.001897
grad_step = 000259, loss = 0.001894
grad_step = 000260, loss = 0.001893
grad_step = 000261, loss = 0.001904
grad_step = 000262, loss = 0.001909
grad_step = 000263, loss = 0.001916
grad_step = 000264, loss = 0.001907
grad_step = 000265, loss = 0.001894
grad_step = 000266, loss = 0.001870
grad_step = 000267, loss = 0.001856
grad_step = 000268, loss = 0.001859
grad_step = 000269, loss = 0.001868
grad_step = 000270, loss = 0.001872
grad_step = 000271, loss = 0.001867
grad_step = 000272, loss = 0.001861
grad_step = 000273, loss = 0.001855
grad_step = 000274, loss = 0.001848
grad_step = 000275, loss = 0.001841
grad_step = 000276, loss = 0.001838
grad_step = 000277, loss = 0.001840
grad_step = 000278, loss = 0.001843
grad_step = 000279, loss = 0.001845
grad_step = 000280, loss = 0.001848
grad_step = 000281, loss = 0.001872
grad_step = 000282, loss = 0.001909
grad_step = 000283, loss = 0.001943
grad_step = 000284, loss = 0.001910
grad_step = 000285, loss = 0.001866
grad_step = 000286, loss = 0.001842
grad_step = 000287, loss = 0.001849
grad_step = 000288, loss = 0.001844
grad_step = 000289, loss = 0.001821
grad_step = 000290, loss = 0.001819
grad_step = 000291, loss = 0.001837
grad_step = 000292, loss = 0.001834
grad_step = 000293, loss = 0.001812
grad_step = 000294, loss = 0.001803
grad_step = 000295, loss = 0.001806
grad_step = 000296, loss = 0.001799
grad_step = 000297, loss = 0.001796
grad_step = 000298, loss = 0.001800
grad_step = 000299, loss = 0.001805
grad_step = 000300, loss = 0.001800
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001795
grad_step = 000302, loss = 0.001780
grad_step = 000303, loss = 0.001775
grad_step = 000304, loss = 0.001771
grad_step = 000305, loss = 0.001771
grad_step = 000306, loss = 0.001786
grad_step = 000307, loss = 0.001798
grad_step = 000308, loss = 0.001827
grad_step = 000309, loss = 0.001828
grad_step = 000310, loss = 0.001833
grad_step = 000311, loss = 0.001814
grad_step = 000312, loss = 0.001825
grad_step = 000313, loss = 0.001832
grad_step = 000314, loss = 0.001814
grad_step = 000315, loss = 0.001765
grad_step = 000316, loss = 0.001743
grad_step = 000317, loss = 0.001770
grad_step = 000318, loss = 0.001792
grad_step = 000319, loss = 0.001776
grad_step = 000320, loss = 0.001741
grad_step = 000321, loss = 0.001732
grad_step = 000322, loss = 0.001751
grad_step = 000323, loss = 0.001766
grad_step = 000324, loss = 0.001744
grad_step = 000325, loss = 0.001725
grad_step = 000326, loss = 0.001729
grad_step = 000327, loss = 0.001757
grad_step = 000328, loss = 0.001780
grad_step = 000329, loss = 0.001822
grad_step = 000330, loss = 0.001826
grad_step = 000331, loss = 0.001823
grad_step = 000332, loss = 0.001734
grad_step = 000333, loss = 0.001695
grad_step = 000334, loss = 0.001714
grad_step = 000335, loss = 0.001740
grad_step = 000336, loss = 0.001736
grad_step = 000337, loss = 0.001692
grad_step = 000338, loss = 0.001692
grad_step = 000339, loss = 0.001718
grad_step = 000340, loss = 0.001695
grad_step = 000341, loss = 0.001673
grad_step = 000342, loss = 0.001684
grad_step = 000343, loss = 0.001690
grad_step = 000344, loss = 0.001676
grad_step = 000345, loss = 0.001663
grad_step = 000346, loss = 0.001657
grad_step = 000347, loss = 0.001662
grad_step = 000348, loss = 0.001670
grad_step = 000349, loss = 0.001679
grad_step = 000350, loss = 0.001678
grad_step = 000351, loss = 0.001680
grad_step = 000352, loss = 0.001668
grad_step = 000353, loss = 0.001654
grad_step = 000354, loss = 0.001637
grad_step = 000355, loss = 0.001627
grad_step = 000356, loss = 0.001625
grad_step = 000357, loss = 0.001626
grad_step = 000358, loss = 0.001627
grad_step = 000359, loss = 0.001632
grad_step = 000360, loss = 0.001642
grad_step = 000361, loss = 0.001651
grad_step = 000362, loss = 0.001674
grad_step = 000363, loss = 0.001677
grad_step = 000364, loss = 0.001681
grad_step = 000365, loss = 0.001649
grad_step = 000366, loss = 0.001619
grad_step = 000367, loss = 0.001602
grad_step = 000368, loss = 0.001599
grad_step = 000369, loss = 0.001596
grad_step = 000370, loss = 0.001595
grad_step = 000371, loss = 0.001593
grad_step = 000372, loss = 0.001583
grad_step = 000373, loss = 0.001581
grad_step = 000374, loss = 0.001586
grad_step = 000375, loss = 0.001582
grad_step = 000376, loss = 0.001568
grad_step = 000377, loss = 0.001560
grad_step = 000378, loss = 0.001575
grad_step = 000379, loss = 0.001599
grad_step = 000380, loss = 0.001702
grad_step = 000381, loss = 0.001788
grad_step = 000382, loss = 0.001914
grad_step = 000383, loss = 0.001727
grad_step = 000384, loss = 0.001592
grad_step = 000385, loss = 0.001586
grad_step = 000386, loss = 0.001647
grad_step = 000387, loss = 0.001624
grad_step = 000388, loss = 0.001541
grad_step = 000389, loss = 0.001593
grad_step = 000390, loss = 0.001616
grad_step = 000391, loss = 0.001558
grad_step = 000392, loss = 0.001569
grad_step = 000393, loss = 0.001589
grad_step = 000394, loss = 0.001538
grad_step = 000395, loss = 0.001544
grad_step = 000396, loss = 0.001550
grad_step = 000397, loss = 0.001524
grad_step = 000398, loss = 0.001518
grad_step = 000399, loss = 0.001535
grad_step = 000400, loss = 0.001525
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001511
grad_step = 000402, loss = 0.001513
grad_step = 000403, loss = 0.001503
grad_step = 000404, loss = 0.001506
grad_step = 000405, loss = 0.001518
grad_step = 000406, loss = 0.001535
grad_step = 000407, loss = 0.001510
grad_step = 000408, loss = 0.001508
grad_step = 000409, loss = 0.001498
grad_step = 000410, loss = 0.001490
grad_step = 000411, loss = 0.001487
grad_step = 000412, loss = 0.001492
grad_step = 000413, loss = 0.001484
grad_step = 000414, loss = 0.001486
grad_step = 000415, loss = 0.001492
grad_step = 000416, loss = 0.001504
grad_step = 000417, loss = 0.001498
grad_step = 000418, loss = 0.001508
grad_step = 000419, loss = 0.001481
grad_step = 000420, loss = 0.001474
grad_step = 000421, loss = 0.001482
grad_step = 000422, loss = 0.001507
grad_step = 000423, loss = 0.001524
grad_step = 000424, loss = 0.001517
grad_step = 000425, loss = 0.001493
grad_step = 000426, loss = 0.001459
grad_step = 000427, loss = 0.001442
grad_step = 000428, loss = 0.001461
grad_step = 000429, loss = 0.001487
grad_step = 000430, loss = 0.001537
grad_step = 000431, loss = 0.001473
grad_step = 000432, loss = 0.001431
grad_step = 000433, loss = 0.001421
grad_step = 000434, loss = 0.001435
grad_step = 000435, loss = 0.001462
grad_step = 000436, loss = 0.001455
grad_step = 000437, loss = 0.001448
grad_step = 000438, loss = 0.001422
grad_step = 000439, loss = 0.001416
grad_step = 000440, loss = 0.001425
grad_step = 000441, loss = 0.001450
grad_step = 000442, loss = 0.001478
grad_step = 000443, loss = 0.001507
grad_step = 000444, loss = 0.001520
grad_step = 000445, loss = 0.001549
grad_step = 000446, loss = 0.001471
grad_step = 000447, loss = 0.001420
grad_step = 000448, loss = 0.001384
grad_step = 000449, loss = 0.001387
grad_step = 000450, loss = 0.001405
grad_step = 000451, loss = 0.001422
grad_step = 000452, loss = 0.001440
grad_step = 000453, loss = 0.001408
grad_step = 000454, loss = 0.001383
grad_step = 000455, loss = 0.001364
grad_step = 000456, loss = 0.001357
grad_step = 000457, loss = 0.001362
grad_step = 000458, loss = 0.001372
grad_step = 000459, loss = 0.001392
grad_step = 000460, loss = 0.001410
grad_step = 000461, loss = 0.001449
grad_step = 000462, loss = 0.001436
grad_step = 000463, loss = 0.001451
grad_step = 000464, loss = 0.001418
grad_step = 000465, loss = 0.001380
grad_step = 000466, loss = 0.001341
grad_step = 000467, loss = 0.001328
grad_step = 000468, loss = 0.001335
grad_step = 000469, loss = 0.001356
grad_step = 000470, loss = 0.001407
grad_step = 000471, loss = 0.001424
grad_step = 000472, loss = 0.001459
grad_step = 000473, loss = 0.001420
grad_step = 000474, loss = 0.001394
grad_step = 000475, loss = 0.001340
grad_step = 000476, loss = 0.001307
grad_step = 000477, loss = 0.001314
grad_step = 000478, loss = 0.001313
grad_step = 000479, loss = 0.001311
grad_step = 000480, loss = 0.001331
grad_step = 000481, loss = 0.001332
grad_step = 000482, loss = 0.001331
grad_step = 000483, loss = 0.001322
grad_step = 000484, loss = 0.001356
grad_step = 000485, loss = 0.001332
grad_step = 000486, loss = 0.001368
grad_step = 000487, loss = 0.001435
grad_step = 000488, loss = 0.001553
grad_step = 000489, loss = 0.001535
grad_step = 000490, loss = 0.001487
grad_step = 000491, loss = 0.001318
grad_step = 000492, loss = 0.001267
grad_step = 000493, loss = 0.001317
grad_step = 000494, loss = 0.001380
grad_step = 000495, loss = 0.001353
grad_step = 000496, loss = 0.001262
grad_step = 000497, loss = 0.001252
grad_step = 000498, loss = 0.001308
grad_step = 000499, loss = 0.001325
grad_step = 000500, loss = 0.001317
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001306
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

  date_run                              2020-05-15 02:29:13.005302
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.195433
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 02:29:13.010420
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0809753
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 02:29:13.017547
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    0.1342
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 02:29:13.022019
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.230449
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
0   2020-05-15 02:28:47.495607  ...    mean_absolute_error
1   2020-05-15 02:28:47.498769  ...     mean_squared_error
2   2020-05-15 02:28:47.501797  ...  median_absolute_error
3   2020-05-15 02:28:47.504499  ...               r2_score
4   2020-05-15 02:28:57.303170  ...    mean_absolute_error
5   2020-05-15 02:28:57.306239  ...     mean_squared_error
6   2020-05-15 02:28:57.308972  ...  median_absolute_error
7   2020-05-15 02:28:57.311329  ...               r2_score
8   2020-05-15 02:29:13.005302  ...    mean_absolute_error
9   2020-05-15 02:29:13.010420  ...     mean_squared_error
10  2020-05-15 02:29:13.017547  ...  median_absolute_error
11  2020-05-15 02:29:13.022019  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8df6fb0fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 313261.17it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 404159.68it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 560693.26it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 792155.27it/s] 77%|███████▋  | 7626752/9912422 [00:00<00:02, 1120119.40it/s]9920512it [00:00, 10621833.98it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 149707.48it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 301504.80it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 391251.56it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 542276.42it/s]1654784it [00:00, 2801386.33it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 53753.44it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8df7003b00> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8da67fe0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8da99b2e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8da8f3a080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8da99b2e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8da675eba8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8da99b2e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8da8ef76d8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8da99b2e48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8da8db1048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fe04eeb21d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=44fe9421ce1b198e10f2e06b6c0604b55c45a6dc192468a7ad80f987c6f67c5e
  Stored in directory: /tmp/pip-ephem-wheel-cache-7iitnywx/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fdfe6cad6a0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 43s
   57344/17464789 [..............................] - ETA: 37s
   90112/17464789 [..............................] - ETA: 35s
  212992/17464789 [..............................] - ETA: 19s
  442368/17464789 [..............................] - ETA: 11s
  892928/17464789 [>.............................] - ETA: 6s 
 1794048/17464789 [==>...........................] - ETA: 3s
 3579904/17464789 [=====>........................] - ETA: 1s
 6594560/17464789 [==========>...................] - ETA: 0s
 9314304/17464789 [==============>...............] - ETA: 0s
12296192/17464789 [====================>.........] - ETA: 0s
15278080/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 02:30:39.050222: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 02:30:39.054380: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095210000 Hz
2020-05-15 02:30:39.054494: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558126bd4b20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 02:30:39.054505: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.5440 - accuracy: 0.5080
 2000/25000 [=>............................] - ETA: 7s - loss: 7.6820 - accuracy: 0.4990 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6922 - accuracy: 0.4983
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7165 - accuracy: 0.4967
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7648 - accuracy: 0.4936
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.8251 - accuracy: 0.4897
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7915 - accuracy: 0.4919
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.8008 - accuracy: 0.4913
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.8029 - accuracy: 0.4911
10000/25000 [===========>..................] - ETA: 3s - loss: 7.8000 - accuracy: 0.4913
11000/25000 [============>.................] - ETA: 3s - loss: 7.8116 - accuracy: 0.4905
12000/25000 [=============>................] - ETA: 2s - loss: 7.7778 - accuracy: 0.4927
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7256 - accuracy: 0.4962
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7181 - accuracy: 0.4966
15000/25000 [=================>............] - ETA: 2s - loss: 7.7157 - accuracy: 0.4968
16000/25000 [==================>...........] - ETA: 1s - loss: 7.7107 - accuracy: 0.4971
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7018 - accuracy: 0.4977
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7015 - accuracy: 0.4977
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6820 - accuracy: 0.4990
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6766 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6674 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6875 - accuracy: 0.4986
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6746 - accuracy: 0.4995
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6628 - accuracy: 0.5002
25000/25000 [==============================] - 6s 245us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 02:30:50.632350
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 02:30:50.632350  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<9:44:06, 24.6kB/s].vector_cache/glove.6B.zip:   0%|          | 401k/862M [00:00<6:50:01, 35.0kB/s] .vector_cache/glove.6B.zip:   1%|          | 5.56M/862M [00:00<4:45:22, 50.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.7M/862M [00:00<3:17:27, 71.5kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.7M/862M [00:00<2:16:47, 102kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 33.7M/862M [00:00<1:34:46, 146kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.5M/862M [00:00<1:05:35, 208kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:01<45:43, 295kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:03<33:45, 398kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:03<24:29, 548kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:05<19:08, 698kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:05<14:14, 937kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:07<11:57, 1.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:07<09:00, 1.47MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:09<08:23, 1.58MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:09<06:37, 1.99MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:11<06:39, 1.97MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:11<05:14, 2.51MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:13<05:46, 2.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:13<04:45, 2.75MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:15<05:20, 2.43MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:15<04:25, 2.94MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:15<03:37, 3.57MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<8:34:51, 25.2kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.9M/862M [00:17<5:59:03, 35.9kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:19<4:18:15, 49.9kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:19<3:01:44, 70.9kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:20<2:08:32, 99.7kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:21<1:30:30, 142kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:22<1:05:06, 196kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:23<46:13, 276kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<34:11, 371kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<24:22, 520kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<19:07, 660kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<13:51, 909kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<11:45, 1.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:28<08:38, 1.45MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<08:11, 1.52MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:30<06:11, 2.01MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<06:24, 1.94MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<05:09, 2.40MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<05:31, 2.23MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:34<04:17, 2.87MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<05:05, 2.41MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<04:05, 3.00MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<04:50, 2.52MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<03:49, 3.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<04:43, 2.57MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<03:43, 3.25MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<04:40, 2.58MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<03:43, 3.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<04:36, 2.60MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:44<03:38, 3.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<04:34, 2.61MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:46<03:37, 3.29MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:48<04:32, 2.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:48<03:53, 3.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<04:31, 2.60MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<04:43, 2.49MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:50<03:22, 3.47MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<2:33:54, 76.1kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<1:48:37, 108kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<1:17:23, 150kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<55:00, 212kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<40:03, 289kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<28:58, 399kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<21:54, 525kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<16:13, 709kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<13:01, 878kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<10:06, 1.13MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:00<07:25, 1.53MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<7:55:08, 24.0kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<5:31:42, 34.2kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<3:54:41, 48.2kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<2:45:06, 68.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<1:56:38, 96.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<1:22:28, 136kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<59:06, 189kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<42:11, 265kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<31:03, 358kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<22:33, 492kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<17:22, 635kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<12:54, 854kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<10:40, 1.03MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<08:13, 1.33MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<07:24, 1.47MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:15<06:04, 1.79MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<05:51, 1.85MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<04:56, 2.19MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<05:04, 2.12MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:19<04:16, 2.52MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:21<04:37, 2.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<04:06, 2.60MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<04:27, 2.38MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<03:54, 2.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<04:19, 2.44MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:50, 2.74MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<04:15, 2.46MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<03:34, 2.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:29<04:06, 2.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:29<03:45, 2.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<04:09, 2.49MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<03:27, 2.99MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<04:01, 2.55MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:33<03:31, 2.91MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<04:00, 2.55MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<03:35, 2.84MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<04:02, 2.51MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<03:29, 2.91MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:37<02:49, 3.57MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<6:36:12, 25.5kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:39<4:36:27, 36.4kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:40<3:15:49, 51.2kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<2:17:47, 72.7kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<1:37:21, 102kB/s] .vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<1:08:50, 144kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<49:23, 200kB/s]  .vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<35:17, 280kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<26:02, 377kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<18:57, 517kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:48<14:40, 664kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<11:00, 884kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<09:07, 1.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<07:07, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<06:24, 1.50MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<05:09, 1.86MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<05:02, 1.89MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<04:13, 2.26MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<04:22, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<03:48, 2.49MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:58<04:04, 2.31MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:58<03:34, 2.63MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:00<03:54, 2.39MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:00<03:20, 2.79MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:02<03:44, 2.48MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<03:20, 2.77MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<03:42, 2.48MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<03:25, 2.69MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:06<03:43, 2.45MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<03:18, 2.75MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<03:40, 2.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<03:16, 2.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<03:38, 2.47MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<03:13, 2.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<03:35, 2.48MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<03:12, 2.78MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<03:34, 2.48MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:14<02:58, 2.98MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:14<02:23, 3.68MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:15<6:02:02, 24.3kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:16<4:12:26, 34.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<2:58:42, 48.8kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<2:05:42, 69.4kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:19<1:28:40, 97.7kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<1:02:34, 138kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:21<44:48, 192kB/s]  .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<32:19, 266kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<23:41, 360kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<17:07, 497kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<13:11, 641kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<09:52, 856kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<08:07, 1.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:27<06:20, 1.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<05:58, 1.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<04:52, 1.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:37, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<03:55, 2.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<03:05, 2.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<02:36, 3.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:32<02:09, 3.80MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:32<01:55, 4.26MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:32<01:44, 4.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<12:26, 657kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<09:29, 861kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<06:58, 1.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<05:13, 1.56MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<03:57, 2.06MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:34<03:07, 2.59MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<09:40, 839kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<07:34, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<05:36, 1.44MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<04:11, 1.93MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:36<03:17, 2.45MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:36<02:36, 3.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<20:21, 395kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<14:59, 536kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<10:43, 748kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<07:49, 1.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:38<05:46, 1.38MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:38<04:18, 1.85MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<54:19, 147kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<38:45, 206kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<27:18, 291kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:40<19:22, 409kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:40<13:46, 575kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<14:05, 561kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<10:35, 745kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<07:38, 1.03MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:42<05:31, 1.42MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:42<04:08, 1.89MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<12:18, 637kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<09:19, 839kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<06:44, 1.16MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<04:57, 1.57MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:44<03:41, 2.11MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<12:32, 619kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<09:28, 818kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<06:48, 1.14MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:46<05:00, 1.54MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:46<03:43, 2.07MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<23:49, 323kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<17:20, 443kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<12:17, 624kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<08:49, 867kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:48<06:22, 1.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<17:01, 448kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<12:33, 607kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<08:58, 847kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:50<06:27, 1.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<08:10, 924kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<06:20, 1.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<04:37, 1.63MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<03:23, 2.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<06:32, 1.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<05:11, 1.44MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<03:48, 1.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:54<02:50, 2.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<06:55, 1.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<05:26, 1.36MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<03:58, 1.86MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:56<02:56, 2.50MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<07:23, 995kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<05:46, 1.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<04:10, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<03:05, 2.36MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<06:00, 1.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<04:49, 1.51MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [02:59<03:29, 2.07MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<02:45, 2.63MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:00<02:05, 3.45MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<52:15, 138kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<37:08, 194kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:01<26:07, 275kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<18:25, 389kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<15:55, 449kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<11:43, 609kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:03<08:21, 852kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<06:02, 1.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<07:03, 1.00MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<05:31, 1.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:05<04:01, 1.75MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<02:58, 2.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<05:51, 1.20MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:07<04:40, 1.50MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:07<03:23, 2.05MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:07<02:34, 2.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<05:14, 1.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<04:14, 1.64MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:09<03:05, 2.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:09<02:20, 2.95MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<04:58, 1.38MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<04:02, 1.70MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:11<02:59, 2.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:11<02:14, 3.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<07:36, 893kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<05:48, 1.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:13<04:13, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:13<03:04, 2.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<06:08, 1.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<04:50, 1.39MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:15<03:30, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:15<02:37, 2.55MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<06:03, 1.10MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<04:47, 1.39MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:17<03:28, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:17<02:32, 2.59MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<03:15, 2.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<5:23:30, 20.4kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<3:46:46, 29.1kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:19<2:38:13, 41.5kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:19<1:50:31, 59.2kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<1:21:45, 79.8kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<57:41, 113kB/s]   .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:21<40:20, 161kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:21<28:17, 229kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<24:35, 263kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<17:43, 364kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:23<12:28, 515kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:23<08:51, 723kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<11:15, 567kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<08:20, 766kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:25<05:57, 1.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:25<04:18, 1.47MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<06:30, 971kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<05:04, 1.25MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:27<03:38, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:27<02:41, 2.33MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<08:26, 741kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<06:21, 982kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<04:35, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:29<03:17, 1.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<09:36, 644kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<07:24, 834kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:31<05:17, 1.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:31<03:48, 1.61MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<06:59, 873kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<05:23, 1.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:33<03:51, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:33<02:49, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<10:59, 550kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<08:10, 738kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<05:48, 1.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:35<04:10, 1.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<14:47, 404kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<10:49, 551kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<07:37, 778kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:37<05:26, 1.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<47:49, 124kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<33:59, 174kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<23:48, 247kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:39<16:39, 351kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<27:40, 211kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<19:48, 295kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<13:56, 417kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:41<09:48, 590kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<11:36, 497kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<08:35, 671kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<06:05, 942kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:43<04:19, 1.32MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<24:28, 233kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<17:35, 324kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<12:20, 459kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:45<08:41, 649kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<54:33, 103kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<38:35, 146kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<26:59, 207kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:47<18:51, 295kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:48<3:00:50, 30.8kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<2:06:52, 43.8kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<1:28:26, 62.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:49<1:01:38, 89.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<1:24:31, 65.0kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<59:32, 92.2kB/s]  .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<41:35, 131kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:51<29:01, 187kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<32:55, 165kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<23:28, 231kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:53<16:25, 328kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<12:57, 414kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<09:29, 564kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<06:41, 795kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:55<04:44, 1.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<1:49:37, 48.2kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<1:17:03, 68.5kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<53:45, 97.7kB/s]  .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<38:43, 135kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<27:30, 190kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:58<19:11, 270kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<15:05, 341kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<10:54, 472kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:00<07:41, 665kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<06:45, 751kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<05:04, 999kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:02<03:37, 1.39MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<03:56, 1.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<03:26, 1.46MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:04<02:28, 2.01MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<02:59, 1.65MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<02:41, 1.84MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<01:57, 2.50MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:06<01:26, 3.40MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<12:01, 405kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<08:49, 551kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:08<06:11, 780kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<06:01, 798kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<04:36, 1.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:10<03:17, 1.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<03:41, 1.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<03:07, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:12<02:15, 2.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:12<01:37, 2.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<41:52, 112kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<29:45, 157kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:14<20:45, 223kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<15:44, 292kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<11:21, 404kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:16<07:58, 572kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<06:50, 662kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<05:07, 882kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:18<03:37, 1.24MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<02:49, 1.58MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<3:50:02, 19.4kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<2:41:01, 27.7kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:20<1:51:43, 39.5kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<1:19:54, 55.0kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<56:12, 78.1kB/s]  .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:22<39:00, 111kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<29:14, 148kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<20:47, 208kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:24<14:26, 296kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<12:30, 340kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<09:05, 468kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:26<06:20, 663kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<06:35, 635kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<04:55, 848kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:28<03:28, 1.19MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<03:44, 1.10MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<02:57, 1.39MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:30<02:05, 1.95MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<03:52, 1.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<03:02, 1.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:32<02:08, 1.86MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<04:10, 953kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<03:15, 1.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:34<02:17, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<04:30, 869kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<03:28, 1.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:36<02:26, 1.58MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<04:25, 867kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<03:24, 1.12MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:38<02:23, 1.58MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<05:13, 723kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<03:58, 950kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:40<02:46, 1.34MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<05:25, 683kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<04:05, 903kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:42<02:51, 1.28MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<06:56, 524kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<05:08, 705kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:44<03:34, 998kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<07:27, 478kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<05:30, 646kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:46<03:50, 915kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<06:40, 524kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<04:56, 706kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:48<03:26, 1.00MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<24:20, 141kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<17:16, 198kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:50<11:54, 282kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<44:10, 76.1kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<31:07, 108kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<21:53, 151kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<15:33, 211kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<11:10, 289kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<08:04, 399kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<06:00, 525kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<04:27, 706kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<03:31, 876kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<02:43, 1.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<02:19, 1.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:52, 1.61MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<01:43, 1.71MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<01:37, 1.81MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:03<01:08, 2.53MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<02:20, 1.23MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:05<01:52, 1.54MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:42, 1.65MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:07<01:25, 1.97MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<01:23, 1.98MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<01:11, 2.29MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<01:13, 2.19MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<01:04, 2.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<01:07, 2.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<00:59, 2.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<01:03, 2.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<00:57, 2.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<01:01, 2.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<00:54, 2.70MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:17<00:41, 3.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<2:10:19, 18.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<1:30:54, 26.3kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<1:02:28, 37.3kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<43:47, 53.1kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<30:10, 75.0kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<21:14, 106kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<14:47, 148kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<10:30, 208kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<07:27, 285kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<05:22, 394kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<03:57, 519kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<02:56, 699kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<02:17, 868kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:45, 1.13MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:28, 1.30MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:14, 1.55MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:33<00:51, 2.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<08:34, 216kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<06:07, 301kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<04:25, 403kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<03:13, 550kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<02:26, 703kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<01:48, 940kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:28, 1.11MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:08, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<01:00, 1.55MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:50, 1.87MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:44<00:47, 1.90MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:40, 2.23MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<00:40, 2.15MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:34, 2.46MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:35, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:29, 2.76MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:32, 2.43MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:28, 2.70MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:30, 2.44MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:52<00:26, 2.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:28, 2.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:25, 2.73MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:26, 2.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:23, 2.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:24, 2.46MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:21, 2.78MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:23, 2.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:20, 2.73MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:21, 2.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:17, 2.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:19, 2.52MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:04<00:17, 2.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:17, 2.51MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:14, 3.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:15, 2.57MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:13, 2.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:14, 2.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:12, 2.91MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:12<00:12, 2.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:11, 2.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:11, 2.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:09, 2.90MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:09, 2.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:08, 2.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:08, 2.52MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:06, 2.89MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:18<00:04, 3.72MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<14:49, 18.2kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<09:49, 25.9kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<05:26, 36.8kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<03:39, 52.4kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<01:46, 73.9kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<01:10, 105kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:25, 146kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:15, 206kB/s].vector_cache/glove.6B.zip: 862MB [06:26, 2.23MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 903/400000 [00:00<00:44, 8911.62it/s]  0%|          | 1835/400000 [00:00<00:44, 9028.15it/s]  1%|          | 2799/400000 [00:00<00:43, 9202.14it/s]  1%|          | 3794/400000 [00:00<00:42, 9414.27it/s]  1%|          | 4775/400000 [00:00<00:41, 9527.30it/s]  1%|▏         | 5653/400000 [00:00<00:42, 9288.69it/s]  2%|▏         | 6626/400000 [00:00<00:41, 9415.46it/s]  2%|▏         | 7610/400000 [00:00<00:41, 9536.49it/s]  2%|▏         | 8635/400000 [00:00<00:40, 9738.95it/s]  2%|▏         | 9642/400000 [00:01<00:39, 9833.52it/s]  3%|▎         | 10643/400000 [00:01<00:39, 9884.14it/s]  3%|▎         | 11616/400000 [00:01<00:39, 9831.02it/s]  3%|▎         | 12588/400000 [00:01<00:39, 9698.39it/s]  3%|▎         | 13551/400000 [00:01<00:40, 9527.38it/s]  4%|▎         | 14542/400000 [00:01<00:39, 9637.23it/s]  4%|▍         | 15604/400000 [00:01<00:38, 9911.90it/s]  4%|▍         | 16596/400000 [00:01<00:38, 9850.44it/s]  4%|▍         | 17582/400000 [00:01<00:39, 9770.21it/s]  5%|▍         | 18611/400000 [00:01<00:38, 9919.80it/s]  5%|▍         | 19700/400000 [00:02<00:37, 10191.05it/s]  5%|▌         | 20722/400000 [00:02<00:37, 10197.13it/s]  5%|▌         | 21744/400000 [00:02<00:37, 10107.91it/s]  6%|▌         | 22757/400000 [00:02<00:37, 9939.09it/s]   6%|▌         | 23811/400000 [00:02<00:37, 10110.00it/s]  6%|▌         | 24824/400000 [00:02<00:37, 10107.83it/s]  6%|▋         | 25837/400000 [00:02<00:36, 10112.59it/s]  7%|▋         | 26850/400000 [00:02<00:37, 9993.44it/s]   7%|▋         | 27851/400000 [00:02<00:38, 9761.70it/s]  7%|▋         | 28830/400000 [00:02<00:38, 9569.50it/s]  7%|▋         | 29813/400000 [00:03<00:38, 9644.16it/s]  8%|▊         | 30833/400000 [00:03<00:37, 9723.11it/s]  8%|▊         | 31827/400000 [00:03<00:37, 9785.13it/s]  8%|▊         | 32815/400000 [00:03<00:37, 9801.31it/s]  8%|▊         | 33796/400000 [00:03<00:37, 9760.91it/s]  9%|▊         | 34773/400000 [00:03<00:38, 9585.07it/s]  9%|▉         | 35819/400000 [00:03<00:37, 9829.10it/s]  9%|▉         | 36805/400000 [00:03<00:37, 9800.84it/s]  9%|▉         | 37810/400000 [00:03<00:36, 9872.43it/s] 10%|▉         | 38799/400000 [00:03<00:36, 9831.71it/s] 10%|▉         | 39897/400000 [00:04<00:35, 10147.04it/s] 10%|█         | 40915/400000 [00:04<00:35, 10131.27it/s] 10%|█         | 41931/400000 [00:04<00:35, 10108.09it/s] 11%|█         | 42995/400000 [00:04<00:34, 10260.81it/s] 11%|█         | 44023/400000 [00:04<00:35, 10156.58it/s] 11%|█▏        | 45117/400000 [00:04<00:34, 10378.35it/s] 12%|█▏        | 46158/400000 [00:04<00:34, 10213.46it/s] 12%|█▏        | 47182/400000 [00:04<00:35, 9950.32it/s]  12%|█▏        | 48189/400000 [00:04<00:35, 9982.52it/s] 12%|█▏        | 49190/400000 [00:04<00:35, 9938.65it/s] 13%|█▎        | 50232/400000 [00:05<00:34, 10073.29it/s] 13%|█▎        | 51324/400000 [00:05<00:33, 10310.53it/s] 13%|█▎        | 52358/400000 [00:05<00:34, 10223.51it/s] 13%|█▎        | 53424/400000 [00:05<00:33, 10348.96it/s] 14%|█▎        | 54524/400000 [00:05<00:32, 10533.99it/s] 14%|█▍        | 55590/400000 [00:05<00:32, 10571.21it/s] 14%|█▍        | 56695/400000 [00:05<00:32, 10709.82it/s] 14%|█▍        | 57768/400000 [00:05<00:32, 10624.06it/s] 15%|█▍        | 58832/400000 [00:05<00:32, 10483.86it/s] 15%|█▍        | 59882/400000 [00:05<00:32, 10457.00it/s] 15%|█▌        | 60947/400000 [00:06<00:32, 10513.55it/s] 16%|█▌        | 62006/400000 [00:06<00:32, 10535.75it/s] 16%|█▌        | 63061/400000 [00:06<00:32, 10369.23it/s] 16%|█▌        | 64099/400000 [00:06<00:32, 10354.11it/s] 16%|█▋        | 65136/400000 [00:06<00:32, 10223.01it/s] 17%|█▋        | 66213/400000 [00:06<00:32, 10379.75it/s] 17%|█▋        | 67272/400000 [00:06<00:31, 10440.74it/s] 17%|█▋        | 68317/400000 [00:06<00:32, 10289.29it/s] 17%|█▋        | 69348/400000 [00:06<00:32, 10187.41it/s] 18%|█▊        | 70386/400000 [00:07<00:32, 10243.77it/s] 18%|█▊        | 71433/400000 [00:07<00:31, 10307.54it/s] 18%|█▊        | 72530/400000 [00:07<00:31, 10496.58it/s] 18%|█▊        | 73581/400000 [00:07<00:31, 10429.02it/s] 19%|█▊        | 74627/400000 [00:07<00:31, 10438.25it/s] 19%|█▉        | 75672/400000 [00:07<00:31, 10411.87it/s] 19%|█▉        | 76742/400000 [00:07<00:30, 10495.40it/s] 19%|█▉        | 77817/400000 [00:07<00:30, 10568.31it/s] 20%|█▉        | 78875/400000 [00:07<00:30, 10442.97it/s] 20%|█▉        | 79929/400000 [00:07<00:30, 10469.15it/s] 20%|██        | 80977/400000 [00:08<00:32, 9938.79it/s]  21%|██        | 82025/400000 [00:08<00:31, 10094.73it/s] 21%|██        | 83047/400000 [00:08<00:31, 10131.32it/s] 21%|██        | 84064/400000 [00:08<00:31, 10130.24it/s] 21%|██▏       | 85108/400000 [00:08<00:30, 10219.36it/s] 22%|██▏       | 86132/400000 [00:08<00:30, 10175.75it/s] 22%|██▏       | 87151/400000 [00:08<00:31, 10015.43it/s] 22%|██▏       | 88155/400000 [00:08<00:31, 9798.48it/s]  22%|██▏       | 89137/400000 [00:08<00:31, 9754.54it/s] 23%|██▎       | 90187/400000 [00:08<00:31, 9966.67it/s] 23%|██▎       | 91225/400000 [00:09<00:30, 10086.55it/s] 23%|██▎       | 92307/400000 [00:09<00:29, 10293.09it/s] 23%|██▎       | 93339/400000 [00:09<00:30, 10149.97it/s] 24%|██▎       | 94357/400000 [00:09<00:30, 10147.22it/s] 24%|██▍       | 95446/400000 [00:09<00:29, 10357.80it/s] 24%|██▍       | 96513/400000 [00:09<00:29, 10447.48it/s] 24%|██▍       | 97596/400000 [00:09<00:28, 10557.24it/s] 25%|██▍       | 98656/400000 [00:09<00:28, 10567.74it/s] 25%|██▍       | 99714/400000 [00:09<00:29, 10330.39it/s] 25%|██▌       | 100749/400000 [00:09<00:29, 10213.04it/s] 25%|██▌       | 101776/400000 [00:10<00:29, 10229.21it/s] 26%|██▌       | 102801/400000 [00:10<00:29, 10051.82it/s] 26%|██▌       | 103818/400000 [00:10<00:29, 10084.97it/s] 26%|██▌       | 104828/400000 [00:10<00:29, 9927.73it/s]  26%|██▋       | 105865/400000 [00:10<00:29, 10054.63it/s] 27%|██▋       | 106897/400000 [00:10<00:28, 10132.23it/s] 27%|██▋       | 107946/400000 [00:10<00:28, 10234.30it/s] 27%|██▋       | 108971/400000 [00:10<00:28, 10065.24it/s] 27%|██▋       | 109979/400000 [00:10<00:29, 9790.63it/s]  28%|██▊       | 110961/400000 [00:11<00:29, 9766.46it/s] 28%|██▊       | 111940/400000 [00:11<00:30, 9600.44it/s] 28%|██▊       | 112928/400000 [00:11<00:29, 9679.83it/s] 28%|██▊       | 113936/400000 [00:11<00:29, 9793.72it/s] 29%|██▊       | 114917/400000 [00:11<00:29, 9679.69it/s] 29%|██▉       | 115916/400000 [00:11<00:29, 9765.21it/s] 29%|██▉       | 116955/400000 [00:11<00:28, 9944.56it/s] 30%|██▉       | 118063/400000 [00:11<00:27, 10258.29it/s] 30%|██▉       | 119101/400000 [00:11<00:27, 10291.96it/s] 30%|███       | 120133/400000 [00:11<00:27, 10028.10it/s] 30%|███       | 121147/400000 [00:12<00:27, 10060.01it/s] 31%|███       | 122203/400000 [00:12<00:27, 10203.45it/s] 31%|███       | 123226/400000 [00:12<00:27, 10197.90it/s] 31%|███       | 124287/400000 [00:12<00:26, 10316.73it/s] 31%|███▏      | 125331/400000 [00:12<00:26, 10351.03it/s] 32%|███▏      | 126402/400000 [00:12<00:26, 10455.96it/s] 32%|███▏      | 127488/400000 [00:12<00:25, 10573.75it/s] 32%|███▏      | 128593/400000 [00:12<00:25, 10711.55it/s] 32%|███▏      | 129666/400000 [00:12<00:25, 10657.70it/s] 33%|███▎      | 130733/400000 [00:12<00:25, 10420.59it/s] 33%|███▎      | 131777/400000 [00:13<00:26, 10288.04it/s] 33%|███▎      | 132848/400000 [00:13<00:25, 10409.09it/s] 33%|███▎      | 133891/400000 [00:13<00:25, 10246.73it/s] 34%|███▎      | 134918/400000 [00:13<00:26, 10156.99it/s] 34%|███▍      | 135935/400000 [00:13<00:26, 10069.70it/s] 34%|███▍      | 136985/400000 [00:13<00:25, 10193.09it/s] 35%|███▍      | 138053/400000 [00:13<00:25, 10332.61it/s] 35%|███▍      | 139088/400000 [00:13<00:25, 10313.05it/s] 35%|███▌      | 140121/400000 [00:13<00:25, 10251.29it/s] 35%|███▌      | 141147/400000 [00:13<00:26, 9911.71it/s]  36%|███▌      | 142142/400000 [00:14<00:26, 9771.62it/s] 36%|███▌      | 143122/400000 [00:14<00:26, 9714.01it/s] 36%|███▌      | 144101/400000 [00:14<00:26, 9735.98it/s] 36%|███▋      | 145076/400000 [00:14<00:26, 9640.34it/s] 37%|███▋      | 146057/400000 [00:14<00:26, 9687.97it/s] 37%|███▋      | 147141/400000 [00:14<00:25, 10005.24it/s] 37%|███▋      | 148216/400000 [00:14<00:24, 10216.80it/s] 37%|███▋      | 149287/400000 [00:14<00:24, 10356.89it/s] 38%|███▊      | 150326/400000 [00:14<00:24, 10252.55it/s] 38%|███▊      | 151354/400000 [00:14<00:25, 9765.12it/s]  38%|███▊      | 152346/400000 [00:15<00:25, 9808.91it/s] 38%|███▊      | 153332/400000 [00:15<00:25, 9741.97it/s] 39%|███▊      | 154367/400000 [00:15<00:24, 9914.34it/s] 39%|███▉      | 155490/400000 [00:15<00:23, 10273.23it/s] 39%|███▉      | 156523/400000 [00:15<00:24, 10103.84it/s] 39%|███▉      | 157542/400000 [00:15<00:23, 10128.03it/s] 40%|███▉      | 158594/400000 [00:15<00:23, 10240.91it/s] 40%|███▉      | 159651/400000 [00:15<00:23, 10336.35it/s] 40%|████      | 160707/400000 [00:15<00:23, 10401.27it/s] 40%|████      | 161749/400000 [00:16<00:23, 10272.69it/s] 41%|████      | 162808/400000 [00:16<00:22, 10363.94it/s] 41%|████      | 163846/400000 [00:16<00:22, 10344.15it/s] 41%|████      | 164887/400000 [00:16<00:22, 10363.25it/s] 41%|████▏     | 165976/400000 [00:16<00:22, 10513.55it/s] 42%|████▏     | 167029/400000 [00:16<00:22, 10438.96it/s] 42%|████▏     | 168074/400000 [00:16<00:22, 10269.15it/s] 42%|████▏     | 169103/400000 [00:16<00:22, 10237.55it/s] 43%|████▎     | 170162/400000 [00:16<00:22, 10339.39it/s] 43%|████▎     | 171197/400000 [00:16<00:22, 10220.75it/s] 43%|████▎     | 172220/400000 [00:17<00:22, 10055.25it/s] 43%|████▎     | 173227/400000 [00:17<00:22, 10055.02it/s] 44%|████▎     | 174283/400000 [00:17<00:22, 10199.99it/s] 44%|████▍     | 175323/400000 [00:17<00:21, 10257.61it/s] 44%|████▍     | 176359/400000 [00:17<00:21, 10285.40it/s] 44%|████▍     | 177389/400000 [00:17<00:22, 10101.85it/s] 45%|████▍     | 178480/400000 [00:17<00:21, 10330.45it/s] 45%|████▍     | 179570/400000 [00:17<00:21, 10492.76it/s] 45%|████▌     | 180631/400000 [00:17<00:20, 10526.67it/s] 45%|████▌     | 181686/400000 [00:17<00:20, 10504.88it/s] 46%|████▌     | 182738/400000 [00:18<00:21, 10321.67it/s] 46%|████▌     | 183772/400000 [00:18<00:21, 9946.28it/s]  46%|████▌     | 184771/400000 [00:18<00:21, 9892.29it/s] 46%|████▋     | 185777/400000 [00:18<00:21, 9939.13it/s] 47%|████▋     | 186773/400000 [00:18<00:21, 9843.43it/s] 47%|████▋     | 187811/400000 [00:18<00:21, 9995.44it/s] 47%|████▋     | 188828/400000 [00:18<00:21, 10042.85it/s] 47%|████▋     | 189852/400000 [00:18<00:20, 10100.59it/s] 48%|████▊     | 190965/400000 [00:18<00:20, 10385.74it/s] 48%|████▊     | 192082/400000 [00:18<00:19, 10607.80it/s] 48%|████▊     | 193160/400000 [00:19<00:19, 10657.86it/s] 49%|████▊     | 194278/400000 [00:19<00:19, 10806.28it/s] 49%|████▉     | 195361/400000 [00:19<00:19, 10373.40it/s] 49%|████▉     | 196404/400000 [00:19<00:20, 9933.53it/s]  49%|████▉     | 197421/400000 [00:19<00:20, 10002.24it/s] 50%|████▉     | 198503/400000 [00:19<00:19, 10234.04it/s] 50%|████▉     | 199599/400000 [00:19<00:19, 10440.09it/s] 50%|█████     | 200679/400000 [00:19<00:18, 10543.13it/s] 50%|█████     | 201737/400000 [00:19<00:19, 10170.01it/s] 51%|█████     | 202760/400000 [00:20<00:19, 9970.39it/s]  51%|█████     | 203828/400000 [00:20<00:19, 10171.89it/s] 51%|█████     | 204877/400000 [00:20<00:19, 10263.75it/s] 51%|█████▏    | 205941/400000 [00:20<00:18, 10371.90it/s] 52%|█████▏    | 206981/400000 [00:20<00:18, 10375.76it/s] 52%|█████▏    | 208021/400000 [00:20<00:18, 10350.50it/s] 52%|█████▏    | 209104/400000 [00:20<00:18, 10487.56it/s] 53%|█████▎    | 210155/400000 [00:20<00:18, 10440.93it/s] 53%|█████▎    | 211207/400000 [00:20<00:18, 10462.56it/s] 53%|█████▎    | 212254/400000 [00:20<00:18, 10323.15it/s] 53%|█████▎    | 213288/400000 [00:21<00:18, 10131.17it/s] 54%|█████▎    | 214369/400000 [00:21<00:17, 10324.23it/s] 54%|█████▍    | 215441/400000 [00:21<00:17, 10439.33it/s] 54%|█████▍    | 216547/400000 [00:21<00:17, 10617.73it/s] 54%|█████▍    | 217675/400000 [00:21<00:16, 10806.06it/s] 55%|█████▍    | 218758/400000 [00:21<00:17, 10515.94it/s] 55%|█████▍    | 219813/400000 [00:21<00:17, 10341.90it/s] 55%|█████▌    | 220894/400000 [00:21<00:17, 10476.61it/s] 55%|█████▌    | 221987/400000 [00:21<00:16, 10606.75it/s] 56%|█████▌    | 223050/400000 [00:21<00:16, 10594.91it/s] 56%|█████▌    | 224111/400000 [00:22<00:16, 10447.39it/s] 56%|█████▋    | 225168/400000 [00:22<00:16, 10480.54it/s] 57%|█████▋    | 226221/400000 [00:22<00:16, 10493.52it/s] 57%|█████▋    | 227329/400000 [00:22<00:16, 10662.01it/s] 57%|█████▋    | 228445/400000 [00:22<00:15, 10806.48it/s] 57%|█████▋    | 229527/400000 [00:22<00:15, 10699.62it/s] 58%|█████▊    | 230599/400000 [00:22<00:16, 10530.12it/s] 58%|█████▊    | 231670/400000 [00:22<00:15, 10582.88it/s] 58%|█████▊    | 232788/400000 [00:22<00:15, 10752.93it/s] 58%|█████▊    | 233883/400000 [00:22<00:15, 10809.64it/s] 59%|█████▊    | 234966/400000 [00:23<00:15, 10378.18it/s] 59%|█████▉    | 236057/400000 [00:23<00:15, 10530.65it/s] 59%|█████▉    | 237114/400000 [00:23<00:16, 10173.67it/s] 60%|█████▉    | 238137/400000 [00:23<00:16, 10097.55it/s] 60%|█████▉    | 239160/400000 [00:23<00:15, 10135.89it/s] 60%|██████    | 240177/400000 [00:23<00:15, 10016.81it/s] 60%|██████    | 241247/400000 [00:23<00:15, 10210.31it/s] 61%|██████    | 242292/400000 [00:23<00:15, 10278.31it/s] 61%|██████    | 243322/400000 [00:23<00:15, 10227.27it/s] 61%|██████    | 244347/400000 [00:23<00:15, 10194.19it/s] 61%|██████▏   | 245390/400000 [00:24<00:15, 10263.52it/s] 62%|██████▏   | 246488/400000 [00:24<00:14, 10465.45it/s] 62%|██████▏   | 247537/400000 [00:24<00:14, 10443.35it/s] 62%|██████▏   | 248614/400000 [00:24<00:14, 10536.65it/s] 62%|██████▏   | 249669/400000 [00:24<00:14, 10113.75it/s] 63%|██████▎   | 250685/400000 [00:24<00:15, 9874.11it/s]  63%|██████▎   | 251677/400000 [00:24<00:15, 9871.82it/s] 63%|██████▎   | 252685/400000 [00:24<00:14, 9931.12it/s] 63%|██████▎   | 253681/400000 [00:24<00:14, 9843.95it/s] 64%|██████▎   | 254677/400000 [00:25<00:14, 9875.63it/s] 64%|██████▍   | 255666/400000 [00:25<00:14, 9794.06it/s] 64%|██████▍   | 256647/400000 [00:25<00:14, 9674.34it/s] 64%|██████▍   | 257677/400000 [00:25<00:14, 9853.77it/s] 65%|██████▍   | 258766/400000 [00:25<00:13, 10143.25it/s] 65%|██████▍   | 259862/400000 [00:25<00:13, 10374.03it/s] 65%|██████▌   | 260903/400000 [00:25<00:13, 10112.93it/s] 65%|██████▌   | 261973/400000 [00:25<00:13, 10280.84it/s] 66%|██████▌   | 263030/400000 [00:25<00:13, 10365.27it/s] 66%|██████▌   | 264070/400000 [00:25<00:13, 10216.58it/s] 66%|██████▋   | 265094/400000 [00:26<00:13, 10163.57it/s] 67%|██████▋   | 266113/400000 [00:26<00:13, 10156.32it/s] 67%|██████▋   | 267146/400000 [00:26<00:13, 10207.80it/s] 67%|██████▋   | 268168/400000 [00:26<00:12, 10211.24it/s] 67%|██████▋   | 269286/400000 [00:26<00:12, 10481.28it/s] 68%|██████▊   | 270337/400000 [00:26<00:12, 10210.85it/s] 68%|██████▊   | 271362/400000 [00:26<00:12, 10041.72it/s] 68%|██████▊   | 272369/400000 [00:26<00:12, 10010.52it/s] 68%|██████▊   | 273409/400000 [00:26<00:12, 10124.00it/s] 69%|██████▊   | 274428/400000 [00:26<00:12, 10142.36it/s] 69%|██████▉   | 275497/400000 [00:27<00:12, 10299.16it/s] 69%|██████▉   | 276529/400000 [00:27<00:12, 10116.75it/s] 69%|██████▉   | 277602/400000 [00:27<00:11, 10292.36it/s] 70%|██████▉   | 278634/400000 [00:27<00:11, 10198.50it/s] 70%|██████▉   | 279656/400000 [00:27<00:11, 10196.34it/s] 70%|███████   | 280677/400000 [00:27<00:11, 10149.57it/s] 70%|███████   | 281693/400000 [00:27<00:12, 9812.73it/s]  71%|███████   | 282678/400000 [00:27<00:12, 9636.18it/s] 71%|███████   | 283645/400000 [00:27<00:12, 9470.12it/s] 71%|███████   | 284595/400000 [00:27<00:12, 9398.03it/s] 71%|███████▏  | 285537/400000 [00:28<00:12, 9085.43it/s] 72%|███████▏  | 286464/400000 [00:28<00:12, 9138.01it/s] 72%|███████▏  | 287389/400000 [00:28<00:12, 9169.88it/s] 72%|███████▏  | 288308/400000 [00:28<00:12, 9167.42it/s] 72%|███████▏  | 289233/400000 [00:28<00:12, 9191.06it/s] 73%|███████▎  | 290154/400000 [00:28<00:12, 9100.82it/s] 73%|███████▎  | 291093/400000 [00:28<00:11, 9184.74it/s] 73%|███████▎  | 292146/400000 [00:28<00:11, 9549.25it/s] 73%|███████▎  | 293192/400000 [00:28<00:10, 9804.04it/s] 74%|███████▎  | 294277/400000 [00:29<00:10, 10095.54it/s] 74%|███████▍  | 295293/400000 [00:29<00:10, 9921.37it/s]  74%|███████▍  | 296290/400000 [00:29<00:10, 9881.30it/s] 74%|███████▍  | 297282/400000 [00:29<00:10, 9763.45it/s] 75%|███████▍  | 298283/400000 [00:29<00:10, 9833.24it/s] 75%|███████▍  | 299271/400000 [00:29<00:10, 9846.61it/s] 75%|███████▌  | 300315/400000 [00:29<00:09, 10014.98it/s] 75%|███████▌  | 301365/400000 [00:29<00:09, 10154.57it/s] 76%|███████▌  | 302417/400000 [00:29<00:09, 10258.81it/s] 76%|███████▌  | 303445/400000 [00:29<00:09, 10023.66it/s] 76%|███████▌  | 304450/400000 [00:30<00:09, 9842.55it/s]  76%|███████▋  | 305478/400000 [00:30<00:09, 9968.79it/s] 77%|███████▋  | 306512/400000 [00:30<00:09, 10076.15it/s] 77%|███████▋  | 307569/400000 [00:30<00:09, 10219.17it/s] 77%|███████▋  | 308603/400000 [00:30<00:08, 10252.11it/s] 77%|███████▋  | 309642/400000 [00:30<00:08, 10290.78it/s] 78%|███████▊  | 310672/400000 [00:30<00:08, 10170.09it/s] 78%|███████▊  | 311735/400000 [00:30<00:08, 10302.17it/s] 78%|███████▊  | 312821/400000 [00:30<00:08, 10462.80it/s] 78%|███████▊  | 313869/400000 [00:30<00:08, 10340.03it/s] 79%|███████▊  | 314905/400000 [00:31<00:08, 10167.03it/s] 79%|███████▉  | 315924/400000 [00:31<00:08, 9991.29it/s]  79%|███████▉  | 316925/400000 [00:31<00:08, 9963.23it/s] 79%|███████▉  | 317997/400000 [00:31<00:08, 10177.90it/s] 80%|███████▉  | 319114/400000 [00:31<00:07, 10455.64it/s] 80%|████████  | 320229/400000 [00:31<00:07, 10652.98it/s] 80%|████████  | 321298/400000 [00:31<00:07, 10480.79it/s] 81%|████████  | 322349/400000 [00:31<00:07, 10325.51it/s] 81%|████████  | 323408/400000 [00:31<00:07, 10401.98it/s] 81%|████████  | 324451/400000 [00:31<00:07, 10188.44it/s] 81%|████████▏ | 325518/400000 [00:32<00:07, 10326.26it/s] 82%|████████▏ | 326553/400000 [00:32<00:07, 10269.62it/s] 82%|████████▏ | 327623/400000 [00:32<00:06, 10394.99it/s] 82%|████████▏ | 328715/400000 [00:32<00:06, 10546.77it/s] 82%|████████▏ | 329839/400000 [00:32<00:06, 10743.67it/s] 83%|████████▎ | 330928/400000 [00:32<00:06, 10784.26it/s] 83%|████████▎ | 332008/400000 [00:32<00:06, 10742.15it/s] 83%|████████▎ | 333091/400000 [00:32<00:06, 10768.04it/s] 84%|████████▎ | 334194/400000 [00:32<00:06, 10843.41it/s] 84%|████████▍ | 335279/400000 [00:32<00:06, 10739.25it/s] 84%|████████▍ | 336354/400000 [00:33<00:06, 10255.92it/s] 84%|████████▍ | 337425/400000 [00:33<00:06, 10385.97it/s] 85%|████████▍ | 338468/400000 [00:33<00:05, 10313.17it/s] 85%|████████▍ | 339503/400000 [00:33<00:05, 10302.39it/s] 85%|████████▌ | 340536/400000 [00:33<00:05, 10270.55it/s] 85%|████████▌ | 341565/400000 [00:33<00:05, 10208.57it/s] 86%|████████▌ | 342587/400000 [00:33<00:05, 10122.64it/s] 86%|████████▌ | 343618/400000 [00:33<00:05, 10177.62it/s] 86%|████████▌ | 344679/400000 [00:33<00:05, 10303.20it/s] 86%|████████▋ | 345715/400000 [00:34<00:05, 10319.78it/s] 87%|████████▋ | 346748/400000 [00:34<00:05, 10281.07it/s] 87%|████████▋ | 347797/400000 [00:34<00:05, 10341.65it/s] 87%|████████▋ | 348862/400000 [00:34<00:04, 10431.40it/s] 87%|████████▋ | 349933/400000 [00:34<00:04, 10512.90it/s] 88%|████████▊ | 350985/400000 [00:34<00:04, 10492.18it/s] 88%|████████▊ | 352035/400000 [00:34<00:04, 10275.71it/s] 88%|████████▊ | 353106/400000 [00:34<00:04, 10401.15it/s] 89%|████████▊ | 354148/400000 [00:34<00:04, 10209.69it/s] 89%|████████▉ | 355171/400000 [00:34<00:04, 9916.05it/s]  89%|████████▉ | 356166/400000 [00:35<00:04, 9840.75it/s] 89%|████████▉ | 357153/400000 [00:35<00:04, 9679.04it/s] 90%|████████▉ | 358124/400000 [00:35<00:04, 9611.23it/s] 90%|████████▉ | 359142/400000 [00:35<00:04, 9774.56it/s] 90%|█████████ | 360207/400000 [00:35<00:03, 10020.83it/s] 90%|█████████ | 361244/400000 [00:35<00:03, 10120.85it/s] 91%|█████████ | 362310/400000 [00:35<00:03, 10276.17it/s] 91%|█████████ | 363340/400000 [00:35<00:03, 10246.93it/s] 91%|█████████ | 364412/400000 [00:35<00:03, 10381.17it/s] 91%|█████████▏| 365452/400000 [00:35<00:03, 10338.24it/s] 92%|█████████▏| 366487/400000 [00:36<00:03, 9875.15it/s]  92%|█████████▏| 367497/400000 [00:36<00:03, 9939.51it/s] 92%|█████████▏| 368497/400000 [00:36<00:03, 9954.57it/s] 92%|█████████▏| 369562/400000 [00:36<00:02, 10151.97it/s] 93%|█████████▎| 370626/400000 [00:36<00:02, 10291.28it/s] 93%|█████████▎| 371702/400000 [00:36<00:02, 10426.08it/s] 93%|█████████▎| 372747/400000 [00:36<00:02, 10409.09it/s] 93%|█████████▎| 373790/400000 [00:36<00:02, 10194.90it/s] 94%|█████████▎| 374839/400000 [00:36<00:02, 10278.92it/s] 94%|█████████▍| 375895/400000 [00:36<00:02, 10359.05it/s] 94%|█████████▍| 376934/400000 [00:37<00:02, 10366.35it/s] 94%|█████████▍| 377972/400000 [00:37<00:02, 10296.74it/s] 95%|█████████▍| 379003/400000 [00:37<00:02, 10078.93it/s] 95%|█████████▌| 380048/400000 [00:37<00:01, 10187.09it/s] 95%|█████████▌| 381077/400000 [00:37<00:01, 10215.38it/s] 96%|█████████▌| 382158/400000 [00:37<00:01, 10385.89it/s] 96%|█████████▌| 383218/400000 [00:37<00:01, 10447.55it/s] 96%|█████████▌| 384264/400000 [00:37<00:01, 10362.59it/s] 96%|█████████▋| 385302/400000 [00:37<00:01, 10327.19it/s] 97%|█████████▋| 386361/400000 [00:37<00:01, 10402.61it/s] 97%|█████████▋| 387402/400000 [00:38<00:01, 9944.71it/s]  97%|█████████▋| 388402/400000 [00:38<00:01, 9690.97it/s] 97%|█████████▋| 389376/400000 [00:38<00:01, 9498.98it/s] 98%|█████████▊| 390331/400000 [00:38<00:01, 9363.50it/s] 98%|█████████▊| 391277/400000 [00:38<00:00, 9389.96it/s] 98%|█████████▊| 392219/400000 [00:38<00:00, 9366.26it/s] 98%|█████████▊| 393158/400000 [00:38<00:00, 9324.65it/s] 99%|█████████▊| 394092/400000 [00:38<00:00, 9230.94it/s] 99%|█████████▉| 395109/400000 [00:38<00:00, 9493.50it/s] 99%|█████████▉| 396146/400000 [00:39<00:00, 9739.77it/s] 99%|█████████▉| 397212/400000 [00:39<00:00, 9997.72it/s]100%|█████████▉| 398287/400000 [00:39<00:00, 10209.08it/s]100%|█████████▉| 399338/400000 [00:39<00:00, 10296.43it/s]100%|█████████▉| 399999/400000 [00:39<00:00, 10152.35it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f6d31b2cc88> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011221596781958497 	 Accuracy: 54
Train Epoch: 1 	 Loss: 0.010998919855392099 	 Accuracy: 57

  model saves at 57% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15761 out of table with 15624 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15761 out of table with 15624 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-15 02:39:29.501398: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 02:39:29.505347: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095210000 Hz
2020-05-15 02:39:29.505463: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563a695863d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 02:39:29.505473: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f6d3d6a8f60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.4980 - accuracy: 0.5110
 2000/25000 [=>............................] - ETA: 7s - loss: 7.6130 - accuracy: 0.5035 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6155 - accuracy: 0.5033
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6858 - accuracy: 0.4988
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6942 - accuracy: 0.4982
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6462 - accuracy: 0.5013
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6228 - accuracy: 0.5029
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5382 - accuracy: 0.5084
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.5661 - accuracy: 0.5066
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5608 - accuracy: 0.5069
11000/25000 [============>.................] - ETA: 3s - loss: 7.5914 - accuracy: 0.5049
12000/25000 [=============>................] - ETA: 2s - loss: 7.6130 - accuracy: 0.5035
13000/25000 [==============>...............] - ETA: 2s - loss: 7.5982 - accuracy: 0.5045
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6206 - accuracy: 0.5030
15000/25000 [=================>............] - ETA: 2s - loss: 7.6298 - accuracy: 0.5024
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6177 - accuracy: 0.5032
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6486 - accuracy: 0.5012
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6504 - accuracy: 0.5011
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6850 - accuracy: 0.4988
21000/25000 [========================>.....] - ETA: 0s - loss: 7.7024 - accuracy: 0.4977
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6924 - accuracy: 0.4983
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6866 - accuracy: 0.4987
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6858 - accuracy: 0.4988
25000/25000 [==============================] - 7s 261us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f6c9f6d0c50> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f6c9e4fb2e8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 944ms/step - loss: 1.9574 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.9353 - val_crf_viterbi_accuracy: 0.0000e+00

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

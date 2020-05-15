
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f7575106fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 03:14:37.556880
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 03:14:37.561312
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 03:14:37.565068
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 03:14:37.569681
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f758111e470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 360698.1875
Epoch 2/10

1/1 [==============================] - 0s 97ms/step - loss: 311898.0625
Epoch 3/10

1/1 [==============================] - 0s 96ms/step - loss: 237689.0781
Epoch 4/10

1/1 [==============================] - 0s 100ms/step - loss: 150072.2031
Epoch 5/10

1/1 [==============================] - 0s 97ms/step - loss: 91037.5938
Epoch 6/10

1/1 [==============================] - 0s 96ms/step - loss: 53672.0000
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 31626.3262
Epoch 8/10

1/1 [==============================] - 0s 94ms/step - loss: 19619.5215
Epoch 9/10

1/1 [==============================] - 0s 94ms/step - loss: 13106.0332
Epoch 10/10

1/1 [==============================] - 0s 98ms/step - loss: 9391.2148

  #### Inference Need return ypred, ytrue ######################### 
[[-6.74439311e-01 -2.85047352e-01 -9.75744724e-01 -1.44867635e+00
   6.64798737e-01  5.33491254e-01 -2.73143589e-01  5.34572363e-01
   1.91975549e-01 -1.40096068e+00  1.89659804e-01 -7.12277651e-01
   7.23578513e-01  2.36415386e-01  8.94835949e-01  2.25366414e-01
  -3.71123075e-01 -5.39404929e-01  2.80683756e-01  1.01928401e+00
   1.26995564e-01 -9.55403447e-02 -1.94524884e+00 -7.34594285e-01
  -2.81045794e-01  1.05408037e+00 -9.55041051e-02 -2.56658345e-02
  -5.21235108e-01  3.11496109e-01 -2.11679935e-02  7.77393341e-01
   1.22337520e-01 -2.02818680e+00  1.45383894e+00 -2.18648523e-01
   2.35559046e-02  3.30529064e-01  2.18122214e-01 -1.09769166e-01
  -2.82479346e-01  5.25853932e-02 -4.56858635e-01 -1.77976418e+00
   1.41486287e-01  6.13275766e-01 -4.08329338e-01 -2.44484484e-01
  -5.06256282e-01 -4.43015695e-01 -1.08301365e+00 -4.73196268e-01
  -6.73894465e-01 -1.60794437e-01  2.71011382e-01 -3.85860950e-02
  -2.84193844e-01  4.60529625e-02 -3.01574171e-01  3.46505731e-01
  -1.34034246e-01  1.07084906e+00  3.67471755e-01  8.97627711e-01
  -6.32380128e-01  7.30673194e-01  1.33358747e-01 -2.38794351e+00
   8.24172258e-01  6.53052449e-01 -1.28479987e-01  1.44796586e+00
  -9.34822083e-01  1.73603296e-02 -4.03534979e-01  9.71350074e-03
   1.41989160e+00  1.00531960e+00 -9.34897602e-01 -5.77289104e-01
  -1.53564942e+00 -1.54743969e-01 -3.70524079e-01 -7.40704179e-01
   1.02349460e-01 -3.27633262e-01 -1.15801692e+00 -1.20442486e+00
   8.65621448e-01 -2.16270417e-01  1.69807053e+00 -7.23991692e-02
   2.28994578e-01 -3.27143967e-01  1.22269809e-01  1.06418884e+00
  -9.38683748e-02 -9.91885185e-01 -1.96080744e-01  7.37492502e-01
  -5.37132323e-01  2.64328241e-01  4.55921888e-01  1.27749884e+00
  -7.39541471e-01 -2.96731293e-03 -5.42059124e-01  6.51814401e-01
  -6.00794435e-01 -4.17104900e-01  4.34267461e-01  1.33290553e+00
  -2.22966939e-01 -8.02332997e-01 -7.09603548e-01  6.78820610e-01
   8.19345891e-01  6.36683822e-01  1.24265850e-02  2.29137123e-01
  -1.69414252e-01  7.50694466e+00  7.39654398e+00  6.46855736e+00
   7.00934124e+00  6.60559845e+00  6.46293259e+00  5.30212021e+00
   6.45392895e+00  7.71521997e+00  7.06515312e+00  5.68227005e+00
   6.51003981e+00  5.82761526e+00  7.07779455e+00  7.37789822e+00
   5.81222153e+00  5.57503557e+00  7.19950867e+00  6.62603521e+00
   6.31326437e+00  6.80821800e+00  7.94928122e+00  6.74139595e+00
   6.65909338e+00  5.80537128e+00  6.48621130e+00  7.37902451e+00
   6.78927279e+00  6.25417089e+00  7.26850557e+00  7.19784880e+00
   6.63950872e+00  7.85926485e+00  7.51505947e+00  6.75750637e+00
   6.60229349e+00  5.67401695e+00  6.77326918e+00  7.22644138e+00
   8.47085571e+00  7.11122274e+00  6.75592947e+00  6.98188829e+00
   7.86309624e+00  6.72992134e+00  7.84541368e+00  6.22505569e+00
   6.47269630e+00  5.28759146e+00  7.00003910e+00  5.22037029e+00
   8.31164742e+00  7.12876081e+00  7.72477961e+00  7.69333506e+00
   7.46817541e+00  6.82268143e+00  7.42206573e+00  5.23431683e+00
   6.98768735e-01  1.90423846e+00  1.53163052e+00  1.38692212e+00
   7.87291527e-01  7.30173647e-01  1.29692101e+00  8.22795451e-01
   4.09845114e-01  2.62549734e+00  6.80476010e-01  1.48731017e+00
   5.72388828e-01  7.02015877e-01  3.62067461e-01  1.03323209e+00
   4.13412571e-01  1.69396138e+00  2.12042212e+00  9.10937190e-01
   1.79646444e+00  5.71917176e-01  1.06604612e+00  8.89571905e-01
   2.07337093e+00  2.41493273e+00  6.72525883e-01  1.43422723e+00
   1.39729273e+00  6.61732435e-01  2.03790474e+00  5.29338062e-01
   8.72986197e-01  5.07735968e-01  4.19527769e-01  1.97973943e+00
   5.09587109e-01  1.98129880e+00  1.73818827e+00  1.63443565e+00
   8.65623236e-01  1.44476974e+00  9.32894170e-01  1.36906636e+00
   8.20165873e-01  6.36625051e-01  1.07417691e+00  6.77782357e-01
   1.63718891e+00  1.31617689e+00  1.10263181e+00  4.09716606e-01
   3.54847670e-01  7.63850451e-01  2.25583410e+00  9.89146650e-01
   1.14710450e+00  2.23932028e+00  7.47009933e-01  3.20276380e-01
   2.34187412e+00  7.30965137e-01  1.16309488e+00  1.57645082e+00
   2.00872731e+00  9.42898214e-01  1.80728889e+00  1.78200531e+00
   1.30118155e+00  1.06961238e+00  6.60832763e-01  7.17202008e-01
   1.14801681e+00  1.45526195e+00  3.39676321e-01  5.21041691e-01
   5.01748025e-01  1.56320238e+00  1.94213402e+00  7.65699387e-01
   1.04659522e+00  7.37947226e-01  3.65588009e-01  1.23794925e+00
   9.93834734e-01  2.28336763e+00  1.96051908e+00  2.98142672e-01
   1.24554574e+00  5.19835353e-01  6.95860505e-01  2.00638914e+00
   9.80305791e-01  8.80533755e-01  3.70164931e-01  7.27628708e-01
   2.09307957e+00  9.40448761e-01  2.13753402e-01  7.91891098e-01
   9.18779731e-01  5.55630684e-01  1.59515285e+00  1.81617081e-01
   5.08508563e-01  2.67079830e-01  1.20861197e+00  6.81740344e-01
   6.06262386e-01  1.01057982e+00  3.20963681e-01  2.06100416e+00
   1.90228784e+00  1.15119696e+00  2.20962381e+00  1.06063080e+00
   7.88046539e-01  4.38799143e-01  5.65826654e-01  1.82302928e+00
   3.36785913e-02  7.86105156e+00  6.49302244e+00  6.89143276e+00
   8.53172970e+00  7.30226326e+00  7.17721748e+00  6.68981552e+00
   6.41804123e+00  7.76999140e+00  6.51342297e+00  7.84811068e+00
   6.03811026e+00  7.56215668e+00  7.10747385e+00  6.18465567e+00
   7.05537653e+00  8.05868340e+00  5.60149717e+00  7.01511908e+00
   7.75839806e+00  6.60604715e+00  7.23659277e+00  6.89511251e+00
   7.20884275e+00  7.50445557e+00  7.49251699e+00  7.13364935e+00
   6.89546967e+00  8.32281971e+00  7.20192289e+00  7.40722370e+00
   7.61954498e+00  6.93389606e+00  6.52464390e+00  7.04454470e+00
   6.51458073e+00  7.60691738e+00  7.01338291e+00  6.25044584e+00
   6.96756840e+00  5.89827824e+00  6.91511583e+00  8.18719769e+00
   7.19532061e+00  6.92342663e+00  6.69460440e+00  6.97090101e+00
   7.68045378e+00  6.60666752e+00  7.28717518e+00  7.62640810e+00
   7.49730825e+00  7.78368139e+00  7.98162556e+00  6.85732746e+00
   6.83944559e+00  7.26379204e+00  7.85185337e+00  6.48303604e+00
  -8.32649899e+00 -6.66285086e+00  5.62099934e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 03:14:46.972146
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.7372
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 03:14:46.976691
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8992.94
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 03:14:46.980479
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.0906
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 03:14:46.984008
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -804.382
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140142095152464
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140140885115456
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140140885115960
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140140885116464
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140140885116968
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140140885117472

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f7575014be0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.541632
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.490588
grad_step = 000002, loss = 0.449648
grad_step = 000003, loss = 0.406099
grad_step = 000004, loss = 0.358631
grad_step = 000005, loss = 0.315709
grad_step = 000006, loss = 0.294681
grad_step = 000007, loss = 0.300604
grad_step = 000008, loss = 0.282483
grad_step = 000009, loss = 0.252559
grad_step = 000010, loss = 0.234601
grad_step = 000011, loss = 0.227259
grad_step = 000012, loss = 0.221915
grad_step = 000013, loss = 0.213795
grad_step = 000014, loss = 0.201922
grad_step = 000015, loss = 0.187659
grad_step = 000016, loss = 0.173090
grad_step = 000017, loss = 0.160671
grad_step = 000018, loss = 0.151541
grad_step = 000019, loss = 0.144023
grad_step = 000020, loss = 0.134877
grad_step = 000021, loss = 0.120795
grad_step = 000022, loss = 0.106843
grad_step = 000023, loss = 0.095117
grad_step = 000024, loss = 0.085536
grad_step = 000025, loss = 0.077803
grad_step = 000026, loss = 0.070761
grad_step = 000027, loss = 0.063164
grad_step = 000028, loss = 0.056830
grad_step = 000029, loss = 0.052590
grad_step = 000030, loss = 0.048697
grad_step = 000031, loss = 0.043653
grad_step = 000032, loss = 0.038249
grad_step = 000033, loss = 0.034047
grad_step = 000034, loss = 0.030978
grad_step = 000035, loss = 0.028283
grad_step = 000036, loss = 0.025693
grad_step = 000037, loss = 0.023099
grad_step = 000038, loss = 0.020382
grad_step = 000039, loss = 0.017864
grad_step = 000040, loss = 0.015997
grad_step = 000041, loss = 0.014684
grad_step = 000042, loss = 0.013489
grad_step = 000043, loss = 0.011920
grad_step = 000044, loss = 0.010238
grad_step = 000045, loss = 0.009069
grad_step = 000046, loss = 0.008313
grad_step = 000047, loss = 0.007612
grad_step = 000048, loss = 0.006950
grad_step = 000049, loss = 0.006362
grad_step = 000050, loss = 0.005815
grad_step = 000051, loss = 0.005259
grad_step = 000052, loss = 0.004774
grad_step = 000053, loss = 0.004449
grad_step = 000054, loss = 0.004116
grad_step = 000055, loss = 0.003780
grad_step = 000056, loss = 0.003599
grad_step = 000057, loss = 0.003513
grad_step = 000058, loss = 0.003387
grad_step = 000059, loss = 0.003180
grad_step = 000060, loss = 0.003002
grad_step = 000061, loss = 0.002915
grad_step = 000062, loss = 0.002852
grad_step = 000063, loss = 0.002825
grad_step = 000064, loss = 0.002811
grad_step = 000065, loss = 0.002763
grad_step = 000066, loss = 0.002700
grad_step = 000067, loss = 0.002668
grad_step = 000068, loss = 0.002667
grad_step = 000069, loss = 0.002635
grad_step = 000070, loss = 0.002599
grad_step = 000071, loss = 0.002589
grad_step = 000072, loss = 0.002577
grad_step = 000073, loss = 0.002557
grad_step = 000074, loss = 0.002535
grad_step = 000075, loss = 0.002516
grad_step = 000076, loss = 0.002487
grad_step = 000077, loss = 0.002471
grad_step = 000078, loss = 0.002466
grad_step = 000079, loss = 0.002448
grad_step = 000080, loss = 0.002420
grad_step = 000081, loss = 0.002396
grad_step = 000082, loss = 0.002380
grad_step = 000083, loss = 0.002364
grad_step = 000084, loss = 0.002352
grad_step = 000085, loss = 0.002340
grad_step = 000086, loss = 0.002326
grad_step = 000087, loss = 0.002313
grad_step = 000088, loss = 0.002301
grad_step = 000089, loss = 0.002288
grad_step = 000090, loss = 0.002274
grad_step = 000091, loss = 0.002265
grad_step = 000092, loss = 0.002257
grad_step = 000093, loss = 0.002248
grad_step = 000094, loss = 0.002239
grad_step = 000095, loss = 0.002231
grad_step = 000096, loss = 0.002223
grad_step = 000097, loss = 0.002217
grad_step = 000098, loss = 0.002211
grad_step = 000099, loss = 0.002204
grad_step = 000100, loss = 0.002197
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002191
grad_step = 000102, loss = 0.002186
grad_step = 000103, loss = 0.002182
grad_step = 000104, loss = 0.002177
grad_step = 000105, loss = 0.002173
grad_step = 000106, loss = 0.002168
grad_step = 000107, loss = 0.002164
grad_step = 000108, loss = 0.002160
grad_step = 000109, loss = 0.002156
grad_step = 000110, loss = 0.002151
grad_step = 000111, loss = 0.002148
grad_step = 000112, loss = 0.002144
grad_step = 000113, loss = 0.002139
grad_step = 000114, loss = 0.002135
grad_step = 000115, loss = 0.002131
grad_step = 000116, loss = 0.002127
grad_step = 000117, loss = 0.002123
grad_step = 000118, loss = 0.002118
grad_step = 000119, loss = 0.002114
grad_step = 000120, loss = 0.002109
grad_step = 000121, loss = 0.002105
grad_step = 000122, loss = 0.002101
grad_step = 000123, loss = 0.002096
grad_step = 000124, loss = 0.002092
grad_step = 000125, loss = 0.002087
grad_step = 000126, loss = 0.002083
grad_step = 000127, loss = 0.002078
grad_step = 000128, loss = 0.002074
grad_step = 000129, loss = 0.002069
grad_step = 000130, loss = 0.002065
grad_step = 000131, loss = 0.002060
grad_step = 000132, loss = 0.002056
grad_step = 000133, loss = 0.002051
grad_step = 000134, loss = 0.002047
grad_step = 000135, loss = 0.002042
grad_step = 000136, loss = 0.002038
grad_step = 000137, loss = 0.002033
grad_step = 000138, loss = 0.002028
grad_step = 000139, loss = 0.002025
grad_step = 000140, loss = 0.002022
grad_step = 000141, loss = 0.002021
grad_step = 000142, loss = 0.002025
grad_step = 000143, loss = 0.002025
grad_step = 000144, loss = 0.002015
grad_step = 000145, loss = 0.002001
grad_step = 000146, loss = 0.001994
grad_step = 000147, loss = 0.001996
grad_step = 000148, loss = 0.001999
grad_step = 000149, loss = 0.001996
grad_step = 000150, loss = 0.001986
grad_step = 000151, loss = 0.001976
grad_step = 000152, loss = 0.001970
grad_step = 000153, loss = 0.001968
grad_step = 000154, loss = 0.001969
grad_step = 000155, loss = 0.001972
grad_step = 000156, loss = 0.001975
grad_step = 000157, loss = 0.001979
grad_step = 000158, loss = 0.001976
grad_step = 000159, loss = 0.001970
grad_step = 000160, loss = 0.001957
grad_step = 000161, loss = 0.001945
grad_step = 000162, loss = 0.001937
grad_step = 000163, loss = 0.001934
grad_step = 000164, loss = 0.001935
grad_step = 000165, loss = 0.001940
grad_step = 000166, loss = 0.001950
grad_step = 000167, loss = 0.001964
grad_step = 000168, loss = 0.001987
grad_step = 000169, loss = 0.001997
grad_step = 000170, loss = 0.001995
grad_step = 000171, loss = 0.001954
grad_step = 000172, loss = 0.001919
grad_step = 000173, loss = 0.001913
grad_step = 000174, loss = 0.001933
grad_step = 000175, loss = 0.001952
grad_step = 000176, loss = 0.001945
grad_step = 000177, loss = 0.001924
grad_step = 000178, loss = 0.001904
grad_step = 000179, loss = 0.001901
grad_step = 000180, loss = 0.001912
grad_step = 000181, loss = 0.001924
grad_step = 000182, loss = 0.001931
grad_step = 000183, loss = 0.001922
grad_step = 000184, loss = 0.001908
grad_step = 000185, loss = 0.001894
grad_step = 000186, loss = 0.001888
grad_step = 000187, loss = 0.001891
grad_step = 000188, loss = 0.001898
grad_step = 000189, loss = 0.001904
grad_step = 000190, loss = 0.001904
grad_step = 000191, loss = 0.001901
grad_step = 000192, loss = 0.001892
grad_step = 000193, loss = 0.001885
grad_step = 000194, loss = 0.001879
grad_step = 000195, loss = 0.001877
grad_step = 000196, loss = 0.001877
grad_step = 000197, loss = 0.001879
grad_step = 000198, loss = 0.001883
grad_step = 000199, loss = 0.001887
grad_step = 000200, loss = 0.001893
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001898
grad_step = 000202, loss = 0.001907
grad_step = 000203, loss = 0.001908
grad_step = 000204, loss = 0.001909
grad_step = 000205, loss = 0.001896
grad_step = 000206, loss = 0.001882
grad_step = 000207, loss = 0.001868
grad_step = 000208, loss = 0.001861
grad_step = 000209, loss = 0.001862
grad_step = 000210, loss = 0.001869
grad_step = 000211, loss = 0.001878
grad_step = 000212, loss = 0.001885
grad_step = 000213, loss = 0.001895
grad_step = 000214, loss = 0.001899
grad_step = 000215, loss = 0.001909
grad_step = 000216, loss = 0.001903
grad_step = 000217, loss = 0.001895
grad_step = 000218, loss = 0.001873
grad_step = 000219, loss = 0.001854
grad_step = 000220, loss = 0.001847
grad_step = 000221, loss = 0.001852
grad_step = 000222, loss = 0.001861
grad_step = 000223, loss = 0.001867
grad_step = 000224, loss = 0.001867
grad_step = 000225, loss = 0.001859
grad_step = 000226, loss = 0.001851
grad_step = 000227, loss = 0.001843
grad_step = 000228, loss = 0.001838
grad_step = 000229, loss = 0.001835
grad_step = 000230, loss = 0.001836
grad_step = 000231, loss = 0.001837
grad_step = 000232, loss = 0.001841
grad_step = 000233, loss = 0.001847
grad_step = 000234, loss = 0.001854
grad_step = 000235, loss = 0.001870
grad_step = 000236, loss = 0.001885
grad_step = 000237, loss = 0.001910
grad_step = 000238, loss = 0.001916
grad_step = 000239, loss = 0.001917
grad_step = 000240, loss = 0.001878
grad_step = 000241, loss = 0.001839
grad_step = 000242, loss = 0.001822
grad_step = 000243, loss = 0.001835
grad_step = 000244, loss = 0.001859
grad_step = 000245, loss = 0.001863
grad_step = 000246, loss = 0.001852
grad_step = 000247, loss = 0.001828
grad_step = 000248, loss = 0.001815
grad_step = 000249, loss = 0.001816
grad_step = 000250, loss = 0.001826
grad_step = 000251, loss = 0.001838
grad_step = 000252, loss = 0.001840
grad_step = 000253, loss = 0.001838
grad_step = 000254, loss = 0.001827
grad_step = 000255, loss = 0.001816
grad_step = 000256, loss = 0.001808
grad_step = 000257, loss = 0.001803
grad_step = 000258, loss = 0.001803
grad_step = 000259, loss = 0.001806
grad_step = 000260, loss = 0.001811
grad_step = 000261, loss = 0.001817
grad_step = 000262, loss = 0.001825
grad_step = 000263, loss = 0.001832
grad_step = 000264, loss = 0.001844
grad_step = 000265, loss = 0.001848
grad_step = 000266, loss = 0.001853
grad_step = 000267, loss = 0.001841
grad_step = 000268, loss = 0.001824
grad_step = 000269, loss = 0.001802
grad_step = 000270, loss = 0.001791
grad_step = 000271, loss = 0.001791
grad_step = 000272, loss = 0.001799
grad_step = 000273, loss = 0.001810
grad_step = 000274, loss = 0.001813
grad_step = 000275, loss = 0.001813
grad_step = 000276, loss = 0.001805
grad_step = 000277, loss = 0.001797
grad_step = 000278, loss = 0.001787
grad_step = 000279, loss = 0.001780
grad_step = 000280, loss = 0.001778
grad_step = 000281, loss = 0.001778
grad_step = 000282, loss = 0.001781
grad_step = 000283, loss = 0.001785
grad_step = 000284, loss = 0.001791
grad_step = 000285, loss = 0.001798
grad_step = 000286, loss = 0.001810
grad_step = 000287, loss = 0.001818
grad_step = 000288, loss = 0.001834
grad_step = 000289, loss = 0.001836
grad_step = 000290, loss = 0.001837
grad_step = 000291, loss = 0.001817
grad_step = 000292, loss = 0.001796
grad_step = 000293, loss = 0.001774
grad_step = 000294, loss = 0.001763
grad_step = 000295, loss = 0.001767
grad_step = 000296, loss = 0.001778
grad_step = 000297, loss = 0.001791
grad_step = 000298, loss = 0.001799
grad_step = 000299, loss = 0.001808
grad_step = 000300, loss = 0.001804
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001804
grad_step = 000302, loss = 0.001792
grad_step = 000303, loss = 0.001780
grad_step = 000304, loss = 0.001765
grad_step = 000305, loss = 0.001754
grad_step = 000306, loss = 0.001750
grad_step = 000307, loss = 0.001750
grad_step = 000308, loss = 0.001754
grad_step = 000309, loss = 0.001760
grad_step = 000310, loss = 0.001768
grad_step = 000311, loss = 0.001776
grad_step = 000312, loss = 0.001792
grad_step = 000313, loss = 0.001804
grad_step = 000314, loss = 0.001832
grad_step = 000315, loss = 0.001839
grad_step = 000316, loss = 0.001850
grad_step = 000317, loss = 0.001816
grad_step = 000318, loss = 0.001777
grad_step = 000319, loss = 0.001743
grad_step = 000320, loss = 0.001739
grad_step = 000321, loss = 0.001758
grad_step = 000322, loss = 0.001777
grad_step = 000323, loss = 0.001785
grad_step = 000324, loss = 0.001768
grad_step = 000325, loss = 0.001750
grad_step = 000326, loss = 0.001733
grad_step = 000327, loss = 0.001727
grad_step = 000328, loss = 0.001730
grad_step = 000329, loss = 0.001738
grad_step = 000330, loss = 0.001751
grad_step = 000331, loss = 0.001761
grad_step = 000332, loss = 0.001776
grad_step = 000333, loss = 0.001778
grad_step = 000334, loss = 0.001780
grad_step = 000335, loss = 0.001765
grad_step = 000336, loss = 0.001746
grad_step = 000337, loss = 0.001726
grad_step = 000338, loss = 0.001716
grad_step = 000339, loss = 0.001716
grad_step = 000340, loss = 0.001724
grad_step = 000341, loss = 0.001734
grad_step = 000342, loss = 0.001738
grad_step = 000343, loss = 0.001740
grad_step = 000344, loss = 0.001736
grad_step = 000345, loss = 0.001732
grad_step = 000346, loss = 0.001724
grad_step = 000347, loss = 0.001717
grad_step = 000348, loss = 0.001710
grad_step = 000349, loss = 0.001705
grad_step = 000350, loss = 0.001701
grad_step = 000351, loss = 0.001700
grad_step = 000352, loss = 0.001699
grad_step = 000353, loss = 0.001699
grad_step = 000354, loss = 0.001701
grad_step = 000355, loss = 0.001703
grad_step = 000356, loss = 0.001708
grad_step = 000357, loss = 0.001715
grad_step = 000358, loss = 0.001730
grad_step = 000359, loss = 0.001750
grad_step = 000360, loss = 0.001792
grad_step = 000361, loss = 0.001826
grad_step = 000362, loss = 0.001885
grad_step = 000363, loss = 0.001870
grad_step = 000364, loss = 0.001836
grad_step = 000365, loss = 0.001745
grad_step = 000366, loss = 0.001690
grad_step = 000367, loss = 0.001701
grad_step = 000368, loss = 0.001747
grad_step = 000369, loss = 0.001780
grad_step = 000370, loss = 0.001754
grad_step = 000371, loss = 0.001711
grad_step = 000372, loss = 0.001680
grad_step = 000373, loss = 0.001683
grad_step = 000374, loss = 0.001711
grad_step = 000375, loss = 0.001733
grad_step = 000376, loss = 0.001750
grad_step = 000377, loss = 0.001731
grad_step = 000378, loss = 0.001709
grad_step = 000379, loss = 0.001681
grad_step = 000380, loss = 0.001668
grad_step = 000381, loss = 0.001671
grad_step = 000382, loss = 0.001684
grad_step = 000383, loss = 0.001695
grad_step = 000384, loss = 0.001692
grad_step = 000385, loss = 0.001682
grad_step = 000386, loss = 0.001668
grad_step = 000387, loss = 0.001660
grad_step = 000388, loss = 0.001659
grad_step = 000389, loss = 0.001664
grad_step = 000390, loss = 0.001670
grad_step = 000391, loss = 0.001672
grad_step = 000392, loss = 0.001673
grad_step = 000393, loss = 0.001668
grad_step = 000394, loss = 0.001662
grad_step = 000395, loss = 0.001656
grad_step = 000396, loss = 0.001650
grad_step = 000397, loss = 0.001647
grad_step = 000398, loss = 0.001646
grad_step = 000399, loss = 0.001646
grad_step = 000400, loss = 0.001646
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001648
grad_step = 000402, loss = 0.001649
grad_step = 000403, loss = 0.001651
grad_step = 000404, loss = 0.001653
grad_step = 000405, loss = 0.001656
grad_step = 000406, loss = 0.001660
grad_step = 000407, loss = 0.001667
grad_step = 000408, loss = 0.001673
grad_step = 000409, loss = 0.001687
grad_step = 000410, loss = 0.001696
grad_step = 000411, loss = 0.001712
grad_step = 000412, loss = 0.001711
grad_step = 000413, loss = 0.001711
grad_step = 000414, loss = 0.001687
grad_step = 000415, loss = 0.001662
grad_step = 000416, loss = 0.001637
grad_step = 000417, loss = 0.001624
grad_step = 000418, loss = 0.001626
grad_step = 000419, loss = 0.001637
grad_step = 000420, loss = 0.001651
grad_step = 000421, loss = 0.001659
grad_step = 000422, loss = 0.001669
grad_step = 000423, loss = 0.001667
grad_step = 000424, loss = 0.001669
grad_step = 000425, loss = 0.001658
grad_step = 000426, loss = 0.001649
grad_step = 000427, loss = 0.001633
grad_step = 000428, loss = 0.001620
grad_step = 000429, loss = 0.001610
grad_step = 000430, loss = 0.001606
grad_step = 000431, loss = 0.001608
grad_step = 000432, loss = 0.001612
grad_step = 000433, loss = 0.001618
grad_step = 000434, loss = 0.001624
grad_step = 000435, loss = 0.001634
grad_step = 000436, loss = 0.001642
grad_step = 000437, loss = 0.001661
grad_step = 000438, loss = 0.001672
grad_step = 000439, loss = 0.001694
grad_step = 000440, loss = 0.001688
grad_step = 000441, loss = 0.001676
grad_step = 000442, loss = 0.001636
grad_step = 000443, loss = 0.001602
grad_step = 000444, loss = 0.001589
grad_step = 000445, loss = 0.001599
grad_step = 000446, loss = 0.001618
grad_step = 000447, loss = 0.001628
grad_step = 000448, loss = 0.001628
grad_step = 000449, loss = 0.001612
grad_step = 000450, loss = 0.001598
grad_step = 000451, loss = 0.001584
grad_step = 000452, loss = 0.001575
grad_step = 000453, loss = 0.001572
grad_step = 000454, loss = 0.001574
grad_step = 000455, loss = 0.001579
grad_step = 000456, loss = 0.001585
grad_step = 000457, loss = 0.001599
grad_step = 000458, loss = 0.001613
grad_step = 000459, loss = 0.001641
grad_step = 000460, loss = 0.001658
grad_step = 000461, loss = 0.001694
grad_step = 000462, loss = 0.001681
grad_step = 000463, loss = 0.001660
grad_step = 000464, loss = 0.001603
grad_step = 000465, loss = 0.001561
grad_step = 000466, loss = 0.001553
grad_step = 000467, loss = 0.001575
grad_step = 000468, loss = 0.001601
grad_step = 000469, loss = 0.001605
grad_step = 000470, loss = 0.001599
grad_step = 000471, loss = 0.001573
grad_step = 000472, loss = 0.001551
grad_step = 000473, loss = 0.001536
grad_step = 000474, loss = 0.001532
grad_step = 000475, loss = 0.001535
grad_step = 000476, loss = 0.001543
grad_step = 000477, loss = 0.001558
grad_step = 000478, loss = 0.001574
grad_step = 000479, loss = 0.001604
grad_step = 000480, loss = 0.001627
grad_step = 000481, loss = 0.001671
grad_step = 000482, loss = 0.001664
grad_step = 000483, loss = 0.001649
grad_step = 000484, loss = 0.001577
grad_step = 000485, loss = 0.001519
grad_step = 000486, loss = 0.001507
grad_step = 000487, loss = 0.001536
grad_step = 000488, loss = 0.001569
grad_step = 000489, loss = 0.001565
grad_step = 000490, loss = 0.001546
grad_step = 000491, loss = 0.001508
grad_step = 000492, loss = 0.001486
grad_step = 000493, loss = 0.001484
grad_step = 000494, loss = 0.001496
grad_step = 000495, loss = 0.001515
grad_step = 000496, loss = 0.001527
grad_step = 000497, loss = 0.001544
grad_step = 000498, loss = 0.001540
grad_step = 000499, loss = 0.001540
grad_step = 000500, loss = 0.001517
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001495
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

  date_run                              2020-05-15 03:15:08.858125
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.254537
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 03:15:08.864316
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.182701
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 03:15:08.872627
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.14298
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 03:15:08.878291
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -1.7762
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
0   2020-05-15 03:14:37.556880  ...    mean_absolute_error
1   2020-05-15 03:14:37.561312  ...     mean_squared_error
2   2020-05-15 03:14:37.565068  ...  median_absolute_error
3   2020-05-15 03:14:37.569681  ...               r2_score
4   2020-05-15 03:14:46.972146  ...    mean_absolute_error
5   2020-05-15 03:14:46.976691  ...     mean_squared_error
6   2020-05-15 03:14:46.980479  ...  median_absolute_error
7   2020-05-15 03:14:46.984008  ...               r2_score
8   2020-05-15 03:15:08.858125  ...    mean_absolute_error
9   2020-05-15 03:15:08.864316  ...     mean_squared_error
10  2020-05-15 03:15:08.872627  ...  median_absolute_error
11  2020-05-15 03:15:08.878291  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f23d4dcbba8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:32, 307659.20it/s]  2%|▏         | 212992/9912422 [00:00<00:24, 397438.19it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 549897.93it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 776787.66it/s] 76%|███████▋  | 7577600/9912422 [00:00<00:02, 1098828.44it/s]9920512it [00:00, 10199384.39it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 145822.28it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 305467.41it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 395754.69it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 547551.84it/s]1654784it [00:00, 2742074.29it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 53913.53it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2387784e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2386db40b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2387784e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2386d0b0b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f23845464a8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2384531c18> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2387784e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2386cca6d8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f23845464a8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f23d4d8eeb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fe1d951b1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=49139c8c81926fa7ce93a4b24f2906eeaafafd62722d35aa8caa53a619090abf
  Stored in directory: /tmp/pip-ephem-wheel-cache-vzr0jxz7/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fe171316710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 46s
   57344/17464789 [..............................] - ETA: 39s
   90112/17464789 [..............................] - ETA: 38s
  212992/17464789 [..............................] - ETA: 21s
  458752/17464789 [..............................] - ETA: 12s
  942080/17464789 [>.............................] - ETA: 6s 
 1884160/17464789 [==>...........................] - ETA: 3s
 3751936/17464789 [=====>........................] - ETA: 1s
 6340608/17464789 [=========>....................] - ETA: 1s
 9093120/17464789 [==============>...............] - ETA: 0s
12189696/17464789 [===================>..........] - ETA: 0s
15253504/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 03:16:41.668329: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 03:16:41.672259: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-05-15 03:16:41.672417: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560f0c8603c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 03:16:41.672434: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5286 - accuracy: 0.5090
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4826 - accuracy: 0.5120 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.4980 - accuracy: 0.5110
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5018 - accuracy: 0.5107
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5256 - accuracy: 0.5092
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5670 - accuracy: 0.5065
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5177 - accuracy: 0.5097
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5440 - accuracy: 0.5080
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5678 - accuracy: 0.5064
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5562 - accuracy: 0.5072
11000/25000 [============>.................] - ETA: 4s - loss: 7.5955 - accuracy: 0.5046
12000/25000 [=============>................] - ETA: 4s - loss: 7.5887 - accuracy: 0.5051
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6206 - accuracy: 0.5030
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6141 - accuracy: 0.5034
15000/25000 [=================>............] - ETA: 3s - loss: 7.6349 - accuracy: 0.5021
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6475 - accuracy: 0.5013
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6522 - accuracy: 0.5009
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6556 - accuracy: 0.5007
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6505 - accuracy: 0.5011
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6444 - accuracy: 0.5015
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6433 - accuracy: 0.5015
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6555 - accuracy: 0.5007
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6706 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 9s 370us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 03:16:58.368517
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 03:16:58.368517  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<11:36:27, 20.6kB/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:00<8:08:25, 29.4kB/s]  .vector_cache/glove.6B.zip:   1%|          | 4.73M/862M [00:00<5:40:17, 42.0kB/s].vector_cache/glove.6B.zip:   1%|          | 8.46M/862M [00:00<3:57:17, 60.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.0M/862M [00:01<2:46:15, 85.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.3M/862M [00:01<1:55:25, 122kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 27.7M/862M [00:01<1:20:02, 174kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.7M/862M [00:01<55:37, 248kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 41.6M/862M [00:01<38:40, 354kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.6M/862M [00:01<26:59, 504kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.7M/862M [00:02<18:53, 716kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:02<13:30, 999kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<09:49, 1.37MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:04<13:33:37, 16.5kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<9:29:55, 23.6kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:06<6:39:14, 33.5kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<4:39:47, 47.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:08<3:17:15, 67.5kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<2:18:30, 96.0kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:11<1:40:03, 132kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:11<1:11:54, 184kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:11<50:17, 262kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:13<43:54, 300kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:13<32:55, 400kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:13<23:32, 559kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<16:56, 776kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.8M/862M [00:14<12:11, 1.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:14<08:51, 1.48MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:16<17:52:39, 12.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:16<12:32:25, 17.4kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:16<8:44:46, 24.9kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:17<6:11:37, 35.1kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:17<4:19:54, 50.1kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:17<3:01:45, 71.4kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:19<2:14:16, 96.5kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:19<1:35:58, 135kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:19<1:07:27, 192kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:19<47:32, 272kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:19<33:38, 384kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:21<30:39, 421kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:21<22:50, 564kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.2M/862M [00:21<16:01, 801kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:23<29:14, 439kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:23<21:23, 599kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.9M/862M [00:23<15:03, 848kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:25<18:10, 702kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:25<13:49, 923kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<09:44, 1.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<22:37, 561kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<16:57, 748kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<12:34, 1.01MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:30<10:55, 1.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:30<09:50, 1.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<07:00, 1.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:32<09:38, 1.30MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:32<08:00, 1.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:40, 2.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:34<1:03:28, 196kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:34<46:20, 269kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<32:26, 382kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:36<30:42, 403kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:36<22:29, 550kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<17:34, 701kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<13:10, 935kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<09:17, 1.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<27:41, 442kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<20:47, 589kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<14:37, 834kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<17:26, 699kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<15:42, 776kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:43<11:22, 1.07MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:43<08:09, 1.49MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<10:13, 1.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:44<08:00, 1.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:46<07:31, 1.60MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:46<07:07, 1.69MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:47<05:12, 2.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<03:51, 3.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:48<12:02, 994kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:48<09:33, 1.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<06:48, 1.75MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<09:52, 1.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<08:29, 1.40MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:51<06:03, 1.96MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:53<09:00, 1.31MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:53<07:38, 1.55MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<05:25, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<2:35:25, 75.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:55<1:49:34, 107kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:55<1:16:51, 153kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<53:44, 218kB/s]  .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:57<1:07:04, 174kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:58<47:47, 244kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:58<33:23, 348kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:59<1:05:34, 177kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:59<46:32, 250kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:59<32:50, 353kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:00<23:11, 499kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:02<23:42, 487kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:02<17:34, 657kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<12:21, 930kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:04<15:11, 756kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:04<12:43, 903kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:04<09:24, 1.22MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:04<06:53, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:06<07:23, 1.54MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:06<05:47, 1.97MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:08<05:48, 1.95MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:08<04:36, 2.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:10<05:19, 2.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:10<04:30, 2.50MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<03:18, 3.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:12<06:17, 1.78MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:12<05:56, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:12<04:27, 2.51MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<03:29, 3.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:12<02:37, 4.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:14<1:21:20, 137kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:14<57:56, 192kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<40:28, 274kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:16<37:34, 295kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:16<27:00, 410kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<18:56, 582kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:18<20:11, 545kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:18<14:45, 745kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:18<10:30, 1.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:20<10:31, 1.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:20<08:26, 1.29MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<06:03, 1.80MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:22<07:31, 1.44MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:22<06:05, 1.78MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<04:20, 2.49MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:24<20:08, 536kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:24<15:13, 709kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<10:42, 1.00MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:26<14:42, 729kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:26<10:54, 982kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:26<07:41, 1.38MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:28<22:30, 473kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:28<18:48, 566kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:28<13:27, 790kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:28<09:30, 1.11MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:30<16:54, 626kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:30<14:10, 746kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:30<10:10, 1.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:32<09:04, 1.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:32<09:42, 1.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:32<07:55, 1.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:32<05:47, 1.81MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:34<06:06, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:34<05:45, 1.81MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:34<04:15, 2.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:36<05:07, 2.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:36<05:43, 1.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:36<04:15, 2.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:38<04:56, 2.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:38<04:15, 2.42MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:40<04:30, 2.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:40<04:00, 2.56MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:42<04:24, 2.31MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:42<03:55, 2.59MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:42<02:48, 3.59MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:44<1:22:26, 123kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:44<58:16, 173kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:46<42:05, 238kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:46<30:19, 331kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:46<21:11, 470kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:48<30:58, 322kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:48<22:20, 446kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:50<17:02, 580kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:50<12:15, 806kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:52<10:13, 962kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:52<07:47, 1.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:55<08:09, 1.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:55<06:34, 1.48MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:56<05:24, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:56<04:03, 2.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<02:57, 3.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:58<08:37, 1.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:58<06:59, 1.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:59<05:01, 1.91MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:00<06:21, 1.50MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:00<05:11, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:00<03:42, 2.56MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:02<08:13, 1.15MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:02<06:29, 1.46MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:02<04:35, 2.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:05<26:33, 354kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:05<19:19, 487kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:07<14:50, 629kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:07<10:57, 852kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:07<07:41, 1.21MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:09<2:25:11, 63.9kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:09<1:42:21, 90.6kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:09<1:11:15, 129kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:11<1:18:13, 118kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:11<55:06, 167kB/s]  .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:14<40:33, 225kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:14<29:11, 313kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:14<20:23, 445kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:16<23:32, 385kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:16<17:02, 532kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:18<13:23, 672kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:18<10:06, 890kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:18<07:07, 1.26MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:20<13:22, 668kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:20<10:12, 874kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:20<09:22, 951kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:21<1:01:44, 144kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:22<44:00, 201kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:22<31:55, 277kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:22<22:20, 394kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:24<19:12, 457kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:24<14:22, 611kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:24<10:07, 863kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:27<10:39, 817kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:27<09:00, 966kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:27<06:25, 1.35MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:29<06:40, 1.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:29<05:49, 1.48MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:29<04:09, 2.07MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:30<06:15, 1.37MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:31<05:19, 1.61MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:31<03:47, 2.25MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:32<07:28, 1.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:33<06:14, 1.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:33<04:26, 1.91MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:34<07:45, 1.09MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:35<06:48, 1.24MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:35<05:19, 1.58MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:35<03:58, 2.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:35<03:00, 2.79MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:35<02:20, 3.58MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:36<14:45, 567kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:37<11:38, 719kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:37<08:23, 994kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:37<06:00, 1.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:38<07:35, 1.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:38<06:10, 1.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:39<04:27, 1.85MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:40<05:02, 1.63MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:40<04:04, 2.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:43<04:15, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:43<04:14, 1.92MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:43<03:09, 2.57MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:43<02:17, 3.52MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:45<32:28, 249kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:45<24:17, 333kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:45<17:09, 470kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:45<12:05, 665kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:47<13:48, 581kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:47<46:08, 174kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:47<32:11, 247kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:49<26:52, 296kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:49<19:09, 414kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:49<13:27, 587kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:51<12:52, 612kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:51<10:02, 785kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:51<07:05, 1.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:53<07:31, 1.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:53<06:00, 1.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:55<05:18, 1.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:55<04:03, 1.90MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:55<02:53, 2.65MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:56<43:28, 177kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:57<30:43, 249kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [03:00<23:54, 318kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [03:00<17:19, 439kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:01<12:26, 606kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:01<08:45, 856kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:03<08:37, 865kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:03<06:53, 1.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:03<04:53, 1.52MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:05<06:05, 1.22MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:05<05:18, 1.39MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:05<03:47, 1.94MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:08<06:28, 1.13MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:08<05:07, 1.43MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:12<05:55, 1.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:12<04:50, 1.50MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:14<04:21, 1.65MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:14<03:43, 1.93MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:14<02:41, 2.66MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:16<04:04, 1.75MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:16<04:14, 1.68MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:16<03:11, 2.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:16<02:18, 3.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:18<09:09, 770kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:18<07:58, 885kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:18<05:44, 1.23MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:18<04:04, 1.72MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:20<26:30, 264kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:20<19:55, 351kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:20<14:09, 492kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:20<09:57, 696kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:22<10:18, 672kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:22<08:12, 843kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:22<05:50, 1.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:24<05:39, 1.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:24<05:57, 1.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:24<04:19, 1.58MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:26<04:16, 1.58MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:26<04:35, 1.48MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:26<03:23, 2.00MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:26<02:26, 2.76MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:28<10:23, 646kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:28<08:08, 824kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:28<05:45, 1.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:30<06:10, 1.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:30<05:20, 1.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:30<03:50, 1.72MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:32<04:13, 1.55MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:32<03:30, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:32<02:29, 2.61MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:34<10:25, 624kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:34<08:02, 809kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:34<05:43, 1.13MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:34<04:04, 1.58MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:36<18:35, 346kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:36<13:25, 479kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:38<10:16, 620kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:38<07:38, 833kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:40<06:34, 957kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:40<05:26, 1.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:40<03:49, 1.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:42<12:41, 491kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:42<09:20, 666kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:42<06:35, 939kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:44<06:32, 942kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:44<05:39, 1.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:44<04:02, 1.51MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:46<04:17, 1.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:46<03:40, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:46<02:36, 2.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:50<11:14, 536kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:50<08:38, 697kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:50<06:02, 986kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:50<05:39, 1.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:50<04:01, 1.47MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:51<02:54, 2.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:52<14:13, 414kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:52<11:30, 512kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:53<08:14, 713kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:54<06:41, 870kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:54<05:05, 1.14MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:56<04:23, 1.31MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:56<03:21, 1.71MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:58<03:13, 1.76MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:58<02:43, 2.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:58<02:00, 2.80MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:59<01:28, 3.80MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [04:00<50:14, 112kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [04:00<36:07, 155kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [04:00<25:15, 221kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [04:02<18:52, 293kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [04:02<14:29, 382kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [04:02<10:21, 533kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:03<07:19, 751kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:04<06:57, 787kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:05<05:23, 1.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:05<03:46, 1.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:06<13:42, 394kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:06<10:01, 539kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:07<06:59, 764kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:08<09:45, 547kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:08<07:14, 736kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:09<05:03, 1.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:10<19:47, 266kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:10<14:21, 366kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:11<09:58, 521kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:13<21:55, 237kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:13<15:45, 329kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:15<11:40, 439kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:15<08:26, 606kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:17<06:37, 763kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:17<04:50, 1.04MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:19<04:10, 1.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:19<03:07, 1.59MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:21<03:00, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:21<02:29, 1.97MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:23<02:27, 1.98MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:23<02:14, 2.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:23<01:35, 3.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:25<07:16, 658kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:26<05:35, 856kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:26<03:54, 1.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:27<06:16, 751kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:27<04:49, 976kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:28<03:24, 1.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:28<02:26, 1.90MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:31<04:00, 1.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:31<03:10, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:31<02:14, 2.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:33<23:48, 190kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:33<17:10, 264kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:33<11:55, 376kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:35<11:04, 403kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:35<08:14, 541kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:35<05:43, 769kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:37<35:38, 123kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:37<25:16, 174kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:37<17:35, 247kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:39<13:49, 313kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:39<10:13, 423kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:39<07:07, 600kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:41<07:02, 605kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:41<05:18, 801kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:41<03:44, 1.13MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:43<04:06, 1.02MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:43<03:21, 1.25MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:43<02:23, 1.74MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:45<02:52, 1.43MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:45<02:26, 1.68MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:45<01:44, 2.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:47<03:14, 1.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:47<02:28, 1.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:47<01:46, 2.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:49<02:39, 1.50MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:49<02:12, 1.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:49<01:35, 2.48MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:51<02:21, 1.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:51<02:04, 1.89MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:51<01:30, 2.58MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:53<02:01, 1.90MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:53<01:46, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:53<01:16, 2.98MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:54<02:26, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:55<02:00, 1.88MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:55<01:25, 2.62MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:56<03:19, 1.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:57<02:47, 1.33MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:57<01:58, 1.86MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:58<03:28, 1.05MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:59<02:43, 1.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:59<01:54, 1.88MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [05:00<1:18:44, 45.4kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [05:01<55:14, 64.5kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [05:02<38:32, 90.9kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [05:02<27:06, 129kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [05:03<18:43, 184kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [05:06<19:19, 178kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [05:06<13:56, 246kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [05:06<09:39, 350kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:07<07:46, 433kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:07<05:31, 606kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:09<04:22, 753kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:09<03:16, 1.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:11<02:44, 1.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:11<02:02, 1.57MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:13<01:55, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:13<01:30, 2.08MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:15<01:31, 2.03MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:15<01:40, 1.84MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:15<01:17, 2.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:15<00:57, 3.19MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:17<01:31, 1.98MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:17<01:37, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:17<01:11, 2.52MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:17<00:51, 3.45MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:19<45:30, 64.9kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:19<31:59, 92.0kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:19<21:59, 131kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:21<20:08, 143kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:21<14:18, 201kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:21<09:50, 287kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:24<13:34, 207kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:24<09:41, 289kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:26<07:02, 389kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:26<05:08, 533kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:28<03:54, 684kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:28<02:51, 931kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:30<02:22, 1.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:30<01:46, 1.46MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:32<01:38, 1.55MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:32<01:15, 2.00MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:34<01:15, 1.96MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:34<00:58, 2.50MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:36<01:04, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:36<01:06, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:36<00:47, 2.96MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:38<01:22, 1.69MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:38<01:13, 1.91MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:38<00:50, 2.67MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:39<05:02, 449kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:40<03:40, 612kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:41<02:50, 773kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:42<02:09, 1.01MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:42<01:29, 1.43MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:44<04:51, 438kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:44<03:33, 594kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:46<02:46, 743kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:46<02:05, 976kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:46<01:26, 1.38MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:48<09:28, 210kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:48<06:54, 287kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:48<04:46, 407kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:50<03:57, 485kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:50<02:53, 660kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:50<01:59, 934kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:52<03:02, 609kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:52<02:36, 710kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:52<01:54, 962kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:52<01:23, 1.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:52<00:59, 1.80MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:53<00:44, 2.40MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:55<12:18, 145kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:55<08:43, 203kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:55<05:59, 289kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:57<05:00, 342kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:57<03:44, 456kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:57<02:35, 646kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:59<02:17, 716kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:59<01:43, 944kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [06:01<01:24, 1.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [06:01<01:03, 1.47MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [06:01<00:44, 2.06MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [06:04<01:50, 818kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [06:04<01:24, 1.07MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [06:06<01:09, 1.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [06:06<00:52, 1.64MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [06:06<00:35, 2.29MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [06:08<03:44, 365kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [06:08<02:40, 506kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:10<01:59, 649kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:10<01:26, 888kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:11<01:09, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:12<00:52, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:12<00:35, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:13<02:49, 410kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:14<02:03, 560kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:15<01:31, 714kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:16<01:08, 945kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:16<00:46, 1.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:18<02:01, 503kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:18<01:31, 664kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:18<01:01, 940kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:20<01:25, 672kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:20<01:03, 897kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:22<00:49, 1.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:22<00:38, 1.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:24<00:32, 1.52MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:24<00:26, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:26<00:23, 1.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:26<00:20, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:28<00:18, 2.14MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:28<00:15, 2.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:30<00:15, 2.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:30<00:13, 2.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:32<00:13, 2.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:32<00:11, 2.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:32<00:07, 3.71MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:34<00:46, 602kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:34<00:34, 801kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:34<00:21, 1.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:36<00:54, 439kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:36<00:40, 588kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:36<00:25, 832kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:38<00:28, 701kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:38<00:23, 853kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:38<00:14, 1.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:40<00:13, 1.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:40<00:13, 1.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:40<00:09, 1.57MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:41<00:07, 1.64MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:42<00:06, 1.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:42<00:09, 1.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:42<00:06, 1.55MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:42<00:03, 2.13MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:45<00:02, 1.80MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:45<00:01, 1.92MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:45<00:00, 2.68MB/s].vector_cache/glove.6B.zip: 862MB [06:45, 2.12MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 760/400000 [00:00<00:52, 7599.54it/s]  0%|          | 1492/400000 [00:00<00:53, 7513.01it/s]  1%|          | 2251/400000 [00:00<00:52, 7534.64it/s]  1%|          | 2980/400000 [00:00<00:53, 7457.10it/s]  1%|          | 3728/400000 [00:00<00:53, 7463.25it/s]  1%|          | 4475/400000 [00:00<00:53, 7462.57it/s]  1%|▏         | 5229/400000 [00:00<00:52, 7484.34it/s]  1%|▏         | 5982/400000 [00:00<00:52, 7496.70it/s]  2%|▏         | 6733/400000 [00:00<00:52, 7498.77it/s]  2%|▏         | 7460/400000 [00:01<00:52, 7427.22it/s]  2%|▏         | 8182/400000 [00:01<00:53, 7362.39it/s]  2%|▏         | 8904/400000 [00:01<00:54, 7157.55it/s]  2%|▏         | 9611/400000 [00:01<00:55, 7051.15it/s]  3%|▎         | 10311/400000 [00:01<00:56, 6901.28it/s]  3%|▎         | 11016/400000 [00:01<00:56, 6943.12it/s]  3%|▎         | 11769/400000 [00:01<00:54, 7109.08it/s]  3%|▎         | 12490/400000 [00:01<00:54, 7137.20it/s]  3%|▎         | 13223/400000 [00:01<00:53, 7192.49it/s]  3%|▎         | 13964/400000 [00:01<00:53, 7253.90it/s]  4%|▎         | 14690/400000 [00:02<00:53, 7227.41it/s]  4%|▍         | 15419/400000 [00:02<00:53, 7244.09it/s]  4%|▍         | 16144/400000 [00:02<00:55, 6863.89it/s]  4%|▍         | 16894/400000 [00:02<00:54, 7042.76it/s]  4%|▍         | 17656/400000 [00:02<00:53, 7206.26it/s]  5%|▍         | 18385/400000 [00:02<00:52, 7229.50it/s]  5%|▍         | 19132/400000 [00:02<00:52, 7297.31it/s]  5%|▍         | 19903/400000 [00:02<00:51, 7415.15it/s]  5%|▌         | 20660/400000 [00:02<00:50, 7459.44it/s]  5%|▌         | 21422/400000 [00:02<00:50, 7506.81it/s]  6%|▌         | 22174/400000 [00:03<00:50, 7445.60it/s]  6%|▌         | 22920/400000 [00:03<00:52, 7244.84it/s]  6%|▌         | 23647/400000 [00:03<00:52, 7102.18it/s]  6%|▌         | 24369/400000 [00:03<00:52, 7135.60it/s]  6%|▋         | 25118/400000 [00:03<00:51, 7236.87it/s]  6%|▋         | 25843/400000 [00:03<00:52, 7106.20it/s]  7%|▋         | 26556/400000 [00:03<00:53, 6989.36it/s]  7%|▋         | 27327/400000 [00:03<00:51, 7188.93it/s]  7%|▋         | 28072/400000 [00:03<00:51, 7263.39it/s]  7%|▋         | 28824/400000 [00:03<00:50, 7336.73it/s]  7%|▋         | 29591/400000 [00:04<00:49, 7431.88it/s]  8%|▊         | 30358/400000 [00:04<00:49, 7499.82it/s]  8%|▊         | 31130/400000 [00:04<00:48, 7563.27it/s]  8%|▊         | 31888/400000 [00:04<00:49, 7434.93it/s]  8%|▊         | 32651/400000 [00:04<00:49, 7489.81it/s]  8%|▊         | 33401/400000 [00:04<00:49, 7452.31it/s]  9%|▊         | 34167/400000 [00:04<00:48, 7510.96it/s]  9%|▊         | 34929/400000 [00:04<00:48, 7540.64it/s]  9%|▉         | 35684/400000 [00:04<00:48, 7513.67it/s]  9%|▉         | 36436/400000 [00:04<00:48, 7497.54it/s]  9%|▉         | 37186/400000 [00:05<00:48, 7477.87it/s]  9%|▉         | 37934/400000 [00:05<00:48, 7447.17it/s] 10%|▉         | 38679/400000 [00:05<00:49, 7269.91it/s] 10%|▉         | 39408/400000 [00:05<00:50, 7159.70it/s] 10%|█         | 40126/400000 [00:05<00:51, 7027.85it/s] 10%|█         | 40842/400000 [00:05<00:50, 7066.94it/s] 10%|█         | 41550/400000 [00:05<00:51, 7003.38it/s] 11%|█         | 42252/400000 [00:05<00:51, 6970.27it/s] 11%|█         | 42950/400000 [00:05<00:51, 6909.35it/s] 11%|█         | 43642/400000 [00:06<00:52, 6795.22it/s] 11%|█         | 44397/400000 [00:06<00:50, 7003.18it/s] 11%|█▏        | 45156/400000 [00:06<00:49, 7168.34it/s] 11%|█▏        | 45923/400000 [00:06<00:48, 7310.02it/s] 12%|█▏        | 46691/400000 [00:06<00:47, 7415.02it/s] 12%|█▏        | 47435/400000 [00:06<00:47, 7386.03it/s] 12%|█▏        | 48176/400000 [00:06<00:48, 7271.57it/s] 12%|█▏        | 48921/400000 [00:06<00:47, 7321.83it/s] 12%|█▏        | 49655/400000 [00:06<00:49, 7141.83it/s] 13%|█▎        | 50371/400000 [00:06<00:49, 7062.87it/s] 13%|█▎        | 51079/400000 [00:07<00:50, 6965.81it/s] 13%|█▎        | 51777/400000 [00:07<00:50, 6907.57it/s] 13%|█▎        | 52469/400000 [00:07<00:50, 6887.54it/s] 13%|█▎        | 53232/400000 [00:07<00:48, 7092.69it/s] 13%|█▎        | 53989/400000 [00:07<00:47, 7228.55it/s] 14%|█▎        | 54714/400000 [00:07<00:48, 7096.45it/s] 14%|█▍        | 55449/400000 [00:07<00:48, 7168.52it/s] 14%|█▍        | 56219/400000 [00:07<00:46, 7317.53it/s] 14%|█▍        | 56973/400000 [00:07<00:46, 7382.32it/s] 14%|█▍        | 57733/400000 [00:07<00:45, 7446.15it/s] 15%|█▍        | 58485/400000 [00:08<00:45, 7468.07it/s] 15%|█▍        | 59239/400000 [00:08<00:45, 7487.64it/s] 15%|█▍        | 59989/400000 [00:08<00:45, 7477.26it/s] 15%|█▌        | 60738/400000 [00:08<00:45, 7434.20it/s] 15%|█▌        | 61482/400000 [00:08<00:46, 7232.10it/s] 16%|█▌        | 62207/400000 [00:08<00:49, 6858.98it/s] 16%|█▌        | 62925/400000 [00:08<00:48, 6950.96it/s] 16%|█▌        | 63690/400000 [00:08<00:47, 7146.01it/s] 16%|█▌        | 64437/400000 [00:08<00:46, 7237.79it/s] 16%|█▋        | 65202/400000 [00:08<00:45, 7356.47it/s] 16%|█▋        | 65941/400000 [00:09<00:45, 7366.15it/s] 17%|█▋        | 66684/400000 [00:09<00:45, 7383.70it/s] 17%|█▋        | 67449/400000 [00:09<00:44, 7460.34it/s] 17%|█▋        | 68197/400000 [00:09<00:44, 7432.05it/s] 17%|█▋        | 68959/400000 [00:09<00:44, 7484.89it/s] 17%|█▋        | 69709/400000 [00:09<00:44, 7458.61it/s] 18%|█▊        | 70456/400000 [00:09<00:45, 7314.50it/s] 18%|█▊        | 71222/400000 [00:09<00:44, 7413.92it/s] 18%|█▊        | 71985/400000 [00:09<00:43, 7475.74it/s] 18%|█▊        | 72734/400000 [00:09<00:43, 7460.07it/s] 18%|█▊        | 73481/400000 [00:10<00:45, 7208.32it/s] 19%|█▊        | 74225/400000 [00:10<00:44, 7274.37it/s] 19%|█▊        | 74988/400000 [00:10<00:44, 7376.25it/s] 19%|█▉        | 75750/400000 [00:10<00:43, 7446.94it/s] 19%|█▉        | 76496/400000 [00:10<00:43, 7429.13it/s] 19%|█▉        | 77240/400000 [00:10<00:43, 7348.17it/s] 20%|█▉        | 78001/400000 [00:10<00:43, 7422.51it/s] 20%|█▉        | 78756/400000 [00:10<00:43, 7458.28it/s] 20%|█▉        | 79503/400000 [00:10<00:43, 7452.47it/s] 20%|██        | 80250/400000 [00:11<00:42, 7456.03it/s] 20%|██        | 80996/400000 [00:11<00:42, 7434.53it/s] 20%|██        | 81747/400000 [00:11<00:42, 7456.71it/s] 21%|██        | 82508/400000 [00:11<00:42, 7499.87it/s] 21%|██        | 83259/400000 [00:11<00:42, 7412.36it/s] 21%|██        | 84001/400000 [00:11<00:43, 7222.16it/s] 21%|██        | 84725/400000 [00:11<00:44, 7057.18it/s] 21%|██▏       | 85487/400000 [00:11<00:43, 7215.23it/s] 22%|██▏       | 86236/400000 [00:11<00:43, 7294.16it/s] 22%|██▏       | 86968/400000 [00:11<00:42, 7281.19it/s] 22%|██▏       | 87733/400000 [00:12<00:42, 7386.71it/s] 22%|██▏       | 88495/400000 [00:12<00:41, 7452.42it/s] 22%|██▏       | 89263/400000 [00:12<00:41, 7518.23it/s] 23%|██▎       | 90016/400000 [00:12<00:41, 7477.16it/s] 23%|██▎       | 90780/400000 [00:12<00:41, 7525.03it/s] 23%|██▎       | 91541/400000 [00:12<00:40, 7547.93it/s] 23%|██▎       | 92297/400000 [00:12<00:42, 7315.25it/s] 23%|██▎       | 93056/400000 [00:12<00:41, 7395.40it/s] 23%|██▎       | 93798/400000 [00:12<00:41, 7316.05it/s] 24%|██▎       | 94531/400000 [00:12<00:42, 7271.50it/s] 24%|██▍       | 95260/400000 [00:13<00:42, 7146.62it/s] 24%|██▍       | 95976/400000 [00:13<00:43, 6980.75it/s] 24%|██▍       | 96694/400000 [00:13<00:43, 7037.53it/s] 24%|██▍       | 97443/400000 [00:13<00:42, 7165.91it/s] 25%|██▍       | 98177/400000 [00:13<00:41, 7214.99it/s] 25%|██▍       | 98903/400000 [00:13<00:41, 7226.57it/s] 25%|██▍       | 99627/400000 [00:13<00:42, 7035.23it/s] 25%|██▌       | 100333/400000 [00:13<00:43, 6961.69it/s] 25%|██▌       | 101031/400000 [00:13<00:43, 6804.78it/s] 25%|██▌       | 101714/400000 [00:13<00:44, 6685.32it/s] 26%|██▌       | 102385/400000 [00:14<00:45, 6527.86it/s] 26%|██▌       | 103087/400000 [00:14<00:44, 6666.05it/s] 26%|██▌       | 103836/400000 [00:14<00:42, 6892.99it/s] 26%|██▌       | 104599/400000 [00:14<00:41, 7097.10it/s] 26%|██▋       | 105356/400000 [00:14<00:40, 7231.44it/s] 27%|██▋       | 106083/400000 [00:14<00:42, 6837.93it/s] 27%|██▋       | 106774/400000 [00:14<00:42, 6831.87it/s] 27%|██▋       | 107462/400000 [00:14<00:43, 6741.01it/s] 27%|██▋       | 108140/400000 [00:14<00:43, 6724.05it/s] 27%|██▋       | 108815/400000 [00:15<00:43, 6730.86it/s] 27%|██▋       | 109490/400000 [00:15<00:43, 6700.34it/s] 28%|██▊       | 110173/400000 [00:15<00:43, 6736.72it/s] 28%|██▊       | 110848/400000 [00:15<00:42, 6731.34it/s] 28%|██▊       | 111558/400000 [00:15<00:42, 6837.69it/s] 28%|██▊       | 112313/400000 [00:15<00:40, 7034.87it/s] 28%|██▊       | 113019/400000 [00:15<00:41, 6944.51it/s] 28%|██▊       | 113781/400000 [00:15<00:40, 7132.08it/s] 29%|██▊       | 114530/400000 [00:15<00:39, 7235.26it/s] 29%|██▉       | 115256/400000 [00:15<00:40, 7101.59it/s] 29%|██▉       | 115969/400000 [00:16<00:41, 6901.42it/s] 29%|██▉       | 116699/400000 [00:16<00:40, 7014.28it/s] 29%|██▉       | 117417/400000 [00:16<00:40, 7060.68it/s] 30%|██▉       | 118169/400000 [00:16<00:39, 7191.44it/s] 30%|██▉       | 118920/400000 [00:16<00:38, 7284.02it/s] 30%|██▉       | 119662/400000 [00:16<00:38, 7322.05it/s] 30%|███       | 120396/400000 [00:16<00:39, 7127.98it/s] 30%|███       | 121161/400000 [00:16<00:38, 7274.42it/s] 30%|███       | 121912/400000 [00:16<00:37, 7341.90it/s] 31%|███       | 122673/400000 [00:16<00:37, 7418.90it/s] 31%|███       | 123417/400000 [00:17<00:37, 7413.82it/s] 31%|███       | 124160/400000 [00:17<00:37, 7415.84it/s] 31%|███       | 124903/400000 [00:17<00:37, 7279.18it/s] 31%|███▏      | 125657/400000 [00:17<00:37, 7355.38it/s] 32%|███▏      | 126394/400000 [00:17<00:37, 7359.51it/s] 32%|███▏      | 127145/400000 [00:17<00:36, 7403.94it/s] 32%|███▏      | 127886/400000 [00:17<00:36, 7379.04it/s] 32%|███▏      | 128638/400000 [00:17<00:36, 7419.77it/s] 32%|███▏      | 129381/400000 [00:17<00:36, 7399.06it/s] 33%|███▎      | 130122/400000 [00:17<00:36, 7374.03it/s] 33%|███▎      | 130882/400000 [00:18<00:36, 7437.93it/s] 33%|███▎      | 131627/400000 [00:18<00:36, 7413.29it/s] 33%|███▎      | 132369/400000 [00:18<00:36, 7282.60it/s] 33%|███▎      | 133098/400000 [00:18<00:37, 7128.53it/s] 33%|███▎      | 133813/400000 [00:18<00:37, 7044.49it/s] 34%|███▎      | 134519/400000 [00:18<00:38, 6957.83it/s] 34%|███▍      | 135216/400000 [00:18<00:38, 6854.45it/s] 34%|███▍      | 135903/400000 [00:18<00:39, 6759.12it/s] 34%|███▍      | 136627/400000 [00:18<00:38, 6895.66it/s] 34%|███▍      | 137318/400000 [00:19<00:38, 6829.09it/s] 35%|███▍      | 138012/400000 [00:19<00:38, 6860.09it/s] 35%|███▍      | 138699/400000 [00:19<00:38, 6794.59it/s] 35%|███▍      | 139393/400000 [00:19<00:38, 6835.63it/s] 35%|███▌      | 140078/400000 [00:19<00:38, 6837.10it/s] 35%|███▌      | 140763/400000 [00:19<00:38, 6809.99it/s] 35%|███▌      | 141499/400000 [00:19<00:37, 6965.58it/s] 36%|███▌      | 142255/400000 [00:19<00:36, 7133.17it/s] 36%|███▌      | 143028/400000 [00:19<00:35, 7300.38it/s] 36%|███▌      | 143761/400000 [00:19<00:35, 7265.70it/s] 36%|███▌      | 144512/400000 [00:20<00:34, 7336.60it/s] 36%|███▋      | 145259/400000 [00:20<00:34, 7375.28it/s] 36%|███▋      | 145998/400000 [00:20<00:34, 7329.03it/s] 37%|███▋      | 146732/400000 [00:20<00:35, 7204.47it/s] 37%|███▋      | 147454/400000 [00:20<00:35, 7099.77it/s] 37%|███▋      | 148166/400000 [00:20<00:35, 7037.55it/s] 37%|███▋      | 148880/400000 [00:20<00:35, 7067.53it/s] 37%|███▋      | 149588/400000 [00:20<00:35, 6961.31it/s] 38%|███▊      | 150285/400000 [00:20<00:36, 6908.47it/s] 38%|███▊      | 150977/400000 [00:20<00:36, 6884.05it/s] 38%|███▊      | 151666/400000 [00:21<00:36, 6837.26it/s] 38%|███▊      | 152351/400000 [00:21<00:36, 6808.28it/s] 38%|███▊      | 153066/400000 [00:21<00:35, 6906.37it/s] 38%|███▊      | 153829/400000 [00:21<00:34, 7107.22it/s] 39%|███▊      | 154596/400000 [00:21<00:33, 7265.11it/s] 39%|███▉      | 155363/400000 [00:21<00:33, 7379.97it/s] 39%|███▉      | 156126/400000 [00:21<00:32, 7452.82it/s] 39%|███▉      | 156873/400000 [00:21<00:33, 7238.25it/s] 39%|███▉      | 157600/400000 [00:21<00:34, 7089.91it/s] 40%|███▉      | 158312/400000 [00:21<00:34, 7067.76it/s] 40%|███▉      | 159021/400000 [00:22<00:34, 7000.78it/s] 40%|███▉      | 159773/400000 [00:22<00:33, 7147.27it/s] 40%|████      | 160518/400000 [00:22<00:33, 7233.60it/s] 40%|████      | 161297/400000 [00:22<00:32, 7390.64it/s] 41%|████      | 162065/400000 [00:22<00:31, 7474.14it/s] 41%|████      | 162831/400000 [00:22<00:31, 7527.58it/s] 41%|████      | 163609/400000 [00:22<00:31, 7600.00it/s] 41%|████      | 164370/400000 [00:22<00:31, 7575.88it/s] 41%|████▏     | 165129/400000 [00:22<00:31, 7376.53it/s] 41%|████▏     | 165869/400000 [00:22<00:32, 7225.20it/s] 42%|████▏     | 166645/400000 [00:23<00:31, 7377.20it/s] 42%|████▏     | 167412/400000 [00:23<00:31, 7460.43it/s] 42%|████▏     | 168169/400000 [00:23<00:30, 7490.95it/s] 42%|████▏     | 168928/400000 [00:23<00:30, 7517.87it/s] 42%|████▏     | 169694/400000 [00:23<00:30, 7559.47it/s] 43%|████▎     | 170451/400000 [00:23<00:31, 7359.33it/s] 43%|████▎     | 171189/400000 [00:23<00:31, 7229.17it/s] 43%|████▎     | 171914/400000 [00:23<00:32, 7076.89it/s] 43%|████▎     | 172624/400000 [00:23<00:32, 7037.75it/s] 43%|████▎     | 173370/400000 [00:24<00:31, 7158.33it/s] 44%|████▎     | 174137/400000 [00:24<00:30, 7302.07it/s] 44%|████▎     | 174906/400000 [00:24<00:30, 7411.59it/s] 44%|████▍     | 175659/400000 [00:24<00:30, 7446.70it/s] 44%|████▍     | 176408/400000 [00:24<00:29, 7458.12it/s] 44%|████▍     | 177155/400000 [00:24<00:30, 7250.21it/s] 44%|████▍     | 177882/400000 [00:24<00:31, 7030.10it/s] 45%|████▍     | 178637/400000 [00:24<00:30, 7178.17it/s] 45%|████▍     | 179379/400000 [00:24<00:30, 7248.91it/s] 45%|████▌     | 180149/400000 [00:24<00:29, 7377.85it/s] 45%|████▌     | 180889/400000 [00:25<00:30, 7218.56it/s] 45%|████▌     | 181614/400000 [00:25<00:30, 7102.46it/s] 46%|████▌     | 182327/400000 [00:25<00:30, 7034.26it/s] 46%|████▌     | 183032/400000 [00:25<00:31, 6875.00it/s] 46%|████▌     | 183722/400000 [00:25<00:31, 6850.34it/s] 46%|████▌     | 184409/400000 [00:25<00:32, 6710.79it/s] 46%|████▋     | 185088/400000 [00:25<00:31, 6732.18it/s] 46%|████▋     | 185784/400000 [00:25<00:31, 6789.72it/s] 47%|████▋     | 186464/400000 [00:25<00:31, 6756.63it/s] 47%|████▋     | 187150/400000 [00:25<00:31, 6787.10it/s] 47%|████▋     | 187839/400000 [00:26<00:31, 6817.45it/s] 47%|████▋     | 188522/400000 [00:26<00:31, 6809.32it/s] 47%|████▋     | 189204/400000 [00:26<00:31, 6779.86it/s] 47%|████▋     | 189966/400000 [00:26<00:29, 7011.20it/s] 48%|████▊     | 190733/400000 [00:26<00:29, 7194.31it/s] 48%|████▊     | 191456/400000 [00:26<00:29, 7166.97it/s] 48%|████▊     | 192175/400000 [00:26<00:29, 7080.06it/s] 48%|████▊     | 192885/400000 [00:26<00:29, 7028.92it/s] 48%|████▊     | 193590/400000 [00:26<00:29, 6909.92it/s] 49%|████▊     | 194283/400000 [00:26<00:29, 6900.16it/s] 49%|████▉     | 195030/400000 [00:27<00:29, 7060.19it/s] 49%|████▉     | 195804/400000 [00:27<00:28, 7250.80it/s] 49%|████▉     | 196580/400000 [00:27<00:27, 7396.23it/s] 49%|████▉     | 197322/400000 [00:27<00:27, 7378.79it/s] 50%|████▉     | 198094/400000 [00:27<00:27, 7475.96it/s] 50%|████▉     | 198864/400000 [00:27<00:26, 7540.06it/s] 50%|████▉     | 199620/400000 [00:27<00:26, 7542.15it/s] 50%|█████     | 200376/400000 [00:27<00:26, 7408.36it/s] 50%|█████     | 201118/400000 [00:27<00:27, 7104.34it/s] 50%|█████     | 201835/400000 [00:28<00:27, 7121.34it/s] 51%|█████     | 202612/400000 [00:28<00:27, 7302.59it/s] 51%|█████     | 203346/400000 [00:28<00:26, 7283.94it/s] 51%|█████     | 204121/400000 [00:28<00:26, 7416.79it/s] 51%|█████     | 204865/400000 [00:28<00:26, 7375.82it/s] 51%|█████▏    | 205647/400000 [00:28<00:25, 7501.84it/s] 52%|█████▏    | 206411/400000 [00:28<00:25, 7540.46it/s] 52%|█████▏    | 207191/400000 [00:28<00:25, 7614.58it/s] 52%|█████▏    | 207961/400000 [00:28<00:25, 7637.82it/s] 52%|█████▏    | 208726/400000 [00:28<00:25, 7458.17it/s] 52%|█████▏    | 209474/400000 [00:29<00:26, 7224.40it/s] 53%|█████▎    | 210200/400000 [00:29<00:26, 7202.63it/s] 53%|█████▎    | 210959/400000 [00:29<00:25, 7313.50it/s] 53%|█████▎    | 211701/400000 [00:29<00:25, 7344.16it/s] 53%|█████▎    | 212442/400000 [00:29<00:25, 7362.67it/s] 53%|█████▎    | 213208/400000 [00:29<00:25, 7448.81it/s] 53%|█████▎    | 213954/400000 [00:29<00:25, 7397.26it/s] 54%|█████▎    | 214705/400000 [00:29<00:24, 7430.11it/s] 54%|█████▍    | 215480/400000 [00:29<00:24, 7520.85it/s] 54%|█████▍    | 216233/400000 [00:29<00:24, 7512.15it/s] 54%|█████▍    | 217001/400000 [00:30<00:24, 7559.42it/s] 54%|█████▍    | 217772/400000 [00:30<00:23, 7602.38it/s] 55%|█████▍    | 218543/400000 [00:30<00:23, 7631.73it/s] 55%|█████▍    | 219319/400000 [00:30<00:23, 7667.20it/s] 55%|█████▌    | 220086/400000 [00:30<00:23, 7657.03it/s] 55%|█████▌    | 220865/400000 [00:30<00:23, 7694.77it/s] 55%|█████▌    | 221640/400000 [00:30<00:23, 7709.82it/s] 56%|█████▌    | 222416/400000 [00:30<00:22, 7724.54it/s] 56%|█████▌    | 223189/400000 [00:30<00:23, 7477.66it/s] 56%|█████▌    | 223939/400000 [00:30<00:24, 7215.68it/s] 56%|█████▌    | 224664/400000 [00:31<00:25, 7006.52it/s] 56%|█████▋    | 225369/400000 [00:31<00:25, 6948.29it/s] 57%|█████▋    | 226136/400000 [00:31<00:24, 7149.60it/s] 57%|█████▋    | 226880/400000 [00:31<00:23, 7233.78it/s] 57%|█████▋    | 227641/400000 [00:31<00:23, 7341.55it/s] 57%|█████▋    | 228416/400000 [00:31<00:23, 7457.34it/s] 57%|█████▋    | 229164/400000 [00:31<00:22, 7435.65it/s] 57%|█████▋    | 229920/400000 [00:31<00:22, 7471.68it/s] 58%|█████▊    | 230669/400000 [00:31<00:22, 7382.51it/s] 58%|█████▊    | 231409/400000 [00:31<00:22, 7385.74it/s] 58%|█████▊    | 232175/400000 [00:32<00:22, 7465.89it/s] 58%|█████▊    | 232943/400000 [00:32<00:22, 7527.71it/s] 58%|█████▊    | 233699/400000 [00:32<00:22, 7534.96it/s] 59%|█████▊    | 234468/400000 [00:32<00:21, 7579.98it/s] 59%|█████▉    | 235229/400000 [00:32<00:21, 7586.39it/s] 59%|█████▉    | 235988/400000 [00:32<00:21, 7535.72it/s] 59%|█████▉    | 236753/400000 [00:32<00:21, 7567.55it/s] 59%|█████▉    | 237529/400000 [00:32<00:21, 7622.08it/s] 60%|█████▉    | 238296/400000 [00:32<00:21, 7635.09it/s] 60%|█████▉    | 239060/400000 [00:33<00:21, 7497.51it/s] 60%|█████▉    | 239836/400000 [00:33<00:21, 7572.32it/s] 60%|██████    | 240594/400000 [00:33<00:21, 7478.89it/s] 60%|██████    | 241370/400000 [00:33<00:20, 7560.69it/s] 61%|██████    | 242127/400000 [00:33<00:21, 7361.09it/s] 61%|██████    | 242865/400000 [00:33<00:22, 7026.47it/s] 61%|██████    | 243640/400000 [00:33<00:21, 7227.65it/s] 61%|██████    | 244400/400000 [00:33<00:21, 7326.49it/s] 61%|██████▏   | 245169/400000 [00:33<00:20, 7430.46it/s] 61%|██████▏   | 245922/400000 [00:33<00:20, 7459.82it/s] 62%|██████▏   | 246680/400000 [00:34<00:20, 7494.09it/s] 62%|██████▏   | 247460/400000 [00:34<00:20, 7583.22it/s] 62%|██████▏   | 248226/400000 [00:34<00:19, 7604.96it/s] 62%|██████▏   | 248999/400000 [00:34<00:19, 7639.38it/s] 62%|██████▏   | 249771/400000 [00:34<00:19, 7661.69it/s] 63%|██████▎   | 250538/400000 [00:34<00:19, 7626.80it/s] 63%|██████▎   | 251311/400000 [00:34<00:19, 7656.02it/s] 63%|██████▎   | 252086/400000 [00:34<00:19, 7681.62it/s] 63%|██████▎   | 252855/400000 [00:34<00:19, 7514.41it/s] 63%|██████▎   | 253608/400000 [00:34<00:20, 7283.73it/s] 64%|██████▎   | 254343/400000 [00:35<00:19, 7302.13it/s] 64%|██████▍   | 255116/400000 [00:35<00:19, 7423.96it/s] 64%|██████▍   | 255890/400000 [00:35<00:19, 7513.96it/s] 64%|██████▍   | 256664/400000 [00:35<00:18, 7579.19it/s] 64%|██████▍   | 257426/400000 [00:35<00:18, 7589.55it/s] 65%|██████▍   | 258187/400000 [00:35<00:18, 7594.13it/s] 65%|██████▍   | 258959/400000 [00:35<00:18, 7628.37it/s] 65%|██████▍   | 259736/400000 [00:35<00:18, 7668.73it/s] 65%|██████▌   | 260504/400000 [00:35<00:18, 7658.90it/s] 65%|██████▌   | 261279/400000 [00:35<00:18, 7685.12it/s] 66%|██████▌   | 262048/400000 [00:36<00:18, 7339.51it/s] 66%|██████▌   | 262786/400000 [00:36<00:19, 7144.58it/s] 66%|██████▌   | 263504/400000 [00:36<00:19, 7034.14it/s] 66%|██████▌   | 264211/400000 [00:36<00:19, 7005.39it/s] 66%|██████▌   | 264956/400000 [00:36<00:18, 7131.79it/s] 66%|██████▋   | 265718/400000 [00:36<00:18, 7270.09it/s] 67%|██████▋   | 266448/400000 [00:36<00:18, 7173.77it/s] 67%|██████▋   | 267168/400000 [00:36<00:18, 7039.03it/s] 67%|██████▋   | 267874/400000 [00:36<00:18, 6957.82it/s] 67%|██████▋   | 268572/400000 [00:37<00:19, 6860.72it/s] 67%|██████▋   | 269260/400000 [00:37<00:19, 6808.55it/s] 67%|██████▋   | 269995/400000 [00:37<00:18, 6959.89it/s] 68%|██████▊   | 270720/400000 [00:37<00:18, 7044.32it/s] 68%|██████▊   | 271447/400000 [00:37<00:18, 7109.58it/s] 68%|██████▊   | 272187/400000 [00:37<00:17, 7192.52it/s] 68%|██████▊   | 272931/400000 [00:37<00:17, 7263.01it/s] 68%|██████▊   | 273702/400000 [00:37<00:17, 7389.42it/s] 69%|██████▊   | 274443/400000 [00:37<00:17, 7346.06it/s] 69%|██████▉   | 275185/400000 [00:37<00:16, 7366.67it/s] 69%|██████▉   | 275923/400000 [00:38<00:17, 7267.68it/s] 69%|██████▉   | 276669/400000 [00:38<00:16, 7322.51it/s] 69%|██████▉   | 277440/400000 [00:38<00:16, 7433.31it/s] 70%|██████▉   | 278191/400000 [00:38<00:16, 7453.95it/s] 70%|██████▉   | 278948/400000 [00:38<00:16, 7486.44it/s] 70%|██████▉   | 279709/400000 [00:38<00:15, 7523.05it/s] 70%|███████   | 280462/400000 [00:38<00:16, 7402.99it/s] 70%|███████   | 281229/400000 [00:38<00:15, 7480.14it/s] 70%|███████   | 281978/400000 [00:38<00:15, 7475.93it/s] 71%|███████   | 282727/400000 [00:38<00:15, 7470.48it/s] 71%|███████   | 283481/400000 [00:39<00:15, 7489.60it/s] 71%|███████   | 284231/400000 [00:39<00:15, 7383.80it/s] 71%|███████   | 284981/400000 [00:39<00:15, 7416.66it/s] 71%|███████▏  | 285724/400000 [00:39<00:15, 7418.95it/s] 72%|███████▏  | 286476/400000 [00:39<00:15, 7447.79it/s] 72%|███████▏  | 287224/400000 [00:39<00:15, 7456.55it/s] 72%|███████▏  | 287970/400000 [00:39<00:15, 7321.67it/s] 72%|███████▏  | 288703/400000 [00:39<00:16, 6955.14it/s] 72%|███████▏  | 289403/400000 [00:39<00:16, 6876.93it/s] 73%|███████▎  | 290125/400000 [00:39<00:15, 6975.48it/s] 73%|███████▎  | 290862/400000 [00:40<00:15, 7087.95it/s] 73%|███████▎  | 291611/400000 [00:40<00:15, 7201.80it/s] 73%|███████▎  | 292357/400000 [00:40<00:14, 7274.95it/s] 73%|███████▎  | 293113/400000 [00:40<00:14, 7357.56it/s] 73%|███████▎  | 293861/400000 [00:40<00:14, 7392.09it/s] 74%|███████▎  | 294617/400000 [00:40<00:14, 7439.54it/s] 74%|███████▍  | 295377/400000 [00:40<00:13, 7484.30it/s] 74%|███████▍  | 296147/400000 [00:40<00:13, 7545.79it/s] 74%|███████▍  | 296903/400000 [00:40<00:13, 7412.55it/s] 74%|███████▍  | 297650/400000 [00:40<00:13, 7427.66it/s] 75%|███████▍  | 298394/400000 [00:41<00:13, 7374.48it/s] 75%|███████▍  | 299144/400000 [00:41<00:13, 7411.22it/s] 75%|███████▍  | 299893/400000 [00:41<00:13, 7432.85it/s] 75%|███████▌  | 300647/400000 [00:41<00:13, 7464.16it/s] 75%|███████▌  | 301394/400000 [00:41<00:13, 7336.44it/s] 76%|███████▌  | 302129/400000 [00:41<00:13, 7196.28it/s] 76%|███████▌  | 302892/400000 [00:41<00:13, 7320.23it/s] 76%|███████▌  | 303661/400000 [00:41<00:12, 7425.52it/s] 76%|███████▌  | 304421/400000 [00:41<00:12, 7475.21it/s] 76%|███████▋  | 305180/400000 [00:41<00:12, 7507.50it/s] 76%|███████▋  | 305932/400000 [00:42<00:12, 7502.21it/s] 77%|███████▋  | 306688/400000 [00:42<00:12, 7516.72it/s] 77%|███████▋  | 307441/400000 [00:42<00:12, 7515.47it/s] 77%|███████▋  | 308199/400000 [00:42<00:12, 7529.33it/s] 77%|███████▋  | 308953/400000 [00:42<00:12, 7336.35it/s] 77%|███████▋  | 309688/400000 [00:42<00:12, 7141.99it/s] 78%|███████▊  | 310428/400000 [00:42<00:12, 7216.68it/s] 78%|███████▊  | 311175/400000 [00:42<00:12, 7289.80it/s] 78%|███████▊  | 311906/400000 [00:42<00:12, 7223.40it/s] 78%|███████▊  | 312682/400000 [00:43<00:11, 7374.19it/s] 78%|███████▊  | 313438/400000 [00:43<00:11, 7427.87it/s] 79%|███████▊  | 314203/400000 [00:43<00:11, 7490.74it/s] 79%|███████▊  | 314957/400000 [00:43<00:11, 7504.26it/s] 79%|███████▉  | 315709/400000 [00:43<00:11, 7479.27it/s] 79%|███████▉  | 316458/400000 [00:43<00:11, 7477.59it/s] 79%|███████▉  | 317207/400000 [00:43<00:11, 7247.10it/s] 79%|███████▉  | 317934/400000 [00:43<00:11, 7150.45it/s] 80%|███████▉  | 318651/400000 [00:43<00:11, 7033.77it/s] 80%|███████▉  | 319381/400000 [00:43<00:11, 7110.34it/s] 80%|████████  | 320103/400000 [00:44<00:11, 7142.44it/s] 80%|████████  | 320849/400000 [00:44<00:10, 7233.51it/s] 80%|████████  | 321592/400000 [00:44<00:10, 7290.04it/s] 81%|████████  | 322335/400000 [00:44<00:10, 7329.73it/s] 81%|████████  | 323070/400000 [00:44<00:10, 7334.39it/s] 81%|████████  | 323823/400000 [00:44<00:10, 7389.51it/s] 81%|████████  | 324563/400000 [00:44<00:10, 7170.92it/s] 81%|████████▏ | 325282/400000 [00:44<00:10, 7079.14it/s] 81%|████████▏ | 325992/400000 [00:44<00:10, 7012.48it/s] 82%|████████▏ | 326764/400000 [00:44<00:10, 7208.35it/s] 82%|████████▏ | 327534/400000 [00:45<00:09, 7348.55it/s] 82%|████████▏ | 328289/400000 [00:45<00:09, 7406.46it/s] 82%|████████▏ | 329056/400000 [00:45<00:09, 7481.71it/s] 82%|████████▏ | 329827/400000 [00:45<00:09, 7546.32it/s] 83%|████████▎ | 330583/400000 [00:45<00:09, 7476.76it/s] 83%|████████▎ | 331343/400000 [00:45<00:09, 7513.30it/s] 83%|████████▎ | 332101/400000 [00:45<00:09, 7532.14it/s] 83%|████████▎ | 332867/400000 [00:45<00:08, 7567.21it/s] 83%|████████▎ | 333643/400000 [00:45<00:08, 7623.56it/s] 84%|████████▎ | 334415/400000 [00:45<00:08, 7650.99it/s] 84%|████████▍ | 335181/400000 [00:46<00:08, 7495.69it/s] 84%|████████▍ | 335932/400000 [00:46<00:08, 7280.21it/s] 84%|████████▍ | 336663/400000 [00:46<00:08, 7109.60it/s] 84%|████████▍ | 337377/400000 [00:46<00:08, 7011.48it/s] 85%|████████▍ | 338081/400000 [00:46<00:08, 6952.92it/s] 85%|████████▍ | 338813/400000 [00:46<00:08, 7056.81it/s] 85%|████████▍ | 339525/400000 [00:46<00:08, 7072.67it/s] 85%|████████▌ | 340288/400000 [00:46<00:08, 7228.59it/s] 85%|████████▌ | 341048/400000 [00:46<00:08, 7333.56it/s] 85%|████████▌ | 341810/400000 [00:46<00:07, 7415.97it/s] 86%|████████▌ | 342563/400000 [00:47<00:07, 7447.96it/s] 86%|████████▌ | 343309/400000 [00:47<00:07, 7285.00it/s] 86%|████████▌ | 344039/400000 [00:47<00:07, 7173.39it/s] 86%|████████▌ | 344758/400000 [00:47<00:07, 7103.84it/s] 86%|████████▋ | 345470/400000 [00:47<00:07, 6914.86it/s] 87%|████████▋ | 346164/400000 [00:47<00:07, 6867.34it/s] 87%|████████▋ | 346853/400000 [00:47<00:07, 6775.18it/s] 87%|████████▋ | 347551/400000 [00:47<00:07, 6833.43it/s] 87%|████████▋ | 348296/400000 [00:47<00:07, 7005.49it/s] 87%|████████▋ | 349024/400000 [00:48<00:07, 7084.38it/s] 87%|████████▋ | 349783/400000 [00:48<00:06, 7227.93it/s] 88%|████████▊ | 350531/400000 [00:48<00:06, 7301.22it/s] 88%|████████▊ | 351271/400000 [00:48<00:06, 7328.26it/s] 88%|████████▊ | 352033/400000 [00:48<00:06, 7410.70it/s] 88%|████████▊ | 352795/400000 [00:48<00:06, 7471.48it/s] 88%|████████▊ | 353559/400000 [00:48<00:06, 7519.18it/s] 89%|████████▊ | 354312/400000 [00:48<00:06, 7506.47it/s] 89%|████████▉ | 355087/400000 [00:48<00:05, 7576.98it/s] 89%|████████▉ | 355846/400000 [00:48<00:05, 7554.61it/s] 89%|████████▉ | 356607/400000 [00:49<00:05, 7569.04it/s] 89%|████████▉ | 357365/400000 [00:49<00:05, 7548.52it/s] 90%|████████▉ | 358121/400000 [00:49<00:05, 7363.23it/s] 90%|████████▉ | 358859/400000 [00:49<00:05, 7196.45it/s] 90%|████████▉ | 359581/400000 [00:49<00:05, 7115.76it/s] 90%|█████████ | 360294/400000 [00:49<00:05, 7027.71it/s] 90%|█████████ | 360998/400000 [00:49<00:05, 6944.56it/s] 90%|█████████ | 361736/400000 [00:49<00:05, 7068.17it/s] 91%|█████████ | 362496/400000 [00:49<00:05, 7218.28it/s] 91%|█████████ | 363220/400000 [00:49<00:05, 6948.24it/s] 91%|█████████ | 363963/400000 [00:50<00:05, 7084.25it/s] 91%|█████████ | 364726/400000 [00:50<00:04, 7238.59it/s] 91%|█████████▏| 365482/400000 [00:50<00:04, 7332.00it/s] 92%|█████████▏| 366218/400000 [00:50<00:04, 7200.82it/s] 92%|█████████▏| 366941/400000 [00:50<00:04, 7004.29it/s] 92%|█████████▏| 367708/400000 [00:50<00:04, 7189.49it/s] 92%|█████████▏| 368456/400000 [00:50<00:04, 7272.62it/s] 92%|█████████▏| 369212/400000 [00:50<00:04, 7353.91it/s] 92%|█████████▏| 369973/400000 [00:50<00:04, 7428.88it/s] 93%|█████████▎| 370750/400000 [00:50<00:03, 7527.55it/s] 93%|█████████▎| 371514/400000 [00:51<00:03, 7558.42it/s] 93%|█████████▎| 372287/400000 [00:51<00:03, 7606.85it/s] 93%|█████████▎| 373049/400000 [00:51<00:03, 7581.38it/s] 93%|█████████▎| 373819/400000 [00:51<00:03, 7615.74it/s] 94%|█████████▎| 374582/400000 [00:51<00:03, 7207.03it/s] 94%|█████████▍| 375335/400000 [00:51<00:03, 7298.59it/s] 94%|█████████▍| 376074/400000 [00:51<00:03, 7312.80it/s] 94%|█████████▍| 376814/400000 [00:51<00:03, 7337.77it/s] 94%|█████████▍| 377565/400000 [00:51<00:03, 7387.03it/s] 95%|█████████▍| 378322/400000 [00:52<00:02, 7439.43it/s] 95%|█████████▍| 379067/400000 [00:52<00:02, 7423.24it/s] 95%|█████████▍| 379834/400000 [00:52<00:02, 7495.09it/s] 95%|█████████▌| 380585/400000 [00:52<00:02, 7496.24it/s] 95%|█████████▌| 381357/400000 [00:52<00:02, 7560.81it/s] 96%|█████████▌| 382135/400000 [00:52<00:02, 7623.95it/s] 96%|█████████▌| 382898/400000 [00:52<00:02, 7603.95it/s] 96%|█████████▌| 383669/400000 [00:52<00:02, 7632.99it/s] 96%|█████████▌| 384433/400000 [00:52<00:02, 7630.34it/s] 96%|█████████▋| 385202/400000 [00:52<00:01, 7647.29it/s] 96%|█████████▋| 385981/400000 [00:53<00:01, 7687.37it/s] 97%|█████████▋| 386755/400000 [00:53<00:01, 7702.20it/s] 97%|█████████▋| 387526/400000 [00:53<00:01, 7681.93it/s] 97%|█████████▋| 388295/400000 [00:53<00:01, 7458.37it/s] 97%|█████████▋| 389043/400000 [00:53<00:01, 7397.26it/s] 97%|█████████▋| 389810/400000 [00:53<00:01, 7475.69it/s] 98%|█████████▊| 390579/400000 [00:53<00:01, 7536.15it/s] 98%|█████████▊| 391351/400000 [00:53<00:01, 7589.87it/s] 98%|█████████▊| 392111/400000 [00:53<00:01, 7540.25it/s] 98%|█████████▊| 392873/400000 [00:53<00:00, 7562.22it/s] 98%|█████████▊| 393655/400000 [00:54<00:00, 7635.05it/s] 99%|█████████▊| 394427/400000 [00:54<00:00, 7659.99it/s] 99%|█████████▉| 395198/400000 [00:54<00:00, 7672.87it/s] 99%|█████████▉| 395966/400000 [00:54<00:00, 7661.48it/s] 99%|█████████▉| 396733/400000 [00:54<00:00, 7648.99it/s] 99%|█████████▉| 397514/400000 [00:54<00:00, 7696.23it/s]100%|█████████▉| 398284/400000 [00:54<00:00, 7672.73it/s]100%|█████████▉| 399052/400000 [00:54<00:00, 7568.51it/s]100%|█████████▉| 399810/400000 [00:54<00:00, 7503.06it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7291.87it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f81104a9240> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01110815984837828 	 Accuracy: 50
Train Epoch: 1 	 Loss: 0.011000253683747256 	 Accuracy: 72

  model saves at 72% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15706 out of table with 15693 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15706 out of table with 15693 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-15 03:26:25.355871: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 03:26:25.360285: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-05-15 03:26:25.360442: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561d69de1a10 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 03:26:25.360460: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f80b4fb1b00> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5593 - accuracy: 0.5070
 2000/25000 [=>............................] - ETA: 10s - loss: 7.2756 - accuracy: 0.5255
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.4060 - accuracy: 0.5170 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.4941 - accuracy: 0.5113
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5409 - accuracy: 0.5082
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5440 - accuracy: 0.5080
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5330 - accuracy: 0.5087
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5133 - accuracy: 0.5100
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5337 - accuracy: 0.5087
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5394 - accuracy: 0.5083
11000/25000 [============>.................] - ETA: 4s - loss: 7.5579 - accuracy: 0.5071
12000/25000 [=============>................] - ETA: 4s - loss: 7.5670 - accuracy: 0.5065
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5758 - accuracy: 0.5059
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5878 - accuracy: 0.5051
15000/25000 [=================>............] - ETA: 3s - loss: 7.6022 - accuracy: 0.5042
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6139 - accuracy: 0.5034
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6369 - accuracy: 0.5019
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6155 - accuracy: 0.5033
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6239 - accuracy: 0.5028
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6375 - accuracy: 0.5019
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6316 - accuracy: 0.5023
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6408 - accuracy: 0.5017
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6500 - accuracy: 0.5011
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
25000/25000 [==============================] - 9s 378us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f8085618c50> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f80856634a8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.7966 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.7139 - val_crf_viterbi_accuracy: 0.0267

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

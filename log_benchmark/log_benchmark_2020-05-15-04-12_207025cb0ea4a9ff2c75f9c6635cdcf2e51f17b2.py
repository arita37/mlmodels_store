
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fdd264d7fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 04:12:29.849592
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 04:12:29.853055
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 04:12:29.856180
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 04:12:29.859079
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fdd324ee4a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354176.9688
Epoch 2/10

1/1 [==============================] - 0s 104ms/step - loss: 309481.7500
Epoch 3/10

1/1 [==============================] - 0s 94ms/step - loss: 215695.4844
Epoch 4/10

1/1 [==============================] - 0s 94ms/step - loss: 141338.5938
Epoch 5/10

1/1 [==============================] - 0s 100ms/step - loss: 85879.3125
Epoch 6/10

1/1 [==============================] - 0s 104ms/step - loss: 51279.4609
Epoch 7/10

1/1 [==============================] - 0s 92ms/step - loss: 31763.5039
Epoch 8/10

1/1 [==============================] - 0s 102ms/step - loss: 20753.8672
Epoch 9/10

1/1 [==============================] - 0s 95ms/step - loss: 14375.8262
Epoch 10/10

1/1 [==============================] - 0s 93ms/step - loss: 10580.9854

  #### Inference Need return ypred, ytrue ######################### 
[[-0.804341    0.00932997 -1.1500252  -0.35164607 -0.8287134  -0.26949993
   0.5038769  -0.8249967   0.00884283 -0.33879548  0.72154766 -0.07563341
  -0.8327896   0.38989794 -0.9815205  -0.15642211  0.9781268  -0.24276175
  -0.38605368  0.2815434   0.19350123  0.4965034  -0.6456026   0.01138207
   1.8739262   1.5836968  -0.53811383  0.5247141  -0.85862666  0.28116244
   0.57155854 -1.3411114  -0.159244    0.2074151   1.1072842   0.6815835
   0.21262088 -0.9525882  -0.4318217  -0.12969786 -1.0310347  -0.31292987
  -0.88164914 -0.15795606  0.06928605  1.0601189  -0.2223224  -0.5137639
  -0.50429094 -0.27438486  0.50897586  1.1933563   0.08586064  0.06664379
   0.04663545  1.4473816  -0.97321033  0.6234653   1.0108237   0.13063323
  -0.02070314 -1.3201697   0.9610534   0.76393664  0.42877144 -0.9356669
   0.61567587  0.30826145 -0.93883866  0.81330717  0.07264553 -1.3516734
   0.9415059  -0.7569955  -0.16816488 -0.10661962 -1.4253676  -0.6936481
   0.11278065 -1.1690917  -0.93966395  1.1375045   0.68558705  1.7471241
  -0.44749275  0.16074343 -0.18107358 -0.68516976 -0.51071537  1.2842422
   0.5003234   1.4514675  -0.43519652 -0.11301909 -1.2821203  -0.56178653
  -0.3829039  -0.15021427 -0.21580833  0.44804204  0.05952404  0.80244684
  -0.13246685 -0.6980786  -0.47248253  0.3189448   0.69889295  1.3029555
   1.2105284  -0.09871468 -0.9558893   1.4211428   0.02088282 -0.9721544
  -0.2877857   1.1816747  -0.31490585 -1.0279427  -0.0938319  -0.49179664
  -0.10063006  5.5081434   4.9732704   6.3833547   7.404354    7.6207943
   5.5682173   6.7227545   7.050153    6.430678    6.098614    5.814884
   6.550947    5.4697576   6.678646    6.3123274   7.4861      6.405501
   4.1049194   6.507984    6.5276284   6.0243487   5.640422    5.696513
   5.8538637   7.4227924   5.7772818   5.7847414   6.26764     5.033425
   6.572919    6.118624    4.7955337   5.729824    5.3860826   7.247592
   5.723868    6.2814884   6.3578863   5.067916    6.5440464   5.6230354
   5.9929934   6.369809    5.2441473   5.653842    6.0632043   6.6131034
   6.671998    5.991453    5.6905937   3.9544373   5.4888477   5.9067054
   4.8386326   6.2807007   6.374881    6.4374704   4.9951024   4.7338467
   1.1034572   0.38618183  2.5770788   1.2771277   0.44532996  0.8910461
   0.60797274  0.31539488  0.9818595   1.0755343   0.92385393  0.9334419
   1.3673556   2.0752466   0.99710554  0.46808195  2.0174298   1.9692686
   0.7755534   0.7516096   1.4924684   1.3288636   0.62960935  1.565047
   0.69456     1.3230741   1.0355821   0.40971577  1.801336    1.5196289
   0.8086748   0.29533565  0.33618355  0.66150343  1.9780638   1.1164454
   1.5875847   0.80335206  1.3658919   0.5344489   0.4844421   0.37109172
   2.0730176   0.9110716   0.50312275  0.3791138   1.2628919   0.98116255
   1.6996818   1.6353791   1.1395354   0.6594722   0.68440706  1.2083142
   1.7034602   2.3715513   0.2448262   0.92834294  1.5578169   1.6116651
   1.8927964   2.108214    1.6082003   0.64319193  0.27848375  0.5468856
   0.3557539   0.29047883  0.5666896   1.6603781   0.5535547   0.6494683
   2.265656    2.0421696   0.33075047  0.88488716  1.370501    2.5186975
   0.32128483  1.7196021   0.32017136  2.3177187   0.734687    0.4802984
   0.89893305  0.9008478   1.5013658   0.60621554  0.33549988  1.4885103
   1.4637574   1.4631934   0.7097156   0.7648295   0.7029399   2.8158698
   0.60166013  1.8495858   1.6086407   1.2785442   0.72025573  0.8363898
   0.75895387  0.9470871   2.1033263   0.22228867  1.0887141   1.048139
   0.33571428  1.6557436   0.945431    2.1251984   0.33471954  0.9145283
   0.81585705  0.7310296   1.2847974   0.2931242   0.64083606  0.94870764
   0.0459516   5.930858    6.3006415   6.116192    7.5468597   7.153093
   5.8151197   7.388033    6.391628    6.679525    6.25098     6.3708854
   6.571673    6.68734     7.033926    7.395572    7.012331    6.592471
   6.2226725   5.2221317   7.191213    6.346519    5.805517    6.6041274
   7.771456    6.3358536   5.936315    6.708671    7.060966    5.271005
   7.556344    8.014986    6.861403    6.278537    6.254307    7.453386
   7.128025    7.0838175   6.7444243   7.1066294   7.1461453   6.4228864
   6.69621     6.555469    7.66763     7.2801642   6.1413593   5.674098
   6.325636    6.336843    6.9257      5.963325    5.990771    6.525727
   7.084184    7.4393373   5.989395    6.999707    6.7847033   7.818555
  -7.574286   -3.7598743   5.0079713 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 04:12:39.393301
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.7959
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 04:12:39.397557
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9387.21
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 04:12:39.401203
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.6824
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 04:12:39.404641
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -839.691
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140587450507336
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140586508919416
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140586508919920
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140586508920424
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140586508920928
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140586508921432

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fdd2e36fef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.604960
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.566590
grad_step = 000002, loss = 0.535323
grad_step = 000003, loss = 0.502535
grad_step = 000004, loss = 0.468098
grad_step = 000005, loss = 0.438103
grad_step = 000006, loss = 0.420559
grad_step = 000007, loss = 0.409862
grad_step = 000008, loss = 0.391049
grad_step = 000009, loss = 0.367679
grad_step = 000010, loss = 0.349560
grad_step = 000011, loss = 0.337609
grad_step = 000012, loss = 0.328280
grad_step = 000013, loss = 0.318126
grad_step = 000014, loss = 0.305952
grad_step = 000015, loss = 0.292691
grad_step = 000016, loss = 0.280767
grad_step = 000017, loss = 0.270765
grad_step = 000018, loss = 0.260980
grad_step = 000019, loss = 0.250078
grad_step = 000020, loss = 0.238605
grad_step = 000021, loss = 0.228050
grad_step = 000022, loss = 0.218776
grad_step = 000023, loss = 0.209495
grad_step = 000024, loss = 0.199520
grad_step = 000025, loss = 0.189603
grad_step = 000026, loss = 0.180662
grad_step = 000027, loss = 0.172383
grad_step = 000028, loss = 0.164000
grad_step = 000029, loss = 0.155300
grad_step = 000030, loss = 0.146804
grad_step = 000031, loss = 0.139124
grad_step = 000032, loss = 0.131895
grad_step = 000033, loss = 0.124603
grad_step = 000034, loss = 0.117222
grad_step = 000035, loss = 0.110088
grad_step = 000036, loss = 0.103535
grad_step = 000037, loss = 0.097247
grad_step = 000038, loss = 0.091090
grad_step = 000039, loss = 0.085160
grad_step = 000040, loss = 0.079550
grad_step = 000041, loss = 0.074200
grad_step = 000042, loss = 0.068966
grad_step = 000043, loss = 0.063996
grad_step = 000044, loss = 0.059430
grad_step = 000045, loss = 0.055064
grad_step = 000046, loss = 0.050716
grad_step = 000047, loss = 0.046639
grad_step = 000048, loss = 0.042950
grad_step = 000049, loss = 0.039459
grad_step = 000050, loss = 0.036100
grad_step = 000051, loss = 0.032962
grad_step = 000052, loss = 0.030101
grad_step = 000053, loss = 0.027345
grad_step = 000054, loss = 0.024747
grad_step = 000055, loss = 0.022432
grad_step = 000056, loss = 0.020306
grad_step = 000057, loss = 0.018331
grad_step = 000058, loss = 0.016525
grad_step = 000059, loss = 0.014898
grad_step = 000060, loss = 0.013406
grad_step = 000061, loss = 0.012038
grad_step = 000062, loss = 0.010793
grad_step = 000063, loss = 0.009681
grad_step = 000064, loss = 0.008681
grad_step = 000065, loss = 0.007782
grad_step = 000066, loss = 0.006989
grad_step = 000067, loss = 0.006262
grad_step = 000068, loss = 0.005620
grad_step = 000069, loss = 0.005073
grad_step = 000070, loss = 0.004593
grad_step = 000071, loss = 0.004171
grad_step = 000072, loss = 0.003819
grad_step = 000073, loss = 0.003512
grad_step = 000074, loss = 0.003248
grad_step = 000075, loss = 0.003036
grad_step = 000076, loss = 0.002858
grad_step = 000077, loss = 0.002711
grad_step = 000078, loss = 0.002589
grad_step = 000079, loss = 0.002497
grad_step = 000080, loss = 0.002423
grad_step = 000081, loss = 0.002368
grad_step = 000082, loss = 0.002333
grad_step = 000083, loss = 0.002328
grad_step = 000084, loss = 0.002404
grad_step = 000085, loss = 0.002667
grad_step = 000086, loss = 0.003116
grad_step = 000087, loss = 0.003207
grad_step = 000088, loss = 0.002514
grad_step = 000089, loss = 0.002309
grad_step = 000090, loss = 0.002865
grad_step = 000091, loss = 0.002683
grad_step = 000092, loss = 0.002239
grad_step = 000093, loss = 0.002599
grad_step = 000094, loss = 0.002588
grad_step = 000095, loss = 0.002217
grad_step = 000096, loss = 0.002493
grad_step = 000097, loss = 0.002451
grad_step = 000098, loss = 0.002199
grad_step = 000099, loss = 0.002408
grad_step = 000100, loss = 0.002337
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002176
grad_step = 000102, loss = 0.002345
grad_step = 000103, loss = 0.002236
grad_step = 000104, loss = 0.002152
grad_step = 000105, loss = 0.002271
grad_step = 000106, loss = 0.002156
grad_step = 000107, loss = 0.002132
grad_step = 000108, loss = 0.002203
grad_step = 000109, loss = 0.002103
grad_step = 000110, loss = 0.002107
grad_step = 000111, loss = 0.002147
grad_step = 000112, loss = 0.002064
grad_step = 000113, loss = 0.002083
grad_step = 000114, loss = 0.002098
grad_step = 000115, loss = 0.002039
grad_step = 000116, loss = 0.002058
grad_step = 000117, loss = 0.002064
grad_step = 000118, loss = 0.002019
grad_step = 000119, loss = 0.002036
grad_step = 000120, loss = 0.002037
grad_step = 000121, loss = 0.002004
grad_step = 000122, loss = 0.002015
grad_step = 000123, loss = 0.002017
grad_step = 000124, loss = 0.001992
grad_step = 000125, loss = 0.001997
grad_step = 000126, loss = 0.002000
grad_step = 000127, loss = 0.001980
grad_step = 000128, loss = 0.001980
grad_step = 000129, loss = 0.001985
grad_step = 000130, loss = 0.001971
grad_step = 000131, loss = 0.001965
grad_step = 000132, loss = 0.001970
grad_step = 000133, loss = 0.001961
grad_step = 000134, loss = 0.001952
grad_step = 000135, loss = 0.001954
grad_step = 000136, loss = 0.001950
grad_step = 000137, loss = 0.001941
grad_step = 000138, loss = 0.001938
grad_step = 000139, loss = 0.001938
grad_step = 000140, loss = 0.001931
grad_step = 000141, loss = 0.001925
grad_step = 000142, loss = 0.001924
grad_step = 000143, loss = 0.001920
grad_step = 000144, loss = 0.001913
grad_step = 000145, loss = 0.001909
grad_step = 000146, loss = 0.001906
grad_step = 000147, loss = 0.001901
grad_step = 000148, loss = 0.001895
grad_step = 000149, loss = 0.001890
grad_step = 000150, loss = 0.001886
grad_step = 000151, loss = 0.001881
grad_step = 000152, loss = 0.001875
grad_step = 000153, loss = 0.001869
grad_step = 000154, loss = 0.001864
grad_step = 000155, loss = 0.001859
grad_step = 000156, loss = 0.001853
grad_step = 000157, loss = 0.001846
grad_step = 000158, loss = 0.001840
grad_step = 000159, loss = 0.001833
grad_step = 000160, loss = 0.001827
grad_step = 000161, loss = 0.001820
grad_step = 000162, loss = 0.001812
grad_step = 000163, loss = 0.001806
grad_step = 000164, loss = 0.001810
grad_step = 000165, loss = 0.001811
grad_step = 000166, loss = 0.001816
grad_step = 000167, loss = 0.001793
grad_step = 000168, loss = 0.001775
grad_step = 000169, loss = 0.001774
grad_step = 000170, loss = 0.001780
grad_step = 000171, loss = 0.001775
grad_step = 000172, loss = 0.001756
grad_step = 000173, loss = 0.001742
grad_step = 000174, loss = 0.001737
grad_step = 000175, loss = 0.001739
grad_step = 000176, loss = 0.001744
grad_step = 000177, loss = 0.001751
grad_step = 000178, loss = 0.001766
grad_step = 000179, loss = 0.001772
grad_step = 000180, loss = 0.001764
grad_step = 000181, loss = 0.001724
grad_step = 000182, loss = 0.001695
grad_step = 000183, loss = 0.001698
grad_step = 000184, loss = 0.001722
grad_step = 000185, loss = 0.001753
grad_step = 000186, loss = 0.001754
grad_step = 000187, loss = 0.001736
grad_step = 000188, loss = 0.001702
grad_step = 000189, loss = 0.001684
grad_step = 000190, loss = 0.001686
grad_step = 000191, loss = 0.001710
grad_step = 000192, loss = 0.001750
grad_step = 000193, loss = 0.001776
grad_step = 000194, loss = 0.001793
grad_step = 000195, loss = 0.001759
grad_step = 000196, loss = 0.001739
grad_step = 000197, loss = 0.001783
grad_step = 000198, loss = 0.001862
grad_step = 000199, loss = 0.001907
grad_step = 000200, loss = 0.001886
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001859
grad_step = 000202, loss = 0.001836
grad_step = 000203, loss = 0.001800
grad_step = 000204, loss = 0.001736
grad_step = 000205, loss = 0.001637
grad_step = 000206, loss = 0.001602
grad_step = 000207, loss = 0.001646
grad_step = 000208, loss = 0.001696
grad_step = 000209, loss = 0.001709
grad_step = 000210, loss = 0.001684
grad_step = 000211, loss = 0.001675
grad_step = 000212, loss = 0.001677
grad_step = 000213, loss = 0.001652
grad_step = 000214, loss = 0.001604
grad_step = 000215, loss = 0.001576
grad_step = 000216, loss = 0.001580
grad_step = 000217, loss = 0.001598
grad_step = 000218, loss = 0.001605
grad_step = 000219, loss = 0.001600
grad_step = 000220, loss = 0.001596
grad_step = 000221, loss = 0.001602
grad_step = 000222, loss = 0.001612
grad_step = 000223, loss = 0.001618
grad_step = 000224, loss = 0.001617
grad_step = 000225, loss = 0.001606
grad_step = 000226, loss = 0.001591
grad_step = 000227, loss = 0.001578
grad_step = 000228, loss = 0.001570
grad_step = 000229, loss = 0.001568
grad_step = 000230, loss = 0.001571
grad_step = 000231, loss = 0.001576
grad_step = 000232, loss = 0.001581
grad_step = 000233, loss = 0.001587
grad_step = 000234, loss = 0.001594
grad_step = 000235, loss = 0.001599
grad_step = 000236, loss = 0.001594
grad_step = 000237, loss = 0.001578
grad_step = 000238, loss = 0.001561
grad_step = 000239, loss = 0.001554
grad_step = 000240, loss = 0.001574
grad_step = 000241, loss = 0.001627
grad_step = 000242, loss = 0.001727
grad_step = 000243, loss = 0.001809
grad_step = 000244, loss = 0.001900
grad_step = 000245, loss = 0.001936
grad_step = 000246, loss = 0.001964
grad_step = 000247, loss = 0.001915
grad_step = 000248, loss = 0.001751
grad_step = 000249, loss = 0.001539
grad_step = 000250, loss = 0.001454
grad_step = 000251, loss = 0.001534
grad_step = 000252, loss = 0.001660
grad_step = 000253, loss = 0.001708
grad_step = 000254, loss = 0.001613
grad_step = 000255, loss = 0.001483
grad_step = 000256, loss = 0.001432
grad_step = 000257, loss = 0.001492
grad_step = 000258, loss = 0.001572
grad_step = 000259, loss = 0.001584
grad_step = 000260, loss = 0.001520
grad_step = 000261, loss = 0.001461
grad_step = 000262, loss = 0.001500
grad_step = 000263, loss = 0.001640
grad_step = 000264, loss = 0.001762
grad_step = 000265, loss = 0.001629
grad_step = 000266, loss = 0.001437
grad_step = 000267, loss = 0.001404
grad_step = 000268, loss = 0.001513
grad_step = 000269, loss = 0.001517
grad_step = 000270, loss = 0.001409
grad_step = 000271, loss = 0.001427
grad_step = 000272, loss = 0.001481
grad_step = 000273, loss = 0.001406
grad_step = 000274, loss = 0.001370
grad_step = 000275, loss = 0.001429
grad_step = 000276, loss = 0.001440
grad_step = 000277, loss = 0.001373
grad_step = 000278, loss = 0.001362
grad_step = 000279, loss = 0.001397
grad_step = 000280, loss = 0.001392
grad_step = 000281, loss = 0.001351
grad_step = 000282, loss = 0.001329
grad_step = 000283, loss = 0.001348
grad_step = 000284, loss = 0.001371
grad_step = 000285, loss = 0.001364
grad_step = 000286, loss = 0.001335
grad_step = 000287, loss = 0.001310
grad_step = 000288, loss = 0.001302
grad_step = 000289, loss = 0.001309
grad_step = 000290, loss = 0.001321
grad_step = 000291, loss = 0.001327
grad_step = 000292, loss = 0.001323
grad_step = 000293, loss = 0.001310
grad_step = 000294, loss = 0.001294
grad_step = 000295, loss = 0.001279
grad_step = 000296, loss = 0.001268
grad_step = 000297, loss = 0.001260
grad_step = 000298, loss = 0.001255
grad_step = 000299, loss = 0.001252
grad_step = 000300, loss = 0.001251
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001251
grad_step = 000302, loss = 0.001254
grad_step = 000303, loss = 0.001263
grad_step = 000304, loss = 0.001284
grad_step = 000305, loss = 0.001333
grad_step = 000306, loss = 0.001443
grad_step = 000307, loss = 0.001646
grad_step = 000308, loss = 0.001941
grad_step = 000309, loss = 0.001950
grad_step = 000310, loss = 0.001902
grad_step = 000311, loss = 0.001769
grad_step = 000312, loss = 0.001536
grad_step = 000313, loss = 0.001376
grad_step = 000314, loss = 0.001260
grad_step = 000315, loss = 0.001322
grad_step = 000316, loss = 0.001441
grad_step = 000317, loss = 0.001531
grad_step = 000318, loss = 0.001419
grad_step = 000319, loss = 0.001224
grad_step = 000320, loss = 0.001166
grad_step = 000321, loss = 0.001288
grad_step = 000322, loss = 0.001395
grad_step = 000323, loss = 0.001347
grad_step = 000324, loss = 0.001214
grad_step = 000325, loss = 0.001138
grad_step = 000326, loss = 0.001188
grad_step = 000327, loss = 0.001262
grad_step = 000328, loss = 0.001254
grad_step = 000329, loss = 0.001180
grad_step = 000330, loss = 0.001122
grad_step = 000331, loss = 0.001124
grad_step = 000332, loss = 0.001170
grad_step = 000333, loss = 0.001187
grad_step = 000334, loss = 0.001158
grad_step = 000335, loss = 0.001108
grad_step = 000336, loss = 0.001082
grad_step = 000337, loss = 0.001082
grad_step = 000338, loss = 0.001110
grad_step = 000339, loss = 0.001124
grad_step = 000340, loss = 0.001129
grad_step = 000341, loss = 0.001113
grad_step = 000342, loss = 0.001098
grad_step = 000343, loss = 0.001069
grad_step = 000344, loss = 0.001063
grad_step = 000345, loss = 0.001060
grad_step = 000346, loss = 0.001055
grad_step = 000347, loss = 0.001043
grad_step = 000348, loss = 0.001029
grad_step = 000349, loss = 0.001017
grad_step = 000350, loss = 0.001013
grad_step = 000351, loss = 0.001012
grad_step = 000352, loss = 0.001013
grad_step = 000353, loss = 0.001010
grad_step = 000354, loss = 0.001005
grad_step = 000355, loss = 0.000996
grad_step = 000356, loss = 0.000992
grad_step = 000357, loss = 0.000999
grad_step = 000358, loss = 0.001031
grad_step = 000359, loss = 0.001123
grad_step = 000360, loss = 0.001134
grad_step = 000361, loss = 0.001166
grad_step = 000362, loss = 0.001021
grad_step = 000363, loss = 0.000960
grad_step = 000364, loss = 0.000959
grad_step = 000365, loss = 0.000991
grad_step = 000366, loss = 0.001059
grad_step = 000367, loss = 0.001014
grad_step = 000368, loss = 0.000977
grad_step = 000369, loss = 0.000942
grad_step = 000370, loss = 0.000926
grad_step = 000371, loss = 0.000924
grad_step = 000372, loss = 0.000936
grad_step = 000373, loss = 0.000954
grad_step = 000374, loss = 0.000963
grad_step = 000375, loss = 0.000971
grad_step = 000376, loss = 0.000944
grad_step = 000377, loss = 0.000919
grad_step = 000378, loss = 0.000897
grad_step = 000379, loss = 0.000884
grad_step = 000380, loss = 0.000881
grad_step = 000381, loss = 0.000881
grad_step = 000382, loss = 0.000886
grad_step = 000383, loss = 0.000888
grad_step = 000384, loss = 0.000896
grad_step = 000385, loss = 0.000901
grad_step = 000386, loss = 0.000915
grad_step = 000387, loss = 0.000919
grad_step = 000388, loss = 0.000934
grad_step = 000389, loss = 0.000931
grad_step = 000390, loss = 0.000939
grad_step = 000391, loss = 0.000945
grad_step = 000392, loss = 0.000984
grad_step = 000393, loss = 0.001070
grad_step = 000394, loss = 0.001223
grad_step = 000395, loss = 0.001447
grad_step = 000396, loss = 0.001652
grad_step = 000397, loss = 0.001687
grad_step = 000398, loss = 0.001432
grad_step = 000399, loss = 0.001033
grad_step = 000400, loss = 0.000822
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000936
grad_step = 000402, loss = 0.001181
grad_step = 000403, loss = 0.001243
grad_step = 000404, loss = 0.001073
grad_step = 000405, loss = 0.000852
grad_step = 000406, loss = 0.000864
grad_step = 000407, loss = 0.001030
grad_step = 000408, loss = 0.001117
grad_step = 000409, loss = 0.001029
grad_step = 000410, loss = 0.000892
grad_step = 000411, loss = 0.000861
grad_step = 000412, loss = 0.000944
grad_step = 000413, loss = 0.000943
grad_step = 000414, loss = 0.000844
grad_step = 000415, loss = 0.000758
grad_step = 000416, loss = 0.000779
grad_step = 000417, loss = 0.000848
grad_step = 000418, loss = 0.000847
grad_step = 000419, loss = 0.000783
grad_step = 000420, loss = 0.000738
grad_step = 000421, loss = 0.000757
grad_step = 000422, loss = 0.000786
grad_step = 000423, loss = 0.000774
grad_step = 000424, loss = 0.000737
grad_step = 000425, loss = 0.000725
grad_step = 000426, loss = 0.000751
grad_step = 000427, loss = 0.000781
grad_step = 000428, loss = 0.000794
grad_step = 000429, loss = 0.000811
grad_step = 000430, loss = 0.000893
grad_step = 000431, loss = 0.000881
grad_step = 000432, loss = 0.000887
grad_step = 000433, loss = 0.000770
grad_step = 000434, loss = 0.000698
grad_step = 000435, loss = 0.000699
grad_step = 000436, loss = 0.000742
grad_step = 000437, loss = 0.000765
grad_step = 000438, loss = 0.000726
grad_step = 000439, loss = 0.000695
grad_step = 000440, loss = 0.000694
grad_step = 000441, loss = 0.000708
grad_step = 000442, loss = 0.000706
grad_step = 000443, loss = 0.000687
grad_step = 000444, loss = 0.000681
grad_step = 000445, loss = 0.000685
grad_step = 000446, loss = 0.000680
grad_step = 000447, loss = 0.000663
grad_step = 000448, loss = 0.000651
grad_step = 000449, loss = 0.000656
grad_step = 000450, loss = 0.000670
grad_step = 000451, loss = 0.000675
grad_step = 000452, loss = 0.000668
grad_step = 000453, loss = 0.000658
grad_step = 000454, loss = 0.000653
grad_step = 000455, loss = 0.000654
grad_step = 000456, loss = 0.000651
grad_step = 000457, loss = 0.000644
grad_step = 000458, loss = 0.000630
grad_step = 000459, loss = 0.000620
grad_step = 000460, loss = 0.000617
grad_step = 000461, loss = 0.000616
grad_step = 000462, loss = 0.000613
grad_step = 000463, loss = 0.000608
grad_step = 000464, loss = 0.000603
grad_step = 000465, loss = 0.000600
grad_step = 000466, loss = 0.000599
grad_step = 000467, loss = 0.000598
grad_step = 000468, loss = 0.000596
grad_step = 000469, loss = 0.000594
grad_step = 000470, loss = 0.000592
grad_step = 000471, loss = 0.000593
grad_step = 000472, loss = 0.000604
grad_step = 000473, loss = 0.000627
grad_step = 000474, loss = 0.000687
grad_step = 000475, loss = 0.000780
grad_step = 000476, loss = 0.000973
grad_step = 000477, loss = 0.000942
grad_step = 000478, loss = 0.000960
grad_step = 000479, loss = 0.000896
grad_step = 000480, loss = 0.000812
grad_step = 000481, loss = 0.000715
grad_step = 000482, loss = 0.000635
grad_step = 000483, loss = 0.000598
grad_step = 000484, loss = 0.000605
grad_step = 000485, loss = 0.000665
grad_step = 000486, loss = 0.000705
grad_step = 000487, loss = 0.000673
grad_step = 000488, loss = 0.000598
grad_step = 000489, loss = 0.000554
grad_step = 000490, loss = 0.000570
grad_step = 000491, loss = 0.000610
grad_step = 000492, loss = 0.000621
grad_step = 000493, loss = 0.000596
grad_step = 000494, loss = 0.000560
grad_step = 000495, loss = 0.000548
grad_step = 000496, loss = 0.000564
grad_step = 000497, loss = 0.000579
grad_step = 000498, loss = 0.000580
grad_step = 000499, loss = 0.000556
grad_step = 000500, loss = 0.000533
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000528
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

  date_run                              2020-05-15 04:12:57.006910
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.237334
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 04:12:57.012334
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.138724
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 04:12:57.019271
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.135787
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 04:12:57.024062
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.10796
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
0   2020-05-15 04:12:29.849592  ...    mean_absolute_error
1   2020-05-15 04:12:29.853055  ...     mean_squared_error
2   2020-05-15 04:12:29.856180  ...  median_absolute_error
3   2020-05-15 04:12:29.859079  ...               r2_score
4   2020-05-15 04:12:39.393301  ...    mean_absolute_error
5   2020-05-15 04:12:39.397557  ...     mean_squared_error
6   2020-05-15 04:12:39.401203  ...  median_absolute_error
7   2020-05-15 04:12:39.404641  ...               r2_score
8   2020-05-15 04:12:57.006910  ...    mean_absolute_error
9   2020-05-15 04:12:57.012334  ...     mean_squared_error
10  2020-05-15 04:12:57.019271  ...  median_absolute_error
11  2020-05-15 04:12:57.024062  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f81250449b0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:19, 124611.68it/s] 56%|█████▌    | 5562368/9912422 [00:00<00:24, 177845.25it/s]9920512it [00:00, 34657893.56it/s]                           
0it [00:00, ?it/s]32768it [00:00, 636665.79it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 141269.91it/s]1654784it [00:00, 10240657.63it/s]                         
0it [00:00, ?it/s]8192it [00:00, 223544.85it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f80d79f3e10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f80d48400b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f80d79f3e10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f80d6f7b080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f80d47b4470> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f80d479fbe0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f80d79f3e10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f80d6f396a0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f80d47b4470> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f8124ffce48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f6fa8f081d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=08fd429a6eb977b4e188bb2c33fef5635e7b8a6764f8e2ff301bcf4e0b62505e
  Stored in directory: /tmp/pip-ephem-wheel-cache-nhsc5kwa/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f6f9f28e080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2490368/17464789 [===>..........................] - ETA: 0s
 6094848/17464789 [=========>....................] - ETA: 0s
13369344/17464789 [=====================>........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 04:14:22.733552: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 04:14:22.737391: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095225000 Hz
2020-05-15 04:14:22.738078: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556c1424deb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 04:14:22.738092: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8200 - accuracy: 0.4900
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8353 - accuracy: 0.4890 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7535 - accuracy: 0.4943
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7203 - accuracy: 0.4965
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6850 - accuracy: 0.4988
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6334 - accuracy: 0.5022
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6075 - accuracy: 0.5039
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6034 - accuracy: 0.5041
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6343 - accuracy: 0.5021
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6574 - accuracy: 0.5006
11000/25000 [============>.................] - ETA: 3s - loss: 7.6750 - accuracy: 0.4995
12000/25000 [=============>................] - ETA: 3s - loss: 7.6641 - accuracy: 0.5002
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6548 - accuracy: 0.5008
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6469 - accuracy: 0.5013
15000/25000 [=================>............] - ETA: 2s - loss: 7.6421 - accuracy: 0.5016
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6264 - accuracy: 0.5026
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6405 - accuracy: 0.5017
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6683 - accuracy: 0.4999
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6650 - accuracy: 0.5001
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6597 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6624 - accuracy: 0.5003
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6833 - accuracy: 0.4989
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6724 - accuracy: 0.4996
25000/25000 [==============================] - 7s 265us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 04:14:35.643945
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 04:14:35.643945  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<47:37:12, 5.03kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<33:34:01, 7.13kB/s].vector_cache/glove.6B.zip:   0%|          | 205k/862M [00:01<23:33:43, 10.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 541k/862M [00:02<16:33:02, 14.5kB/s].vector_cache/glove.6B.zip:   0%|          | 909k/862M [00:02<11:36:25, 20.6kB/s].vector_cache/glove.6B.zip:   0%|          | 1.28M/862M [00:02<8:08:50, 29.4kB/s].vector_cache/glove.6B.zip:   0%|          | 1.65M/862M [00:02<5:43:36, 41.7kB/s].vector_cache/glove.6B.zip:   0%|          | 2.02M/862M [00:02<4:01:52, 59.3kB/s].vector_cache/glove.6B.zip:   0%|          | 2.39M/862M [00:02<2:50:46, 83.9kB/s].vector_cache/glove.6B.zip:   0%|          | 2.77M/862M [00:03<2:01:02, 118kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.15M/862M [00:03<1:26:13, 166kB/s].vector_cache/glove.6B.zip:   0%|          | 3.54M/862M [00:03<1:01:50, 231kB/s].vector_cache/glove.6B.zip:   0%|          | 3.92M/862M [00:03<44:45, 320kB/s]  .vector_cache/glove.6B.zip:   1%|          | 4.32M/862M [00:03<32:47, 436kB/s].vector_cache/glove.6B.zip:   1%|          | 4.71M/862M [00:03<24:23, 586kB/s].vector_cache/glove.6B.zip:   1%|          | 5.10M/862M [00:03<18:32, 771kB/s].vector_cache/glove.6B.zip:   1%|          | 5.51M/862M [00:03<14:24, 991kB/s].vector_cache/glove.6B.zip:   1%|          | 5.91M/862M [00:04<11:28, 1.24MB/s].vector_cache/glove.6B.zip:   1%|          | 6.32M/862M [00:04<09:26, 1.51MB/s].vector_cache/glove.6B.zip:   1%|          | 6.73M/862M [00:04<08:01, 1.78MB/s].vector_cache/glove.6B.zip:   1%|          | 7.13M/862M [00:04<06:41, 2.13MB/s].vector_cache/glove.6B.zip:   1%|          | 7.36M/862M [00:04<06:35, 2.16MB/s].vector_cache/glove.6B.zip:   1%|          | 7.65M/862M [00:04<06:04, 2.35MB/s].vector_cache/glove.6B.zip:   1%|          | 7.98M/862M [00:04<05:35, 2.54MB/s].vector_cache/glove.6B.zip:   1%|          | 8.27M/862M [00:04<05:24, 2.63MB/s].vector_cache/glove.6B.zip:   1%|          | 8.61M/862M [00:04<05:04, 2.80MB/s].vector_cache/glove.6B.zip:   1%|          | 8.90M/862M [00:05<05:06, 2.79MB/s].vector_cache/glove.6B.zip:   1%|          | 9.26M/862M [00:05<04:45, 2.99MB/s].vector_cache/glove.6B.zip:   1%|          | 9.54M/862M [00:05<04:52, 2.92MB/s].vector_cache/glove.6B.zip:   1%|          | 9.90M/862M [00:05<04:38, 3.06MB/s].vector_cache/glove.6B.zip:   1%|          | 10.2M/862M [00:05<04:49, 2.94MB/s].vector_cache/glove.6B.zip:   1%|          | 10.6M/862M [00:05<04:33, 3.11MB/s].vector_cache/glove.6B.zip:   1%|▏         | 10.9M/862M [00:05<04:37, 3.06MB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.2M/862M [00:05<04:26, 3.20MB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.5M/862M [00:05<04:50, 2.93MB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.9M/862M [00:05<04:26, 3.19MB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:06<04:33, 3.11MB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.6M/862M [00:06<04:17, 3.30MB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.8M/862M [00:06<04:41, 3.01MB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.2M/862M [00:06<04:26, 3.19MB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.5M/862M [00:06<04:41, 3.01MB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.0M/862M [00:06<04:28, 3.15MB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.4M/862M [00:06<04:12, 3.36MB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.7M/862M [00:06<04:26, 3.18MB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.1M/862M [00:06<04:14, 3.33MB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.5M/862M [00:07<04:16, 3.30MB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.9M/862M [00:07<04:00, 3.52MB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.2M/862M [00:07<04:06, 3.43MB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.7M/862M [00:07<03:50, 3.67MB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.0M/862M [00:07<04:06, 3.43MB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.5M/862M [00:07<03:43, 3.78MB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.8M/862M [00:07<03:58, 3.54MB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.3M/862M [00:07<03:36, 3.90MB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.7M/862M [00:07<03:44, 3.76MB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.2M/862M [00:07<03:29, 4.03MB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.6M/862M [00:08<03:31, 3.99MB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.1M/862M [00:08<03:19, 4.23MB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.5M/862M [00:08<03:23, 4.15MB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.0M/862M [00:08<03:16, 4.28MB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.5M/862M [00:08<03:08, 4.47MB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.0M/862M [00:08<03:05, 4.52MB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.5M/862M [00:08<02:58, 4.69MB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.0M/862M [00:08<02:58, 4.71MB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.6M/862M [00:08<02:48, 4.97MB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.1M/862M [00:09<02:49, 4.94MB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.6M/862M [00:09<02:43, 5.11MB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.2M/862M [00:09<02:39, 5.25MB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.8M/862M [00:09<02:35, 5.37MB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:09<02:31, 5.51MB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.0M/862M [00:09<02:28, 5.62MB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.6M/862M [00:09<02:23, 5.83MB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.2M/862M [00:09<02:22, 5.87MB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.9M/862M [00:09<02:17, 6.07MB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:09<02:14, 6.21MB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.2M/862M [00:10<02:12, 6.29MB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.0M/862M [00:10<02:04, 6.69MB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.7M/862M [00:10<02:07, 6.53MB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.5M/862M [00:10<01:59, 6.95MB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.1M/862M [00:10<02:03, 6.73MB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.9M/862M [00:10<01:59, 6.91MB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.7M/862M [00:10<01:55, 7.19MB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.4M/862M [00:10<01:53, 7.31MB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.3M/862M [00:10<01:48, 7.63MB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.1M/862M [00:10<01:46, 7.74MB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.8M/862M [00:11<01:55, 7.14MB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.0M/862M [00:11<01:49, 7.55MB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.1M/862M [00:11<01:39, 8.28MB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.8M/862M [00:11<01:44, 7.83MB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.1M/862M [00:11<01:39, 8.27MB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.4M/862M [00:11<01:34, 8.69MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.8M/862M [00:11<01:30, 9.05MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.2M/862M [00:11<01:26, 9.47MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.7M/862M [00:12<01:22, 9.82MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.2M/862M [00:12<01:20, 10.2MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.7M/862M [00:12<01:17, 10.5MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:12<01:17, 10.4MB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.1M/862M [00:12<01:12, 11.1MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.0M/862M [00:13<04:49, 2.79MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.3M/862M [00:13<05:04, 2.65MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.5M/862M [00:13<04:01, 3.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.3M/862M [00:13<03:07, 4.30MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.2M/862M [00:15<09:04, 1.48MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:15<12:55, 1.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.6M/862M [00:15<10:36, 1.26MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:15<07:50, 1.71MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:15<05:41, 2.35MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:17<13:29, 988kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:17<14:55, 893kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.8M/862M [00:17<11:53, 1.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:17<08:42, 1.53MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.3M/862M [00:17<06:19, 2.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:19<43:09, 307kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:19<35:15, 376kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.0M/862M [00:19<25:45, 514kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:19<18:20, 721kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:21<15:48, 834kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:21<15:24, 856kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:21<11:43, 1.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:21<08:41, 1.52MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:21<06:14, 2.10MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.8M/862M [00:23<15:11, 864kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:23<14:40, 894kB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.4M/862M [00:23<11:09, 1.18MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:23<08:05, 1.62MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:25<09:17, 1.41MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:25<10:22, 1.26MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.5M/862M [00:25<08:08, 1.60MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:25<06:06, 2.13MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:25<04:25, 2.93MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:27<34:56, 372kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:27<27:53, 465kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.8M/862M [00:27<20:16, 640kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:27<14:23, 900kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:29<13:46, 938kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:29<12:53, 1.00MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.1M/862M [00:29<09:41, 1.33MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:29<06:57, 1.85MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.4M/862M [00:31<09:29, 1.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:31<09:59, 1.29MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.2M/862M [00:31<07:48, 1.64MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:31<05:40, 2.26MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:33<09:24, 1.36MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:33<09:22, 1.36MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.5M/862M [00:33<07:15, 1.76MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:33<05:16, 2.41MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.7M/862M [00:35<11:33, 1.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:35<11:35, 1.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:35<08:50, 1.44MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:35<06:23, 1.98MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:37<08:16, 1.53MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:37<08:11, 1.54MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:37<06:20, 1.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:37<04:35, 2.74MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:39<35:41, 352kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:39<27:16, 461kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:39<19:38, 639kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:39<13:51, 902kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:41<3:08:25, 66.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:41<2:13:58, 93.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:41<1:34:12, 132kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:41<1:05:54, 189kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:43<55:18, 225kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 116M/862M [00:43<41:37, 299kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:43<29:43, 418kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:43<20:56, 591kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:45<19:09, 645kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:45<15:18, 808kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:45<11:06, 1.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:45<07:54, 1.56MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:47<17:18, 710kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:47<14:38, 840kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:47<10:46, 1.14MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:47<07:55, 1.55MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:49<08:19, 1.47MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:49<08:20, 1.46MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:49<06:23, 1.91MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:49<04:38, 2.62MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:51<08:26, 1.44MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:51<08:24, 1.44MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:51<06:30, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:51<04:40, 2.59MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:52<15:23, 785kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:53<13:09, 919kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:53<09:44, 1.24MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:53<06:58, 1.73MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:54<10:27, 1.15MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:55<09:49, 1.22MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:55<07:24, 1.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:55<05:25, 2.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:56<07:07, 1.68MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:57<07:27, 1.60MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:57<05:44, 2.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:57<04:15, 2.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:58<06:25, 1.85MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:59<06:57, 1.71MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:59<05:28, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:59<03:56, 3.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [01:00<34:01, 347kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [01:00<26:13, 450kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [01:01<18:52, 625kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [01:01<13:19, 883kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [01:02<16:35, 708kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:02<13:56, 842kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:03<10:21, 1.13MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [01:03<07:21, 1.59MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [01:04<15:46, 740kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [01:04<13:25, 869kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:05<09:54, 1.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:05<07:08, 1.63MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:06<08:25, 1.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:06<08:16, 1.40MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:06<06:18, 1.84MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:07<04:34, 2.53MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:08<08:05, 1.43MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:08<08:01, 1.44MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:08<06:12, 1.86MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:09<04:27, 2.58MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:10<16:29, 695kB/s] .vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:10<13:53, 825kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:10<10:17, 1.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:11<07:17, 1.56MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:12<30:40, 372kB/s] .vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:12<23:50, 478kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:12<17:09, 664kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:12<12:11, 932kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:14<12:12, 928kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:14<10:51, 1.04MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:14<08:06, 1.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:14<05:47, 1.95MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:16<12:15, 920kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:16<10:52, 1.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:16<08:07, 1.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:16<05:52, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:18<07:41, 1.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:18<07:40, 1.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:18<05:56, 1.88MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:18<04:15, 2.61MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:20<3:37:17, 51.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:20<2:34:22, 72.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:20<1:48:30, 102kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:22<1:17:23, 143kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:22<56:26, 196kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:22<39:56, 276kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:22<28:04, 392kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:24<23:20, 471kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:24<18:35, 591kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:24<13:28, 814kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:24<09:35, 1.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:26<10:34, 1.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:26<09:39, 1.13MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:26<07:14, 1.51MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:26<05:14, 2.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:28<07:24, 1.47MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:28<07:24, 1.47MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:28<05:39, 1.91MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:28<04:07, 2.62MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:30<07:14, 1.49MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:30<06:32, 1.65MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:30<04:53, 2.20MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:32<05:28, 1.96MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:32<05:46, 1.85MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:32<04:31, 2.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:32<03:16, 3.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:34<15:00, 710kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:34<12:27, 854kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:34<09:11, 1.16MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:34<06:31, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:36<1:00:44, 174kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:36<44:25, 238kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:36<31:28, 335kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:36<22:11, 475kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:38<18:46, 560kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:38<15:02, 698kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:38<10:56, 960kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:38<07:47, 1.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:40<10:11, 1.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:40<09:06, 1.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:40<06:50, 1.52MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:42<06:30, 1.59MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:42<06:32, 1.58MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:42<05:00, 2.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:42<03:36, 2.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:44<26:45, 385kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:44<20:40, 498kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:44<14:52, 691kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:44<10:28, 977kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:45<24:34, 416kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:46<19:08, 535kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:46<13:51, 738kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:47<11:21, 895kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:48<09:47, 1.04MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:48<07:19, 1.39MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:48<05:19, 1.90MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:49<06:50, 1.47MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:50<06:39, 1.52MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:50<05:07, 1.97MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:51<05:14, 1.91MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:52<05:29, 1.82MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:52<04:14, 2.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:52<03:07, 3.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:53<05:55, 1.68MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:53<05:57, 1.67MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:54<04:37, 2.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:55<04:52, 2.03MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:55<05:13, 1.89MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:56<04:06, 2.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:57<04:30, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:57<04:57, 1.98MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:58<03:52, 2.54MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:58<02:50, 3.45MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:59<07:15, 1.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:59<06:51, 1.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:59<05:11, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [02:00<03:45, 2.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [02:01<07:41, 1.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [02:01<07:08, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [02:01<05:24, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:02<03:52, 2.49MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:03<32:27, 296kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:03<24:27, 393kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [02:03<17:32, 547kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:05<13:49, 691kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:05<11:28, 832kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:05<08:27, 1.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:07<07:28, 1.27MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:07<07:01, 1.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:07<05:19, 1.78MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:09<05:17, 1.78MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:09<05:28, 1.72MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:09<04:15, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:09<03:04, 3.04MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:11<17:19, 539kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:11<13:53, 673kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:11<10:08, 920kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:13<08:36, 1.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:13<07:43, 1.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:13<05:50, 1.59MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:15<05:36, 1.64MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:15<05:35, 1.64MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:15<04:17, 2.14MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:15<03:05, 2.95MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:17<11:18, 808kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:17<09:34, 954kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:17<07:07, 1.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:19<06:28, 1.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:19<06:12, 1.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:19<04:42, 1.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:19<03:24, 2.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:21<07:07, 1.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:21<06:37, 1.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:21<05:00, 1.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:21<03:35, 2.49MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:23<11:55, 749kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:23<09:57, 896kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:23<07:22, 1.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:25<06:38, 1.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:25<06:16, 1.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:25<04:48, 1.84MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:27<04:48, 1.83MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:27<04:58, 1.77MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:27<03:52, 2.26MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:29<04:10, 2.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:29<04:30, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:29<03:30, 2.48MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:29<02:32, 3.40MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:31<12:52, 672kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:31<10:36, 816kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:31<07:48, 1.11MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:31<05:31, 1.55MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:33<31:40, 271kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:33<23:43, 362kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:33<16:58, 505kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:33<11:55, 715kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:35<15:52, 536kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:35<12:39, 672kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:35<09:14, 920kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:37<07:50, 1.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:37<07:05, 1.19MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:37<05:20, 1.58MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:39<05:07, 1.64MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:39<05:10, 1.62MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:39<03:57, 2.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:39<02:51, 2.90MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:41<08:37, 964kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:41<07:36, 1.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:41<05:39, 1.47MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:41<04:03, 2.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:42<07:44, 1.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:43<06:58, 1.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:43<05:15, 1.56MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:43<03:45, 2.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:44<10:51, 753kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:45<09:05, 898kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:45<06:40, 1.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:45<04:47, 1.70MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:46<06:36, 1.23MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:47<06:05, 1.33MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:47<04:38, 1.74MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:47<03:18, 2.43MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:48<31:41, 254kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:49<23:37, 340kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:49<16:53, 475kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:50<13:05, 609kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:50<10:37, 750kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:51<07:47, 1.02MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:51<05:31, 1.43MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:52<09:13, 856kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:52<07:54, 999kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:53<05:50, 1.35MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:53<04:11, 1.87MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:54<06:00, 1.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:54<05:37, 1.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:55<04:14, 1.84MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:55<03:04, 2.53MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:56<05:58, 1.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:56<05:35, 1.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:56<04:12, 1.84MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:57<03:03, 2.52MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:58<05:23, 1.43MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:58<05:11, 1.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:58<03:58, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:59<02:50, 2.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [03:00<23:11, 329kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [03:00<17:37, 432kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [03:00<12:39, 601kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:01<08:53, 851kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:02<55:15, 137kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:02<40:05, 188kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:02<28:20, 266kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:02<19:50, 378kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:04<18:39, 401kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:04<14:28, 517kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:04<10:27, 715kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:04<07:20, 1.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:06<44:37, 166kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:06<32:37, 227kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:06<23:08, 320kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:06<16:10, 455kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:08<39:58, 184kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:08<29:21, 250kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:08<20:49, 352kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:08<14:36, 500kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:10<13:45, 529kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:10<10:57, 665kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:10<07:55, 916kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:10<05:40, 1.27MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:12<06:10, 1.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:12<05:38, 1.28MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:12<04:16, 1.68MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:14<04:10, 1.71MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:14<04:13, 1.69MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:14<03:14, 2.20MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:14<02:23, 2.97MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:16<04:02, 1.75MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:16<04:08, 1.71MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:16<03:13, 2.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:18<03:24, 2.05MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:18<03:39, 1.91MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:18<02:50, 2.46MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:18<02:06, 3.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:20<03:49, 1.81MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:20<03:57, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:20<03:05, 2.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:22<03:17, 2.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:22<03:33, 1.93MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:22<02:47, 2.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:22<02:00, 3.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:24<32:03, 212kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:24<23:39, 287kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:24<16:49, 403kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:24<11:47, 572kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:26<13:26, 501kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:26<10:38, 633kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:26<07:44, 868kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:28<06:29, 1.03MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:28<05:48, 1.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:28<04:21, 1.52MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:28<03:05, 2.13MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:30<34:22, 192kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:30<25:19, 260kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:30<17:59, 365kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:32<13:35, 480kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:32<10:44, 607kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:32<07:45, 838kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:32<05:30, 1.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:34<06:58, 926kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:34<06:05, 1.06MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:34<04:33, 1.41MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:34<03:14, 1.97MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:36<07:18, 873kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:36<06:17, 1.02MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:36<04:39, 1.37MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:36<03:20, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:38<04:50, 1.31MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:38<04:32, 1.39MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:38<03:26, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:38<02:28, 2.53MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:39<04:57, 1.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:40<04:36, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:40<03:28, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:40<02:29, 2.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:41<05:35, 1.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:42<05:02, 1.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:42<03:45, 1.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:42<02:45, 2.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:43<03:36, 1.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:44<03:39, 1.67MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:44<02:47, 2.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:44<02:01, 2.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:45<05:23, 1.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:46<04:52, 1.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:46<03:40, 1.64MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:47<03:33, 1.68MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:47<03:34, 1.67MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:48<02:46, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:49<02:55, 2.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:49<03:08, 1.88MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:50<02:25, 2.42MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:50<01:51, 3.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:51<02:48, 2.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:51<03:01, 1.93MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:52<02:23, 2.44MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:53<02:37, 2.20MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:53<02:55, 1.97MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:53<02:16, 2.53MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:54<01:40, 3.42MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:55<03:38, 1.56MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:55<03:38, 1.57MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:55<02:46, 2.05MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:56<01:59, 2.82MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:57<05:52, 957kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:57<05:11, 1.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:57<03:52, 1.45MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:59<03:37, 1.53MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:59<03:36, 1.54MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:59<02:43, 2.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:59<01:58, 2.78MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:01<04:13, 1.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:01<03:57, 1.39MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:01<03:01, 1.82MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:03<03:00, 1.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:03<03:05, 1.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:03<02:22, 2.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:03<01:43, 3.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:05<05:09, 1.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:05<04:26, 1.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:05<03:24, 1.56MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:05<02:30, 2.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:07<03:06, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:07<03:08, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:07<02:24, 2.19MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:07<01:45, 2.99MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:09<03:44, 1.40MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:09<03:33, 1.46MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:09<02:42, 1.93MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:09<01:55, 2.67MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:11<08:26, 611kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:11<06:50, 752kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:11<05:00, 1.02MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:13<04:19, 1.17MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:13<03:55, 1.29MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:13<02:58, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:15<02:54, 1.72MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:15<02:59, 1.68MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:15<02:18, 2.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:17<02:26, 2.03MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:17<02:38, 1.87MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:17<02:02, 2.41MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:17<01:33, 3.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:19<02:22, 2.05MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:19<02:33, 1.90MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:19<02:01, 2.41MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:19<01:26, 3.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:21<17:31, 275kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:21<13:07, 366kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:21<09:22, 511kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:21<06:33, 724kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:23<08:37, 550kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:23<06:53, 687kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:23<05:01, 941kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:23<03:31, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:25<17:46, 263kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:25<13:17, 351kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:25<09:27, 492kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:25<06:36, 697kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:27<10:39, 432kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:27<08:18, 554kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:27<05:59, 765kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:27<04:12, 1.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:29<06:41, 677kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:29<05:30, 822kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:29<04:02, 1.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:29<02:51, 1.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:31<06:22, 700kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:31<05:16, 846kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:31<03:52, 1.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:33<03:25, 1.28MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:33<03:11, 1.37MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:33<02:24, 1.82MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:33<01:43, 2.51MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:35<03:19, 1.30MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:35<03:08, 1.37MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:35<02:23, 1.80MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:36<02:22, 1.80MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:37<02:27, 1.73MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:37<01:54, 2.21MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:38<02:01, 2.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:39<02:12, 1.89MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:39<01:44, 2.41MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:40<01:53, 2.18MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:41<02:06, 1.96MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:41<01:39, 2.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:42<01:49, 2.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:43<02:00, 2.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:43<01:35, 2.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:44<01:46, 2.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:44<01:57, 2.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:45<01:33, 2.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:45<01:09, 3.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:46<02:01, 1.93MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:46<02:08, 1.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:47<01:40, 2.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:48<01:48, 2.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:48<01:57, 1.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:49<01:31, 2.52MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:49<01:06, 3.44MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:50<03:13, 1.17MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:50<02:56, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:51<02:13, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:52<02:09, 1.72MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:52<02:11, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:52<01:41, 2.17MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:54<01:47, 2.04MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:54<01:54, 1.91MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:54<01:29, 2.42MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:56<01:38, 2.19MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:56<01:47, 1.99MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:56<01:24, 2.51MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:58<01:33, 2.24MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:58<01:43, 2.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:58<01:22, 2.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [05:00<01:31, 2.26MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [05:00<01:41, 2.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [05:00<01:20, 2.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:02<01:29, 2.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:02<01:39, 2.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:02<01:18, 2.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:04<01:27, 2.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:04<01:36, 2.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:04<01:16, 2.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [05:06<01:25, 2.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:06<01:36, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:06<01:15, 2.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:06<00:54, 3.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:08<02:24, 1.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:08<02:16, 1.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:08<01:44, 1.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:10<01:42, 1.80MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:10<01:45, 1.75MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:10<01:21, 2.26MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:12<01:26, 2.09MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:12<01:34, 1.92MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:12<01:13, 2.44MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:12<00:54, 3.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:14<01:41, 1.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:14<01:43, 1.72MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:14<01:18, 2.23MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:14<00:56, 3.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:16<03:10, 911kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:16<02:44, 1.05MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:16<02:02, 1.40MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:18<01:52, 1.50MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:18<01:49, 1.54MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:18<01:23, 2.00MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:18<01:01, 2.71MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:20<01:51, 1.48MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:20<01:48, 1.52MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:20<01:21, 2.00MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:20<00:57, 2.78MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:22<23:16, 115kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:22<16:46, 159kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:22<11:48, 225kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:24<08:32, 306kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:24<06:28, 403kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:24<04:37, 561kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:26<03:35, 706kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:26<03:00, 844kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:26<02:12, 1.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:26<01:32, 1.61MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:28<37:08, 66.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:28<26:25, 93.5kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:28<18:30, 133kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:30<13:05, 184kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:30<09:36, 250kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:30<06:46, 352kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:30<04:42, 499kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:32<04:26, 526kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:32<03:31, 661kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:32<02:32, 910kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:32<01:46, 1.28MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:33<02:55, 776kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:34<02:27, 923kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:34<01:48, 1.24MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:35<01:36, 1.37MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:36<01:31, 1.44MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:36<01:09, 1.88MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:37<01:09, 1.85MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:38<01:11, 1.78MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:38<00:55, 2.30MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:38<00:38, 3.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:39<07:35, 271kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:40<05:42, 360kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:40<04:06, 499kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:40<02:52, 703kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:41<02:29, 798kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:41<02:06, 944kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:42<01:32, 1.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:43<01:22, 1.39MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:43<01:18, 1.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:44<00:59, 1.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:44<00:42, 2.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:45<01:40, 1.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:45<01:30, 1.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:46<01:07, 1.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:46<00:50, 2.18MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:47<01:02, 1.72MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:47<01:03, 1.68MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:47<00:49, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:49<00:50, 2.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:49<00:55, 1.87MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:49<00:42, 2.38MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:51<00:45, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:51<00:50, 1.95MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:51<00:39, 2.47MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:52<00:28, 3.39MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:53<01:50, 859kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:53<01:35, 995kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:53<01:10, 1.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:55<01:02, 1.44MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:55<01:00, 1.48MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:55<00:46, 1.93MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:57<00:45, 1.89MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:57<00:47, 1.80MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:57<00:37, 2.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:59<00:38, 2.12MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:59<00:42, 1.95MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:59<00:32, 2.47MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:01<00:35, 2.21MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:01<00:38, 2.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:01<00:30, 2.55MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [06:01<00:21, 3.49MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:03<01:11, 1.04MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:03<01:03, 1.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:03<00:46, 1.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:03<00:32, 2.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:05<11:53, 98.3kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:05<08:30, 137kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:05<05:55, 194kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:05<04:02, 276kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:07<03:18, 332kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:07<02:30, 437kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:07<01:47, 606kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:09<01:21, 756kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:09<01:08, 903kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:09<00:49, 1.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:11<00:42, 1.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:11<00:40, 1.42MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:11<00:30, 1.86MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:13<00:29, 1.84MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:13<00:30, 1.75MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:13<00:23, 2.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:15<00:24, 2.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:15<00:35, 1.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:15<00:29, 1.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:15<00:21, 2.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:17<00:25, 1.79MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:17<00:26, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:17<00:20, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:19<00:19, 2.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:19<00:21, 1.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:19<00:16, 2.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:21<00:16, 2.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:21<00:18, 1.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:21<00:14, 2.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:21<00:09, 3.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:23<01:16, 433kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:23<00:59, 555kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:23<00:41, 764kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:23<00:26, 1.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:25<02:28, 195kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:25<01:48, 264kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:25<01:14, 371kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:27<00:50, 487kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:27<00:39, 616kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:27<00:27, 845kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:29<00:20, 1.00MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:29<00:17, 1.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:29<00:12, 1.51MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:31<00:10, 1.58MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:31<00:10, 1.58MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:31<00:07, 2.05MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:33<00:06, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:33<00:06, 1.83MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:33<00:04, 2.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:33<00:03, 3.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:35<00:04, 1.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:35<00:04, 1.67MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:35<00:03, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:35<00:01, 2.99MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:37<00:04, 887kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:37<00:03, 1.01MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:37<00:02, 1.37MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:37<00:00, 1.88MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:38<00:00, 1.52MB/s].vector_cache/glove.6B.zip: 862MB [06:38, 2.16MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 894/400000 [00:00<00:44, 8937.66it/s]  0%|          | 1827/400000 [00:00<00:43, 9051.85it/s]  1%|          | 2751/400000 [00:00<00:43, 9105.40it/s]  1%|          | 3663/400000 [00:00<00:43, 9107.59it/s]  1%|          | 4565/400000 [00:00<00:43, 9079.95it/s]  1%|▏         | 5470/400000 [00:00<00:43, 9070.71it/s]  2%|▏         | 6319/400000 [00:00<00:44, 8885.29it/s]  2%|▏         | 7199/400000 [00:00<00:44, 8858.46it/s]  2%|▏         | 8112/400000 [00:00<00:43, 8936.73it/s]  2%|▏         | 9007/400000 [00:01<00:43, 8939.94it/s]  2%|▏         | 9892/400000 [00:01<00:43, 8910.68it/s]  3%|▎         | 10795/400000 [00:01<00:43, 8945.82it/s]  3%|▎         | 11710/400000 [00:01<00:43, 9004.66it/s]  3%|▎         | 12631/400000 [00:01<00:42, 9065.08it/s]  3%|▎         | 13532/400000 [00:01<00:42, 9023.93it/s]  4%|▎         | 14438/400000 [00:01<00:42, 9033.50it/s]  4%|▍         | 15362/400000 [00:01<00:42, 9091.55it/s]  4%|▍         | 16281/400000 [00:01<00:42, 9120.73it/s]  4%|▍         | 17199/400000 [00:01<00:41, 9136.14it/s]  5%|▍         | 18126/400000 [00:02<00:41, 9174.57it/s]  5%|▍         | 19043/400000 [00:02<00:41, 9147.18it/s]  5%|▍         | 19958/400000 [00:02<00:41, 9103.15it/s]  5%|▌         | 20869/400000 [00:02<00:42, 9017.01it/s]  5%|▌         | 21794/400000 [00:02<00:41, 9084.32it/s]  6%|▌         | 22703/400000 [00:02<00:41, 9074.19it/s]  6%|▌         | 23620/400000 [00:02<00:41, 9102.01it/s]  6%|▌         | 24531/400000 [00:02<00:41, 9072.65it/s]  6%|▋         | 25439/400000 [00:02<00:41, 9008.25it/s]  7%|▋         | 26340/400000 [00:02<00:41, 8986.98it/s]  7%|▋         | 27254/400000 [00:03<00:41, 9030.09it/s]  7%|▋         | 28165/400000 [00:03<00:41, 9051.34it/s]  7%|▋         | 29071/400000 [00:03<00:41, 9029.15it/s]  7%|▋         | 29975/400000 [00:03<00:41, 8975.85it/s]  8%|▊         | 30873/400000 [00:03<00:41, 8973.91it/s]  8%|▊         | 31771/400000 [00:03<00:41, 8963.45it/s]  8%|▊         | 32686/400000 [00:03<00:40, 9017.20it/s]  8%|▊         | 33605/400000 [00:03<00:40, 9067.32it/s]  9%|▊         | 34527/400000 [00:03<00:40, 9110.21it/s]  9%|▉         | 35465/400000 [00:03<00:39, 9187.96it/s]  9%|▉         | 36385/400000 [00:04<00:39, 9163.90it/s]  9%|▉         | 37302/400000 [00:04<00:39, 9107.19it/s] 10%|▉         | 38213/400000 [00:04<00:39, 9056.85it/s] 10%|▉         | 39119/400000 [00:04<00:39, 9025.81it/s] 10%|█         | 40039/400000 [00:04<00:39, 9074.83it/s] 10%|█         | 40947/400000 [00:04<00:39, 9066.06it/s] 10%|█         | 41874/400000 [00:04<00:39, 9125.56it/s] 11%|█         | 42814/400000 [00:04<00:38, 9203.49it/s] 11%|█         | 43735/400000 [00:04<00:38, 9185.99it/s] 11%|█         | 44654/400000 [00:04<00:38, 9184.82it/s] 11%|█▏        | 45577/400000 [00:05<00:38, 9197.98it/s] 12%|█▏        | 46505/400000 [00:05<00:38, 9220.73it/s] 12%|█▏        | 47432/400000 [00:05<00:38, 9232.91it/s] 12%|█▏        | 48356/400000 [00:05<00:38, 9224.31it/s] 12%|█▏        | 49279/400000 [00:05<00:38, 9194.93it/s] 13%|█▎        | 50199/400000 [00:05<00:38, 9173.75it/s] 13%|█▎        | 51121/400000 [00:05<00:37, 9187.18it/s] 13%|█▎        | 52045/400000 [00:05<00:37, 9202.62it/s] 13%|█▎        | 52966/400000 [00:05<00:37, 9198.41it/s] 13%|█▎        | 53886/400000 [00:05<00:37, 9166.94it/s] 14%|█▎        | 54812/400000 [00:06<00:37, 9192.61it/s] 14%|█▍        | 55733/400000 [00:06<00:37, 9195.96it/s] 14%|█▍        | 56653/400000 [00:06<00:37, 9171.27it/s] 14%|█▍        | 57572/400000 [00:06<00:37, 9176.58it/s] 15%|█▍        | 58490/400000 [00:06<00:37, 9161.03it/s] 15%|█▍        | 59407/400000 [00:06<00:37, 9090.10it/s] 15%|█▌        | 60317/400000 [00:06<00:37, 9086.04it/s] 15%|█▌        | 61234/400000 [00:06<00:37, 9108.73it/s] 16%|█▌        | 62160/400000 [00:06<00:36, 9151.80it/s] 16%|█▌        | 63076/400000 [00:06<00:36, 9121.81it/s] 16%|█▌        | 63989/400000 [00:07<00:36, 9100.53it/s] 16%|█▌        | 64921/400000 [00:07<00:36, 9165.06it/s] 16%|█▋        | 65851/400000 [00:07<00:36, 9204.98it/s] 17%|█▋        | 66772/400000 [00:07<00:36, 9198.83it/s] 17%|█▋        | 67696/400000 [00:07<00:36, 9209.48it/s] 17%|█▋        | 68618/400000 [00:07<00:35, 9208.44it/s] 17%|█▋        | 69539/400000 [00:07<00:35, 9189.43it/s] 18%|█▊        | 70458/400000 [00:07<00:35, 9180.98it/s] 18%|█▊        | 71382/400000 [00:07<00:35, 9196.32it/s] 18%|█▊        | 72302/400000 [00:07<00:35, 9191.28it/s] 18%|█▊        | 73222/400000 [00:08<00:35, 9190.73it/s] 19%|█▊        | 74142/400000 [00:08<00:35, 9179.60it/s] 19%|█▉        | 75060/400000 [00:08<00:35, 9172.11it/s] 19%|█▉        | 75983/400000 [00:08<00:35, 9186.78it/s] 19%|█▉        | 76902/400000 [00:08<00:35, 9162.70it/s] 19%|█▉        | 77830/400000 [00:08<00:35, 9197.02it/s] 20%|█▉        | 78750/400000 [00:08<00:35, 9157.10it/s] 20%|█▉        | 79671/400000 [00:08<00:34, 9171.72it/s] 20%|██        | 80601/400000 [00:08<00:34, 9208.44it/s] 20%|██        | 81528/400000 [00:08<00:34, 9224.33it/s] 21%|██        | 82451/400000 [00:09<00:35, 8941.58it/s] 21%|██        | 83381/400000 [00:09<00:35, 9045.55it/s] 21%|██        | 84314/400000 [00:09<00:34, 9127.16it/s] 21%|██▏       | 85229/400000 [00:09<00:34, 9020.76it/s] 22%|██▏       | 86133/400000 [00:09<00:34, 8971.67it/s] 22%|██▏       | 87052/400000 [00:09<00:34, 9034.11it/s] 22%|██▏       | 87977/400000 [00:09<00:34, 9095.71it/s] 22%|██▏       | 88891/400000 [00:09<00:34, 9108.40it/s] 22%|██▏       | 89817/400000 [00:09<00:33, 9150.44it/s] 23%|██▎       | 90733/400000 [00:09<00:34, 9072.98it/s] 23%|██▎       | 91641/400000 [00:10<00:34, 9034.99it/s] 23%|██▎       | 92560/400000 [00:10<00:33, 9078.52it/s] 23%|██▎       | 93476/400000 [00:10<00:33, 9100.54it/s] 24%|██▎       | 94401/400000 [00:10<00:33, 9144.28it/s] 24%|██▍       | 95316/400000 [00:10<00:33, 8969.06it/s] 24%|██▍       | 96214/400000 [00:10<00:34, 8902.29it/s] 24%|██▍       | 97140/400000 [00:10<00:33, 9004.17it/s] 25%|██▍       | 98042/400000 [00:10<00:33, 8985.37it/s] 25%|██▍       | 98975/400000 [00:10<00:33, 9085.03it/s] 25%|██▍       | 99901/400000 [00:10<00:32, 9134.59it/s] 25%|██▌       | 100849/400000 [00:11<00:32, 9235.39it/s] 25%|██▌       | 101797/400000 [00:11<00:32, 9304.98it/s] 26%|██▌       | 102742/400000 [00:11<00:31, 9346.11it/s] 26%|██▌       | 103678/400000 [00:11<00:31, 9335.13it/s] 26%|██▌       | 104612/400000 [00:11<00:31, 9251.82it/s] 26%|██▋       | 105538/400000 [00:11<00:31, 9234.59it/s] 27%|██▋       | 106462/400000 [00:11<00:32, 9161.98it/s] 27%|██▋       | 107379/400000 [00:11<00:31, 9147.90it/s] 27%|██▋       | 108303/400000 [00:11<00:31, 9174.49it/s] 27%|██▋       | 109221/400000 [00:11<00:31, 9102.04it/s] 28%|██▊       | 110138/400000 [00:12<00:31, 9119.81it/s] 28%|██▊       | 111063/400000 [00:12<00:31, 9157.61it/s] 28%|██▊       | 111986/400000 [00:12<00:31, 9178.37it/s] 28%|██▊       | 112910/400000 [00:12<00:31, 9195.41it/s] 28%|██▊       | 113830/400000 [00:12<00:31, 9148.98it/s] 29%|██▊       | 114751/400000 [00:12<00:31, 9164.52it/s] 29%|██▉       | 115668/400000 [00:12<00:31, 9159.90it/s] 29%|██▉       | 116585/400000 [00:12<00:30, 9146.71it/s] 29%|██▉       | 117511/400000 [00:12<00:30, 9180.39it/s] 30%|██▉       | 118430/400000 [00:12<00:30, 9130.81it/s] 30%|██▉       | 119347/400000 [00:13<00:30, 9142.44it/s] 30%|███       | 120264/400000 [00:13<00:30, 9149.05it/s] 30%|███       | 121179/400000 [00:13<00:30, 9130.98it/s] 31%|███       | 122093/400000 [00:13<00:30, 9069.96it/s] 31%|███       | 123001/400000 [00:13<00:30, 9005.74it/s] 31%|███       | 123920/400000 [00:13<00:30, 9059.22it/s] 31%|███       | 124827/400000 [00:13<00:30, 9034.64it/s] 31%|███▏      | 125740/400000 [00:13<00:30, 9062.02it/s] 32%|███▏      | 126667/400000 [00:13<00:29, 9123.10it/s] 32%|███▏      | 127583/400000 [00:13<00:29, 9133.11it/s] 32%|███▏      | 128504/400000 [00:14<00:29, 9155.46it/s] 32%|███▏      | 129421/400000 [00:14<00:29, 9156.93it/s] 33%|███▎      | 130355/400000 [00:14<00:29, 9210.02it/s] 33%|███▎      | 131277/400000 [00:14<00:29, 9170.77it/s] 33%|███▎      | 132195/400000 [00:14<00:29, 8994.97it/s] 33%|███▎      | 133096/400000 [00:14<00:30, 8799.15it/s] 34%|███▎      | 134014/400000 [00:14<00:29, 8908.06it/s] 34%|███▎      | 134907/400000 [00:14<00:30, 8760.35it/s] 34%|███▍      | 135827/400000 [00:14<00:29, 8886.15it/s] 34%|███▍      | 136738/400000 [00:15<00:29, 8949.81it/s] 34%|███▍      | 137662/400000 [00:15<00:29, 9032.06it/s] 35%|███▍      | 138588/400000 [00:15<00:28, 9097.63it/s] 35%|███▍      | 139503/400000 [00:15<00:28, 9112.02it/s] 35%|███▌      | 140418/400000 [00:15<00:28, 9122.05it/s] 35%|███▌      | 141331/400000 [00:15<00:29, 8867.44it/s] 36%|███▌      | 142251/400000 [00:15<00:28, 8962.98it/s] 36%|███▌      | 143150/400000 [00:15<00:28, 8969.24it/s] 36%|███▌      | 144068/400000 [00:15<00:28, 9030.15it/s] 36%|███▌      | 144995/400000 [00:15<00:28, 9098.63it/s] 36%|███▋      | 145906/400000 [00:16<00:28, 9063.55it/s] 37%|███▋      | 146813/400000 [00:16<00:28, 9001.98it/s] 37%|███▋      | 147744/400000 [00:16<00:27, 9091.52it/s] 37%|███▋      | 148666/400000 [00:16<00:27, 9128.86it/s] 37%|███▋      | 149597/400000 [00:16<00:27, 9179.40it/s] 38%|███▊      | 150516/400000 [00:16<00:27, 9138.15it/s] 38%|███▊      | 151442/400000 [00:16<00:27, 9173.00it/s] 38%|███▊      | 152360/400000 [00:16<00:26, 9173.23it/s] 38%|███▊      | 153278/400000 [00:16<00:27, 9108.43it/s] 39%|███▊      | 154198/400000 [00:16<00:26, 9135.53it/s] 39%|███▉      | 155116/400000 [00:17<00:26, 9147.27it/s] 39%|███▉      | 156043/400000 [00:17<00:26, 9181.26it/s] 39%|███▉      | 156989/400000 [00:17<00:26, 9261.16it/s] 39%|███▉      | 157916/400000 [00:17<00:26, 9238.31it/s] 40%|███▉      | 158862/400000 [00:17<00:25, 9301.50it/s] 40%|███▉      | 159793/400000 [00:17<00:25, 9250.63it/s] 40%|████      | 160719/400000 [00:17<00:25, 9228.01it/s] 40%|████      | 161642/400000 [00:17<00:25, 9216.62it/s] 41%|████      | 162564/400000 [00:17<00:26, 9125.94it/s] 41%|████      | 163489/400000 [00:17<00:25, 9162.45it/s] 41%|████      | 164406/400000 [00:18<00:25, 9138.15it/s] 41%|████▏     | 165320/400000 [00:18<00:25, 9131.69it/s] 42%|████▏     | 166242/400000 [00:18<00:25, 9156.41it/s] 42%|████▏     | 167158/400000 [00:18<00:25, 9112.69it/s] 42%|████▏     | 168078/400000 [00:18<00:25, 9136.91it/s] 42%|████▏     | 168992/400000 [00:18<00:25, 9131.19it/s] 42%|████▏     | 169906/400000 [00:18<00:25, 9090.92it/s] 43%|████▎     | 170816/400000 [00:18<00:25, 9018.44it/s] 43%|████▎     | 171727/400000 [00:18<00:25, 9042.67it/s] 43%|████▎     | 172632/400000 [00:18<00:25, 8918.58it/s] 43%|████▎     | 173525/400000 [00:19<00:25, 8851.67it/s] 44%|████▎     | 174442/400000 [00:19<00:25, 8942.27it/s] 44%|████▍     | 175362/400000 [00:19<00:24, 9015.98it/s] 44%|████▍     | 176287/400000 [00:19<00:24, 9084.20it/s] 44%|████▍     | 177196/400000 [00:19<00:24, 9078.70it/s] 45%|████▍     | 178105/400000 [00:19<00:24, 8984.55it/s] 45%|████▍     | 179004/400000 [00:19<00:24, 8980.47it/s] 45%|████▍     | 179903/400000 [00:19<00:24, 8969.70it/s] 45%|████▌     | 180801/400000 [00:19<00:24, 8945.32it/s] 45%|████▌     | 181719/400000 [00:19<00:24, 9012.22it/s] 46%|████▌     | 182621/400000 [00:20<00:24, 8968.38it/s] 46%|████▌     | 183533/400000 [00:20<00:24, 9011.41it/s] 46%|████▌     | 184452/400000 [00:20<00:23, 9062.95it/s] 46%|████▋     | 185372/400000 [00:20<00:23, 9103.26it/s] 47%|████▋     | 186283/400000 [00:20<00:23, 9070.67it/s] 47%|████▋     | 187191/400000 [00:20<00:23, 9050.80it/s] 47%|████▋     | 188107/400000 [00:20<00:23, 9082.64it/s] 47%|████▋     | 189021/400000 [00:20<00:23, 9096.68it/s] 47%|████▋     | 189931/400000 [00:20<00:23, 8972.96it/s] 48%|████▊     | 190829/400000 [00:20<00:23, 8970.26it/s] 48%|████▊     | 191727/400000 [00:21<00:23, 8970.54it/s] 48%|████▊     | 192654/400000 [00:21<00:22, 9057.90it/s] 48%|████▊     | 193561/400000 [00:21<00:22, 9051.42it/s] 49%|████▊     | 194482/400000 [00:21<00:22, 9097.73it/s] 49%|████▉     | 195397/400000 [00:21<00:22, 9111.28it/s] 49%|████▉     | 196309/400000 [00:21<00:22, 9059.72it/s] 49%|████▉     | 197235/400000 [00:21<00:22, 9116.48it/s] 50%|████▉     | 198147/400000 [00:21<00:22, 9083.72it/s] 50%|████▉     | 199056/400000 [00:21<00:22, 9024.81it/s] 50%|████▉     | 199963/400000 [00:21<00:22, 9035.31it/s] 50%|█████     | 200867/400000 [00:22<00:22, 9009.52it/s] 50%|█████     | 201800/400000 [00:22<00:21, 9100.33it/s] 51%|█████     | 202711/400000 [00:22<00:21, 9102.06it/s] 51%|█████     | 203642/400000 [00:22<00:21, 9163.44it/s] 51%|█████     | 204559/400000 [00:22<00:21, 9138.00it/s] 51%|█████▏    | 205473/400000 [00:22<00:21, 9077.41it/s] 52%|█████▏    | 206393/400000 [00:22<00:21, 9111.84it/s] 52%|█████▏    | 207305/400000 [00:22<00:21, 9114.00it/s] 52%|█████▏    | 208234/400000 [00:22<00:20, 9163.55it/s] 52%|█████▏    | 209151/400000 [00:22<00:20, 9145.76it/s] 53%|█████▎    | 210066/400000 [00:23<00:20, 9087.32it/s] 53%|█████▎    | 210988/400000 [00:23<00:20, 9124.40it/s] 53%|█████▎    | 211920/400000 [00:23<00:20, 9180.13it/s] 53%|█████▎    | 212849/400000 [00:23<00:20, 9210.40it/s] 53%|█████▎    | 213771/400000 [00:23<00:20, 9188.76it/s] 54%|█████▎    | 214690/400000 [00:23<00:20, 9156.75it/s] 54%|█████▍    | 215606/400000 [00:23<00:20, 9145.18it/s] 54%|█████▍    | 216521/400000 [00:23<00:20, 8961.34it/s] 54%|█████▍    | 217431/400000 [00:23<00:20, 9002.12it/s] 55%|█████▍    | 218344/400000 [00:23<00:20, 9038.11it/s] 55%|█████▍    | 219251/400000 [00:24<00:19, 9045.83it/s] 55%|█████▌    | 220166/400000 [00:24<00:19, 9073.89it/s] 55%|█████▌    | 221079/400000 [00:24<00:19, 9088.12it/s] 55%|█████▌    | 221989/400000 [00:24<00:19, 9046.98it/s] 56%|█████▌    | 222894/400000 [00:24<00:19, 9043.71it/s] 56%|█████▌    | 223799/400000 [00:24<00:19, 9004.89it/s] 56%|█████▌    | 224714/400000 [00:24<00:19, 9047.87it/s] 56%|█████▋    | 225629/400000 [00:24<00:19, 9075.42it/s] 57%|█████▋    | 226547/400000 [00:24<00:19, 9104.18it/s] 57%|█████▋    | 227458/400000 [00:25<00:19, 9057.63it/s] 57%|█████▋    | 228364/400000 [00:25<00:19, 9015.29it/s] 57%|█████▋    | 229293/400000 [00:25<00:18, 9095.89it/s] 58%|█████▊    | 230209/400000 [00:25<00:18, 9114.74it/s] 58%|█████▊    | 231121/400000 [00:25<00:18, 9105.78it/s] 58%|█████▊    | 232032/400000 [00:25<00:19, 8801.12it/s] 58%|█████▊    | 232938/400000 [00:25<00:18, 8875.75it/s] 58%|█████▊    | 233857/400000 [00:25<00:18, 8965.93it/s] 59%|█████▊    | 234790/400000 [00:25<00:18, 9069.18it/s] 59%|█████▉    | 235702/400000 [00:25<00:18, 9083.46it/s] 59%|█████▉    | 236639/400000 [00:26<00:17, 9165.03it/s] 59%|█████▉    | 237557/400000 [00:26<00:17, 9165.40it/s] 60%|█████▉    | 238475/400000 [00:26<00:17, 9146.82it/s] 60%|█████▉    | 239391/400000 [00:26<00:17, 9082.55it/s] 60%|██████    | 240317/400000 [00:26<00:17, 9133.55it/s] 60%|██████    | 241239/400000 [00:26<00:17, 9157.54it/s] 61%|██████    | 242162/400000 [00:26<00:17, 9175.78it/s] 61%|██████    | 243080/400000 [00:26<00:17, 8901.10it/s] 61%|██████    | 244002/400000 [00:26<00:17, 8994.09it/s] 61%|██████    | 244935/400000 [00:26<00:17, 9089.75it/s] 61%|██████▏   | 245849/400000 [00:27<00:16, 9104.01it/s] 62%|██████▏   | 246761/400000 [00:27<00:16, 9035.90it/s] 62%|██████▏   | 247666/400000 [00:27<00:17, 8828.45it/s] 62%|██████▏   | 248554/400000 [00:27<00:17, 8834.79it/s] 62%|██████▏   | 249462/400000 [00:27<00:16, 8905.41it/s] 63%|██████▎   | 250354/400000 [00:27<00:16, 8887.45it/s] 63%|██████▎   | 251275/400000 [00:27<00:16, 8978.93it/s] 63%|██████▎   | 252174/400000 [00:27<00:16, 8965.86it/s] 63%|██████▎   | 253098/400000 [00:27<00:16, 9044.74it/s] 64%|██████▎   | 254037/400000 [00:27<00:15, 9143.24it/s] 64%|██████▎   | 254958/400000 [00:28<00:15, 9162.48it/s] 64%|██████▍   | 255875/400000 [00:28<00:15, 9158.87it/s] 64%|██████▍   | 256792/400000 [00:28<00:15, 9159.88it/s] 64%|██████▍   | 257719/400000 [00:28<00:15, 9191.02it/s] 65%|██████▍   | 258646/400000 [00:28<00:15, 9213.94it/s] 65%|██████▍   | 259568/400000 [00:28<00:15, 9184.14it/s] 65%|██████▌   | 260487/400000 [00:28<00:15, 9041.74it/s] 65%|██████▌   | 261392/400000 [00:28<00:15, 8983.45it/s] 66%|██████▌   | 262306/400000 [00:28<00:15, 9028.51it/s] 66%|██████▌   | 263224/400000 [00:28<00:15, 9071.60it/s] 66%|██████▌   | 264144/400000 [00:29<00:14, 9107.96it/s] 66%|██████▋   | 265056/400000 [00:29<00:14, 9064.84it/s] 66%|██████▋   | 265963/400000 [00:29<00:14, 8993.49it/s] 67%|██████▋   | 266863/400000 [00:29<00:15, 8793.05it/s] 67%|██████▋   | 267798/400000 [00:29<00:14, 8951.65it/s] 67%|██████▋   | 268730/400000 [00:29<00:14, 9057.50it/s] 67%|██████▋   | 269654/400000 [00:29<00:14, 9111.35it/s] 68%|██████▊   | 270567/400000 [00:29<00:14, 9060.78it/s] 68%|██████▊   | 271492/400000 [00:29<00:14, 9116.45it/s] 68%|██████▊   | 272438/400000 [00:29<00:13, 9214.71it/s] 68%|██████▊   | 273362/400000 [00:30<00:13, 9221.36it/s] 69%|██████▊   | 274285/400000 [00:30<00:13, 9115.88it/s] 69%|██████▉   | 275198/400000 [00:30<00:13, 9012.53it/s] 69%|██████▉   | 276136/400000 [00:30<00:13, 9117.51it/s] 69%|██████▉   | 277080/400000 [00:30<00:13, 9209.36it/s] 70%|██████▉   | 278002/400000 [00:30<00:13, 9147.66it/s] 70%|██████▉   | 278918/400000 [00:30<00:13, 9110.25it/s] 70%|██████▉   | 279830/400000 [00:30<00:13, 8927.69it/s] 70%|███████   | 280727/400000 [00:30<00:13, 8937.61it/s] 70%|███████   | 281628/400000 [00:30<00:13, 8956.73it/s] 71%|███████   | 282534/400000 [00:31<00:13, 8987.20it/s] 71%|███████   | 283455/400000 [00:31<00:12, 9050.96it/s] 71%|███████   | 284362/400000 [00:31<00:12, 9056.04it/s] 71%|███████▏  | 285272/400000 [00:31<00:12, 9067.47it/s] 72%|███████▏  | 286179/400000 [00:31<00:12, 9002.71it/s] 72%|███████▏  | 287080/400000 [00:31<00:12, 8852.36it/s] 72%|███████▏  | 287990/400000 [00:31<00:12, 8923.21it/s] 72%|███████▏  | 288883/400000 [00:31<00:12, 8717.40it/s] 72%|███████▏  | 289791/400000 [00:31<00:12, 8822.91it/s] 73%|███████▎  | 290688/400000 [00:32<00:12, 8865.93it/s] 73%|███████▎  | 291605/400000 [00:32<00:12, 8953.44it/s] 73%|███████▎  | 292502/400000 [00:32<00:12, 8905.77it/s] 73%|███████▎  | 293394/400000 [00:32<00:12, 8713.60it/s] 74%|███████▎  | 294267/400000 [00:32<00:12, 8334.08it/s] 74%|███████▍  | 295141/400000 [00:32<00:12, 8451.83it/s] 74%|███████▍  | 296036/400000 [00:32<00:12, 8594.22it/s] 74%|███████▍  | 296924/400000 [00:32<00:11, 8675.47it/s] 74%|███████▍  | 297838/400000 [00:32<00:11, 8807.80it/s] 75%|███████▍  | 298750/400000 [00:32<00:11, 8898.27it/s] 75%|███████▍  | 299642/400000 [00:33<00:11, 8852.22it/s] 75%|███████▌  | 300539/400000 [00:33<00:11, 8886.63it/s] 75%|███████▌  | 301440/400000 [00:33<00:11, 8921.21it/s] 76%|███████▌  | 302333/400000 [00:33<00:10, 8891.82it/s] 76%|███████▌  | 303258/400000 [00:33<00:10, 8995.46it/s] 76%|███████▌  | 304159/400000 [00:33<00:10, 8998.75it/s] 76%|███████▋  | 305060/400000 [00:33<00:10, 8979.40it/s] 76%|███████▋  | 305977/400000 [00:33<00:10, 9035.52it/s] 77%|███████▋  | 306881/400000 [00:33<00:10, 8972.22it/s] 77%|███████▋  | 307792/400000 [00:33<00:10, 9012.08it/s] 77%|███████▋  | 308722/400000 [00:34<00:10, 9095.50it/s] 77%|███████▋  | 309648/400000 [00:34<00:09, 9143.82it/s] 78%|███████▊  | 310580/400000 [00:34<00:09, 9194.47it/s] 78%|███████▊  | 311500/400000 [00:34<00:09, 9150.13it/s] 78%|███████▊  | 312455/400000 [00:34<00:09, 9266.06it/s] 78%|███████▊  | 313383/400000 [00:34<00:09, 9059.29it/s] 79%|███████▊  | 314309/400000 [00:34<00:09, 9117.53it/s] 79%|███████▉  | 315243/400000 [00:34<00:09, 9180.91it/s] 79%|███████▉  | 316166/400000 [00:34<00:09, 9194.91it/s] 79%|███████▉  | 317100/400000 [00:34<00:08, 9237.27it/s] 80%|███████▉  | 318025/400000 [00:35<00:08, 9204.66it/s] 80%|███████▉  | 318946/400000 [00:35<00:08, 9071.67it/s] 80%|███████▉  | 319871/400000 [00:35<00:08, 9122.73it/s] 80%|████████  | 320784/400000 [00:35<00:08, 9047.82it/s] 80%|████████  | 321693/400000 [00:35<00:08, 9059.08it/s] 81%|████████  | 322630/400000 [00:35<00:08, 9149.43it/s] 81%|████████  | 323558/400000 [00:35<00:08, 9185.84it/s] 81%|████████  | 324482/400000 [00:35<00:08, 9199.39it/s] 81%|████████▏ | 325412/400000 [00:35<00:08, 9227.56it/s] 82%|████████▏ | 326335/400000 [00:35<00:07, 9216.97it/s] 82%|████████▏ | 327270/400000 [00:36<00:07, 9255.59it/s] 82%|████████▏ | 328210/400000 [00:36<00:07, 9297.46it/s] 82%|████████▏ | 329140/400000 [00:36<00:07, 9293.91it/s] 83%|████████▎ | 330070/400000 [00:36<00:07, 9247.39it/s] 83%|████████▎ | 330995/400000 [00:36<00:07, 9239.14it/s] 83%|████████▎ | 331920/400000 [00:36<00:07, 9114.03it/s] 83%|████████▎ | 332836/400000 [00:36<00:07, 9126.79it/s] 83%|████████▎ | 333749/400000 [00:36<00:07, 9062.90it/s] 84%|████████▎ | 334656/400000 [00:36<00:07, 9013.76it/s] 84%|████████▍ | 335558/400000 [00:36<00:07, 8975.61it/s] 84%|████████▍ | 336476/400000 [00:37<00:07, 9034.62it/s] 84%|████████▍ | 337420/400000 [00:37<00:06, 9150.64it/s] 85%|████████▍ | 338358/400000 [00:37<00:06, 9217.69it/s] 85%|████████▍ | 339281/400000 [00:37<00:06, 9157.29it/s] 85%|████████▌ | 340198/400000 [00:37<00:06, 9138.69it/s] 85%|████████▌ | 341113/400000 [00:37<00:06, 9139.87it/s] 86%|████████▌ | 342028/400000 [00:37<00:06, 9059.54it/s] 86%|████████▌ | 342935/400000 [00:37<00:06, 8857.08it/s] 86%|████████▌ | 343822/400000 [00:37<00:06, 8855.09it/s] 86%|████████▌ | 344714/400000 [00:37<00:06, 8872.26it/s] 86%|████████▋ | 345620/400000 [00:38<00:06, 8925.45it/s] 87%|████████▋ | 346533/400000 [00:38<00:05, 8981.65it/s] 87%|████████▋ | 347454/400000 [00:38<00:05, 9048.82it/s] 87%|████████▋ | 348360/400000 [00:38<00:05, 8985.19it/s] 87%|████████▋ | 349259/400000 [00:38<00:05, 8794.32it/s] 88%|████████▊ | 350175/400000 [00:38<00:05, 8900.88it/s] 88%|████████▊ | 351067/400000 [00:38<00:05, 8769.23it/s] 88%|████████▊ | 351987/400000 [00:38<00:05, 8893.50it/s] 88%|████████▊ | 352914/400000 [00:38<00:05, 9002.09it/s] 88%|████████▊ | 353822/400000 [00:38<00:05, 9025.14it/s] 89%|████████▊ | 354726/400000 [00:39<00:05, 8998.36it/s] 89%|████████▉ | 355636/400000 [00:39<00:04, 9028.19it/s] 89%|████████▉ | 356564/400000 [00:39<00:04, 9102.16it/s] 89%|████████▉ | 357475/400000 [00:39<00:04, 9079.33it/s] 90%|████████▉ | 358384/400000 [00:39<00:04, 8936.42it/s] 90%|████████▉ | 359301/400000 [00:39<00:04, 9003.36it/s] 90%|█████████ | 360229/400000 [00:39<00:04, 9081.84it/s] 90%|█████████ | 361138/400000 [00:39<00:04, 8945.44it/s] 91%|█████████ | 362044/400000 [00:39<00:04, 8978.00it/s] 91%|█████████ | 362954/400000 [00:40<00:04, 9010.31it/s] 91%|█████████ | 363879/400000 [00:40<00:03, 9078.28it/s] 91%|█████████ | 364819/400000 [00:40<00:03, 9170.78it/s] 91%|█████████▏| 365748/400000 [00:40<00:03, 9205.42it/s] 92%|█████████▏| 366669/400000 [00:40<00:03, 9184.92it/s] 92%|█████████▏| 367588/400000 [00:40<00:03, 8853.33it/s] 92%|█████████▏| 368501/400000 [00:40<00:03, 8932.01it/s] 92%|█████████▏| 369397/400000 [00:40<00:03, 8789.51it/s] 93%|█████████▎| 370278/400000 [00:40<00:03, 8625.48it/s] 93%|█████████▎| 371190/400000 [00:40<00:03, 8767.79it/s] 93%|█████████▎| 372119/400000 [00:41<00:03, 8916.21it/s] 93%|█████████▎| 373043/400000 [00:41<00:02, 9009.76it/s] 93%|█████████▎| 373967/400000 [00:41<00:02, 9077.61it/s] 94%|█████████▎| 374877/400000 [00:41<00:02, 8995.20it/s] 94%|█████████▍| 375778/400000 [00:41<00:02, 8990.14it/s] 94%|█████████▍| 376693/400000 [00:41<00:02, 9035.39it/s] 94%|█████████▍| 377605/400000 [00:41<00:02, 9060.15it/s] 95%|█████████▍| 378537/400000 [00:41<00:02, 9133.92it/s] 95%|█████████▍| 379451/400000 [00:41<00:02, 9117.68it/s] 95%|█████████▌| 380364/400000 [00:41<00:02, 9054.33it/s] 95%|█████████▌| 381270/400000 [00:42<00:02, 9045.15it/s] 96%|█████████▌| 382197/400000 [00:42<00:01, 9110.16it/s] 96%|█████████▌| 383109/400000 [00:42<00:01, 9044.57it/s] 96%|█████████▌| 384029/400000 [00:42<00:01, 9088.62it/s] 96%|█████████▌| 384939/400000 [00:42<00:01, 9090.33it/s] 96%|█████████▋| 385849/400000 [00:42<00:01, 9055.76it/s] 97%|█████████▋| 386779/400000 [00:42<00:01, 9126.72it/s] 97%|█████████▋| 387692/400000 [00:42<00:01, 9125.66it/s] 97%|█████████▋| 388629/400000 [00:42<00:01, 9195.91it/s] 97%|█████████▋| 389555/400000 [00:42<00:01, 9215.03it/s] 98%|█████████▊| 390477/400000 [00:43<00:01, 9202.65it/s] 98%|█████████▊| 391401/400000 [00:43<00:00, 9213.58it/s] 98%|█████████▊| 392339/400000 [00:43<00:00, 9261.03it/s] 98%|█████████▊| 393302/400000 [00:43<00:00, 9367.12it/s] 99%|█████████▊| 394243/400000 [00:43<00:00, 9379.86it/s] 99%|█████████▉| 395182/400000 [00:43<00:00, 9323.37it/s] 99%|█████████▉| 396115/400000 [00:43<00:00, 9319.01it/s] 99%|█████████▉| 397048/400000 [00:43<00:00, 9302.84it/s] 99%|█████████▉| 397987/400000 [00:43<00:00, 9325.66it/s]100%|█████████▉| 398920/400000 [00:43<00:00, 9293.61it/s]100%|█████████▉| 399854/400000 [00:44<00:00, 9306.49it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 9076.87it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f89d00eb940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011113713824527287 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.010933687256331428 	 Accuracy: 57

  model saves at 57% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15819 out of table with 15679 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15819 out of table with 15679 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-15 04:23:43.136971: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 04:23:43.141487: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095225000 Hz
2020-05-15 04:23:43.141625: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560daf9ebb30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 04:23:43.141639: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f8975b62860> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8660 - accuracy: 0.4870
 2000/25000 [=>............................] - ETA: 7s - loss: 7.6206 - accuracy: 0.5030 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6717 - accuracy: 0.4997
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6743 - accuracy: 0.4995
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6942 - accuracy: 0.4982
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5976 - accuracy: 0.5045
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6294 - accuracy: 0.5024
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5995 - accuracy: 0.5044
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6002 - accuracy: 0.5043
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5915 - accuracy: 0.5049
11000/25000 [============>.................] - ETA: 3s - loss: 7.6206 - accuracy: 0.5030
12000/25000 [=============>................] - ETA: 3s - loss: 7.6347 - accuracy: 0.5021
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6607 - accuracy: 0.5004
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6425 - accuracy: 0.5016
15000/25000 [=================>............] - ETA: 2s - loss: 7.6799 - accuracy: 0.4991
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6829 - accuracy: 0.4989
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6973 - accuracy: 0.4980
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6964 - accuracy: 0.4981
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7062 - accuracy: 0.4974
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6843 - accuracy: 0.4988
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6747 - accuracy: 0.4995
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6882 - accuracy: 0.4986
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6980 - accuracy: 0.4980
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6839 - accuracy: 0.4989
25000/25000 [==============================] - 7s 264us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f894c00b198> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f894d269128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.8308 - crf_viterbi_accuracy: 0.3333 - val_loss: 1.7884 - val_crf_viterbi_accuracy: 0.2800

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

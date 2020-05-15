
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fbf9cfe4fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 05:13:28.515104
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 05:13:28.518604
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 05:13:28.521786
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 05:13:28.524971
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fbfa8ffc4a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 350396.1562
Epoch 2/10

1/1 [==============================] - 0s 96ms/step - loss: 221784.2656
Epoch 3/10

1/1 [==============================] - 0s 88ms/step - loss: 119250.6250
Epoch 4/10

1/1 [==============================] - 0s 85ms/step - loss: 58785.0820
Epoch 5/10

1/1 [==============================] - 0s 86ms/step - loss: 30625.8398
Epoch 6/10

1/1 [==============================] - 0s 84ms/step - loss: 17704.3809
Epoch 7/10

1/1 [==============================] - 0s 88ms/step - loss: 11413.2090
Epoch 8/10

1/1 [==============================] - 0s 84ms/step - loss: 8013.2290
Epoch 9/10

1/1 [==============================] - 0s 89ms/step - loss: 5996.6206
Epoch 10/10

1/1 [==============================] - 0s 85ms/step - loss: 4744.1128

  #### Inference Need return ypred, ytrue ######################### 
[[-0.47225946 -0.10601866 -0.2904994  -0.23552269 -2.3373933   0.569896
  -0.9917861  -0.6355439  -2.123188   -0.7299081   0.91464627 -0.52593446
  -0.37812462 -1.6747283   0.1612292   1.8075671  -1.57161    -1.249431
  -0.4528244  -0.5666383  -0.45850164 -0.74297434 -1.3516245  -0.8944909
  -0.14512268 -1.049003   -1.3322035  -0.29321292  0.86609817 -1.7210591
  -0.0882827   1.5537322   0.36129367 -1.1342208  -1.7570333   0.39060235
   1.1084852  -0.19114062 -0.11623436  0.8819446   0.36523598 -0.28957367
  -1.270217    1.3175523  -1.4116666   0.4737289   0.07020748  0.9684163
  -1.0108396  -0.47359747 -0.26382804  0.27231073 -1.1899637  -1.0808164
  -1.1318476   0.32472622 -2.009007   -0.5848963  -0.96926653  0.6187209
  -0.05090103 -1.9750253  -0.5112495  -2.211348    0.2081213  -0.7139995
  -0.9109078  -0.1042591   2.33879    -0.15488917  0.60232556 -0.62937987
  -0.4343749   0.03615808  0.20945537  0.56665266 -1.1746273  -0.67351836
  -2.6068025   0.174395    0.3519126  -0.89958423 -1.1192584   0.14680064
  -0.1025492   1.3322576   0.4900689   0.7977155  -1.7819724  -1.5221436
   0.11576667 -1.152274    0.16065717 -0.19482434 -1.7298791  -0.8421843
   0.80254865  0.43472517 -0.12649885  0.67610186 -1.1149145  -1.0015082
  -1.5458288  -0.47363183 -0.03270796  1.4164685  -1.1364167   1.9072187
  -0.30749834  2.3338053  -0.39630878  1.720299    0.0701654   0.9556853
   0.28226805  0.6137736   0.775102    0.901003    0.36182588 -0.5779345
   0.3488239   9.170291    9.32218     9.755447   10.450108    9.886322
  11.935778    9.641296   10.3101015  10.880411    9.584409    9.919319
   9.942488   11.198782   10.666417   10.002041   10.049326    9.617054
  10.22506     9.420172    9.15539    10.441774   10.009385   10.640718
  10.262914   12.123521    8.302155    7.93426     8.812665   10.537295
   9.822716    8.530348   10.2932005   8.426936    9.599971    9.756339
   9.019606   10.191362   12.103466    9.037555   10.555982    9.73509
  11.250991   10.7782     11.624022    9.620365    9.627053    8.918632
   8.087481   10.718044    8.759456    9.955648    8.428554    9.291771
   9.868008    9.191261   10.282398    9.524602   11.115536    9.105983
   1.7858741   0.53358006  1.0100378   0.40153497  1.7521906   0.59785104
   0.27161705  2.0622296   0.31864107  0.41627192  0.6334593   0.46969694
   1.6015983   0.36411452  0.45892453  3.5719523   0.67395544  0.29672807
   2.1149511   0.32662547  2.4814458   1.8016431   2.5980244   0.20861197
   0.8885864   1.1364883   0.66984     2.8786793   1.063923    3.013864
   0.47069335  0.6418826   1.3610244   1.573299    0.12160254  1.5179101
   1.0445821   0.12320101  1.4725429   0.7179204   0.80791116  0.7182685
   0.23748183  1.1907358   0.12650764  1.7559419   2.506371    1.8212867
   0.6850029   0.21216369  2.3549786   2.252105    0.8715894   1.7432394
   2.6224113   0.66572076  1.2793031   0.93962777  0.88363904  0.8944716
   0.21683311  1.3585749   1.1475481   0.81102777  0.23863125  1.8172123
   0.29578424  1.3342363   0.48903883  2.2592037   1.4700844   2.096696
   0.9518299   1.6716856   4.274399    0.2200864   0.66989374  1.5614637
   1.5162401   0.6940613   0.67026865  1.1890992   1.5939591   0.93006355
   1.1461025   0.21367753  2.7600849   0.91717845  0.9886555   2.2709327
   0.28851223  2.6561613   1.1609085   2.15163     0.2835166   0.45764518
   0.65463114  0.6248006   1.041218    2.702981    2.0491672   0.1837998
   0.6470287   0.50633556  0.81668377  0.13984811  2.6443367   0.4031272
   2.226774    1.6320859   0.94995666  0.12898248  1.9594332   1.2289422
   1.8653516   0.7289118   2.0091057   0.48446333  1.5012466   1.0891151
   0.28941798 10.351122    9.716722    9.729166   11.041384   11.301892
   9.776771   11.239159   10.448043   10.082039    8.399381    8.594254
   9.577534    9.692429    9.270344   10.83763     8.019043   10.452581
  10.000103   11.463698   10.532259   10.231159   10.530336   10.414548
  10.990569   11.47039    10.396824   10.567442   10.562771   10.926405
  11.159912    8.385554    9.377821    8.688243   10.689294    8.6195135
   8.915635   10.222719    9.323247   10.733625    9.757029    7.455559
   8.254567   10.373428   10.957064   10.892696   10.886352    9.403321
   9.270677    8.246822    9.957633    9.324193   10.460925   10.037
  10.72142     9.619564    8.627459    9.300906    8.968287    9.507581
  -6.585805   -6.772427    9.438368  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 05:13:38.256923
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.7321
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 05:13:38.260230
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8622.15
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 05:13:38.262983
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.1999
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 05:13:38.265451
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -771.174
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140460609888832
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140459382731384
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140459382731888
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140459382732392
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140459382732896
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140459382733400

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fbfa4e7def0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.523552
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.496968
grad_step = 000002, loss = 0.473391
grad_step = 000003, loss = 0.447402
grad_step = 000004, loss = 0.419490
grad_step = 000005, loss = 0.392268
grad_step = 000006, loss = 0.370725
grad_step = 000007, loss = 0.347629
grad_step = 000008, loss = 0.320170
grad_step = 000009, loss = 0.296781
grad_step = 000010, loss = 0.272409
grad_step = 000011, loss = 0.248362
grad_step = 000012, loss = 0.227733
grad_step = 000013, loss = 0.208971
grad_step = 000014, loss = 0.192508
grad_step = 000015, loss = 0.177202
grad_step = 000016, loss = 0.162545
grad_step = 000017, loss = 0.148753
grad_step = 000018, loss = 0.135174
grad_step = 000019, loss = 0.121951
grad_step = 000020, loss = 0.109975
grad_step = 000021, loss = 0.099347
grad_step = 000022, loss = 0.089145
grad_step = 000023, loss = 0.078787
grad_step = 000024, loss = 0.069409
grad_step = 000025, loss = 0.061100
grad_step = 000026, loss = 0.053707
grad_step = 000027, loss = 0.047206
grad_step = 000028, loss = 0.041466
grad_step = 000029, loss = 0.036331
grad_step = 000030, loss = 0.031517
grad_step = 000031, loss = 0.027210
grad_step = 000032, loss = 0.023310
grad_step = 000033, loss = 0.019856
grad_step = 000034, loss = 0.016853
grad_step = 000035, loss = 0.014475
grad_step = 000036, loss = 0.012502
grad_step = 000037, loss = 0.010905
grad_step = 000038, loss = 0.009596
grad_step = 000039, loss = 0.008451
grad_step = 000040, loss = 0.007382
grad_step = 000041, loss = 0.006420
grad_step = 000042, loss = 0.005706
grad_step = 000043, loss = 0.005211
grad_step = 000044, loss = 0.004901
grad_step = 000045, loss = 0.004608
grad_step = 000046, loss = 0.004390
grad_step = 000047, loss = 0.004128
grad_step = 000048, loss = 0.003920
grad_step = 000049, loss = 0.003739
grad_step = 000050, loss = 0.003575
grad_step = 000051, loss = 0.003461
grad_step = 000052, loss = 0.003398
grad_step = 000053, loss = 0.003346
grad_step = 000054, loss = 0.003278
grad_step = 000055, loss = 0.003233
grad_step = 000056, loss = 0.003164
grad_step = 000057, loss = 0.003095
grad_step = 000058, loss = 0.003025
grad_step = 000059, loss = 0.002984
grad_step = 000060, loss = 0.002936
grad_step = 000061, loss = 0.002900
grad_step = 000062, loss = 0.002855
grad_step = 000063, loss = 0.002818
grad_step = 000064, loss = 0.002781
grad_step = 000065, loss = 0.002743
grad_step = 000066, loss = 0.002696
grad_step = 000067, loss = 0.002661
grad_step = 000068, loss = 0.002625
grad_step = 000069, loss = 0.002585
grad_step = 000070, loss = 0.002549
grad_step = 000071, loss = 0.002517
grad_step = 000072, loss = 0.002479
grad_step = 000073, loss = 0.002445
grad_step = 000074, loss = 0.002414
grad_step = 000075, loss = 0.002384
grad_step = 000076, loss = 0.002356
grad_step = 000077, loss = 0.002330
grad_step = 000078, loss = 0.002306
grad_step = 000079, loss = 0.002285
grad_step = 000080, loss = 0.002263
grad_step = 000081, loss = 0.002242
grad_step = 000082, loss = 0.002226
grad_step = 000083, loss = 0.002211
grad_step = 000084, loss = 0.002198
grad_step = 000085, loss = 0.002188
grad_step = 000086, loss = 0.002178
grad_step = 000087, loss = 0.002168
grad_step = 000088, loss = 0.002161
grad_step = 000089, loss = 0.002153
grad_step = 000090, loss = 0.002147
grad_step = 000091, loss = 0.002141
grad_step = 000092, loss = 0.002136
grad_step = 000093, loss = 0.002132
grad_step = 000094, loss = 0.002128
grad_step = 000095, loss = 0.002124
grad_step = 000096, loss = 0.002122
grad_step = 000097, loss = 0.002122
grad_step = 000098, loss = 0.002128
grad_step = 000099, loss = 0.002150
grad_step = 000100, loss = 0.002207
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002332
grad_step = 000102, loss = 0.002506
grad_step = 000103, loss = 0.002604
grad_step = 000104, loss = 0.002400
grad_step = 000105, loss = 0.002128
grad_step = 000106, loss = 0.002150
grad_step = 000107, loss = 0.002342
grad_step = 000108, loss = 0.002329
grad_step = 000109, loss = 0.002127
grad_step = 000110, loss = 0.002112
grad_step = 000111, loss = 0.002249
grad_step = 000112, loss = 0.002227
grad_step = 000113, loss = 0.002093
grad_step = 000114, loss = 0.002101
grad_step = 000115, loss = 0.002189
grad_step = 000116, loss = 0.002156
grad_step = 000117, loss = 0.002069
grad_step = 000118, loss = 0.002093
grad_step = 000119, loss = 0.002147
grad_step = 000120, loss = 0.002109
grad_step = 000121, loss = 0.002056
grad_step = 000122, loss = 0.002080
grad_step = 000123, loss = 0.002112
grad_step = 000124, loss = 0.002082
grad_step = 000125, loss = 0.002047
grad_step = 000126, loss = 0.002063
grad_step = 000127, loss = 0.002085
grad_step = 000128, loss = 0.002065
grad_step = 000129, loss = 0.002039
grad_step = 000130, loss = 0.002044
grad_step = 000131, loss = 0.002061
grad_step = 000132, loss = 0.002054
grad_step = 000133, loss = 0.002033
grad_step = 000134, loss = 0.002028
grad_step = 000135, loss = 0.002038
grad_step = 000136, loss = 0.002042
grad_step = 000137, loss = 0.002031
grad_step = 000138, loss = 0.002019
grad_step = 000139, loss = 0.002018
grad_step = 000140, loss = 0.002023
grad_step = 000141, loss = 0.002024
grad_step = 000142, loss = 0.002018
grad_step = 000143, loss = 0.002009
grad_step = 000144, loss = 0.002004
grad_step = 000145, loss = 0.002005
grad_step = 000146, loss = 0.002007
grad_step = 000147, loss = 0.002006
grad_step = 000148, loss = 0.002002
grad_step = 000149, loss = 0.001996
grad_step = 000150, loss = 0.001990
grad_step = 000151, loss = 0.001987
grad_step = 000152, loss = 0.001986
grad_step = 000153, loss = 0.001985
grad_step = 000154, loss = 0.001985
grad_step = 000155, loss = 0.001985
grad_step = 000156, loss = 0.001984
grad_step = 000157, loss = 0.001983
grad_step = 000158, loss = 0.001981
grad_step = 000159, loss = 0.001983
grad_step = 000160, loss = 0.001989
grad_step = 000161, loss = 0.002003
grad_step = 000162, loss = 0.002030
grad_step = 000163, loss = 0.002076
grad_step = 000164, loss = 0.002152
grad_step = 000165, loss = 0.002249
grad_step = 000166, loss = 0.002345
grad_step = 000167, loss = 0.002353
grad_step = 000168, loss = 0.002237
grad_step = 000169, loss = 0.002046
grad_step = 000170, loss = 0.001942
grad_step = 000171, loss = 0.001986
grad_step = 000172, loss = 0.002096
grad_step = 000173, loss = 0.002145
grad_step = 000174, loss = 0.002076
grad_step = 000175, loss = 0.001967
grad_step = 000176, loss = 0.001927
grad_step = 000177, loss = 0.001976
grad_step = 000178, loss = 0.002037
grad_step = 000179, loss = 0.002036
grad_step = 000180, loss = 0.001975
grad_step = 000181, loss = 0.001921
grad_step = 000182, loss = 0.001921
grad_step = 000183, loss = 0.001959
grad_step = 000184, loss = 0.001985
grad_step = 000185, loss = 0.001972
grad_step = 000186, loss = 0.001933
grad_step = 000187, loss = 0.001903
grad_step = 000188, loss = 0.001901
grad_step = 000189, loss = 0.001920
grad_step = 000190, loss = 0.001938
grad_step = 000191, loss = 0.001939
grad_step = 000192, loss = 0.001924
grad_step = 000193, loss = 0.001902
grad_step = 000194, loss = 0.001885
grad_step = 000195, loss = 0.001879
grad_step = 000196, loss = 0.001883
grad_step = 000197, loss = 0.001892
grad_step = 000198, loss = 0.001901
grad_step = 000199, loss = 0.001907
grad_step = 000200, loss = 0.001910
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001911
grad_step = 000202, loss = 0.001910
grad_step = 000203, loss = 0.001908
grad_step = 000204, loss = 0.001905
grad_step = 000205, loss = 0.001904
grad_step = 000206, loss = 0.001904
grad_step = 000207, loss = 0.001907
grad_step = 000208, loss = 0.001914
grad_step = 000209, loss = 0.001923
grad_step = 000210, loss = 0.001938
grad_step = 000211, loss = 0.001960
grad_step = 000212, loss = 0.001988
grad_step = 000213, loss = 0.002020
grad_step = 000214, loss = 0.002047
grad_step = 000215, loss = 0.002053
grad_step = 000216, loss = 0.002030
grad_step = 000217, loss = 0.001976
grad_step = 000218, loss = 0.001906
grad_step = 000219, loss = 0.001848
grad_step = 000220, loss = 0.001819
grad_step = 000221, loss = 0.001821
grad_step = 000222, loss = 0.001847
grad_step = 000223, loss = 0.001885
grad_step = 000224, loss = 0.001931
grad_step = 000225, loss = 0.001980
grad_step = 000226, loss = 0.002025
grad_step = 000227, loss = 0.002052
grad_step = 000228, loss = 0.002041
grad_step = 000229, loss = 0.001986
grad_step = 000230, loss = 0.001902
grad_step = 000231, loss = 0.001829
grad_step = 000232, loss = 0.001797
grad_step = 000233, loss = 0.001806
grad_step = 000234, loss = 0.001841
grad_step = 000235, loss = 0.001878
grad_step = 000236, loss = 0.001899
grad_step = 000237, loss = 0.001895
grad_step = 000238, loss = 0.001869
grad_step = 000239, loss = 0.001831
grad_step = 000240, loss = 0.001796
grad_step = 000241, loss = 0.001775
grad_step = 000242, loss = 0.001770
grad_step = 000243, loss = 0.001777
grad_step = 000244, loss = 0.001791
grad_step = 000245, loss = 0.001808
grad_step = 000246, loss = 0.001831
grad_step = 000247, loss = 0.001864
grad_step = 000248, loss = 0.001915
grad_step = 000249, loss = 0.001988
grad_step = 000250, loss = 0.002067
grad_step = 000251, loss = 0.002126
grad_step = 000252, loss = 0.002112
grad_step = 000253, loss = 0.002016
grad_step = 000254, loss = 0.001887
grad_step = 000255, loss = 0.001802
grad_step = 000256, loss = 0.001788
grad_step = 000257, loss = 0.001817
grad_step = 000258, loss = 0.001838
grad_step = 000259, loss = 0.001837
grad_step = 000260, loss = 0.001830
grad_step = 000261, loss = 0.001822
grad_step = 000262, loss = 0.001809
grad_step = 000263, loss = 0.001774
grad_step = 000264, loss = 0.001742
grad_step = 000265, loss = 0.001737
grad_step = 000266, loss = 0.001763
grad_step = 000267, loss = 0.001793
grad_step = 000268, loss = 0.001795
grad_step = 000269, loss = 0.001774
grad_step = 000270, loss = 0.001749
grad_step = 000271, loss = 0.001743
grad_step = 000272, loss = 0.001752
grad_step = 000273, loss = 0.001758
grad_step = 000274, loss = 0.001748
grad_step = 000275, loss = 0.001733
grad_step = 000276, loss = 0.001724
grad_step = 000277, loss = 0.001730
grad_step = 000278, loss = 0.001745
grad_step = 000279, loss = 0.001762
grad_step = 000280, loss = 0.001781
grad_step = 000281, loss = 0.001820
grad_step = 000282, loss = 0.001898
grad_step = 000283, loss = 0.002040
grad_step = 000284, loss = 0.002202
grad_step = 000285, loss = 0.002339
grad_step = 000286, loss = 0.002291
grad_step = 000287, loss = 0.002079
grad_step = 000288, loss = 0.001848
grad_step = 000289, loss = 0.001778
grad_step = 000290, loss = 0.001826
grad_step = 000291, loss = 0.001882
grad_step = 000292, loss = 0.001900
grad_step = 000293, loss = 0.001878
grad_step = 000294, loss = 0.001821
grad_step = 000295, loss = 0.001724
grad_step = 000296, loss = 0.001695
grad_step = 000297, loss = 0.001758
grad_step = 000298, loss = 0.001828
grad_step = 000299, loss = 0.001821
grad_step = 000300, loss = 0.001729
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001669
grad_step = 000302, loss = 0.001679
grad_step = 000303, loss = 0.001702
grad_step = 000304, loss = 0.001702
grad_step = 000305, loss = 0.001701
grad_step = 000306, loss = 0.001724
grad_step = 000307, loss = 0.001739
grad_step = 000308, loss = 0.001718
grad_step = 000309, loss = 0.001677
grad_step = 000310, loss = 0.001655
grad_step = 000311, loss = 0.001653
grad_step = 000312, loss = 0.001646
grad_step = 000313, loss = 0.001630
grad_step = 000314, loss = 0.001625
grad_step = 000315, loss = 0.001638
grad_step = 000316, loss = 0.001651
grad_step = 000317, loss = 0.001660
grad_step = 000318, loss = 0.001672
grad_step = 000319, loss = 0.001708
grad_step = 000320, loss = 0.001774
grad_step = 000321, loss = 0.001880
grad_step = 000322, loss = 0.002021
grad_step = 000323, loss = 0.002210
grad_step = 000324, loss = 0.002336
grad_step = 000325, loss = 0.002304
grad_step = 000326, loss = 0.002023
grad_step = 000327, loss = 0.001713
grad_step = 000328, loss = 0.001605
grad_step = 000329, loss = 0.001740
grad_step = 000330, loss = 0.001920
grad_step = 000331, loss = 0.001919
grad_step = 000332, loss = 0.001759
grad_step = 000333, loss = 0.001604
grad_step = 000334, loss = 0.001607
grad_step = 000335, loss = 0.001724
grad_step = 000336, loss = 0.001801
grad_step = 000337, loss = 0.001761
grad_step = 000338, loss = 0.001643
grad_step = 000339, loss = 0.001573
grad_step = 000340, loss = 0.001593
grad_step = 000341, loss = 0.001659
grad_step = 000342, loss = 0.001697
grad_step = 000343, loss = 0.001669
grad_step = 000344, loss = 0.001608
grad_step = 000345, loss = 0.001562
grad_step = 000346, loss = 0.001559
grad_step = 000347, loss = 0.001588
grad_step = 000348, loss = 0.001618
grad_step = 000349, loss = 0.001625
grad_step = 000350, loss = 0.001603
grad_step = 000351, loss = 0.001570
grad_step = 000352, loss = 0.001544
grad_step = 000353, loss = 0.001534
grad_step = 000354, loss = 0.001538
grad_step = 000355, loss = 0.001551
grad_step = 000356, loss = 0.001567
grad_step = 000357, loss = 0.001579
grad_step = 000358, loss = 0.001589
grad_step = 000359, loss = 0.001593
grad_step = 000360, loss = 0.001597
grad_step = 000361, loss = 0.001599
grad_step = 000362, loss = 0.001605
grad_step = 000363, loss = 0.001611
grad_step = 000364, loss = 0.001625
grad_step = 000365, loss = 0.001643
grad_step = 000366, loss = 0.001675
grad_step = 000367, loss = 0.001707
grad_step = 000368, loss = 0.001757
grad_step = 000369, loss = 0.001785
grad_step = 000370, loss = 0.001813
grad_step = 000371, loss = 0.001772
grad_step = 000372, loss = 0.001709
grad_step = 000373, loss = 0.001602
grad_step = 000374, loss = 0.001514
grad_step = 000375, loss = 0.001477
grad_step = 000376, loss = 0.001494
grad_step = 000377, loss = 0.001545
grad_step = 000378, loss = 0.001610
grad_step = 000379, loss = 0.001687
grad_step = 000380, loss = 0.001733
grad_step = 000381, loss = 0.001767
grad_step = 000382, loss = 0.001733
grad_step = 000383, loss = 0.001672
grad_step = 000384, loss = 0.001583
grad_step = 000385, loss = 0.001507
grad_step = 000386, loss = 0.001458
grad_step = 000387, loss = 0.001447
grad_step = 000388, loss = 0.001469
grad_step = 000389, loss = 0.001512
grad_step = 000390, loss = 0.001567
grad_step = 000391, loss = 0.001620
grad_step = 000392, loss = 0.001681
grad_step = 000393, loss = 0.001724
grad_step = 000394, loss = 0.001756
grad_step = 000395, loss = 0.001732
grad_step = 000396, loss = 0.001674
grad_step = 000397, loss = 0.001571
grad_step = 000398, loss = 0.001478
grad_step = 000399, loss = 0.001424
grad_step = 000400, loss = 0.001422
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001459
grad_step = 000402, loss = 0.001508
grad_step = 000403, loss = 0.001554
grad_step = 000404, loss = 0.001574
grad_step = 000405, loss = 0.001585
grad_step = 000406, loss = 0.001564
grad_step = 000407, loss = 0.001536
grad_step = 000408, loss = 0.001493
grad_step = 000409, loss = 0.001452
grad_step = 000410, loss = 0.001415
grad_step = 000411, loss = 0.001390
grad_step = 000412, loss = 0.001380
grad_step = 000413, loss = 0.001381
grad_step = 000414, loss = 0.001390
grad_step = 000415, loss = 0.001406
grad_step = 000416, loss = 0.001436
grad_step = 000417, loss = 0.001485
grad_step = 000418, loss = 0.001581
grad_step = 000419, loss = 0.001723
grad_step = 000420, loss = 0.001974
grad_step = 000421, loss = 0.002149
grad_step = 000422, loss = 0.002253
grad_step = 000423, loss = 0.001927
grad_step = 000424, loss = 0.001524
grad_step = 000425, loss = 0.001378
grad_step = 000426, loss = 0.001578
grad_step = 000427, loss = 0.001790
grad_step = 000428, loss = 0.001678
grad_step = 000429, loss = 0.001436
grad_step = 000430, loss = 0.001364
grad_step = 000431, loss = 0.001504
grad_step = 000432, loss = 0.001649
grad_step = 000433, loss = 0.001593
grad_step = 000434, loss = 0.001450
grad_step = 000435, loss = 0.001343
grad_step = 000436, loss = 0.001366
grad_step = 000437, loss = 0.001468
grad_step = 000438, loss = 0.001521
grad_step = 000439, loss = 0.001495
grad_step = 000440, loss = 0.001398
grad_step = 000441, loss = 0.001327
grad_step = 000442, loss = 0.001322
grad_step = 000443, loss = 0.001370
grad_step = 000444, loss = 0.001424
grad_step = 000445, loss = 0.001433
grad_step = 000446, loss = 0.001404
grad_step = 000447, loss = 0.001344
grad_step = 000448, loss = 0.001302
grad_step = 000449, loss = 0.001300
grad_step = 000450, loss = 0.001326
grad_step = 000451, loss = 0.001359
grad_step = 000452, loss = 0.001370
grad_step = 000453, loss = 0.001358
grad_step = 000454, loss = 0.001327
grad_step = 000455, loss = 0.001298
grad_step = 000456, loss = 0.001278
grad_step = 000457, loss = 0.001274
grad_step = 000458, loss = 0.001282
grad_step = 000459, loss = 0.001295
grad_step = 000460, loss = 0.001314
grad_step = 000461, loss = 0.001332
grad_step = 000462, loss = 0.001359
grad_step = 000463, loss = 0.001381
grad_step = 000464, loss = 0.001413
grad_step = 000465, loss = 0.001423
grad_step = 000466, loss = 0.001437
grad_step = 000467, loss = 0.001414
grad_step = 000468, loss = 0.001379
grad_step = 000469, loss = 0.001322
grad_step = 000470, loss = 0.001272
grad_step = 000471, loss = 0.001243
grad_step = 000472, loss = 0.001240
grad_step = 000473, loss = 0.001258
grad_step = 000474, loss = 0.001284
grad_step = 000475, loss = 0.001314
grad_step = 000476, loss = 0.001334
grad_step = 000477, loss = 0.001358
grad_step = 000478, loss = 0.001367
grad_step = 000479, loss = 0.001389
grad_step = 000480, loss = 0.001384
grad_step = 000481, loss = 0.001383
grad_step = 000482, loss = 0.001345
grad_step = 000483, loss = 0.001302
grad_step = 000484, loss = 0.001252
grad_step = 000485, loss = 0.001218
grad_step = 000486, loss = 0.001206
grad_step = 000487, loss = 0.001212
grad_step = 000488, loss = 0.001229
grad_step = 000489, loss = 0.001252
grad_step = 000490, loss = 0.001283
grad_step = 000491, loss = 0.001315
grad_step = 000492, loss = 0.001375
grad_step = 000493, loss = 0.001425
grad_step = 000494, loss = 0.001518
grad_step = 000495, loss = 0.001523
grad_step = 000496, loss = 0.001490
grad_step = 000497, loss = 0.001348
grad_step = 000498, loss = 0.001226
grad_step = 000499, loss = 0.001185
grad_step = 000500, loss = 0.001234
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001315
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

  date_run                              2020-05-15 05:13:55.546620
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.235165
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 05:13:55.551980
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.137971
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 05:13:55.558584
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.137907
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 05:13:55.562918
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.09652
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
0   2020-05-15 05:13:28.515104  ...    mean_absolute_error
1   2020-05-15 05:13:28.518604  ...     mean_squared_error
2   2020-05-15 05:13:28.521786  ...  median_absolute_error
3   2020-05-15 05:13:28.524971  ...               r2_score
4   2020-05-15 05:13:38.256923  ...    mean_absolute_error
5   2020-05-15 05:13:38.260230  ...     mean_squared_error
6   2020-05-15 05:13:38.262983  ...  median_absolute_error
7   2020-05-15 05:13:38.265451  ...               r2_score
8   2020-05-15 05:13:55.546620  ...    mean_absolute_error
9   2020-05-15 05:13:55.551980  ...     mean_squared_error
10  2020-05-15 05:13:55.558584  ...  median_absolute_error
11  2020-05-15 05:13:55.562918  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb3752d3cc0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 27%|██▋       | 2695168/9912422 [00:00<00:00, 26121591.75it/s]9920512it [00:00, 32473385.93it/s]                             
0it [00:00, ?it/s]32768it [00:00, 656440.53it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 157735.04it/s]1654784it [00:00, 11408628.22it/s]                         
0it [00:00, ?it/s]8192it [00:00, 224621.90it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb327c8de80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb3272be0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb324a4e4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb375296eb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb3752dea20> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb375296eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb327c8de80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb375296eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb324a4e4e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb3752daa20> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fbef476c1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=2a8cb22d743cf8f56b97020b4a068ea4ab4b8720067fb8da8d737ceadb89547e
  Stored in directory: /tmp/pip-ephem-wheel-cache-_yn2j8c0/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fbe8c567748> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 5s
  647168/17464789 [>.............................] - ETA: 1s
 2998272/17464789 [====>.........................] - ETA: 0s
 6184960/17464789 [=========>....................] - ETA: 0s
 9158656/17464789 [==============>...............] - ETA: 0s
12468224/17464789 [====================>.........] - ETA: 0s
14966784/17464789 [========================>.....] - ETA: 0s
17432576/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 05:15:19.952288: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 05:15:19.956924: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-15 05:15:19.957371: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559ef0b48e40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 05:15:19.957566: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 8.2493 - accuracy: 0.4620
 2000/25000 [=>............................] - ETA: 7s - loss: 7.9580 - accuracy: 0.4810 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.9784 - accuracy: 0.4797
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.8890 - accuracy: 0.4855
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.8506 - accuracy: 0.4880
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.8148 - accuracy: 0.4903
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.8024 - accuracy: 0.4911
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7950 - accuracy: 0.4916
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7791 - accuracy: 0.4927
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7448 - accuracy: 0.4949
11000/25000 [============>.................] - ETA: 3s - loss: 7.7280 - accuracy: 0.4960
12000/25000 [=============>................] - ETA: 3s - loss: 7.6922 - accuracy: 0.4983
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6961 - accuracy: 0.4981
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6644 - accuracy: 0.5001
15000/25000 [=================>............] - ETA: 2s - loss: 7.6390 - accuracy: 0.5018
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6455 - accuracy: 0.5014
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6522 - accuracy: 0.5009
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6586 - accuracy: 0.5005
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6528 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6513 - accuracy: 0.5010
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6562 - accuracy: 0.5007
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6720 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6832 - accuracy: 0.4989
25000/25000 [==============================] - 7s 261us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 05:15:32.486221
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 05:15:32.486221  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:36:25, 12.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:15:34, 18.1kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:20:09, 25.6kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:32:39, 36.6kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:34:12, 52.2kB/s].vector_cache/glove.6B.zip:   1%|          | 9.59M/862M [00:01<3:10:40, 74.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:01<2:13:03, 106kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.9M/862M [00:01<1:32:42, 152kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.7M/862M [00:01<1:04:43, 216kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.4M/862M [00:01<45:06, 308kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 32.7M/862M [00:01<31:26, 440kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.8M/862M [00:01<22:04, 624kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.6M/862M [00:02<15:26, 886kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.1M/862M [00:02<10:48, 1.26MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.3M/862M [00:02<07:40, 1.76MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:02<06:03, 2.23MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<06:08, 2.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<06:08, 2.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.3M/862M [00:04<04:45, 2.82MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:06<05:52, 2.28MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<05:45, 2.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:06<04:26, 3.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<05:48, 2.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<07:03, 1.88MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<05:40, 2.34MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:08<04:08, 3.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<23:39, 559kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<17:54, 738kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.3M/862M [00:10<12:51, 1.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:12<12:05, 1.09MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<09:48, 1.34MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:12<07:11, 1.83MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<08:06, 1.61MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<07:01, 1.86MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<05:14, 2.49MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<06:45, 1.93MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<06:04, 2.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:16<04:31, 2.87MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<06:14, 2.07MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<07:02, 1.84MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<05:28, 2.36MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:18<04:01, 3.20MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<07:42, 1.67MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<06:42, 1.92MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:20<05:01, 2.56MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<06:31, 1.97MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<05:54, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<04:24, 2.90MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<06:04, 2.10MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<05:34, 2.29MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:24<04:12, 3.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:56, 2.13MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:27, 2.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<04:08, 3.05MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<05:52, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:24, 2.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<04:06, 3.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:49, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:22, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:04, 3.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:47, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:20, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:03, 3.07MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<05:45, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:18, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:01, 3.07MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<05:43, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:16, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:00, 3.08MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<05:41, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:14, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<03:58, 3.08MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:40, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:20, 2.28MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:04, 2.99MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:27, 2.22MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<06:33, 1.85MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:12, 2.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<03:49, 3.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<06:55, 1.74MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<06:11, 1.95MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<04:37, 2.61MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:49, 2.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<06:54, 1.74MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:24, 2.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<03:59, 3.00MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<06:29, 1.84MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<05:54, 2.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<04:28, 2.66MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:40, 2.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<06:45, 1.75MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:25, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<03:57, 2.98MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<14:19, 823kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<11:22, 1.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<08:16, 1.42MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<08:17, 1.41MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<08:33, 1.37MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:34, 1.78MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<04:44, 2.46MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<08:25, 1.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<07:14, 1.61MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:23, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:15, 1.85MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:59, 1.65MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:27, 2.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:58, 2.90MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<07:11, 1.60MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<06:20, 1.81MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:45, 2.41MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:47, 1.97MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:36, 1.73MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<05:11, 2.20MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:47, 3.00MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:53, 1.65MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:06, 1.86MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<04:35, 2.47MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:37, 2.01MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:28, 1.75MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<05:05, 2.22MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<03:41, 3.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<08:35, 1.31MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<07:16, 1.54MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<05:24, 2.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:10, 1.81MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:57, 1.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:26, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:55, 2.83MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<11:07, 996kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<09:03, 1.22MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<06:38, 1.66MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:00, 1.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:10, 1.79MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<04:37, 2.38MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:35, 1.96MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:22, 1.72MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:59, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<03:38, 2.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:32, 1.66MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:49, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<04:19, 2.51MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:21, 2.02MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:13, 1.74MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:53, 2.21MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<03:34, 3.01MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:29, 1.66MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:44, 1.87MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<04:16, 2.51MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:17, 2.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:06, 1.75MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<04:47, 2.22MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<03:34, 2.98MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:16, 2.01MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:52, 2.17MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<03:42, 2.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:51, 2.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:52, 1.79MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:37, 2.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<03:23, 3.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:02, 1.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<05:25, 1.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<04:05, 2.55MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:05, 2.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:00, 1.73MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:41, 2.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<03:27, 3.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:44, 1.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:11, 1.99MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<03:52, 2.65MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:55, 2.09MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:51, 1.75MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<04:36, 2.22MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<03:20, 3.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:18, 1.39MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:16, 1.62MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<04:40, 2.17MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:26, 1.86MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:05, 1.66MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:50, 2.09MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:39<03:30, 2.86MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<12:20, 814kB/s] .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<09:47, 1.03MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<07:07, 1.41MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<07:07, 1.40MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<07:12, 1.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:32, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<03:59, 2.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<08:08, 1.22MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:47, 1.46MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<04:58, 1.98MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:38, 1.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<06:09, 1.60MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:46, 2.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:30, 2.79MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:28, 1.78MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:55, 1.98MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<03:42, 2.62MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<04:43, 2.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<05:34, 1.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:22, 2.21MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<03:10, 3.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:40, 1.44MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:46, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<04:15, 2.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:02, 1.90MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<04:36, 2.08MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<03:29, 2.74MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:28, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:13, 2.25MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<03:13, 2.94MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:16, 2.21MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:12, 1.81MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<04:05, 2.30MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<03:01, 3.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:05, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:38, 2.01MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<03:28, 2.69MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<04:25, 2.10MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:10, 1.79MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:05, 2.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<02:58, 3.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<07:05, 1.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:01, 1.53MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<04:25, 2.08MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:03, 1.81MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:35, 1.63MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:26, 2.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<03:13, 2.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<11:19, 801kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<08:53, 1.02MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<06:27, 1.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:30, 1.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:40, 1.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<05:11, 1.73MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<03:44, 2.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<11:23, 784kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<08:59, 992kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<06:32, 1.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<06:27, 1.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:32, 1.60MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<04:07, 2.14MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<04:46, 1.84MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:18, 1.66MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:08, 2.12MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<03:00, 2.90MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:40, 1.54MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:57, 1.76MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:42, 2.34MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:26, 1.95MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:02, 1.71MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:02, 2.14MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<02:55, 2.94MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<09:53, 868kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<07:53, 1.09MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<05:45, 1.49MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:50, 1.46MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:55, 1.44MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:35, 1.85MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<03:17, 2.56MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<26:38, 317kB/s] .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<19:33, 432kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<13:50, 608kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<11:31, 727kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<09:57, 842kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<07:26, 1.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<05:18, 1.57MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<11:52, 700kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<09:14, 899kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<06:41, 1.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<06:25, 1.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<06:26, 1.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:54, 1.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:33, 2.30MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<05:12, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:35, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:26, 2.37MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:06, 1.97MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:41, 1.73MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:41, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:42, 2.97MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:21, 1.85MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:55, 2.04MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<02:55, 2.74MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<03:49, 2.08MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:35, 2.22MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<02:43, 2.90MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<03:36, 2.19MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:22, 1.81MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:27, 2.28MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:31, 3.11MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:46, 1.64MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:14, 1.85MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:09, 2.47MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:51, 2.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:31, 1.71MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:36, 2.14MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:47<02:37, 2.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<09:22, 819kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<07:26, 1.03MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<05:24, 1.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:24, 1.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:29, 1.39MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:13, 1.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<03:02, 2.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<06:14, 1.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:14, 1.44MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:52, 1.94MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:18, 1.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:51, 1.93MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<02:54, 2.56MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<03:37, 2.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:16, 1.73MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:21, 2.20MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<02:26, 3.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<04:39, 1.57MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:06, 1.79MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:02, 2.40MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:41, 1.97MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:18, 1.69MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:21, 2.16MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:27, 2.93MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:59, 1.81MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:37, 1.99MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:41, 2.66MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:25, 2.09MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:04, 1.75MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:11, 2.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:20, 3.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<04:03, 1.74MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:39, 1.93MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:43, 2.59MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:24, 2.05MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:57, 1.77MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:06, 2.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<02:15, 3.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<04:39, 1.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:03, 1.70MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:01, 2.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:35, 1.91MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<04:02, 1.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:09, 2.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:19, 2.94MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:47, 1.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<03:22, 2.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:32, 2.66MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:15, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<03:43, 1.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:54, 2.30MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:07, 3.13MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:18<04:11, 1.59MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<03:41, 1.80MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:44, 2.41MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:19, 1.98MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:44, 1.76MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:57, 2.21MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<02:08, 3.03MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<21:01, 310kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<15:27, 421kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<10:56, 592kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<08:59, 716kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<07:40, 838kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<05:39, 1.14MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<04:03, 1.58MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:45, 1.34MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:03, 1.57MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<03:00, 2.11MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:27, 1.82MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:49, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<03:02, 2.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:11, 2.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<07:14, 861kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<05:43, 1.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<04:09, 1.49MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:16, 1.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:26, 1.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:24, 1.80MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:27, 2.48MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:03, 1.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:32, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:38, 2.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:07, 1.92MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:36, 1.67MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:49, 2.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<02:02, 2.93MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<04:26, 1.34MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:47, 1.57MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<02:48, 2.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:13, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:54, 2.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<02:12, 2.66MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:46, 2.09MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<03:11, 1.82MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:29, 2.32MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<01:50, 3.12MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<03:04, 1.87MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:48, 2.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:07, 2.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:41, 2.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:14, 1.75MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:33, 2.21MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<01:53, 2.98MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:55, 1.91MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:41, 2.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<02:02, 2.74MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:36, 2.12MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:03, 1.80MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:24, 2.29MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<01:45, 3.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:05, 1.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:48, 1.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<02:06, 2.58MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:37, 2.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<03:02, 1.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:24, 2.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:54<01:44, 3.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<04:45, 1.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:55, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:52, 1.85MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:07, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:25, 1.54MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:39, 1.98MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:58<01:54, 2.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<04:36, 1.13MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:48, 1.36MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<02:47, 1.85MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:02, 1.68MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:19, 1.54MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:34, 1.99MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<01:52, 2.72MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:06, 1.62MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:45, 1.83MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:03, 2.44MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:30, 1.99MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:17, 2.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:43, 2.87MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:18, 2.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:43, 1.81MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:08, 2.30MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<01:33, 3.11MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:43, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:26, 1.98MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:10<01:50, 2.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:18, 2.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:44, 1.74MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:10, 2.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:12<01:34, 2.98MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<05:43, 821kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<04:32, 1.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<03:16, 1.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<03:16, 1.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<03:22, 1.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:34, 1.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<01:52, 2.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:43, 1.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:25, 1.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:49, 2.49MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<02:13, 2.01MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<02:37, 1.72MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:03, 2.18MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<01:28, 3.00MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:18, 1.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:49, 1.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:03, 2.13MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:22, 1.83MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:38, 1.65MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:03, 2.11MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:24<01:29, 2.88MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:36, 1.64MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:19, 1.85MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:44, 2.45MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:06, 2.00MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:56, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:27, 2.88MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:54, 2.17MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:13, 1.86MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<01:44, 2.37MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<01:15, 3.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:42, 1.51MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:21, 1.73MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:45, 2.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:04, 1.92MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:21, 1.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:52, 2.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:34<01:20, 2.93MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<04:36, 855kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<03:40, 1.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<02:40, 1.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:41, 1.44MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:18, 1.67MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:43, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:00, 1.89MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:17, 1.65MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:47, 2.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<01:17, 2.89MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:16, 1.64MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:59, 1.88MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:28, 2.52MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<01:50, 1.99MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:06, 1.73MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:40, 2.17MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<01:12, 2.98MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<04:11, 858kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<03:20, 1.07MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:25, 1.47MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<02:26, 1.44MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:05, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:33, 2.24MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:49, 1.90MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:06, 1.64MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:38, 2.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:13, 2.78MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:37, 2.07MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:52<01:31, 2.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:09, 2.91MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:30, 2.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:50, 1.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:26, 2.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<01:03, 3.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:46, 1.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:36, 2.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:12, 2.65MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:31, 2.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:25, 2.22MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:04, 2.91MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:25, 2.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:20, 2.31MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:01, 3.02MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:21, 2.23MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:40, 1.81MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:19, 2.29MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<00:56, 3.14MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:22, 1.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:59, 1.48MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:27, 2.02MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:38, 1.77MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:49, 1.59MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:24, 2.04MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<01:01, 2.79MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:49, 1.55MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:35, 1.76MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:10, 2.38MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:24, 1.95MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:34, 1.75MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:13, 2.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<00:53, 3.05MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:40, 1.60MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:28, 1.82MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:12<01:05, 2.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:19, 1.98MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:30, 1.73MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:11, 2.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:50, 3.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<02:10, 1.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:48, 1.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:19, 1.90MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:26, 1.71MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:35, 1.55MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:13, 2.00MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<00:53, 2.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:27, 1.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:17, 1.87MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<00:57, 2.48MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<01:09, 2.01MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:21, 1.72MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:04, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<00:46, 2.95MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<02:38, 858kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<02:06, 1.07MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:31, 1.47MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:31, 1.45MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:19, 1.67MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:58, 2.23MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:07, 1.89MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:16, 1.68MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:00, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<00:42, 2.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<02:26, 844kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<01:56, 1.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:24, 1.45MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:23, 1.43MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:25, 1.40MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:06, 1.80MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<00:46, 2.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<02:26, 786kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:55, 996kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:23, 1.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:20, 1.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:22, 1.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:02, 1.76MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:44, 2.42MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<01:11, 1.50MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<01:01, 1.73MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:45, 2.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:53, 1.93MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:01, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:47, 2.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:35, 2.86MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:48, 2.05MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:44, 2.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:33, 2.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:43, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:51, 1.82MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:40, 2.31MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:29, 3.14MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:52, 1.72MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:46, 1.93MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:34, 2.55MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:42, 2.04MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:48, 1.79MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:37, 2.29MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:26, 3.14MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<01:03, 1.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:49<00:53, 1.53MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:39, 2.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:43, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:39, 1.98MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<00:29, 2.62MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:35, 2.07MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:41, 1.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:32, 2.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<00:23, 3.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:40, 1.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:36, 1.91MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<00:26, 2.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:32, 2.03MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:37, 1.76MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:28, 2.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:57<00:20, 3.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:37, 1.65MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:32, 1.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:59<00:24, 2.47MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:28, 2.01MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:33, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:26, 2.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:01<00:18, 2.94MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<01:04, 818kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:50, 1.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:35, 1.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:34, 1.40MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:35, 1.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:27, 1.75MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:05<00:18, 2.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:56, 785kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:44, 994kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:07<00:31, 1.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:29, 1.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:29, 1.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:22, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:09<00:15, 2.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:46, 786kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:36, 995kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:25, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:23, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:23, 1.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:17, 1.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:12, 2.44MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:17, 1.62MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:15, 1.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:10, 2.44MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 1.99MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:13, 1.71MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:10, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:07, 2.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.67MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:10, 1.88MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:07, 2.52MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 2.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:08, 1.72MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:06, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<00:04, 2.99MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:07, 1.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:06, 1.69MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:04, 2.26MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.13MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:25<00:01, 2.91MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.60MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.81MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 2.41MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 895/400000 [00:00<00:44, 8945.64it/s]  0%|          | 1831/400000 [00:00<00:43, 9064.90it/s]  1%|          | 2747/400000 [00:00<00:43, 9092.46it/s]  1%|          | 3684/400000 [00:00<00:43, 9172.85it/s]  1%|          | 4682/400000 [00:00<00:42, 9398.79it/s]  1%|▏         | 5627/400000 [00:00<00:41, 9412.43it/s]  2%|▏         | 6577/400000 [00:00<00:41, 9436.61it/s]  2%|▏         | 7511/400000 [00:00<00:41, 9399.34it/s]  2%|▏         | 8499/400000 [00:00<00:41, 9538.23it/s]  2%|▏         | 9500/400000 [00:01<00:40, 9673.20it/s]  3%|▎         | 10501/400000 [00:01<00:39, 9770.86it/s]  3%|▎         | 11462/400000 [00:01<00:40, 9684.76it/s]  3%|▎         | 12419/400000 [00:01<00:40, 9553.80it/s]  3%|▎         | 13367/400000 [00:01<00:41, 9414.43it/s]  4%|▎         | 14338/400000 [00:01<00:40, 9499.59it/s]  4%|▍         | 15350/400000 [00:01<00:39, 9675.26it/s]  4%|▍         | 16359/400000 [00:01<00:39, 9793.98it/s]  4%|▍         | 17338/400000 [00:01<00:40, 9446.25it/s]  5%|▍         | 18309/400000 [00:01<00:40, 9523.67it/s]  5%|▍         | 19300/400000 [00:02<00:39, 9635.20it/s]  5%|▌         | 20284/400000 [00:02<00:39, 9694.34it/s]  5%|▌         | 21320/400000 [00:02<00:38, 9884.01it/s]  6%|▌         | 22311/400000 [00:02<00:38, 9858.00it/s]  6%|▌         | 23299/400000 [00:02<00:38, 9820.33it/s]  6%|▌         | 24287/400000 [00:02<00:38, 9835.49it/s]  6%|▋         | 25272/400000 [00:02<00:38, 9822.96it/s]  7%|▋         | 26255/400000 [00:02<00:38, 9697.54it/s]  7%|▋         | 27226/400000 [00:02<00:38, 9643.64it/s]  7%|▋         | 28202/400000 [00:02<00:38, 9675.50it/s]  7%|▋         | 29210/400000 [00:03<00:37, 9790.70it/s]  8%|▊         | 30206/400000 [00:03<00:37, 9839.81it/s]  8%|▊         | 31191/400000 [00:03<00:38, 9685.64it/s]  8%|▊         | 32161/400000 [00:03<00:38, 9582.15it/s]  8%|▊         | 33121/400000 [00:03<00:38, 9471.35it/s]  9%|▊         | 34081/400000 [00:03<00:38, 9508.93it/s]  9%|▉         | 35097/400000 [00:03<00:37, 9694.24it/s]  9%|▉         | 36068/400000 [00:03<00:37, 9579.15it/s]  9%|▉         | 37028/400000 [00:03<00:38, 9418.48it/s] 10%|▉         | 38000/400000 [00:03<00:38, 9506.74it/s] 10%|▉         | 38983/400000 [00:04<00:37, 9600.71it/s] 10%|█         | 40019/400000 [00:04<00:36, 9815.03it/s] 10%|█         | 41033/400000 [00:04<00:36, 9909.29it/s] 11%|█         | 42073/400000 [00:04<00:35, 10049.09it/s] 11%|█         | 43080/400000 [00:04<00:35, 9982.04it/s]  11%|█         | 44080/400000 [00:04<00:35, 9970.16it/s] 11%|█▏        | 45078/400000 [00:04<00:36, 9849.01it/s] 12%|█▏        | 46080/400000 [00:04<00:35, 9899.52it/s] 12%|█▏        | 47096/400000 [00:04<00:35, 9970.75it/s] 12%|█▏        | 48094/400000 [00:04<00:35, 9959.45it/s] 12%|█▏        | 49091/400000 [00:05<00:35, 9842.22it/s] 13%|█▎        | 50076/400000 [00:05<00:35, 9742.43it/s] 13%|█▎        | 51075/400000 [00:05<00:35, 9814.86it/s] 13%|█▎        | 52069/400000 [00:05<00:35, 9851.02it/s] 13%|█▎        | 53055/400000 [00:05<00:35, 9724.27it/s] 14%|█▎        | 54079/400000 [00:05<00:35, 9872.99it/s] 14%|█▍        | 55068/400000 [00:05<00:35, 9732.57it/s] 14%|█▍        | 56043/400000 [00:05<00:36, 9519.26it/s] 14%|█▍        | 56997/400000 [00:05<00:36, 9355.44it/s] 14%|█▍        | 57981/400000 [00:05<00:36, 9495.34it/s] 15%|█▍        | 58996/400000 [00:06<00:35, 9681.45it/s] 15%|█▌        | 60008/400000 [00:06<00:34, 9808.89it/s] 15%|█▌        | 61020/400000 [00:06<00:34, 9898.45it/s] 16%|█▌        | 62015/400000 [00:06<00:34, 9910.82it/s] 16%|█▌        | 63008/400000 [00:06<00:34, 9636.97it/s] 16%|█▌        | 64008/400000 [00:06<00:34, 9741.07it/s] 16%|█▌        | 64985/400000 [00:06<00:34, 9676.69it/s] 16%|█▋        | 65955/400000 [00:06<00:35, 9411.79it/s] 17%|█▋        | 66937/400000 [00:06<00:34, 9528.60it/s] 17%|█▋        | 67922/400000 [00:07<00:34, 9622.69it/s] 17%|█▋        | 68887/400000 [00:07<00:34, 9585.83it/s] 17%|█▋        | 69868/400000 [00:07<00:34, 9649.81it/s] 18%|█▊        | 70881/400000 [00:07<00:33, 9788.60it/s] 18%|█▊        | 71862/400000 [00:07<00:33, 9748.55it/s] 18%|█▊        | 72838/400000 [00:07<00:33, 9659.25it/s] 18%|█▊        | 73826/400000 [00:07<00:33, 9724.25it/s] 19%|█▊        | 74800/400000 [00:07<00:33, 9726.25it/s] 19%|█▉        | 75774/400000 [00:07<00:33, 9589.79it/s] 19%|█▉        | 76743/400000 [00:07<00:33, 9617.39it/s] 19%|█▉        | 77729/400000 [00:08<00:33, 9688.22it/s] 20%|█▉        | 78709/400000 [00:08<00:33, 9720.95it/s] 20%|█▉        | 79682/400000 [00:08<00:33, 9682.42it/s] 20%|██        | 80651/400000 [00:08<00:33, 9596.25it/s] 20%|██        | 81612/400000 [00:08<00:33, 9585.48it/s] 21%|██        | 82571/400000 [00:08<00:34, 9156.01it/s] 21%|██        | 83587/400000 [00:08<00:33, 9433.75it/s] 21%|██        | 84580/400000 [00:08<00:32, 9575.40it/s] 21%|██▏       | 85557/400000 [00:08<00:32, 9632.79it/s] 22%|██▏       | 86556/400000 [00:08<00:32, 9736.45it/s] 22%|██▏       | 87546/400000 [00:09<00:31, 9782.62it/s] 22%|██▏       | 88587/400000 [00:09<00:31, 9959.93it/s] 22%|██▏       | 89597/400000 [00:09<00:31, 10000.50it/s] 23%|██▎       | 90604/400000 [00:09<00:30, 10020.88it/s] 23%|██▎       | 91614/400000 [00:09<00:30, 10041.55it/s] 23%|██▎       | 92619/400000 [00:09<00:32, 9563.71it/s]  23%|██▎       | 93581/400000 [00:09<00:32, 9320.57it/s] 24%|██▎       | 94568/400000 [00:09<00:32, 9478.50it/s] 24%|██▍       | 95521/400000 [00:09<00:32, 9353.42it/s] 24%|██▍       | 96460/400000 [00:09<00:32, 9324.97it/s] 24%|██▍       | 97472/400000 [00:10<00:31, 9547.76it/s] 25%|██▍       | 98476/400000 [00:10<00:31, 9689.16it/s] 25%|██▍       | 99461/400000 [00:10<00:30, 9736.25it/s] 25%|██▌       | 100437/400000 [00:10<00:31, 9560.85it/s] 25%|██▌       | 101415/400000 [00:10<00:31, 9623.31it/s] 26%|██▌       | 102379/400000 [00:10<00:31, 9551.12it/s] 26%|██▌       | 103336/400000 [00:10<00:31, 9539.38it/s] 26%|██▌       | 104312/400000 [00:10<00:30, 9604.35it/s] 26%|██▋       | 105288/400000 [00:10<00:30, 9650.49it/s] 27%|██▋       | 106254/400000 [00:10<00:30, 9580.91it/s] 27%|██▋       | 107213/400000 [00:11<00:30, 9573.00it/s] 27%|██▋       | 108210/400000 [00:11<00:30, 9686.86it/s] 27%|██▋       | 109226/400000 [00:11<00:29, 9822.49it/s] 28%|██▊       | 110210/400000 [00:11<00:29, 9797.01it/s] 28%|██▊       | 111191/400000 [00:11<00:29, 9694.97it/s] 28%|██▊       | 112162/400000 [00:11<00:30, 9587.23it/s] 28%|██▊       | 113168/400000 [00:11<00:29, 9723.87it/s] 29%|██▊       | 114142/400000 [00:11<00:29, 9640.71it/s] 29%|██▉       | 115143/400000 [00:11<00:29, 9747.35it/s] 29%|██▉       | 116157/400000 [00:12<00:28, 9859.70it/s] 29%|██▉       | 117154/400000 [00:12<00:28, 9891.85it/s] 30%|██▉       | 118204/400000 [00:12<00:28, 10063.80it/s] 30%|██▉       | 119212/400000 [00:12<00:28, 9995.90it/s]  30%|███       | 120213/400000 [00:12<00:28, 9895.54it/s] 30%|███       | 121204/400000 [00:12<00:28, 9780.64it/s] 31%|███       | 122200/400000 [00:12<00:28, 9833.23it/s] 31%|███       | 123185/400000 [00:12<00:28, 9837.21it/s] 31%|███       | 124179/400000 [00:12<00:27, 9866.61it/s] 31%|███▏      | 125167/400000 [00:12<00:27, 9861.17it/s] 32%|███▏      | 126158/400000 [00:13<00:27, 9873.81it/s] 32%|███▏      | 127166/400000 [00:13<00:27, 9933.91it/s] 32%|███▏      | 128160/400000 [00:13<00:27, 9807.19it/s] 32%|███▏      | 129159/400000 [00:13<00:27, 9860.39it/s] 33%|███▎      | 130199/400000 [00:13<00:26, 10014.93it/s] 33%|███▎      | 131202/400000 [00:13<00:27, 9796.39it/s]  33%|███▎      | 132232/400000 [00:13<00:26, 9939.77it/s] 33%|███▎      | 133228/400000 [00:13<00:27, 9834.65it/s] 34%|███▎      | 134213/400000 [00:13<00:27, 9772.59it/s] 34%|███▍      | 135223/400000 [00:13<00:26, 9868.13it/s] 34%|███▍      | 136211/400000 [00:14<00:27, 9736.04it/s] 34%|███▍      | 137186/400000 [00:14<00:27, 9728.83it/s] 35%|███▍      | 138160/400000 [00:14<00:26, 9725.16it/s] 35%|███▍      | 139203/400000 [00:14<00:26, 9925.42it/s] 35%|███▌      | 140251/400000 [00:14<00:25, 10083.74it/s] 35%|███▌      | 141261/400000 [00:14<00:25, 10045.33it/s] 36%|███▌      | 142267/400000 [00:14<00:25, 9941.35it/s]  36%|███▌      | 143263/400000 [00:14<00:26, 9710.72it/s] 36%|███▌      | 144255/400000 [00:14<00:26, 9772.47it/s] 36%|███▋      | 145250/400000 [00:14<00:25, 9823.20it/s] 37%|███▋      | 146234/400000 [00:15<00:26, 9728.79it/s] 37%|███▋      | 147208/400000 [00:15<00:25, 9728.91it/s] 37%|███▋      | 148247/400000 [00:15<00:25, 9915.91it/s] 37%|███▋      | 149240/400000 [00:15<00:25, 9900.81it/s] 38%|███▊      | 150241/400000 [00:15<00:25, 9931.41it/s] 38%|███▊      | 151250/400000 [00:15<00:24, 9977.72it/s] 38%|███▊      | 152249/400000 [00:15<00:25, 9858.71it/s] 38%|███▊      | 153236/400000 [00:15<00:25, 9659.87it/s] 39%|███▊      | 154246/400000 [00:15<00:25, 9787.17it/s] 39%|███▉      | 155259/400000 [00:15<00:24, 9885.69it/s] 39%|███▉      | 156249/400000 [00:16<00:25, 9643.27it/s] 39%|███▉      | 157216/400000 [00:16<00:25, 9535.09it/s] 40%|███▉      | 158172/400000 [00:16<00:26, 9243.84it/s] 40%|███▉      | 159100/400000 [00:16<00:26, 9041.38it/s] 40%|████      | 160021/400000 [00:16<00:26, 9088.85it/s] 40%|████      | 160948/400000 [00:16<00:26, 9142.34it/s] 40%|████      | 161987/400000 [00:16<00:25, 9481.72it/s] 41%|████      | 162940/400000 [00:16<00:25, 9466.44it/s] 41%|████      | 163914/400000 [00:16<00:24, 9546.24it/s] 41%|████      | 164975/400000 [00:17<00:23, 9840.10it/s] 41%|████▏     | 165963/400000 [00:17<00:24, 9722.48it/s] 42%|████▏     | 166999/400000 [00:17<00:23, 9902.09it/s] 42%|████▏     | 168003/400000 [00:17<00:23, 9941.45it/s] 42%|████▏     | 169000/400000 [00:17<00:24, 9543.42it/s] 43%|████▎     | 170023/400000 [00:17<00:23, 9738.76it/s] 43%|████▎     | 171008/400000 [00:17<00:23, 9770.63it/s] 43%|████▎     | 172055/400000 [00:17<00:22, 9969.64it/s] 43%|████▎     | 173056/400000 [00:17<00:23, 9812.75it/s] 44%|████▎     | 174041/400000 [00:17<00:23, 9779.69it/s] 44%|████▍     | 175021/400000 [00:18<00:23, 9741.59it/s] 44%|████▍     | 175997/400000 [00:18<00:23, 9654.56it/s] 44%|████▍     | 176964/400000 [00:18<00:23, 9522.38it/s] 44%|████▍     | 177990/400000 [00:18<00:22, 9731.61it/s] 45%|████▍     | 178975/400000 [00:18<00:22, 9765.21it/s] 45%|████▍     | 179961/400000 [00:18<00:22, 9791.22it/s] 45%|████▌     | 180942/400000 [00:18<00:22, 9711.78it/s] 45%|████▌     | 181915/400000 [00:18<00:22, 9622.98it/s] 46%|████▌     | 182882/400000 [00:18<00:22, 9636.74it/s] 46%|████▌     | 183886/400000 [00:18<00:22, 9752.85it/s] 46%|████▌     | 184863/400000 [00:19<00:22, 9727.11it/s] 46%|████▋     | 185837/400000 [00:19<00:22, 9539.44it/s] 47%|████▋     | 186814/400000 [00:19<00:22, 9604.82it/s] 47%|████▋     | 187776/400000 [00:19<00:22, 9566.01it/s] 47%|████▋     | 188734/400000 [00:19<00:22, 9514.73it/s] 47%|████▋     | 189687/400000 [00:19<00:22, 9385.91it/s] 48%|████▊     | 190627/400000 [00:19<00:22, 9160.84it/s] 48%|████▊     | 191545/400000 [00:19<00:22, 9143.74it/s] 48%|████▊     | 192461/400000 [00:19<00:22, 9117.44it/s] 48%|████▊     | 193374/400000 [00:19<00:22, 9075.89it/s] 49%|████▊     | 194296/400000 [00:20<00:22, 9117.50it/s] 49%|████▉     | 195209/400000 [00:20<00:22, 9102.05it/s] 49%|████▉     | 196120/400000 [00:20<00:22, 9000.46it/s] 49%|████▉     | 197029/400000 [00:20<00:22, 9024.51it/s] 49%|████▉     | 197932/400000 [00:20<00:22, 8997.07it/s] 50%|████▉     | 198832/400000 [00:20<00:22, 8972.14it/s] 50%|████▉     | 199730/400000 [00:20<00:22, 8953.77it/s] 50%|█████     | 200629/400000 [00:20<00:22, 8963.07it/s] 50%|█████     | 201526/400000 [00:20<00:22, 8834.40it/s] 51%|█████     | 202441/400000 [00:20<00:22, 8926.67it/s] 51%|█████     | 203385/400000 [00:21<00:21, 9072.50it/s] 51%|█████     | 204310/400000 [00:21<00:21, 9124.19it/s] 51%|█████▏    | 205224/400000 [00:21<00:21, 9094.34it/s] 52%|█████▏    | 206134/400000 [00:21<00:21, 9095.91it/s] 52%|█████▏    | 207044/400000 [00:21<00:21, 9072.68it/s] 52%|█████▏    | 207952/400000 [00:21<00:21, 8957.28it/s] 52%|█████▏    | 208849/400000 [00:21<00:21, 8838.33it/s] 52%|█████▏    | 209734/400000 [00:21<00:21, 8796.69it/s] 53%|█████▎    | 210661/400000 [00:21<00:21, 8931.20it/s] 53%|█████▎    | 211579/400000 [00:21<00:20, 9003.55it/s] 53%|█████▎    | 212498/400000 [00:22<00:20, 9056.28it/s] 53%|█████▎    | 213405/400000 [00:22<00:20, 9060.05it/s] 54%|█████▎    | 214312/400000 [00:22<00:20, 9053.25it/s] 54%|█████▍    | 215243/400000 [00:22<00:20, 9126.02it/s] 54%|█████▍    | 216184/400000 [00:22<00:19, 9206.98it/s] 54%|█████▍    | 217106/400000 [00:22<00:19, 9205.87it/s] 55%|█████▍    | 218027/400000 [00:22<00:19, 9155.63it/s] 55%|█████▍    | 218943/400000 [00:22<00:20, 8943.74it/s] 55%|█████▍    | 219839/400000 [00:22<00:20, 8899.83it/s] 55%|█████▌    | 220730/400000 [00:23<00:20, 8881.98it/s] 55%|█████▌    | 221635/400000 [00:23<00:19, 8930.67it/s] 56%|█████▌    | 222536/400000 [00:23<00:19, 8954.03it/s] 56%|█████▌    | 223432/400000 [00:23<00:19, 8873.12it/s] 56%|█████▌    | 224408/400000 [00:23<00:19, 9120.73it/s] 56%|█████▋    | 225323/400000 [00:23<00:19, 8950.92it/s] 57%|█████▋    | 226323/400000 [00:23<00:18, 9240.41it/s] 57%|█████▋    | 227317/400000 [00:23<00:18, 9437.64it/s] 57%|█████▋    | 228315/400000 [00:23<00:17, 9593.53it/s] 57%|█████▋    | 229340/400000 [00:23<00:17, 9779.28it/s] 58%|█████▊    | 230342/400000 [00:24<00:17, 9847.58it/s] 58%|█████▊    | 231330/400000 [00:24<00:17, 9704.17it/s] 58%|█████▊    | 232323/400000 [00:24<00:17, 9768.97it/s] 58%|█████▊    | 233303/400000 [00:24<00:17, 9778.15it/s] 59%|█████▊    | 234282/400000 [00:24<00:17, 9683.85it/s] 59%|█████▉    | 235252/400000 [00:24<00:17, 9590.29it/s] 59%|█████▉    | 236230/400000 [00:24<00:16, 9645.90it/s] 59%|█████▉    | 237196/400000 [00:24<00:16, 9591.08it/s] 60%|█████▉    | 238165/400000 [00:24<00:16, 9618.01it/s] 60%|█████▉    | 239129/400000 [00:24<00:16, 9623.95it/s] 60%|██████    | 240102/400000 [00:25<00:16, 9655.16it/s] 60%|██████    | 241068/400000 [00:25<00:16, 9592.64it/s] 61%|██████    | 242028/400000 [00:25<00:16, 9580.23it/s] 61%|██████    | 243016/400000 [00:25<00:16, 9667.47it/s] 61%|██████    | 244023/400000 [00:25<00:15, 9783.29it/s] 61%|██████▏   | 245055/400000 [00:25<00:15, 9937.88it/s] 62%|██████▏   | 246087/400000 [00:25<00:15, 10048.52it/s] 62%|██████▏   | 247094/400000 [00:25<00:15, 10053.16it/s] 62%|██████▏   | 248101/400000 [00:25<00:15, 9977.00it/s]  62%|██████▏   | 249100/400000 [00:25<00:15, 9549.16it/s] 63%|██████▎   | 250060/400000 [00:26<00:15, 9450.69it/s] 63%|██████▎   | 251009/400000 [00:26<00:15, 9402.55it/s] 63%|██████▎   | 251952/400000 [00:26<00:15, 9362.95it/s] 63%|██████▎   | 252890/400000 [00:26<00:15, 9359.83it/s] 63%|██████▎   | 253832/400000 [00:26<00:15, 9376.37it/s] 64%|██████▎   | 254775/400000 [00:26<00:15, 9390.45it/s] 64%|██████▍   | 255793/400000 [00:26<00:15, 9612.24it/s] 64%|██████▍   | 256760/400000 [00:26<00:14, 9627.12it/s] 64%|██████▍   | 257751/400000 [00:26<00:14, 9708.48it/s] 65%|██████▍   | 258760/400000 [00:26<00:14, 9818.95it/s] 65%|██████▍   | 259754/400000 [00:27<00:14, 9853.99it/s] 65%|██████▌   | 260741/400000 [00:27<00:14, 9856.44it/s] 65%|██████▌   | 261736/400000 [00:27<00:13, 9884.28it/s] 66%|██████▌   | 262748/400000 [00:27<00:13, 9952.79it/s] 66%|██████▌   | 263744/400000 [00:27<00:14, 9486.46it/s] 66%|██████▌   | 264759/400000 [00:27<00:13, 9674.32it/s] 66%|██████▋   | 265741/400000 [00:27<00:13, 9715.50it/s] 67%|██████▋   | 266718/400000 [00:27<00:13, 9729.57it/s] 67%|██████▋   | 267734/400000 [00:27<00:13, 9852.85it/s] 67%|██████▋   | 268733/400000 [00:27<00:13, 9891.57it/s] 67%|██████▋   | 269724/400000 [00:28<00:13, 9846.51it/s] 68%|██████▊   | 270730/400000 [00:28<00:13, 9907.98it/s] 68%|██████▊   | 271722/400000 [00:28<00:13, 9759.12it/s] 68%|██████▊   | 272699/400000 [00:28<00:13, 9732.54it/s] 68%|██████▊   | 273674/400000 [00:28<00:13, 9708.86it/s] 69%|██████▊   | 274646/400000 [00:28<00:13, 9529.66it/s] 69%|██████▉   | 275627/400000 [00:28<00:12, 9609.71it/s] 69%|██████▉   | 276589/400000 [00:28<00:12, 9611.60it/s] 69%|██████▉   | 277551/400000 [00:28<00:12, 9493.51it/s] 70%|██████▉   | 278585/400000 [00:29<00:12, 9731.88it/s] 70%|██████▉   | 279561/400000 [00:29<00:12, 9612.28it/s] 70%|███████   | 280524/400000 [00:29<00:12, 9532.31it/s] 70%|███████   | 281479/400000 [00:29<00:12, 9531.08it/s] 71%|███████   | 282434/400000 [00:29<00:12, 9527.15it/s] 71%|███████   | 283388/400000 [00:29<00:12, 9499.51it/s] 71%|███████   | 284396/400000 [00:29<00:11, 9664.30it/s] 71%|███████▏  | 285381/400000 [00:29<00:11, 9717.12it/s] 72%|███████▏  | 286357/400000 [00:29<00:11, 9729.15it/s] 72%|███████▏  | 287331/400000 [00:29<00:11, 9661.57it/s] 72%|███████▏  | 288298/400000 [00:30<00:11, 9659.82it/s] 72%|███████▏  | 289265/400000 [00:30<00:11, 9648.88it/s] 73%|███████▎  | 290276/400000 [00:30<00:11, 9782.13it/s] 73%|███████▎  | 291255/400000 [00:30<00:11, 9728.61it/s] 73%|███████▎  | 292229/400000 [00:30<00:11, 9497.25it/s] 73%|███████▎  | 293223/400000 [00:30<00:11, 9623.92it/s] 74%|███████▎  | 294240/400000 [00:30<00:10, 9779.20it/s] 74%|███████▍  | 295220/400000 [00:30<00:10, 9749.13it/s] 74%|███████▍  | 296197/400000 [00:30<00:10, 9652.41it/s] 74%|███████▍  | 297164/400000 [00:30<00:10, 9465.02it/s] 75%|███████▍  | 298159/400000 [00:31<00:10, 9604.36it/s] 75%|███████▍  | 299122/400000 [00:31<00:10, 9477.87it/s] 75%|███████▌  | 300119/400000 [00:31<00:10, 9619.50it/s] 75%|███████▌  | 301083/400000 [00:31<00:10, 9615.66it/s] 76%|███████▌  | 302046/400000 [00:31<00:10, 9435.79it/s] 76%|███████▌  | 303018/400000 [00:31<00:10, 9519.27it/s] 76%|███████▌  | 303972/400000 [00:31<00:10, 9494.84it/s] 76%|███████▌  | 304923/400000 [00:31<00:10, 9208.14it/s] 76%|███████▋  | 305898/400000 [00:31<00:10, 9363.31it/s] 77%|███████▋  | 306866/400000 [00:31<00:09, 9455.01it/s] 77%|███████▋  | 307864/400000 [00:32<00:09, 9606.22it/s] 77%|███████▋  | 308860/400000 [00:32<00:09, 9708.28it/s] 77%|███████▋  | 309833/400000 [00:32<00:09, 9487.09it/s] 78%|███████▊  | 310784/400000 [00:32<00:09, 9137.30it/s] 78%|███████▊  | 311725/400000 [00:32<00:09, 9215.94it/s] 78%|███████▊  | 312713/400000 [00:32<00:09, 9404.84it/s] 78%|███████▊  | 313703/400000 [00:32<00:09, 9546.43it/s] 79%|███████▊  | 314690/400000 [00:32<00:08, 9641.23it/s] 79%|███████▉  | 315662/400000 [00:32<00:08, 9663.48it/s] 79%|███████▉  | 316630/400000 [00:32<00:08, 9322.93it/s] 79%|███████▉  | 317674/400000 [00:33<00:08, 9629.64it/s] 80%|███████▉  | 318681/400000 [00:33<00:08, 9754.59it/s] 80%|███████▉  | 319668/400000 [00:33<00:08, 9786.46it/s] 80%|████████  | 320705/400000 [00:33<00:07, 9953.31it/s] 80%|████████  | 321703/400000 [00:33<00:08, 9715.28it/s] 81%|████████  | 322678/400000 [00:33<00:08, 9626.82it/s] 81%|████████  | 323644/400000 [00:33<00:08, 9459.05it/s] 81%|████████  | 324593/400000 [00:33<00:08, 9331.81it/s] 81%|████████▏ | 325529/400000 [00:33<00:08, 9175.91it/s] 82%|████████▏ | 326482/400000 [00:34<00:07, 9278.26it/s] 82%|████████▏ | 327434/400000 [00:34<00:07, 9349.17it/s] 82%|████████▏ | 328435/400000 [00:34<00:07, 9535.03it/s] 82%|████████▏ | 329473/400000 [00:34<00:07, 9772.53it/s] 83%|████████▎ | 330512/400000 [00:34<00:06, 9947.45it/s] 83%|████████▎ | 331510/400000 [00:34<00:06, 9802.72it/s] 83%|████████▎ | 332548/400000 [00:34<00:06, 9967.08it/s] 83%|████████▎ | 333548/400000 [00:34<00:06, 9608.18it/s] 84%|████████▎ | 334585/400000 [00:34<00:06, 9822.75it/s] 84%|████████▍ | 335572/400000 [00:34<00:06, 9553.60it/s] 84%|████████▍ | 336591/400000 [00:35<00:06, 9733.27it/s] 84%|████████▍ | 337621/400000 [00:35<00:06, 9894.21it/s] 85%|████████▍ | 338641/400000 [00:35<00:06, 9982.11it/s] 85%|████████▍ | 339654/400000 [00:35<00:06, 10025.86it/s] 85%|████████▌ | 340659/400000 [00:35<00:06, 9731.30it/s]  85%|████████▌ | 341636/400000 [00:35<00:06, 9708.63it/s] 86%|████████▌ | 342610/400000 [00:35<00:06, 9512.83it/s] 86%|████████▌ | 343564/400000 [00:35<00:05, 9514.20it/s] 86%|████████▌ | 344537/400000 [00:35<00:05, 9577.21it/s] 86%|████████▋ | 345497/400000 [00:35<00:05, 9271.21it/s] 87%|████████▋ | 346481/400000 [00:36<00:05, 9434.71it/s] 87%|████████▋ | 347428/400000 [00:36<00:05, 9272.56it/s] 87%|████████▋ | 348374/400000 [00:36<00:05, 9326.25it/s] 87%|████████▋ | 349401/400000 [00:36<00:05, 9588.86it/s] 88%|████████▊ | 350380/400000 [00:36<00:05, 9647.85it/s] 88%|████████▊ | 351390/400000 [00:36<00:04, 9778.25it/s] 88%|████████▊ | 352395/400000 [00:36<00:04, 9857.84it/s] 88%|████████▊ | 353383/400000 [00:36<00:04, 9751.93it/s] 89%|████████▊ | 354360/400000 [00:36<00:04, 9650.53it/s] 89%|████████▉ | 355327/400000 [00:37<00:04, 9373.88it/s] 89%|████████▉ | 356277/400000 [00:37<00:04, 9410.62it/s] 89%|████████▉ | 357235/400000 [00:37<00:04, 9460.35it/s] 90%|████████▉ | 358183/400000 [00:37<00:04, 9229.19it/s] 90%|████████▉ | 359118/400000 [00:37<00:04, 9263.38it/s] 90%|█████████ | 360046/400000 [00:37<00:04, 9143.62it/s] 90%|█████████ | 361003/400000 [00:37<00:04, 9265.62it/s] 91%|█████████ | 362030/400000 [00:37<00:03, 9545.21it/s] 91%|█████████ | 363012/400000 [00:37<00:03, 9625.96it/s] 91%|█████████ | 364009/400000 [00:37<00:03, 9724.97it/s] 91%|█████████▏| 365017/400000 [00:38<00:03, 9828.60it/s] 92%|█████████▏| 366060/400000 [00:38<00:03, 10000.70it/s] 92%|█████████▏| 367062/400000 [00:38<00:03, 9870.52it/s]  92%|█████████▏| 368051/400000 [00:38<00:03, 9791.99it/s] 92%|█████████▏| 369069/400000 [00:38<00:03, 9904.99it/s] 93%|█████████▎| 370061/400000 [00:38<00:03, 9572.56it/s] 93%|█████████▎| 371022/400000 [00:38<00:03, 9492.64it/s] 93%|█████████▎| 372012/400000 [00:38<00:02, 9609.18it/s] 93%|█████████▎| 372975/400000 [00:38<00:02, 9598.34it/s] 93%|█████████▎| 373937/400000 [00:38<00:02, 9307.16it/s] 94%|█████████▎| 374871/400000 [00:39<00:02, 9255.01it/s] 94%|█████████▍| 375884/400000 [00:39<00:02, 9499.11it/s] 94%|█████████▍| 376856/400000 [00:39<00:02, 9562.70it/s] 94%|█████████▍| 377815/400000 [00:39<00:02, 9558.55it/s] 95%|█████████▍| 378857/400000 [00:39<00:02, 9800.64it/s] 95%|█████████▍| 379840/400000 [00:39<00:02, 9577.08it/s] 95%|█████████▌| 380801/400000 [00:39<00:02, 9395.33it/s] 95%|█████████▌| 381754/400000 [00:39<00:01, 9434.51it/s] 96%|█████████▌| 382746/400000 [00:39<00:01, 9574.13it/s] 96%|█████████▌| 383771/400000 [00:39<00:01, 9766.89it/s] 96%|█████████▌| 384750/400000 [00:40<00:01, 9740.48it/s] 96%|█████████▋| 385744/400000 [00:40<00:01, 9796.77it/s] 97%|█████████▋| 386725/400000 [00:40<00:01, 9597.08it/s] 97%|█████████▋| 387687/400000 [00:40<00:01, 9409.97it/s] 97%|█████████▋| 388631/400000 [00:40<00:01, 9240.32it/s] 97%|█████████▋| 389558/400000 [00:40<00:01, 9164.92it/s] 98%|█████████▊| 390533/400000 [00:40<00:01, 9332.71it/s] 98%|█████████▊| 391529/400000 [00:40<00:00, 9512.02it/s] 98%|█████████▊| 392527/400000 [00:40<00:00, 9644.52it/s] 98%|█████████▊| 393508/400000 [00:40<00:00, 9691.26it/s] 99%|█████████▊| 394479/400000 [00:41<00:00, 9692.72it/s] 99%|█████████▉| 395450/400000 [00:41<00:00, 9672.18it/s] 99%|█████████▉| 396418/400000 [00:41<00:00, 9505.37it/s] 99%|█████████▉| 397384/400000 [00:41<00:00, 9551.01it/s]100%|█████████▉| 398348/400000 [00:41<00:00, 9576.12it/s]100%|█████████▉| 399307/400000 [00:41<00:00, 9380.47it/s]100%|█████████▉| 399999/400000 [00:41<00:00, 9595.90it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7ff431895940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010983674652416137 	 Accuracy: 55
Train Epoch: 1 	 Loss: 0.011081745592646774 	 Accuracy: 51

  model saves at 51% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 16047 out of table with 15890 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 16047 out of table with 15890 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-15 05:24:23.010311: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 05:24:23.015776: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-15 05:24:23.015999: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ea5f91e7f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 05:24:23.016014: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ff3e4dc8160> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.5133 - accuracy: 0.5100
 2000/25000 [=>............................] - ETA: 7s - loss: 7.4290 - accuracy: 0.5155 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5184 - accuracy: 0.5097
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.5248 - accuracy: 0.5092
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5378 - accuracy: 0.5084
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5235 - accuracy: 0.5093
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5702 - accuracy: 0.5063
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.5823 - accuracy: 0.5055
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6070 - accuracy: 0.5039
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6268 - accuracy: 0.5026
11000/25000 [============>.................] - ETA: 3s - loss: 7.6262 - accuracy: 0.5026
12000/25000 [=============>................] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6690 - accuracy: 0.4998
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6776 - accuracy: 0.4993
15000/25000 [=================>............] - ETA: 2s - loss: 7.6687 - accuracy: 0.4999
16000/25000 [==================>...........] - ETA: 1s - loss: 7.6791 - accuracy: 0.4992
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6865 - accuracy: 0.4987
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7050 - accuracy: 0.4975
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7013 - accuracy: 0.4977
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7042 - accuracy: 0.4976
21000/25000 [========================>.....] - ETA: 0s - loss: 7.7017 - accuracy: 0.4977
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6980 - accuracy: 0.4980
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6886 - accuracy: 0.4986
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6775 - accuracy: 0.4993
25000/25000 [==============================] - 6s 255us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7ff3ad80af60> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7ff3e52ebf98> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4511 - crf_viterbi_accuracy: 0.4667 - val_loss: 1.4191 - val_crf_viterbi_accuracy: 0.4933

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
